import os, json, random, numpy as np, pandas as pd, torch, torch.nn as nn
from tqdm import tqdm
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

from models import SimpleSocialLSTM, ImprovedTrajectoryTransformer
from ngsim_dataloader import NGSIMDataset
from utils import ade_fde, combined_loss, save_json
from evaluate import compute_comprehensive_metrics, plot_error_distribution, plot_sample_predictions

# ----------------- Setup -----------------
SEED = 42
def set_seed(seed=SEED):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
set_seed()

device = torch.device("cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

def train_val_split(df, ratio=0.8):
    vids = df["Vehicle_ID"].unique(); np.random.shuffle(vids)
    n = int(len(vids) * ratio)
    return df[df.Vehicle_ID.isin(vids[:n])], df[df.Vehicle_ID.isin(vids[n:])]

def exponential_scheduled_sampling(p_init=1.0, decay=0.9, step=1):
    return max(0.05, p_init * (decay ** step))

def make_dataloader_from_df(df, batch_size=32, obs_len=20, pred_len=25, shuffle=True):
    dataset = NGSIMDataset(
        df, obs_len=obs_len, pred_len=pred_len,
        filter_smooth=False   #  disable over-strict filter
    )
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=True)


# ----------------- Evaluation -----------------
def evaluate(model, loader, name="val"):
    model.eval(); preds, gts = [], []
    with torch.no_grad():
        for b in loader:
            obs=b["target"].to(device); fut=b["gt"].to(device)
            nd=b["neighbors_dyn"].to(device); ns=b["neighbors_spatial"].to(device); lane=b["lane"].to(device)
            pred,_=model(obs,nd,ns,lane) if hasattr(model,"multi_att") else (model(obs,nd,ns,lane),None)
            preds.append(pred.cpu().numpy()); gts.append(fut.cpu().numpy())
    preds,gts=np.concatenate(preds),np.concatenate(gts)
    ADE,FDE=ade_fde(preds,gts)
    print(f"{name} ADE={ADE:.3f}  FDE={FDE:.3f}")
    return ADE,FDE,preds,gts

def plot_training_curves(history, save_dir):
    plt.figure(figsize=(7,4))
    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["val_ADE"], label="Val ADE")
    plt.xlabel("Epoch"); plt.ylabel("Metric"); plt.legend(); plt.grid(True)
    plt.tight_layout(); plt.savefig(os.path.join(save_dir,"training_curves.png"),dpi=150); plt.close()

# ----------------- Training -----------------
def train_ngsim(csv_path, save_dir="./results_ngsim", model_type="transformer", train_ratio=0.8):
    os.makedirs(save_dir,exist_ok=True)
    df=pd.read_csv(csv_path)
    df_train, df_val = train_val_split(df,ratio=train_ratio)

    obs_len, pred_len, batch_size = 15, 20, 32

    # Build full dataset once, then split samples directly
    full_dataset = NGSIMDataset(
    df, obs_len=obs_len, pred_len=pred_len,
    filter_smooth=False  # disable over-strict filtering
    )
    print(f"Loaded NGSIMDataset with {len(full_dataset)} valid samples.")

    num_samples = len(full_dataset)
    if num_samples == 0:
        raise RuntimeError("No valid NGSIM samples found! Try smaller obs/pred lengths or disable filter_smooth.")

    train_size = int(train_ratio * num_samples)
    val_size   = num_samples - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader   = torch.utils.data.DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, drop_last=True)


    model = SimpleSocialLSTM(pred_len=pred_len) if model_type=="slstm" else ImprovedTrajectoryTransformer(pred_len=pred_len)
    model.to(device)

    opt=torch.optim.AdamW(model.parameters(),lr=5e-4,weight_decay=5e-5)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=10,eta_min=1e-5)
    best_ade=float("inf"); history={"train_loss":[],"val_ADE":[]}; patience,wait=5,0

    print(f"\n Training {model_type.upper()} on NGSIM ({len(df_train)} train / {len(df_val)} val rows)")

    for epoch in range(1,11):
        model.train(); epoch_loss=0.0
        alpha = exponential_scheduled_sampling(1.0,0.9,epoch)
        warmup_alpha = max(0.0, 1.0 - epoch / 5.0)  # 1→0 over first 5 epochs

        for b in tqdm(train_loader, desc=f"Epoch {epoch}/10"):
            obs=b["target"].to(device); fut=b["gt"].to(device)
            nd=b["neighbors_dyn"].to(device); ns=b["neighbors_spatial"].to(device); lane=b["lane"].to(device)
            opt.zero_grad()

            # forward with CV warmup (transformer only)
            if model_type == "transformer":
                pred, _ = model(obs, nd, ns, lane, cv_warmup_alpha=warmup_alpha)
            else:
                pred, _ = model(obs, nd, ns, lane), None

            # scheduled sampling blend
            pred = alpha * fut + (1 - alpha) * pred

            loss, details = combined_loss(pred, fut, w_pos=1.0, w_vel=0.3, w_acc=0.1)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); epoch_loss += loss.item()

        scheduler.step()

        # Evaluate periodically
        if epoch % 2 == 0 or epoch == 1:
            model.eval()
            val_epoch_loss = 0.0  # Track total validation loss
            with torch.no_grad():
                for b in val_loader:
                    obs = b["target"].to(device)
                    fut = b["gt"].to(device)
                    nd  = b["neighbors_dyn"].to(device)
                    ns  = b["neighbors_spatial"].to(device)
                    lane = b["lane"].to(device)

                    if model_type == "transformer":
                        pred, _ = model(obs, nd, ns, lane, cv_warmup_alpha=0.0)
                    else:
                        pred, _ = model(obs, nd, ns, lane), None

                    loss, _ = combined_loss(pred, fut, w_pos=1.0, w_vel=0.3, w_acc=0.1)
                    val_epoch_loss += loss.item()

            val_loss = val_epoch_loss / max(1, len(val_loader))  # Mean Val (Test) loss

            train_ADE, train_FDE, _, _ = evaluate(model, train_loader, "Train")
            val_ADE, val_FDE, _, _     = evaluate(model, val_loader, "Val")
            history["train_loss"].append(epoch_loss / len(train_loader))
            history.setdefault("val_loss", []).append(val_loss)  #  Save it
            history["val_ADE"].append(val_ADE)

            print(f"Epoch {epoch}: "
                f"TrainLoss={epoch_loss/len(train_loader):.4f}, "
                f"ValLoss={val_loss:.4f}, "
                f"TrainADE={train_ADE:.3f}, "
                f"ValADE={val_ADE:.3f}")

            if val_ADE < best_ade:
                best_ade, wait = val_ADE, 0
                torch.save(model.state_dict(), os.path.join(save_dir, f"best_{model_type}.pt"))
            else:
                wait += 1
            if wait >= patience:
                print("⏸ Early stopping triggered."); break
        
        if epoch % 5 == 0 or epoch == 10:
            obs_np = obs[0, :, :2].cpu().numpy()
            gt_np  = fut[0].cpu().numpy()
            pred_np = pred[0].detach().cpu().numpy()
            plt.figure(figsize=(6,5))
            plt.plot(obs_np[:,0], obs_np[:,1], 'ko-', label='Observed')
            plt.plot(gt_np[:,0], gt_np[:,1], 'g-', label='Ground Truth')
            plt.plot(pred_np[:,0], pred_np[:,1], 'r--', label='Predicted')
            plt.legend(); plt.grid(True); plt.axis('equal')
            plt.savefig(os.path.join(save_dir, f"checkpoint_epoch{epoch}.png"), dpi=150)
            plt.close()        

    plot_training_curves(history, save_dir)
    save_json(history, os.path.join(save_dir, "training_history.json"))
    print(f" Training complete. Best Val ADE={best_ade:.3f}")

    # Final Evaluation
    model.load_state_dict(torch.load(os.path.join(save_dir,f"best_{model_type}.pt"),map_location=device))
    model.eval(); _,_,preds,gts = evaluate(model,val_loader,"Final Val")

    metrics = compute_comprehensive_metrics(preds,gts)
    save_json(metrics, os.path.join(save_dir, "metrics.json"))
    plot_error_distribution(preds, gts, save_dir)
    plot_sample_predictions(preds, gts, [np.zeros((obs_len,2))]*len(preds), save_dir)
    print(" All evaluation artifacts saved to:", save_dir)
    return model, metrics

if __name__=="__main__":
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument("tracks_csv")
    parser.add_argument("--model_type",choices=["transformer","slstm"],default="transformer")
    parser.add_argument("--save_dir",default="./results_ngsim")
    parser.add_argument("--train_ratio",type=float,default=0.8)
    args=parser.parse_args()
    train_ngsim(args.tracks_csv,save_dir=args.save_dir,model_type=args.model_type,train_ratio=args.train_ratio)
