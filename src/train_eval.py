"""
FINAL TRAINING SCRIPT FOR NGSIM DATASET
---------------------------------------
Compatible with:
  - ./Dataset/NGSIM/US101_cleaned.csv
  - models_improved.py (ImprovedTrajectoryTransformer)
  - ngsim_dataloader.py
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import random
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

from models import ImprovedTrajectoryTransformer
from ngsim_dataloader import make_dataloader_ngsim as make_dataloader
from utils import ade_fde


SEED = 42
def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
set_seed()

if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using Apple Metal (MPS) acceleration")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using NVIDIA CUDA GPU")
else:
    device = torch.device("cpu")
    print("Using CPU (no GPU acceleration available)")


def trajectory_loss_improved(pred, gt, lambda_vel=0.5, lambda_acc=0.2, lambda_diversity=0.1):
    """Position + velocity + acceleration + diversity loss"""
    pos_loss = nn.functional.mse_loss(pred, gt)

    if pred.shape[1] > 1:
        pred_vel = pred[:, 1:, :] - pred[:, :-1, :]
        gt_vel = gt[:, 1:, :] - gt[:, :-1, :]
        vel_loss = nn.functional.mse_loss(pred_vel, gt_vel)
    else:
        vel_loss = torch.tensor(0.0, device=pred.device)

    if pred.shape[1] > 2:
        pred_acc = pred_vel[:, 1:, :] - pred_vel[:, :-1, :]
        gt_acc = gt_vel[:, 1:, :] - gt_vel[:, :-1, :]
        acc_loss = nn.functional.mse_loss(pred_acc, gt_acc)
    else:
        acc_loss = torch.tensor(0.0, device=pred.device)

    pred_std = pred.std(dim=0).mean()
    gt_std = gt.std(dim=0).mean()
    diversity_loss = torch.abs(pred_std - gt_std)

    total_loss = (
        pos_loss
        + lambda_vel * vel_loss
        + lambda_acc * acc_loss
        + lambda_diversity * diversity_loss
    )

    return total_loss, {
        'pos': pos_loss.item(),
        'vel': vel_loss.item(),
        'acc': acc_loss.item(),
        'div': diversity_loss.item()
    }


def plot_sample_improved(obs, gt, pred, save_path, title="Prediction"):
    plt.figure(figsize=(6, 5))
    plt.plot(obs[:, 0], obs[:, 1], 'bo-', label='Observed', markersize=3)
    plt.plot(gt[:, 0], gt[:, 1], 'g-', label='Ground Truth', linewidth=2)
    plt.plot(pred[:, 0], pred[:, 1], 'r--', label='Predicted', linewidth=2)
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def train_model(csv_path, save_dir='./results_NGSIM/Checkpoints'):
    os.makedirs(save_dir, exist_ok=True)

    curriculum = [
        (10, 5e-4, 2),
        (15, 3e-4, 4),
        (25, 2e-4, 6)
    ]
    
    obs_len = 10
    batch_size = 32
    k_neighbors = 8

    model = ImprovedTrajectoryTransformer(
        d_model=256, nhead=8, num_layers=4, pred_len=25, k_neighbors=k_neighbors
    ).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    best_ade = float('inf')

    # ----------------------------------------------------------------------
    for stage, (pred_len, lr, epochs) in enumerate(curriculum):
        # print(f"\n{'='*60}")
        print(f"Stage {stage+1}: pred_len={pred_len}, lr={lr:.0e}, epochs={epochs}")
        # print(f"{'='*60}")
        
        for pg in optimizer.param_groups:
            pg['lr'] = lr

        model.pred_len = pred_len
        loader = make_dataloader(csv_path, batch_size=batch_size,
                                 obs_len=obs_len, pred_len=pred_len)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=lr/10
        )

        for epoch in range(1, epochs + 1):
            model.train()
            epoch_loss = 0.0
            pbar = tqdm(loader, desc=f"Epoch {epoch}/{epochs}")

            for batch in pbar:
                obs = batch["target"].to(device)              # (B, T, 7)
                fut = batch["gt"].to(device)                 # (B, T_pred, 2)
                neigh_dyn = batch["neighbors_dyn"].to(device)  # (B, K, T, 7)
                neigh_spatial = batch["neighbors_spatial"].to(device)
                lane = batch["lane"].to(device)  # (B, 1)

                optimizer.zero_grad()
                pred = model(obs, neigh_dyn, neigh_spatial, lane,
                             last_obs_pos=None, pred_len=pred_len)

                loss, _ = trajectory_loss_improved(pred, fut)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                epoch_loss += loss.item()
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            
            scheduler.step()
            avg_loss = epoch_loss / len(loader)
            print(f"Epoch {epoch} avg_loss={avg_loss:.6f}")

            if epoch % 2 == 0 or epoch == epochs:
                model.eval()
                preds_all, gts_all = [], []
                with torch.no_grad():
                    for i, batch in enumerate(loader):
                        obs = batch["target"].to(device)
                        fut = batch["gt"].to(device)
                        neigh_dyn = batch["neighbors_dyn"].to(device)
                        neigh_spatial = batch["neighbors_spatial"].to(device)
                        lane = batch["lane"].to(device)
                        
                        pred = model(obs, neigh_dyn, neigh_spatial, lane, pred_len=pred_len)
                        preds_all.append(pred.cpu().numpy())
                        gts_all.append(fut.cpu().numpy())
                        if i >= 100:
                            break
                            
                preds_all = np.concatenate(preds_all)
                gts_all = np.concatenate(gts_all)
                ADE, FDE = ade_fde(preds_all, gts_all)
                print(f"  Eval: ADE={ADE:.3f}m  FDE={FDE:.3f}m")

                if ADE < best_ade:
                    best_ade = ADE
                    torch.save(model.state_dict(), os.path.join(save_dir, 'best_model.pt'))
                    print(f"New best ADE: {best_ade:.3f}m")

                # Plot one representative sample - FIX: Don't add extra batch dimension
                try:
                    # Get a fresh batch for plotting
                    plot_loader = make_dataloader(csv_path, batch_size=1,
                                                  obs_len=obs_len, pred_len=pred_len, 
                                                  shuffle=False)
                    sample = next(iter(plot_loader))
                    
                    with torch.no_grad():
                        obs_s = sample["target"].to(device)       # (1, T_obs, 7)
                        neigh_dyn_s = sample["neighbors_dyn"].to(device)  # (1, K, T_obs, 7)
                        neigh_spatial_s = sample["neighbors_spatial"].to(device)
                        lane_s = sample["lane"].to(device)        # (1, 1)
                        
                        pred_s = model(obs_s, neigh_dyn_s, neigh_spatial_s, lane_s, pred_len=pred_len)
                        
                        # Convert to numpy (remove batch dimension)
                        obs_np = obs_s[0].cpu().numpy()  # (T_obs, 7)
                        fut_np = sample["gt"][0].cpu().numpy()  # (T_pred, 2)
                        pred_np = pred_s[0].cpu().numpy()  # (T_pred, 2)

                    plot_sample_improved(
                        obs_np[:, :2],  # Only x, y coordinates
                        fut_np,
                        pred_np,
                        save_path=os.path.join(save_dir, f'stage{stage+1}_epoch{epoch:02d}.png'),
                        title=f"Stage {stage+1} Epoch {epoch} (ADE={ADE:.2f})"
                    )
                except Exception as e:
                    print(f"Warning: Could not generate plot: {e}")

        torch.save(model.state_dict(), os.path.join(save_dir, f'model_stage{stage+1}.pt'))

    return model


if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print("Usage: python train_ngsim_final.py path/to/US101_cleaned.csv")
        sys.exit(1)
    
    csv_path = sys.argv[1]
    print(f"Loading CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows, {df['Vehicle_ID'].nunique()} unique vehicles.")
    
    model = train_model(csv_path, save_dir='./results_NGSIM/Checkpoints')
    print("\n Training complete for NGSIM dataset!")