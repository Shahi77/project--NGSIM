"""
train_eval.py - training loop with curriculum, improved loss, checkpoints,
and evaluation hooks. Uses make_dataloader_ngsim from ngsim_dataloader.py
"""

import os
import random
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from ngsim_dataloader import make_dataloader_ngsim
from models import ImprovedTrajectoryTransformer
from utils import combined_loss, ade_fde, torch_ade_fde

SEED = 42
def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
set_seed()

# device selection including MPS
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using Apple Metal (MPS) acceleration")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using CUDA")
else:
    device = torch.device("cpu")
    print("Using CPU")

def plot_sample(obs_xy, gt_xy, pred_xy, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.figure(figsize=(6,5))
    plt.plot(obs_xy[:,0], obs_xy[:,1], 'bo-', label='Observed')
    plt.plot(gt_xy[:,0], gt_xy[:,1], 'g-', label='GT')
    plt.plot(pred_xy[:,0], pred_xy[:,1], 'r--', label='Pred')
    plt.legend(); plt.axis('equal'); plt.grid(True)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()

def train_model(csv_path, save_dir='./results_NGSIM_new/Checkpoints'):
    os.makedirs(save_dir, exist_ok=True)
    # curriculum: (pred_len, lr, epochs)
    curriculum = [
        (10, 5e-4, 3),
        (15, 3e-4, 4),
        (25, 2e-4, 6)
    ]

    obs_len = 10
    batch_size = 32
    k_neighbors = 8

    model = ImprovedTrajectoryTransformer(d_model=384, nhead=12, num_layers=4,
                                          pred_len=25, k_neighbors=k_neighbors, dropout=0.1).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    best_ade = float('inf')

    for stage, (pred_len, lr, epochs) in enumerate(curriculum):
        print(f"\nStage {stage+1}: pred_len={pred_len}, lr={lr}, epochs={epochs}")
        for pg in optimizer.param_groups:
            pg['lr'] = lr

        model.pred_len = pred_len
        loader = make_dataloader_ngsim(csv_path, batch_size=batch_size, obs_len=obs_len,
                                       pred_len=pred_len, stride=5, k_neighbors=k_neighbors,
                                       augment=True, shuffle=True)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr/10)

        for epoch in range(1, epochs + 1):
            model.train()
            total_loss = 0.0
            pbar = tqdm(loader, desc=f"Epoch {epoch}/{epochs}")
            for batch in pbar:
                obs = batch['target'].to(device)                # (B, T_obs, 7)
                gt = batch['gt'].to(device)                     # (B, T_pred, 2)
                neigh_dyn = batch['neighbors_dyn'].to(device)   # (B, K, T_obs, 7)
                neigh_spatial = batch['neighbors_spatial'].to(device)
                lane = batch['lane'].to(device)                 # (B,1) or (B,1)

                optimizer.zero_grad()
                pred = model(obs, neigh_dyn, neigh_spatial, lane, pred_len=pred_len)
                loss, details = combined_loss(pred, gt,
                                              w_pos=1.0, w_vel=0.6, w_acc=0.2, w_ang=0.6, w_horizon=0.6)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                total_loss += loss.item()
                pbar.set_postfix({'loss': f"{loss.item():.4f}", 'pos': f"{details['pos']:.4f}", 'ang': f"{details['ang']:.4f}"})

            scheduler.step()
            avg_loss = total_loss / len(loader)
            print(f"Epoch {epoch} avg_loss={avg_loss:.6f}")

            # Evaluation checkpoint
            if epoch % 2 == 0 or epoch == epochs:
                model.eval()
                preds_all, gts_all = [], []
                with torch.no_grad():
                    for i, batch in enumerate(loader):
                        obs = batch['target'].to(device)
                        gt = batch['gt'].to(device)
                        neigh_dyn = batch['neighbors_dyn'].to(device)
                        neigh_spatial = batch['neighbors_spatial'].to(device)
                        lane = batch['lane'].to(device)

                        pred = model(obs, neigh_dyn, neigh_spatial, lane, pred_len=pred_len)
                        preds_all.append(pred.cpu().numpy())
                        gts_all.append(gt.cpu().numpy())
                        if i >= 100:
                            break

                preds_all = np.concatenate(preds_all, axis=0)
                gts_all = np.concatenate(gts_all, axis=0)
                ADE, FDE = ade_fde(preds_all, gts_all)
                print(f"  Eval (n={len(preds_all)}): ADE={ADE:.3f}m FDE={FDE:.3f}m")

                if ADE < best_ade:
                    best_ade = ADE
                    torch.save(model.state_dict(), os.path.join(save_dir, 'best_model.pt'))
                    print(f"  New best ADE: {best_ade:.4f}m")

                # plot one sample (use a separate batch_size=1 loader to avoid shape issues)
                try:
                    plot_loader = make_dataloader_ngsim(csv_path, batch_size=1, obs_len=obs_len, pred_len=pred_len, shuffle=False, augment=False, k_neighbors=k_neighbors)
                    sample = next(iter(plot_loader))
                    obs_s = sample['target'][0].numpy()[:, :2]
                    gt_s = sample['gt'][0].numpy()
                    neigh_dyn_s = sample['neighbors_dyn'][0:1]
                    with torch.no_grad():
                        pred_s = model(sample['target'].to(device),
                                       sample['neighbors_dyn'].to(device),
                                       sample['neighbors_spatial'].to(device),
                                       sample['lane'].to(device),
                                       pred_len=pred_len)[0].cpu().numpy()
                    plot_sample(obs_s, gt_s, pred_s, os.path.join(save_dir, f'stage{stage+1}_epoch{epoch:02d}.png'))
                except Exception as e:
                    print(f"  Plot warning: {e}")

        # end stage
        torch.save(model.state_dict(), os.path.join(save_dir, f'model_stage{stage+1}.pt'))

    return model

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python train_eval.py path/to/US101_cleaned.csv")
        sys.exit(1)
    csv_path = sys.argv[1]
    print("Loading CSV:", csv_path)
    df = None
    # quick check
    import pandas as pd
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows, {df['Vehicle_ID'].nunique()} unique vehicles.")
    model = train_model(csv_path, save_dir='./results_NGSIM_new/Checkpoints')
    print("Training finished.")
