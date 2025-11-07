"""
Training and evaluation for NGSIM with road-aligned normalization and world-frame metrics.
"""

import os
import json
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

from models import ImprovedTrajectoryTransformer
from ngsim_dataloader import make_dataloader_ngsim as make_dataloader
from utils import ade_fde  # expect ade_fde(preds, gts) -> floats in meters


def compute_comprehensive_metrics(preds, gts):
    """
    Compute comprehensive metrics beyond ADE/FDE
    preds, gts: (N, T, 2) numpy arrays
    """
    assert preds.shape == gts.shape
    N, T, _ = preds.shape

    dists = np.linalg.norm(preds - gts, axis=-1)
    ADE = dists.mean()
    FDE = dists[:, -1].mean()

    mae_x = np.abs(preds[:, :, 0] - gts[:, :, 0]).mean()
    mae_y = np.abs(preds[:, :, 1] - gts[:, :, 1]).mean()
    MAE = (mae_x + mae_y) / 2

    mse = ((preds - gts) ** 2).mean()
    RMSE = np.sqrt(mse)

    pred_vel = np.diff(preds, axis=1)
    gt_vel = np.diff(gts, axis=1)
    vel_error = np.linalg.norm(pred_vel - gt_vel, axis=-1).mean()

    if T > 2:
        pred_acc = np.diff(pred_vel, axis=1)
        gt_acc = np.diff(gt_vel, axis=1)
        acc_error = np.linalg.norm(pred_acc - gt_acc, axis=-1).mean()
    else:
        acc_error = 0.0

    pred_angles = np.arctan2(pred_vel[:, :, 1], pred_vel[:, :, 0])
    gt_angles = np.arctan2(gt_vel[:, :, 1], gt_vel[:, :, 0])
    angle_diff = np.abs(pred_angles - gt_angles)
    angle_diff = np.minimum(angle_diff, 2*np.pi - angle_diff)
    direction_error = np.degrees(angle_diff.mean())

    miss_rate = (dists[:, -1] > 2.0).mean() * 100

    lon_error = np.abs(preds[:, :, 0] - gts[:, :, 0]).mean()
    lat_error = np.abs(preds[:, :, 1] - gts[:, :, 1]).mean()

    per_step_error = dists.mean(axis=0)

    pred_variance = preds.var(axis=0).mean()
    gt_variance = gts.var(axis=0).mean()
    variance_ratio = pred_variance / (gt_variance + 1e-8)

    return {
        'ADE': float(ADE),
        'FDE': float(FDE),
        'MAE': float(MAE),
        'RMSE': float(RMSE),
        'Velocity_Error': float(vel_error),
        'Acceleration_Error': float(acc_error),
        'Direction_Error_deg': float(direction_error),
        'Miss_Rate_%': float(miss_rate),
        'Longitudinal_Error': float(lon_error),
        'Lateral_Error': float(lat_error),
        'Variance_Ratio': float(variance_ratio),
        'Per_Step_Error': per_step_error.tolist()
    }


def plot_error_distribution(preds, gts, save_dir='eval_results'):
    os.makedirs(save_dir, exist_ok=True)
    dists = np.linalg.norm(preds - gts, axis=-1)
    final_errors = dists[:, -1]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes[0, 0].hist(final_errors, bins=50, edgecolor='black', alpha=0.7, color='skyblue')
    axes[0, 0].axvline(final_errors.mean(), color='r', linestyle='--', label=f'Mean: {final_errors.mean():.2f}m')
    axes[0, 0].axvline(np.median(final_errors), color='g', linestyle='--', label=f'Median: {np.median(final_errors):.2f}m')
    axes[0, 0].set_title('FDE Distribution'); axes[0, 0].legend(); axes[0, 0].grid(True)

    per_step_error = dists.mean(axis=0)
    timesteps = np.arange(len(per_step_error))
    axes[0, 1].plot(timesteps, per_step_error, 'b-o', linewidth=2)
    axes[0, 1].fill_between(timesteps, np.percentile(dists, 25, axis=0),
                            np.percentile(dists, 75, axis=0), alpha=0.3, color='blue')
    axes[0, 1].set_title('Error Evolution Over Time'); axes[0, 1].grid(True)

    lon_errors = np.abs(preds[:, :, 0] - gts[:, :, 0]).flatten()
    lat_errors = np.abs(preds[:, :, 1] - gts[:, :, 1]).flatten()
    axes[1, 0].hexbin(lon_errors, lat_errors, gridsize=50, cmap='YlOrRd', mincnt=1)
    axes[1, 0].set_title('Longitudinal vs Lateral Error'); axes[1, 0].grid(True)

    pred_std = preds.std(axis=0)
    gt_std = gts.std(axis=0)
    axes[1, 1].plot(timesteps, pred_std[:, 0], 'r-', label='Pred X')
    axes[1, 1].plot(timesteps, gt_std[:, 0], 'b-', label='GT X')
    axes[1, 1].plot(timesteps, pred_std[:, 1], 'r--', label='Pred Y')
    axes[1, 1].plot(timesteps, gt_std[:, 1], 'b--', label='GT Y')
    axes[1, 1].set_title('Variance Check'); axes[1, 1].legend(); axes[1, 1].grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'error_analysis.png'), dpi=150)
    plt.close()
    print(f"Saved error analysis to {save_dir}/error_analysis.png")


def plot_sample_predictions(preds, gts, obs_list, save_dir='eval_results', n_samples=12):
    os.makedirs(save_dir, exist_ok=True)
    final_errors = np.linalg.norm(preds[:, -1, :] - gts[:, -1, :], axis=-1)
    n_per_cat = max(1, n_samples // 3)
    worst_idx = np.argsort(final_errors)[-n_per_cat:]
    median_start = len(final_errors) // 2 - n_per_cat // 2
    median_idx = np.argsort(final_errors)[max(0, median_start):max(0, median_start) + n_per_cat]
    best_idx = np.argsort(final_errors)[:n_per_cat]

    rows = 3
    cols = max(1, n_per_cat)
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 12))
    if rows == 1 or cols == 1:
        axes = np.atleast_2d(axes)

    categories = [('Worst', worst_idx, 0), ('Median', median_idx, 1), ('Best', best_idx, 2)]
    for cat_name, indices, row in categories:
        for col, idx in enumerate(indices):
            ax = axes[row, col]
            obs = obs_list[idx][:, :2]
            gt = gts[idx]
            pred = preds[idx]
            error = final_errors[idx]
            ax.plot(obs[:, 0], obs[:, 1], 'ko-', label='Observed')
            ax.plot(gt[:, 0], gt[:, 1], 'g-', label='Ground Truth')
            ax.plot(pred[:, 0], pred[:, 1], 'r--', label='Predicted')
            ax.set_title(f'{cat_name} FDE={error:.2f}m'); ax.legend(); ax.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'sample_predictions.png'), dpi=150)
    plt.close()
    print(f"Saved sample predictions to {save_dir}/sample_predictions.png")


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
    print("Using Apple Metal (MPS)")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using NVIDIA CUDA")
else:
    device = torch.device("cpu")
    print("Using CPU")


def to_world(pred_xy, fut_xy, origin, heading):
    """
    pred_xy: (B,T,2) normalized
    fut_xy:  (B,T,2) normalized
    origin:  (B,2)
    heading: (B,)
    returns preds_w, futs_w in world frame
    """
    B, T, _ = pred_xy.shape
    cos = np.cos(heading).reshape(B, 1, 1)
    sin = np.sin(heading).reshape(B, 1, 1)
    R_inv = np.stack(
        [np.stack([cos, -sin], axis=-1), np.stack([sin, cos], axis=-1)],
        axis=-2
    )  # (B,1,2,2)

    def inv(x):
        x = x[..., None]             # (B,T,2,1)
        xr = (R_inv @ x).squeeze(-1) # (B,T,2)
        return xr + origin.reshape(B, 1, 2)

    return inv(pred_xy), inv(fut_xy)


def trajectory_loss(pred, gt, lambda_vel=0.5, lambda_acc=0.2, lambda_lateral=5.0,
                    cv=None, lambda_motion_prior=0.2):
    lon = nn.functional.mse_loss(pred[:, :, 0], gt[:, :, 0])
    lat = nn.functional.mse_loss(pred[:, :, 1], gt[:, :, 1])
    pos = lon + lambda_lateral * lat

    if pred.size(1) > 1:
        pred_vel = pred[:, 1:, :] - pred[:, :-1, :]
        gt_vel = gt[:, 1:, :] - gt[:, :-1, :]
        vel = nn.functional.mse_loss(pred_vel, gt_vel)
    else:
        vel = torch.tensor(0.0, device=pred.device)

    if pred.size(1) > 2:
        pred_acc = pred_vel[:, 1:, :] - pred_vel[:, :-1, :]
        gt_acc = gt_vel[:, 1:, :] - gt_vel[:, :-1, :]
        acc = nn.functional.mse_loss(pred_acc, gt_acc)
    else:
        acc = torch.tensor(0.0, device=pred.device)

    motion_prior = torch.tensor(0.0, device=pred.device)
    if cv is not None:
        motion_prior = nn.functional.mse_loss(pred, cv)

    total = pos + lambda_vel * vel + lambda_acc * acc + lambda_motion_prior * motion_prior
    return total, {"pos": pos.item(), "lon": lon.item(), "lat": lat.item(),
                   "vel": float(vel), "acc": float(acc), "mp": float(motion_prior)}


def plot_sample(obs_xy, fut_xy, pred_xy, save_path, title):
    plt.figure(figsize=(6, 5))
    plt.plot(obs_xy[:, 0], obs_xy[:, 1], 'bo-', label='Observed', markersize=3)
    plt.plot(fut_xy[:, 0], fut_xy[:, 1], 'g-', label='Ground Truth', linewidth=2)
    plt.plot(pred_xy[:, 0], pred_xy[:, 1], 'r--', label='Predicted', linewidth=2)
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def train_model(csv_path, save_dir='./results_final/Checkpoints'):
    os.makedirs(save_dir, exist_ok=True)

    curriculum = [
        (5, 7e-4, 4),
        (10, 5e-4, 10),
        (15, 3e-4, 14),
        (20, 2e-4, 20)
    ]

    obs_len = 20
    batch_size = 32
    k_neighbors = 8

    model = ImprovedTrajectoryTransformer(
        d_model=256, nhead=8, num_layers=4, pred_len=25, k_neighbors=k_neighbors, dt=0.1
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=5e-5)
    best_ade = float('inf')
    best_pred_len = None

    for stage, (pred_len, lr, epochs) in enumerate(curriculum, 1):
        if stage >= 3:
            for p in model.encoder.parameters():
                p.requires_grad = False
        print(f"Stage {stage}: pred_len={pred_len} lr={lr} epochs={epochs}")
        for pg in optimizer.param_groups:
            pg['lr'] = lr
        model.pred_len = pred_len

        loader = make_dataloader(csv_path, batch_size=batch_size,
                                 obs_len=obs_len, pred_len=pred_len)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=lr / 10.0
        )

        for epoch in range(1, epochs + 1):
            if stage >= 3 and epoch == 2:
                for p in model.encoder.parameters():
                    p.requires_grad = True
            model.train()
            epoch_loss = 0.0
            pbar = tqdm(loader, desc=f"Epoch {epoch}/{epochs}")

            for batch in pbar:
                obs = batch["target"].to(device)                 # (B,T,7) normalized
                fut = batch["gt"].to(device)                     # (B,Tp,2) normalized
                neigh_dyn = batch["neighbors_dyn"].to(device)    # (B,K,T,7)
                neigh_spatial = batch["neighbors_spatial"].to(device)
                lane = batch["lane"].to(device)                  # (B,1)

                if np.random.rand() < 0.5:
                    noise = torch.randn_like(obs[:, :, :2]) * 0.05
                    obs = obs.clone()
                    obs[:, :, :2] += noise
                if np.random.rand() < 0.5:
                    scale = float(1.0 + np.random.randn() * 0.1)
                    obs = obs.clone()
                    obs[:, :, 2:4] *= scale

                optimizer.zero_grad()
                pred, cv = model(obs, neigh_dyn, neigh_spatial, lane, pred_len=pred_len)
                loss, _ = trajectory_loss(pred, fut, cv=cv)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                epoch_loss += loss.item()
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})

            scheduler.step()
            print(f"Epoch {epoch} avg_loss={epoch_loss / max(1, len(loader)):.6f}")

            if epoch % 2 == 0 or epoch == epochs:
                model.eval()
                preds_world, gts_world = [], []

                with torch.no_grad():
                    for i, batch in enumerate(loader):
                        obs = batch["target"].to(device)
                        fut = batch["gt"].to(device)
                        neigh_dyn = batch["neighbors_dyn"].to(device)
                        neigh_spatial = batch["neighbors_spatial"].to(device)
                        lane = batch["lane"].to(device)

                        pred, _ = model(obs, neigh_dyn, neigh_spatial, lane, pred_len=pred_len)

                        origin = batch["origin"].numpy()
                        heading = batch["heading"].numpy()
                        pred_np = pred.cpu().numpy()
                        fut_np = fut.cpu().numpy()

                        pred_w, fut_w = to_world(pred_np, fut_np, origin, heading)
                        preds_world.append(pred_w)
                        gts_world.append(fut_w)

                        if i >= 100:
                            break

                preds_world = np.concatenate(preds_world, axis=0)
                gts_world = np.concatenate(gts_world, axis=0)
                ADE, FDE = ade_fde(preds_world, gts_world)
                print(f"Eval world-frame: ADE={ADE:.3f}m FDE={FDE:.3f}m")

                if ADE < best_ade:
                    best_ade = ADE
                    best_pred_len = pred_len
                    torch.save({
                            "pred_len": pred_len,
                            "state_dict": model.state_dict()
                        }, os.path.join(save_dir, 'best_model.pt'))
                    print(f"New best ADE {best_ade:.3f}m")

                try:
                    plot_loader = make_dataloader(csv_path, batch_size=1,
                                                  obs_len=obs_len, pred_len=pred_len,
                                                  shuffle=False)
                    sample = next(iter(plot_loader))
                    with torch.no_grad():
                        obs_s = sample["target"].to(device)
                        neigh_dyn_s = sample["neighbors_dyn"].to(device)
                        neigh_spatial_s = sample["neighbors_spatial"].to(device)
                        lane_s = sample["lane"].to(device)
                        pred_s, _ = model(obs_s, neigh_dyn_s, neigh_spatial_s, lane_s, pred_len=pred_len)

                    origin = sample["origin"].numpy()
                    heading = sample["heading"].numpy()
                    obs_np = obs_s[0, :, :2].cpu().numpy()
                    fut_np = sample["gt"][0].cpu().numpy()
                    pred_np = pred_s[0].cpu().numpy()

                    def inv_one(x):
                        cos = np.cos(heading[0]); sin = np.sin(heading[0])
                        R_inv = np.array([[cos, -sin], [sin, cos]])
                        xr = (R_inv @ x.T).T + origin[0]
                        return xr

                    obs_w = inv_one(obs_np)
                    fut_w = inv_one(fut_np)
                    pred_w = inv_one(pred_np)

                    plot_sample(
                        obs_w, fut_w, pred_w,
                        save_path=os.path.join(save_dir, f'stage{stage}_epoch{epoch:02d}.png'),
                        title=f"Stage {stage} Epoch {epoch} (ADE={ADE:.2f} m)"
                    )
                except Exception as e:
                    print(f"Plot warning: {e}")

        torch.save(model.state_dict(), os.path.join(save_dir, f'model_stage{stage}.pt'))

    # final comprehensive evaluation using BEST stage horizon
    if best_pred_len is None:
        best_pred_len = curriculum[-1][0]

    root_dir = os.path.abspath(os.path.join(save_dir, os.pardir))  # ./results_final
    final_eval_dir = os.path.join(root_dir, "eval_results_ngsim")
    os.makedirs(final_eval_dir, exist_ok=True)

    ckpt = torch.load(os.path.join(save_dir, "best_model.pt"), map_location=device)
    best_pred_len = ckpt["pred_len"]
    best_model = ImprovedTrajectoryTransformer(
        d_model=256, nhead=8, num_layers=4, pred_len=best_pred_len, k_neighbors=k_neighbors, dt=0.1
    ).to(device)
    best_model.load_state_dict(ckpt["state_dict"])
    best_model.eval()

    

    eval_loader = make_dataloader(
        csv_path, batch_size=1, obs_len=obs_len, pred_len=best_pred_len, shuffle=False, stride=10
    )

    preds_eval, gts_eval, obs_eval = [], [], []

    with torch.no_grad():
        for batch in tqdm(eval_loader, desc="Final Eval"):
            obs = batch["target"].to(device)
            fut = batch["gt"].cpu().numpy()
            neigh_dyn = batch["neighbors_dyn"].to(device)
            neigh_spatial = batch["neighbors_spatial"].to(device)
            lane = batch["lane"].to(device)

            pred, _ = best_model(obs, neigh_dyn, neigh_spatial, lane, pred_len=best_pred_len)
            pred = pred.cpu().numpy()

            origin = batch["origin"].numpy()
            heading = batch["heading"].numpy()

            pred_w, fut_w = to_world(pred, fut, origin, heading)

            preds_eval.append(pred_w[0])
            gts_eval.append(fut_w[0])
            obs_eval.append(obs[0, :, :2].cpu().numpy())

    preds_eval = np.array(preds_eval)
    gts_eval = np.array(gts_eval)
    obs_eval = np.array(obs_eval)

    metrics = compute_comprehensive_metrics(preds_eval, gts_eval)
    print("FINAL COMPREHENSIVE METRICS")
    for k, v in metrics.items():
        if k != 'Per_Step_Error':
            print(f"{k}: {v}")

    with open(os.path.join(final_eval_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)

    plot_error_distribution(preds_eval, gts_eval, save_dir=final_eval_dir)
    plot_sample_predictions(preds_eval, gts_eval, obs_eval, save_dir=final_eval_dir)
    print(f"Saved final eval results to: {final_eval_dir}")

    return model


if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print("Usage: python train_eval.py path/to/US101_cleaned.csv")
        sys.exit(1)

    csv_path = sys.argv[1]
    print(f"Loading CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows, {df['Vehicle_ID'].nunique()} vehicles")

    model = train_model(csv_path, save_dir='./results_final/Checkpoints')
    print("Training complete")
