"""
Comprehensive evaluation script for NGSIM dataset models
"""
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import json

from models import ImprovedTrajectoryTransformer
from ngsim_dataloader import make_dataloader_ngsim as make_dataloader


def compute_comprehensive_metrics(preds, gts):
    """
    Compute comprehensive metrics beyond ADE/FDE
    preds, gts: (N, T, 2) numpy arrays
    """
    assert preds.shape == gts.shape
    N, T, _ = preds.shape
    
    # 1. ADE & FDE
    dists = np.linalg.norm(preds - gts, axis=-1)
    ADE = dists.mean()
    FDE = dists[:, -1].mean()
    
    # 2. MAE
    mae_x = np.abs(preds[:, :, 0] - gts[:, :, 0]).mean()
    mae_y = np.abs(preds[:, :, 1] - gts[:, :, 1]).mean()
    MAE = (mae_x + mae_y) / 2
    
    # 3. RMSE
    mse = ((preds - gts) ** 2).mean()
    RMSE = np.sqrt(mse)
    
    # 4. Velocity error
    pred_vel = np.diff(preds, axis=1)
    gt_vel = np.diff(gts, axis=1)
    vel_error = np.linalg.norm(pred_vel - gt_vel, axis=-1).mean()
    
    # 5. Acceleration error
    if T > 2:
        pred_acc = np.diff(pred_vel, axis=1)
        gt_acc = np.diff(gt_vel, axis=1)
        acc_error = np.linalg.norm(pred_acc - gt_acc, axis=-1).mean()
    else:
        acc_error = 0.0
    
    # 6. Direction error
    pred_angles = np.arctan2(pred_vel[:, :, 1], pred_vel[:, :, 0])
    gt_angles = np.arctan2(gt_vel[:, :, 1], gt_vel[:, :, 0])
    angle_diff = np.abs(pred_angles - gt_angles)
    angle_diff = np.minimum(angle_diff, 2*np.pi - angle_diff)
    direction_error = np.degrees(angle_diff.mean())
    
    # 7. Miss Rate (>2m at final step)
    miss_rate = (dists[:, -1] > 2.0).mean() * 100
    
    # 8. Longitudinal vs Lateral
    lon_error = np.abs(preds[:, :, 0] - gts[:, :, 0]).mean()
    lat_error = np.abs(preds[:, :, 1] - gts[:, :, 1]).mean()
    
    # 9. Per-timestep error
    per_step_error = dists.mean(axis=0)
    
    # 10. Variance ratio
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
    n_per_cat = n_samples // 3
    worst_idx = np.argsort(final_errors)[-n_per_cat:]
    median_start = len(final_errors) // 2 - n_per_cat // 2
    median_idx = np.argsort(final_errors)[median_start:median_start + n_per_cat]
    best_idx = np.argsort(final_errors)[:n_per_cat]
    fig, axes = plt.subplots(3, n_per_cat, figsize=(5*n_per_cat, 12))

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


def evaluate_model_comprehensive(model_path, csv_path, n_samples=1000, save_dir='eval_results'):
    os.makedirs(save_dir, exist_ok=True)

    # Device selection
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple Metal (MPS)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA GPU")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    # Load model
    print(f"\nLoading model from {model_path}")
    model = ImprovedTrajectoryTransformer(
        d_model=256, nhead=8, num_layers=4, pred_len=25, k_neighbors=8
    ).to(device)

    # PyTorch 2.6+ compatibility fix
    state_dict = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(state_dict)
    model.eval()

    loader = make_dataloader(csv_path, batch_size=1, shuffle=False, obs_len=10, pred_len=25, stride=10)
    preds_all, gts_all, obs_all = [], [], []

    print(f"\nEvaluating on {min(n_samples, len(loader))} samples...")
    with torch.no_grad():
        for i, batch in enumerate(tqdm(loader, total=min(n_samples, len(loader)))):
            if i >= n_samples:
                break
            target = batch['target'].to(device)
            neigh_dyn = batch['neighbors_dyn'].to(device)
            neigh_spatial = batch['neighbors_spatial'].to(device)
            lane = batch['lane'].to(device)
            gt = batch['gt'][0].cpu().numpy()
            obs = target[0].cpu().numpy()
            pred = model(target, neigh_dyn, neigh_spatial, lane)[0].cpu().numpy()
            preds_all.append(pred)
            gts_all.append(gt)
            obs_all.append(obs)

    preds_all = np.array(preds_all)
    gts_all = np.array(gts_all)
    obs_all = np.array(obs_all)

    metrics = compute_comprehensive_metrics(preds_all, gts_all)
    print("\nEVALUATION RESULTS")
    print(f"ADE: {metrics['ADE']:.4f} m | FDE: {metrics['FDE']:.4f} m | MAE: {metrics['MAE']:.4f} m | RMSE: {metrics['RMSE']:.4f} m")
    print(f"Velocity Error: {metrics['Velocity_Error']:.4f} | Acc Error: {metrics['Acceleration_Error']:.4f}")
    print(f"Direction Error: {metrics['Direction_Error_deg']:.2f}° | Miss Rate: {metrics['Miss_Rate_%']:.2f}%")
    print(f"Variance Ratio: {metrics['Variance_Ratio']:.4f}")

    if metrics['Variance_Ratio'] < 0.1:
        print("Warning: Model collapse detected.")
    elif metrics['Variance_Ratio'] < 0.5:
        print("Warning: Low prediction diversity.")

    with open(os.path.join(save_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics to {save_dir}/metrics.json")

    plot_error_distribution(preds_all, gts_all, save_dir)
    plot_sample_predictions(preds_all, gts_all, obs_all, save_dir)
    print(f"Evaluation complete. Results saved to {save_dir}/")
    return metrics


if __name__ == '__main__':
    import sys
    if len(sys.argv) < 3:
        print("Usage: python evaluate.py path/to/model.pt path/to/US101_cleaned.csv")
        sys.exit(1)
    model_path = sys.argv[1]
    csv_path = sys.argv[2]
    evaluate_model_comprehensive(model_path, csv_path, n_samples=1000, save_dir='./results_NGSIM/eval_results_ngsim')
