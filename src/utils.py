# utils.py
import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import json

def ade_fde(preds, gts):
    """
    preds, gts: numpy arrays (N, T, 2) in same coordinate frame (meters).
    returns ADE, FDE (scalars)
    """
    assert preds.shape == gts.shape
    dists = np.linalg.norm(preds - gts, axis=-1)  # (N, T)
    ade_per = dists.mean(axis=1)  # per sample
    fde_per = dists[:, -1]
    ADE = ade_per.mean()
    FDE = fde_per.mean()
    return float(ADE), float(FDE)

def cumulate_deltas(last_pos, deltas):
    """
    last_pos: (B,2) last observed position (in agent frame) typically (0,0) if agent-centric
    deltas: (B, T_pred, 2)
    returns absolute positions starting from last_pos + cumulative deltas
    """
    cum = np.cumsum(deltas, axis=1)
    return cum + last_pos[:, None, :]

def torch_cumulate_deltas(last_pos, deltas):
    """
    torch version: last_pos: (B,2), deltas: (B, T, 2)
    returns (B, T, 2)
    """
    # cumulative sum along time
    cum = torch.cumsum(deltas, dim=1)
    return cum + last_pos.unsqueeze(1)

def plot_prediction_one(obs_world, gt_world, pred_world, save_path='pred.png'):
    plt.figure(figsize=(6,4))
    obs_world = np.array(obs_world)
    gt_world = np.array(gt_world)
    pred_world = np.array(pred_world)
    plt.plot(obs_world[:,0], obs_world[:,1], '-o', label='obs', markersize=3)
    plt.plot(gt_world[:,0], gt_world[:,1], '-x', label='gt', markersize=3)
    plt.plot(pred_world[:,0], pred_world[:,1], '-s', label='pred', markersize=3)
    plt.legend()
    plt.axis('equal')
    plt.savefig(save_path)
    plt.close()

def save_json(obj, p):
    with open(p, 'w') as f:
        json.dump(obj, f, indent=2)

def load_model_state(model, path):
    model.load_state_dict(torch.load(path, map_location='cpu'))

# ----------------------------
# Trajectory loss (position + velocity + acceleration consistency)
# ----------------------------
def trajectory_loss(preds_abs, gt_abs, weight_vel=0.25, weight_acc=0.1):
    """
    preds_abs, gt_abs: torch tensors (B, T, 2) in same coordinate frame (meters)
    returns: scalar loss (torch)
    """
    mse_pos = torch.mean((preds_abs - gt_abs) ** 2)

    # velocity terms
    if preds_abs.shape[1] < 2:
        # no vel/acc terms possible
        return mse_pos
    vel_pred = preds_abs[:, 1:, :] - preds_abs[:, :-1, :]
    vel_gt = gt_abs[:, 1:, :] - gt_abs[:, :-1, :]
    mse_vel = torch.mean((vel_pred - vel_gt) ** 2)

    # acceleration terms
    if vel_pred.shape[1] < 2:
        mse_acc = torch.tensor(0.0, device=preds_abs.device, dtype=preds_abs.dtype)
    else:
        acc_pred = vel_pred[:, 1:, :] - vel_pred[:, :-1, :]
        acc_gt = vel_gt[:, 1:, :] - vel_gt[:, :-1, :]
        mse_acc = torch.mean((acc_pred - acc_gt) ** 2)

    return mse_pos + weight_vel * mse_vel + weight_acc * mse_acc