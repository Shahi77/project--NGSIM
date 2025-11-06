"""
utils.py - numeric helpers and losses
"""

import numpy as np
import torch
import json

def ade_fde(preds, gts):
    """
    preds, gts: numpy arrays (N, T, 2)
    """
    assert preds.shape == gts.shape
    dists = np.linalg.norm(preds - gts, axis=-1)
    ADE = dists.mean()
    FDE = dists[:, -1].mean()
    return float(ADE), float(FDE)

def torch_ade_fde(pred, gt):
    # pred, gt: torch tensors (B, T, 2)
    d = torch.norm(pred - gt, dim=-1)
    ade = d.mean()
    fde = d[:, -1].mean()
    return ade, fde

def angular_loss(pred, gt):
    """
    Penalize angle difference between predicted velocity vectors and GT velocities.
    pred, gt: torch tensors (B, T, 2)
    """
    if pred.shape[1] < 2:
        return torch.tensor(0.0, device=pred.device)
    pv = pred[:, 1:, :] - pred[:, :-1, :]
    gv = gt[:, 1:, :] - gt[:, :-1, :]
    # normalize
    pv_n = pv / (pv.norm(dim=-1, keepdim=True) + 1e-6)
    gv_n = gv / (gv.norm(dim=-1, keepdim=True) + 1e-6)
    cos_sim = (pv_n * gv_n).sum(dim=-1).clamp(-1.0, 1.0)
    ang = torch.acos(cos_sim)  # radians
    return ang.mean()

def lane_constraint_loss(pred, lane_ids):
    """
    Soft constraint to keep predicted lateral displacement consistent with lane ID.
    lane_ids: (B, T_obs?) or (B,1)
    This is a simple heuristic: penalize large lateral y if lane indicates center lane (optional).
    We'll implement a tiny penalty proportional to std of lateral predictions when lane fixed.
    pred: (B, T_pred, 2)
    """
    # If lane_ids not provided or meaningless, return 0
    return torch.tensor(0.0, device=pred.device)

def weighted_horizon_loss(pred, gt, weights=None):
    """
    Give more weight to later timesteps (to avoid trivial near-zero predictions).
    pred, gt: (B, T, 2)
    weights: None -> linear increasing from 0.5 to 1.5 across horizon
    """
    B, T, _ = pred.shape
    if weights is None:
        w = torch.linspace(0.5, 1.5, steps=T, device=pred.device).unsqueeze(0).unsqueeze(-1)  # (1,T,1)
    else:
        w = weights.view(1, T, 1).to(pred.device)
    return ((pred - gt) ** 2 * w).mean()

def combined_loss(pred, gt, lane_ids=None,
                  w_pos=1.0, w_vel=0.5, w_acc=0.2, w_ang=0.5, w_horizon=0.5):
    """
    Combine position MSE, velocity & acceleration MSE, angular loss, and weighted horizon MSE.
    """
    device = pred.device
    pos_mse = torch.mean((pred - gt) ** 2)

    # vel MSE
    if pred.shape[1] > 1:
        pred_vel = pred[:, 1:, :] - pred[:, :-1, :]
        gt_vel = gt[:, 1:, :] - gt[:, :-1, :]
        vel_mse = torch.mean((pred_vel - gt_vel) ** 2)
    else:
        vel_mse = torch.tensor(0.0, device=device)

    # acc MSE
    if pred.shape[1] > 2:
        pred_acc = pred_vel[:, 1:, :] - pred_vel[:, :-1, :]
        gt_acc = gt_vel[:, 1:, :] - gt_vel[:, :-1, :]
        acc_mse = torch.mean((pred_acc - gt_acc) ** 2)
    else:
        acc_mse = torch.tensor(0.0, device=device)

    ang = angular_loss(pred, gt)
    horizon = weighted_horizon_loss(pred, gt)
    # lane constraint left as placeholder (zero) unless you pass lane info and a rule
    lane_loss = torch.tensor(0.0, device=device)

    total = w_pos * pos_mse + w_vel * vel_mse + w_acc * acc_mse + w_ang * ang + w_horizon * horizon + 0.0 * lane_loss

    details = {
        "pos": pos_mse.item() if isinstance(pos_mse, torch.Tensor) else float(pos_mse),
        "vel": vel_mse.item() if isinstance(vel_mse, torch.Tensor) else float(vel_mse),
        "acc": acc_mse.item() if isinstance(acc_mse, torch.Tensor) else float(acc_mse),
        "ang": ang.item() if isinstance(ang, torch.Tensor) else float(ang),
        "horizon": horizon.item() if isinstance(horizon, torch.Tensor) else float(horizon)
    }
    return total, details

def save_json(obj, p):
    with open(p, 'w') as f:
        json.dump(obj, f, indent=2)
