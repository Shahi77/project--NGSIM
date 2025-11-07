import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import math

def load_ngsim_csv(csv_path):
    """
    Expected columns:
    ['Vehicle_ID', 'Frame_ID', 'Local_X', 'Local_Y', 'Local_Velocity', 'Lane_ID']
    Units: feet -> converted to meters
    """
    df = pd.read_csv(csv_path)
    df["Local_X"] *= 0.3048
    df["Local_Y"] *= 0.3048
    df["Local_Velocity"] *= 0.3048
    df["Frame_ID"] = df["Frame_ID"].astype(int)
    df["Vehicle_ID"] = df["Vehicle_ID"].astype(int)
    df["Lane_ID"] = df["Lane_ID"].astype(int)
    return df


def is_smooth_trajectory(xs, ys, max_accel=5.0):
    vx = np.gradient(xs) * 10.0
    vy = np.gradient(ys) * 10.0
    ax = np.gradient(vx) * 10.0
    ay = np.gradient(vy) * 10.0
    accel_mag = np.sqrt(ax**2 + ay**2)
    return np.max(accel_mag) < max_accel


def is_curved_trajectory(xs, ys, min_curvature=0.005):
    if len(xs) < 3:
        return False
    start = np.array([xs[0], ys[0]])
    end = np.array([xs[-1], ys[-1]])
    line = end - start
    line_len = np.linalg.norm(line)
    if line_len < 1.0:
        return False
    max_dev = 0.0
    for i in range(1, len(xs) - 1):
        p = np.array([xs[i], ys[i]])
        dev = np.abs(np.cross(line, start - p)) / line_len
        max_dev = max(max_dev, dev)
    return max_dev > min_curvature * line_len


class NGSIMDataset(Dataset):
    def __init__(self, df, obs_len=20, pred_len=25, stride=5, filter_smooth=True,
                 balance_curved=False):
        self.df = df
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.stride = stride
        self.samples = []

        vehicle_ids = df["Vehicle_ID"].unique()

        for vid in vehicle_ids:
            v = df[df["Vehicle_ID"] == vid].sort_values("Frame_ID")
            if len(v) < obs_len + pred_len:
                continue

            xs = v["Local_X"].values
            ys = v["Local_Y"].values

            if filter_smooth and not is_smooth_trajectory(xs, ys):
                continue

            curved = is_curved_trajectory(xs, ys)
            if balance_curved and not curved:
                if np.random.rand() > 0.5:
                    continue

            lane = v["Lane_ID"].values
            vx = np.gradient(xs) * 10.0
            vy = np.gradient(ys) * 10.0
            ax = np.gradient(vx) * 10.0
            ay = np.gradient(vy) * 10.0
            heading = np.arctan2(vy, vx)

            for i in range(0, len(xs) - (obs_len + pred_len), stride):
                s = i
                e_obs = i + obs_len
                e_pred = e_obs + pred_len

                obs = np.stack([
                    xs[s:e_obs],
                    ys[s:e_obs],
                    vx[s:e_obs],
                    vy[s:e_obs],
                    ax[s:e_obs],
                    ay[s:e_obs],
                    heading[s:e_obs]
                ], axis=1)

                fut = np.stack([xs[e_obs:e_pred], ys[e_obs:e_pred]], axis=1)

                # road-aligned normalization
                origin = obs[-1, :2].copy()
                last_heading = obs[-1, 6]
                c, s_h = np.cos(-last_heading), np.sin(-last_heading)
                R = np.array([[c, -s_h], [s_h, c]])

                obs_pos = obs[:, :2] - origin
                obs[:, :2] = (R @ obs_pos.T).T
                fut_pos = fut - origin
                fut = (R @ fut_pos.T).T

                vel = obs[:, 2:4]
                obs[:, 2:4] = (R @ vel.T).T
                obs[:, 6] = obs[:, 6] - last_heading

                frame_id = int(v.iloc[e_obs]["Frame_ID"])

                self.samples.append((
                    vid, frame_id,
                    obs.astype(np.float32),
                    fut.astype(np.float32),
                    np.array(lane[s:e_obs], dtype=np.float32),
                    origin.astype(np.float32),
                    np.float32(last_heading)
                ))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        vid, frame_id, obs, fut, lane_obs, origin, last_heading = self.samples[idx]

        # neighbors at current frame (kept simple)
        frame_data = self.df[self.df["Frame_ID"] == frame_id]
        neighbors = frame_data[frame_data["Vehicle_ID"] != vid]

        K = 8
        neigh_dyn = np.zeros((K, self.obs_len, 7), dtype=np.float32)
        neigh_spatial = np.zeros((K, self.obs_len, 18), dtype=np.float32)

        if len(neighbors) > 0:
            # approximate current positions in normalized frame
            # reconstruct inverse rotation used above
            c, s_h = np.cos(-last_heading), np.sin(-last_heading)
            R = np.array([[c, -s_h], [s_h, c]])
            R_inv = R.T

            npos_world = neighbors[["Local_X", "Local_Y"]].values
            # bring to normalized frame with same transform as target
            npos_norm = (R @ (npos_world - origin).T).T

            dists = np.linalg.norm(npos_norm, axis=1)
            pick = np.argsort(dists)[:K]
            nsel = neighbors.iloc[pick]
            for j, (_, row) in enumerate(nsel.iterrows()):
                dx, dy = npos_norm[pick[j]]
                speed = float(row["Local_Velocity"])
                # assume along lane for lack of heading per neighbor; keep zeros in acc and heading deltas
                neigh_dyn[j, :, :] = np.stack([
                    np.full(self.obs_len, dx),
                    np.full(self.obs_len, dy),
                    np.full(self.obs_len, speed),  # vx placeholder
                    np.zeros(self.obs_len),         # vy placeholder
                    np.zeros(self.obs_len),
                    np.zeros(self.obs_len),
                    np.zeros(self.obs_len)
                ], axis=1)

        lane_feat = np.array([[lane_obs[-1]]], dtype=np.float32)  # shape (1,1)

        return {
            "target": obs,                 # (T_obs,7) normalized
            "gt": fut,                     # (T_pred,2) normalized
            "neighbors_dyn": neigh_dyn,    # (K,T_obs,7)
            "neighbors_spatial": neigh_spatial,
            "lane": lane_feat,             # (1,1)
            "origin": origin,              # (2,)
            "heading": last_heading,       # scalar
        }


def make_dataloader_ngsim(csv_path, batch_size=32, obs_len=20, pred_len=25,
                          stride=5, shuffle=True, filter_smooth=True,
                          balance_curved=False):
    df = load_ngsim_csv(csv_path)
    dataset = NGSIMDataset(
        df, obs_len=obs_len, pred_len=pred_len, stride=stride,
        filter_smooth=filter_smooth, balance_curved=balance_curved
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=True)
    return loader
