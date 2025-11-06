"""
ngsim_dataloader.py
Clean, normalized, and augmentation-capable dataloader for NGSIM US101_cleaned.csv
Expected CSV columns: ['Vehicle_ID','Frame_ID','Local_X','Local_Y','Local_Velocity','Lane_ID']
Local_X/Y in feet (converted to meters here).
"""

import numpy as np
import pandas as pd
import math
from torch.utils.data import Dataset, DataLoader

def load_ngsim_csv(csv_path):
    df = pd.read_csv(csv_path)
    # convert feet -> meters for positions and velocity (approx)
    df["Local_X"] = df["Local_X"].astype(float) * 0.3048
    df["Local_Y"] = df["Local_Y"].astype(float) * 0.3048
    df["Local_Velocity"] = df["Local_Velocity"].astype(float) * 0.3048
    df["Frame_ID"] = df["Frame_ID"].astype(int)
    df["Vehicle_ID"] = df["Vehicle_ID"].astype(int)
    df["Lane_ID"] = df["Lane_ID"].astype(int)
    return df

def compute_heading(vx, vy):
    return np.arctan2(vy, vx)

def compute_8dir_features(target_xy, neighbors_xy, max_radius=50.0):
    dirs = np.zeros(16, dtype=np.float32)
    tx, ty = target_xy
    for (nx, ny) in neighbors_xy:
        dx, dy = nx - tx, ny - ty
        dist = math.hypot(dx, dy)
        if dist > max_radius:
            continue
        angle = (math.degrees(math.atan2(dy, dx)) + 360) % 360
        k = int(angle // 45)
        dirs[2 * k] = dx
        dirs[2 * k + 1] = dy
    return dirs

class NGSIMDataset(Dataset):
    def __init__(self, df_or_path, obs_len=10, pred_len=25, stride=5,
                 k_neighbors=8, augment=True, noise_std=0.01, rotation=True):
        if isinstance(df_or_path, str):
            self.df = load_ngsim_csv(df_or_path)
        else:
            self.df = df_or_path.copy()
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.stride = stride
        self.k_neighbors = k_neighbors
        self.augment = augment
        self.noise_std = noise_std
        self.rotation = rotation

        self.samples = []
        self._build_index()

    def _build_index(self):
        vehicle_ids = self.df["Vehicle_ID"].unique()
        for vid in vehicle_ids:
            v = self.df[self.df["Vehicle_ID"] == vid].sort_values("Frame_ID")
            n = len(v)
            if n < self.obs_len + self.pred_len:
                continue
            xs = v["Local_X"].values
            ys = v["Local_Y"].values
            vel = v["Local_Velocity"].values
            lane = v["Lane_ID"].values
            # compute finite-difference velocities/accelerations (smoothed)
            vx = np.gradient(xs) * 10.0
            vy = np.gradient(ys) * 10.0
            ax = np.gradient(vx) * 10.0
            ay = np.gradient(vy) * 10.0
            heading = compute_heading(vx, vy)

            for start in range(0, n - (self.obs_len + self.pred_len) + 1, self.stride):
                end_obs = start + self.obs_len
                end_pred = end_obs + self.pred_len
                frame_id = int(v.iloc[end_obs]["Frame_ID"])
                obs = np.stack([
                    xs[start:end_obs],
                    ys[start:end_obs],
                    vx[start:end_obs],
                    vy[start:end_obs],
                    ax[start:end_obs],
                    ay[start:end_obs],
                    heading[start:end_obs]
                ], axis=1).astype(np.float32)  # (obs_len, 7)

                fut = np.stack([
                    xs[end_obs:end_pred],
                    ys[end_obs:end_pred]
                ], axis=1).astype(np.float32)  # (pred_len, 2)

                self.samples.append({
                    "vid": int(vid),
                    "frame_id": frame_id,
                    "obs": obs,
                    "fut": fut,
                    "lane_obs": lane[start:end_obs].astype(np.int32)
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        obs = s["obs"].copy()  # (T_obs, 7)
        fut = s["fut"].copy()  # (T_pred, 2)
        lane_obs = s["lane_obs"].copy()

        # agent-centric normalization: last observed position => origin
        origin = obs[-1, :2].copy()
        obs[:, :2] -= origin
        fut[:, :2] -= origin

        # ensure equal spacing (NGSIM is ~10Hz). We assume uniform sampling.

        # neighbors: take vehicles in same frame
        frame_rows = self.df[self.df["Frame_ID"] == s["frame_id"]]
        neigh = frame_rows[frame_rows["Vehicle_ID"] != s["vid"]]
        neighbors_xy = neigh[["Local_X", "Local_Y"]].values
        # compute relative to agent last obs absolute pos (origin + obs[-1] == original last pos)
        agent_last_abs = origin + np.array([0.0, 0.0])  # origin is last obs in absolute coords already subtracted

        # compute nearest K neighbors
        K = self.k_neighbors
        neigh_dyn = np.zeros((K, self.obs_len, 7), dtype=np.float32)
        neigh_spatial = np.zeros((K, self.obs_len, 18), dtype=np.float32)
        if len(neighbors_xy) > 0:
            diffs = neighbors_xy - (origin + obs[-1, :2])  # approximate (still world coords)
            dists = np.hypot(diffs[:, 0], diffs[:, 1])
            order = np.argsort(dists)[:K]
            selected = neigh.iloc[order]
            for j, (_, row) in enumerate(selected.iterrows()):
                # represent neighbor as constant over obs window (approximation)
                dx = (row["Local_X"] - origin[0]).astype(np.float32)
                dy = (row["Local_Y"] - origin[1]).astype(np.float32)
                # neighbor velocity approximate from column
                v = float(row.get("Local_Velocity", 0.0))
                vx = v  # already converted in loader if path used
                vy = 0.0
                heading = 0.0
                neigh_dyn[j, :, :] = np.stack([
                    np.full(self.obs_len, dx, dtype=np.float32),
                    np.full(self.obs_len, dy, dtype=np.float32),
                    np.full(self.obs_len, vx, dtype=np.float32),
                    np.full(self.obs_len, vy, dtype=np.float32),
                    np.zeros(self.obs_len, dtype=np.float32),
                    np.zeros(self.obs_len, dtype=np.float32),
                    np.full(self.obs_len, heading, dtype=np.float32)
                ], axis=1)
                neigh_spatial[j, :, :] = 0.0  # placeholder 18-dim zeros

        # lane feature: scalar lane id of last obs
        lane_feat = np.array([lane_obs[-1]], dtype=np.float32)  # shape (1,)

        # augmentations (in agent frame): small gaussian noise, random rotation, velocity jitter
        if self.augment:
            if self.noise_std > 0:
                obs[:, :2] += np.random.normal(scale=self.noise_std, size=obs[:, :2].shape)
            if self.rotation:
                theta = np.random.uniform(-0.04, 0.04)  # small rotation ±~2.3deg
                c, s = math.cos(theta), math.sin(theta)
                R = np.array([[c, -s], [s, c]], dtype=np.float32)
                obs[:, :2] = (R @ obs[:, :2].T).T
                fut[:, :2] = (R @ fut[:, :2].T).T
                neigh_dyn[:, :, 0:2] = (R @ neigh_dyn[:, :, 0:2].reshape(-1, 2).T).T.reshape(neigh_dyn[:, :, 0:2].shape)

            # velocity jitter
            jitter = np.random.normal(scale=0.05, size=obs[:, 2:4].shape)
            obs[:, 2:4] += jitter

        sample = {
            "target": obs,                 # (T_obs, 7)
            "gt": fut,                     # (T_pred, 2)
            "neighbors_dyn": neigh_dyn,    # (K, T_obs, 7)
            "neighbors_spatial": neigh_spatial,  # (K, T_obs, 18)
            "lane": lane_feat               # (1,)
        }
        return sample

def make_dataloader_ngsim(csv_path_or_df, batch_size=32, obs_len=10, pred_len=25, stride=5,
                          k_neighbors=8, augment=True, shuffle=True, drop_last=True, num_workers=0):
    if isinstance(csv_path_or_df, str):
        df = load_ngsim_csv(csv_path_or_df)
    else:
        df = csv_path_or_df
    ds = NGSIMDataset(df, obs_len=obs_len, pred_len=pred_len, stride=stride,
                      k_neighbors=k_neighbors, augment=augment)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers)
    return loader
