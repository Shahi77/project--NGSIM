import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import math
import os
import matplotlib.pyplot as plt

# os.makedirs("./results_NGSIM", exist_ok=True)

# df = pd.read_csv('./Dataset/NGSIM/US101_cleaned.csv')
# plt.figure(figsize=(8, 6))

# # Convert feet → meters for plotting
# df["Local_X"] *= 0.3048
# df["Local_Y"] *= 0.3048

# for vid in df["Vehicle_ID"].unique()[:5]:
#     v = df[df["Vehicle_ID"] == vid]
#     plt.plot(v["Local_X"], v["Local_Y"], lw=1)

# plt.xlabel("X (m)")
# plt.ylabel("Y (m)")
# plt.title("Sample NGSIM Trajectories (converted to meters)")
# plt.tight_layout()
# plt.savefig("./results_NGSIM/Sample_NGSIM_trajectories.png", dpi=200)
# print(" Saved plot to ./results_NGSIM/Sample_NGSIM_trajectories.png")
# plt.close()
def load_ngsim_csv(csv_path):
    """
    Expected columns:
    ['Vehicle_ID', 'Frame_ID', 'Local_X', 'Local_Y', 'Local_Velocity', 'Lane_ID']
    Units: feet → converted to meters
    """
    df = pd.read_csv(csv_path)
    df["Local_X"] *= 0.3048
    df["Local_Y"] *= 0.3048
    df["Local_Velocity"] *= 0.3048
    df["Frame_ID"] = df["Frame_ID"].astype(int)
    df["Vehicle_ID"] = df["Vehicle_ID"].astype(int)
    df["Lane_ID"] = df["Lane_ID"].astype(int)
    return df


def compute_8dir_features(target, neighbors, max_radius=50.0):
    """
    Reproduces compute_8dir_features from highd_dataloader.
    Divides surrounding space into 8 sectors and extracts relative distances.
    """
    dirs = np.zeros(16)  # 8 directions × (dx, dy)
    tx, ty = target
    for i, (nx, ny) in enumerate(neighbors):
        dx, dy = nx - tx, ny - ty
        angle = math.degrees(math.atan2(dy, dx)) % 360
        k = int(angle // 45)
        if np.hypot(dx, dy) < max_radius:
            dirs[2 * k:2 * k + 2] = [dx, dy]
    return dirs

class NGSIMDataset(Dataset):
    def __init__(self, df, obs_len=20, pred_len=25, stride=5):
        self.df = df
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.stride = stride
        self.samples = []

        self.vehicle_ids = df["Vehicle_ID"].unique()
        self.frames = df.groupby("Frame_ID")

        for vid in self.vehicle_ids:
            v = df[df["Vehicle_ID"] == vid].sort_values("Frame_ID")
            if len(v) < obs_len + pred_len:
                continue

            xs = v["Local_X"].values
            ys = v["Local_Y"].values
            lane = v["Lane_ID"].values
            vx = np.gradient(xs) * 10  # 10 Hz
            vy = np.gradient(ys) * 10
            ax = np.gradient(vx) * 10
            ay = np.gradient(vy) * 10

            for i in range(0, len(xs) - (obs_len + pred_len), stride):
                start = i
                end_obs = i + obs_len
                end_pred = end_obs + pred_len
                heading = np.arctan2(vy, vx)
                obs = np.stack([
                            xs[start:end_obs],
                            ys[start:end_obs],
                            vx[start:end_obs],
                            vy[start:end_obs],
                            ax[start:end_obs],
                            ay[start:end_obs],
                            heading[start:end_obs]
                        ], axis=1)
                fut = np.stack([xs[end_obs:end_pred],
                                ys[end_obs:end_pred]], axis=1)
                frame_id = int(v.iloc[end_obs]["Frame_ID"])
                self.samples.append((vid, frame_id, obs, fut, lane[start:end_obs]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        vid, frame_id, obs, fut, lane_obs = self.samples[idx]

        # Agent-centric normalization
        origin = obs[-1, :2].copy()
        obs[:, :2] -= origin
        fut[:, :2] -= origin

        # Current frame neighbors
        frame_data = self.df[self.df["Frame_ID"] == frame_id]
        neighbors = frame_data[frame_data["Vehicle_ID"] != vid]

        # Neighbor dynamics (dummy temporal, as NGSIM frame data is independent)
        K = 8
        neigh_dyn = np.zeros((K, self.obs_len, 7), dtype=np.float32)
        neigh_spatial = np.zeros((K, self.obs_len, 18), dtype=np.float32)

        # Sort by proximity
        if len(neighbors) > 0:
            diffs = neighbors[["Local_X", "Local_Y"]].values - (origin + obs[-1, :2])
            dists = np.hypot(diffs[:, 0], diffs[:, 1])
            nearest_idx = np.argsort(dists)[:K]
            nearest = neighbors.iloc[nearest_idx]

            # For each neighbor, replicate position as constant over obs_len
            for j, (_, row) in enumerate(nearest.iterrows()):
                dx = row["Local_X"] - origin[0]
                dy = row["Local_Y"] - origin[1]
                vx = row["Local_Velocity"] * np.cos(0)
                vy = row["Local_Velocity"] * np.sin(0)
                heading = 0.0
                neigh_dyn[j, :, :] = np.stack([np.full(self.obs_len, dx),
                                            np.full(self.obs_len, dy),
                                            np.full(self.obs_len, vx),
                                            np.full(self.obs_len, vy),
                                            np.zeros(self.obs_len),
                                            np.zeros(self.obs_len),
                                            np.full(self.obs_len, heading)], axis=1)
                neigh_spatial[j, :, :] = np.zeros((self.obs_len, 18))

        lane_feats = np.expand_dims(lane_obs[-1], axis=0).astype(np.float32)

        sample = {
            "target": obs.astype(np.float32),
            "gt": fut.astype(np.float32),
            "neighbors_dyn": neigh_dyn,
            "neighbors_spatial": neigh_spatial,
            "lane": lane_feats
        }
        return sample



def make_dataloader_ngsim(csv_path, batch_size=32, obs_len=20, pred_len=25, stride=5, shuffle=True):
    df = load_ngsim_csv(csv_path)
    dataset = NGSIMDataset(df, obs_len=obs_len, pred_len=pred_len, stride=stride)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=True)
    return loader
