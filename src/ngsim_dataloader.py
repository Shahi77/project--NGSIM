# ngsim_dataloader.py
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np

def load_ngsim_csv(csv_path):
    """
    Load and preprocess NGSIM CSV file.
    Expected columns: ['Vehicle_ID', 'Frame_ID', 'Local_X', 'Local_Y', 'Local_Velocity', 'Lane_ID']
    Units: feet -> converted to meters.
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
    start, end = np.array([xs[0], ys[0]]), np.array([xs[-1], ys[-1]])
    line = end - start
    line_len = np.linalg.norm(line)
    if line_len < 1.0:
        return False
    max_dev = 0.0
    for i in range(1, len(xs) - 1):
        p = np.array([xs[i], ys[i]])
        # Use 2D cross product magnitude formula for scalars
        dev = abs(line[0]*(start[1]-p[1]) - line[1]*(start[0]-p[0])) / line_len
        max_dev = max(max_dev, dev)
    return max_dev > min_curvature * line_len


class NGSIMDataset(Dataset):
    def __init__(self, df, obs_len=20, pred_len=25, stride=5, filter_smooth=True, balance_curved=False):
        self.df = df
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.stride = stride
        self.samples = []

        for vid in df["Vehicle_ID"].unique():
            v = df[df["Vehicle_ID"] == vid].sort_values("Frame_ID")
            if len(v) < obs_len + pred_len:
                continue

            xs, ys = v["Local_X"].values, v["Local_Y"].values
            if filter_smooth and not is_smooth_trajectory(xs, ys):
                continue
            curved = is_curved_trajectory(xs, ys)
            if balance_curved and not curved and np.random.rand() > 0.5:
                continue

            lane = v["Lane_ID"].values
            vx = np.gradient(xs) * 10.0
            vy = np.gradient(ys) * 10.0
            ax = np.gradient(vx) * 10.0
            ay = np.gradient(vy) * 10.0
            heading = np.arctan2(vy, vx)

            for i in range(0, len(xs) - (obs_len + pred_len), stride):
                s, e_obs, e_pred = i, i + obs_len, i + obs_len + pred_len
                obs = np.stack([xs[s:e_obs], ys[s:e_obs], vx[s:e_obs], vy[s:e_obs],
                                ax[s:e_obs], ay[s:e_obs], heading[s:e_obs]], axis=1)
                fut = np.stack([xs[e_obs:e_pred], ys[e_obs:e_pred]], axis=1)

                origin = obs[-1, :2].copy()
                last_heading = obs[-1, 6]
                c, s_h = np.cos(-last_heading), np.sin(-last_heading)
                R = np.array([[c, -s_h], [s_h, c]])
                obs[:, :2] = (R @ (obs[:, :2] - origin).T).T
                fut = (R @ (fut - origin).T).T
                obs[:, 2:4] = (R @ obs[:, 2:4].T).T
                obs[:, 6] = obs[:, 6] - last_heading

                frame_id = int(v.iloc[e_obs]["Frame_ID"])
                self.samples.append((vid, frame_id, obs.astype(np.float32), fut.astype(np.float32),
                                     np.array(lane[s:e_obs], dtype=np.float32),
                                     origin.astype(np.float32), np.float32(last_heading)))

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        vid, frame_id, obs, fut, lane_obs, origin, last_heading = self.samples[idx]
        frame_data = self.df[self.df["Frame_ID"] == frame_id]
        neighbors = frame_data[frame_data["Vehicle_ID"] != vid]

        K = 8
        neigh_dyn = np.zeros((K, self.obs_len, 7), np.float32)
        neigh_spatial = np.zeros((K, self.obs_len, 18), np.float32)

        if len(neighbors) > 0:
            c, s_h = np.cos(-last_heading), np.sin(-last_heading)
            R = np.array([[c, -s_h], [s_h, c]])
            npos_norm = (R @ (neighbors[["Local_X", "Local_Y"]].values - origin).T).T
            pick = np.argsort(np.linalg.norm(npos_norm, axis=1))[:K]
            for j, (_, row) in enumerate(neighbors.iloc[pick].iterrows()):
                dx, dy = npos_norm[j]
                speed = float(row["Local_Velocity"])
                # Each neighbor feature repeated along obs_len
                neigh_dyn[j, :, :] = np.stack([
                    np.full(self.obs_len, dx),          # x offset
                    np.full(self.obs_len, dy),          # y offset
                    np.full(self.obs_len, speed),       # vx (approx)
                    np.zeros(self.obs_len),             # vy placeholder
                    np.zeros(self.obs_len),             # ax placeholder
                    np.zeros(self.obs_len),             # ay placeholder
                    np.zeros(self.obs_len)              # heading placeholder
                ], axis=1)


        return {
            "target": obs, "gt": fut,
            "neighbors_dyn": neigh_dyn, "neighbors_spatial": neigh_spatial,
            "lane": np.array([[lane_obs[-1]]], np.float32),
            "origin": origin, "heading": last_heading
        }


def make_dataloader_ngsim(csv_path, batch_size=32, obs_len=20, pred_len=25, stride=5, shuffle=True):
    df = load_ngsim_csv(csv_path)
    dataset = NGSIMDataset(df, obs_len=obs_len, pred_len=pred_len, stride=stride, filter_smooth=False) # disable strict smoothness filter

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=True)


if __name__ == "__main__":
    csv_path = "./Dataset/NGSIM/US101_cleaned.csv"
    print("Loading dataset from:", csv_path)
    df = load_ngsim_csv(csv_path)
    print("Loaded rows:", len(df))
    print("Unique vehicles:", df['Vehicle_ID'].nunique())

    dataset = NGSIMDataset(df, obs_len=20, pred_len=25)
    print("Total samples:", len(dataset))

    sample = dataset[0]
    print("Sample keys:", sample.keys())
    print("target shape:", sample["target"].shape)
    print("gt shape:", sample["gt"].shape)
    print("neighbors_dyn shape:", sample["neighbors_dyn"].shape)
    print("origin:", sample["origin"])
    print("heading:", sample["heading"])
