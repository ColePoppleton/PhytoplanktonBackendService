import xarray as xr
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Dict


class OceanDataset(Dataset):
    def __init__(self, data: np.ndarray, window_size: int = 12):
        self.data = torch.FloatTensor(data)
        self.window_size = window_size

    def __len__(self):
        return len(self.data) - self.window_size

    def __getitem__(self, idx):
        x = self.data[idx: idx + self.window_size]
        y = self.data[idx + self.window_size]
        return x.unsqueeze(1), y.unsqueeze(0)


class DataManager:
    def __init__(self, file_path: str):
        self.ds = xr.open_dataset(file_path)

    def extract_subset(self, spatial_ratio: float = 0.4, temporal_ratio: float = 0.4):
        lat_size = int(len(self.ds.latitude) * spatial_ratio)
        lon_size = int(len(self.ds.longitude) * spatial_ratio)
        time_size = int(len(self.ds.time) * temporal_ratio)

        self.subset = self.ds.isel(
            latitude=slice(0, lat_size),
            longitude=slice(0, lon_size),
            time=slice(0, time_size)
        )
        return self.subset

    def get_splits(self, variable: str = "chl") -> Dict[str, DataLoader]:
        data = self.subset[variable].ffill("time").bfill("time").values
        data = (data - data.min()) / (data.max() - data.min())

        n = len(data)
        train_idx = int(n * 0.70)
        val_idx = int(n * 0.85)

        train_data = data[:train_idx]
        val_data = data[train_idx:val_idx]
        test_data = data[val_idx:]

        loaders = {
            "train": DataLoader(OceanDataset(train_data), batch_size=16, shuffle=False),
            "val": DataLoader(OceanDataset(val_data), batch_size=16, shuffle=False),
            "test": DataLoader(OceanDataset(test_data), batch_size=16, shuffle=False)
        }
        return loaders