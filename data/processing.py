import xarray as xr
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Dict
import logging

logger = logging.getLogger("digital_twin")


class OceanDataset(Dataset):
    def __init__(self, data: np.ndarray, window_size: int = 12):
        self.data = torch.FloatTensor(data)
        self.window_size = window_size

        if len(self.data) <= self.window_size:
            raise ValueError(
                f"Split size ({len(self.data)}) must be greater than window_size ({self.window_size}). "
                "Increase your temporal range in main.py to allow 70/15/15 splitting."
            )

    def __len__(self):
        return len(self.data) - self.window_size

    def __getitem__(self, idx):
        x = self.data[idx: idx + self.window_size]
        y = self.data[idx + self.window_size]
        return x.unsqueeze(1), y.unsqueeze(0)


class DataManager:
    def __init__(self, file_path: str):
        self.ds = xr.open_dataset(file_path)

    def extract_subset(self, spatial_ratio: float = 0.15, temporal_ratio: float = 1.0):
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
        target_res = 64
        working_ds = self.subset[variable]
        if "depth" in working_ds.dims:
            working_ds = working_ds.isel(depth=0)

        lat_coords = np.linspace(float(working_ds.latitude[0]), float(working_ds.latitude[-1]), target_res)
        lon_coords = np.linspace(float(working_ds.longitude[0]), float(working_ds.longitude[-1]), target_res)
        resampled = working_ds.interp(latitude=lat_coords, longitude=lon_coords)

        filled = resampled.ffill("time").bfill("time")
        data_values = np.nan_to_num(filled.values, nan=0.0)

        data = (data_values - data_values.min()) / (data_values.max() - data_values.min() + 1e-8)

        n = len(data)
        train_idx = int(n * 0.70)
        val_idx = int(n * 0.85)
        loaders = {
            "train": DataLoader(OceanDataset(data[:train_idx]), batch_size=16, shuffle=False),
            "val": DataLoader(OceanDataset(data[train_idx:val_idx]), batch_size=16, shuffle=False),
            "test": DataLoader(OceanDataset(data[val_idx:]), batch_size=16, shuffle=False)
        }
        return loaders