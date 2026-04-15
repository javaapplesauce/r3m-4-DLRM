"""HDF5-backed demonstration dataset.

Storage layout (written by cavr.data.collector):
    <save_dir>/demos.hdf5
        /demo_0/
            images   (T, 3, H, W) uint8
            proprio  (T, proprio_dim) float32
            actions  (T, action_dim) float32
        /demo_1/
            ...
        attrs: env_name, action_dim, proprio_dim, image_size
"""
import os
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class DemoDataset(Dataset):
    def __init__(self, save_dir, filename="demos.hdf5"):
        path = Path(save_dir)
        if path.is_file():
            self.h5_path = str(path)
        else:
            self.h5_path = str(path / filename)
        if not os.path.exists(self.h5_path):
            raise FileNotFoundError(f"No demo file at {self.h5_path}")

        self._file = None
        self._index = []
        with h5py.File(self.h5_path, "r") as f:
            for key in sorted(f.keys()):
                T = f[key]["actions"].shape[0]
                for t in range(T):
                    self._index.append((key, t))

    def _handle(self):
        if self._file is None:
            self._file = h5py.File(self.h5_path, "r", swmr=True)
        return self._file

    def __len__(self):
        return len(self._index)

    def __getitem__(self, idx):
        key, t = self._index[idx]
        g = self._handle()[key]
        image = torch.from_numpy(np.asarray(g["images"][t], dtype=np.uint8))
        proprio = torch.from_numpy(np.asarray(g["proprio"][t], dtype=np.float32))
        action = torch.from_numpy(np.asarray(g["actions"][t], dtype=np.float32))
        return image, proprio, action

    def __del__(self):
        try:
            if self._file is not None:
                self._file.close()
        except Exception:
            pass
