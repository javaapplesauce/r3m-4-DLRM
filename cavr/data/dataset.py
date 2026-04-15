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

Optional precomputed masks (written by scripts/precompute_masks.py):
    <save_dir>/masks.hdf5
        /demo_0/masks  (T, feature_h, feature_w) uint8
        /demo_1/masks  (T, feature_h, feature_w) uint8
        attrs: feature_h, feature_w, task_description, backbone

When `masks.hdf5` is present, __getitem__ returns a 4-tuple (image, proprio,
action, mask). Otherwise the fourth element is an empty tensor sentinel
(shape (0,)) that the trainer recognizes as "no cached mask".
"""
import os
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class DemoDataset(Dataset):
    def __init__(self, save_dir, filename="demos.hdf5",
                 mask_filename="masks.hdf5", mask_path=None):
        """
        Args:
            save_dir: directory containing demos.hdf5 (or a direct path to
                demos.hdf5).
            filename: name of the demo HDF5 file inside save_dir.
            mask_filename: name of the optional cached-mask HDF5 file inside
                save_dir. Pass None to disable auto-detection.
            mask_path: explicit path to a mask HDF5 file, overrides
                mask_filename-based auto-detection.
        """
        path = Path(save_dir)
        if path.is_file():
            self.h5_path = str(path)
            base_dir = path.parent
        else:
            self.h5_path = str(path / filename)
            base_dir = path
        if not os.path.exists(self.h5_path):
            raise FileNotFoundError(f"No demo file at {self.h5_path}")

        self._file = None
        self._mask_file = None
        self._index = []
        with h5py.File(self.h5_path, "r") as f:
            for key in sorted(f.keys()):
                T = f[key]["actions"].shape[0]
                for t in range(T):
                    self._index.append((key, t))

        if mask_path is not None:
            self.mask_path = str(mask_path)
        elif mask_filename is not None:
            candidate = base_dir / mask_filename
            self.mask_path = str(candidate) if candidate.exists() else None
        else:
            self.mask_path = None

        self.has_masks = self.mask_path is not None and os.path.exists(self.mask_path)
        if self.has_masks:
            with h5py.File(self.mask_path, "r") as mf:
                self._mask_feature_h = int(mf.attrs.get("feature_h", 0))
                self._mask_feature_w = int(mf.attrs.get("feature_w", 0))

    def _handle(self):
        if self._file is None:
            self._file = h5py.File(self.h5_path, "r", swmr=True)
        return self._file

    def _mask_handle(self):
        if self._mask_file is None:
            self._mask_file = h5py.File(self.mask_path, "r", swmr=True)
        return self._mask_file

    def __len__(self):
        return len(self._index)

    def __getitem__(self, idx):
        key, t = self._index[idx]
        g = self._handle()[key]
        image = torch.from_numpy(np.asarray(g["images"][t], dtype=np.uint8))
        proprio = torch.from_numpy(np.asarray(g["proprio"][t], dtype=np.float32))
        action = torch.from_numpy(np.asarray(g["actions"][t], dtype=np.float32))

        if self.has_masks:
            mg = self._mask_handle()[key]
            mask = torch.from_numpy(np.asarray(mg["masks"][t], dtype=np.uint8)).float()
        else:
            mask = torch.empty(0)

        return image, proprio, action, mask

    def __del__(self):
        for attr in ("_file", "_mask_file"):
            try:
                f = getattr(self, attr, None)
                if f is not None:
                    f.close()
            except Exception:
                pass
