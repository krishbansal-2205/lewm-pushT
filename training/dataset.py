"""
PushT dataset loader for LeWM training.

Two backends:
  1. CachedPushTDataset  - reads from pre-processed memmap files (fast)
  2. PushTDataset        - lazy HDF5 loader (fallback)

GPU utilization change:
  - CachedPushTDataset returns uint8 tensors from memmap
  - NormalizeBatch performs uint8->float32->ImageNet normalization on GPU
  - Workers avoid heavy per-sample preprocessing
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


class NormalizeBatch(nn.Module):
    """Convert uint8 image batches to ImageNet-normalized float tensors."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("mean", IMAGENET_MEAN.clone())
        self.register_buffer("std", IMAGENET_STD.clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float().mul_(1.0 / 255.0)
        x.sub_(self.mean).div_(self.std)
        return x

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        return x.mul(self.std).add(self.mean).clamp(0.0, 1.0)


class CachedPushTDataset(Dataset):
    """PushT dataset backed by memory-mapped arrays produced by data.preprocess."""

    def __init__(
        self,
        cache_dir: str | Path,
        augmentation: bool = False,
        split_indices: Optional[np.ndarray] = None,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.augmentation = augmentation

        meta_path = self.cache_dir / "meta.pt"
        if not meta_path.exists():
            raise FileNotFoundError(
                f"Cache not found: {meta_path}\n"
                "Run first: python -m data.preprocess --h5_path <dataset>/pusht_expert_train.h5"
            )

        meta = torch.load(meta_path, weights_only=False)
        self._n_total = meta["n_samples"]

        self._obs = np.load(self.cache_dir / "obs.npy", mmap_mode="r")
        self._nxt = np.load(self.cache_dir / "next_obs.npy", mmap_mode="r")
        self._act = np.load(self.cache_dir / "actions.npy", mmap_mode="r")

        if split_indices is not None:
            self._indices = split_indices.astype(np.int64)
        else:
            self._indices = np.arange(self._n_total, dtype=np.int64)

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        real_idx = int(self._indices[idx])

        obs = np.array(self._obs[real_idx])
        nxt = np.array(self._nxt[real_idx])
        action = np.array(self._act[real_idx])

        if self.augmentation and np.random.random() > 0.5:
            obs = obs[:, :, ::-1].copy()
            nxt = nxt[:, :, ::-1].copy()
            action = action.copy()
            action[0] = -action[0]

        return {
            "obs": torch.from_numpy(obs),
            "next_obs": torch.from_numpy(nxt),
            "action": torch.from_numpy(action),
        }


class PushTDataset(Dataset):
    """Lazy HDF5 loader fallback when memmap cache is unavailable."""

    def __init__(
        self,
        h5_path: str | Path,
        augmentation: bool = True,
        image_size: int = 224,
    ) -> None:
        import h5py

        self.h5_path = Path(h5_path)
        self.augmentation = augmentation
        self.image_size = image_size

        if not self.h5_path.exists():
            raise FileNotFoundError(
                f"Dataset file not found: {self.h5_path}\n"
                "Download with: python -m data.download"
            )

        self._h5_file: Optional[h5py.File] = None

        with h5py.File(self.h5_path, "r") as f:
            self._keys = list(f.keys())

            if "pixels" in f:
                self._obs_key = "pixels"
            elif "observations" in f:
                self._obs_key = "observations"
            else:
                raise KeyError(f"No obs key found. Keys: {self._keys}")

            self._action_key = "action" if "action" in f else "actions"

            self._has_next_obs = False
            for k in f.keys():
                if "next" in k.lower() and "obs" in k.lower():
                    self._next_obs_key = k
                    self._has_next_obs = True
                    break

            self._length = f[self._obs_key].shape[0]
            if "ep_offset" in f and "ep_len" in f:
                ep_offsets = f["ep_offset"][:]
                ep_lens = f["ep_len"][:]
                valid: list[int] = []
                for offset, length in zip(ep_offsets, ep_lens):
                    if length > 1:
                        valid.extend(range(int(offset), int(offset) + int(length) - 1))
                self._valid_indices = np.array(valid, dtype=np.int64)
            else:
                self._valid_indices = np.arange(
                    self._length - 1 if not self._has_next_obs else self._length,
                    dtype=np.int64,
                )

    def _get_h5(self):
        import h5py

        if self._h5_file is None:
            try:
                self._h5_file = h5py.File(
                    self.h5_path,
                    "r",
                    swmr=True,
                    libver="latest",
                    rdcc_nbytes=512 * 1024 * 1024,
                )
            except OSError:
                self._h5_file = h5py.File(
                    self.h5_path,
                    "r",
                    rdcc_nbytes=512 * 1024 * 1024,
                )
        return self._h5_file

    def __len__(self) -> int:
        return len(self._valid_indices)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        h5 = self._get_h5()
        real_idx = int(self._valid_indices[idx])

        obs_np = h5[self._obs_key][real_idx]
        act_np = h5[self._action_key][real_idx]

        if self._has_next_obs:
            nxt_np = h5[self._next_obs_key][real_idx]
        else:
            nxt_np = h5[self._obs_key][real_idx + 1]

        obs = torch.from_numpy(obs_np.astype(np.float32)).permute(2, 0, 1) / 255.0
        nxt = torch.from_numpy(nxt_np.astype(np.float32)).permute(2, 0, 1) / 255.0
        action = torch.from_numpy(act_np.astype(np.float32))

        obs_raw = obs.clone()
        nxt_raw = nxt.clone()

        if self.augmentation and torch.rand(1).item() > 0.5:
            obs = torch.flip(obs, dims=[-1])
            nxt = torch.flip(nxt, dims=[-1])
            action = action.clone()
            action[0] = -action[0]

        obs = (obs - IMAGENET_MEAN) / IMAGENET_STD
        nxt = (nxt - IMAGENET_MEAN) / IMAGENET_STD

        return {
            "obs": obs,
            "next_obs": nxt,
            "action": action,
            "obs_raw": obs_raw,
            "next_obs_raw": nxt_raw,
        }

    def __del__(self):
        if getattr(self, "_h5_file", None) is not None:
            try:
                self._h5_file.close()
            except Exception:
                pass


def get_dataloaders(
    h5_path: Optional[str | Path] = None,
    cache_dir: Optional[str | Path] = None,
    batch_size: int = 256,
    train_split: float = 0.9,
    augmentation: bool = True,
    num_workers: int = 8,
    image_size: int = 224,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, NormalizeBatch]:
    """Create optimized train/val DataLoaders with cache auto-detection."""
    use_cache = False

    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        meta_path = cache_dir / "meta.pt"
        if meta_path.exists():
            use_cache = True
        else:
            print(f"⚠ Cache not found at {cache_dir}, falling back to HDF5.")
            print(f"  Run: python -m data.preprocess --h5_path {h5_path}")

    if use_cache:
        print(f"✓ Using pre-processed cache: {cache_dir}")

        meta = torch.load(cache_dir / "meta.pt", weights_only=False)
        n = meta["n_samples"]
        rng = np.random.RandomState(seed)
        indices = rng.permutation(n)
        n_train = int(n * train_split)
        train_idx = indices[:n_train]
        val_idx = indices[n_train:]

        train_ds = CachedPushTDataset(cache_dir, augmentation=augmentation, split_indices=train_idx)
        val_ds = CachedPushTDataset(cache_dir, augmentation=False, split_indices=val_idx)

        use_cuda = torch.cuda.is_available()
        worker_kwargs = dict(
            num_workers=num_workers,
            pin_memory=use_cuda,
            prefetch_factor=8 if num_workers > 0 else None,
            persistent_workers=num_workers > 0,
        )

        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            **worker_kwargs,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size * 2,
            shuffle=False,
            drop_last=False,
            **worker_kwargs,
        )

        print(f"Dataset loaded: {n:,} total samples")
        print(f"  Train: {n_train:,} | Val: {n - n_train:,}")
        print(f"  Batch size: {batch_size} train / {batch_size * 2} val")
        print(f"  Workers: {num_workers} | prefetch_factor: 8")
        print("  Backend: CachedPushTDataset (memmap, uint8)")

    else:
        if h5_path is None:
            raise ValueError("Either cache_dir or h5_path must be provided.")
        h5_path = Path(h5_path)
        print(f"⚠ Using slow HDF5 backend: {h5_path}")
        print(f"  Speed up by running: python -m data.preprocess --h5_path {h5_path}")

        full_ds = PushTDataset(h5_path=h5_path, augmentation=False, image_size=image_size)
        n = len(full_ds)
        rng = np.random.RandomState(seed)
        indices = rng.permutation(n)
        n_train = int(n * train_split)

        train_ds = PushTDataset(h5_path=h5_path, augmentation=augmentation, image_size=image_size)
        train_subset = Subset(train_ds, indices[:n_train].tolist())
        val_subset = Subset(full_ds, indices[n_train:].tolist())

        use_cuda = torch.cuda.is_available()
        worker_kwargs = dict(
            num_workers=num_workers,
            pin_memory=use_cuda,
            prefetch_factor=4 if num_workers > 0 else None,
            persistent_workers=num_workers > 0,
        )

        train_loader = DataLoader(
            train_subset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            **worker_kwargs,
        )
        val_loader = DataLoader(
            val_subset,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            **worker_kwargs,
        )

        print(f"Dataset loaded: {n:,} total | Train: {n_train:,} | Val: {n - n_train:,}")

    normalizer = NormalizeBatch()
    return train_loader, val_loader, normalizer
