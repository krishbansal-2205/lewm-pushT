"""
PushT HDF5 dataset loader for LeWM training.

Loads the PushT expert demonstrations from HDF5 file with lazy loading
to minimize RAM usage. Provides train/val splits and data augmentation.

The HDF5 file (from quentinll/lewm-pusht) contains:
  - observations:      (N, H, W, 3) uint8 images
  - actions:           (N, action_dim) float32
  - next_observations: (N, H, W, 3) uint8 images
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import h5py
import hdf5plugin
import numpy as np
import torch
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader, Dataset, Subset

# ImageNet normalization constants
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


class PushTDataset(Dataset):
    """PushT dataset from HDF5 file.

    Loads observations, actions, and next observations with lazy loading
    (only reads individual samples from disk on __getitem__).

    Args:
        h5_path: Path to the HDF5 file.
        augmentation: If True, apply random horizontal flips.
        image_size: Expected image size (default: 96).
    """

    def __init__(
        self,
        h5_path: str | Path,
        augmentation: bool = True,
        image_size: int = 224,
    ) -> None:
        self.h5_path = Path(h5_path)
        self.augmentation = augmentation
        self.image_size = image_size

        if not self.h5_path.exists():
            raise FileNotFoundError(
                f"Dataset file not found: {self.h5_path}\n"
                f"Download it with: python -m lewm_pusht.data.download"
            )

        # Open file to get dataset length and verify structure
        with h5py.File(self.h5_path, "r") as f:
            self._keys = list(f.keys())

            if "pixels" in f:
                self._obs_key = "pixels"
            elif "observations" in f:
                self._obs_key = "observations"
            elif "observation" in f:
                self._obs_key = "observation"
            else:
                self._obs_key = None
                for k in f.keys():
                    if "obs" in k.lower() and "next" not in k.lower():
                        self._obs_key = k
                        break
                if not self._obs_key:
                    raise KeyError(
                        f"Cannot find observation keys in HDF5 file. Keys: {self._keys}")

            if "actions" in f:
                self._action_key = "actions"
            elif "action" in f:
                self._action_key = "action"
            else:
                raise KeyError(f"Cannot find action key. Keys: {self._keys}")

            self._has_next_obs = False
            for k in f.keys():
                if "next" in k.lower() and "obs" in k.lower():
                    self._next_obs_key = k
                    self._has_next_obs = True

            # Setup valid indices correctly for episode-based datasets
            self._length = f[self._obs_key].shape[0]
            if "ep_offset" in f and "ep_len" in f:
                # LeRobot style dataset
                ep_offsets = f["ep_offset"][:]
                ep_lens = f["ep_len"][:]
                valid_indices = []
                for offset, length in zip(ep_offsets, ep_lens):
                    if length > 1:
                        # Exclude the last frame of each episode since it has no next_obs
                        valid_indices.extend(
                            range(offset, offset + length - 1))
                self._valid_indices = np.array(valid_indices)
            else:
                # Assuming generic transitions or _has_next_obs is True
                if self._has_next_obs:
                    self._valid_indices = np.arange(self._length)
                else:
                    self._valid_indices = np.arange(self._length - 1)

        # h5py file handle for lazy loading (opened per-worker in DataLoader)
        self._h5_file: Optional[h5py.File] = None

    def _get_h5(self) -> h5py.File:
        """Get or open the HDF5 file handle (thread-safe for DataLoader workers)."""
        if self._h5_file is None:
            # Open with SWMR (Single Writer Multiple Reader) for faster concurrent reads
            self._h5_file = h5py.File(self.h5_path, "r", swmr=True, libver='latest', rdcc_nbytes=1024**3,
                                      rdcc_nslots=1e6)
        return self._h5_file

    def __len__(self) -> int:
        return len(self._valid_indices)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single sample.

        Returns:
            Dict with keys:
                - 'obs': Normalized observation, shape (3, H, W), float32.
                - 'action': Action vector, shape (action_dim,), float32.
                - 'next_obs': Normalized next observation, shape (3, H, W), float32.
                - 'obs_raw': Un-normalized observation for visualization, (3, H, W).
                - 'next_obs_raw': Un-normalized next observation, (3, H, W).
        """
        h5 = self._get_h5()
        real_idx = self._valid_indices[idx]

        # Load data from HDF5
        obs = h5[self._obs_key][real_idx]           # (H, W, 3) uint8
        action = h5[self._action_key][real_idx]     # (action_dim,) float32

        if self._has_next_obs:
            next_obs = h5[self._next_obs_key][real_idx]  # (H, W, 3) uint8
        else:
            next_obs = h5[self._obs_key][real_idx + 1]

        # Convert to float tensors: HWC → CHW, [0, 255] → [0, 1]
        obs = torch.from_numpy(obs.astype(np.float32)).permute(2, 0, 1) 
        next_obs = torch.from_numpy(next_obs.astype(
            np.float32)).permute(2, 0, 1) 
        action = torch.from_numpy(action.astype(np.float32))

        # if obs.shape[-1] != self.image_size or obs.shape[-2] != self.image_size:
        #     obs = TF.resize(
        #         obs, [self.image_size, self.image_size], antialias=True)
        #     next_obs = TF.resize(
        #         next_obs, [self.image_size, self.image_size], antialias=True)

        # Store raw (un-normalized) copies for visualization
        obs_raw = obs.clone()
        next_obs_raw = next_obs.clone()

        # Apply augmentation: random horizontal flip
        if self.augmentation and torch.rand(1).item() > 0.5:
            obs = torch.flip(obs, dims=[-1])
            next_obs = torch.flip(next_obs, dims=[-1])
            # Flip action x-component for consistency (PushT: action[0] is x-velocity)
            action[0] = -action[0]

        # Normalize with ImageNet stats
        obs = (obs - IMAGENET_MEAN) / IMAGENET_STD
        next_obs = (next_obs - IMAGENET_MEAN) / IMAGENET_STD

        return {
            "obs": obs,
            "action": action,
            "next_obs": next_obs,
            "obs_raw": obs_raw,
            "next_obs_raw": next_obs_raw,
        }

    def __del__(self) -> None:
        """Clean up HDF5 file handle."""
        if getattr(self, "_h5_file", None) is not None:
            try:
                self._h5_file.close()
            except Exception:
                pass


def get_dataloaders(
    h5_path: str | Path,
    batch_size: int = 256,
    train_split: float = 0.9,
    augmentation: bool = True,
    num_workers: int = 16,
    image_size: int = 224,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, PushTDataset]:
    """Create train and validation DataLoaders.

    Args:
        h5_path: Path to the HDF5 dataset file.
        batch_size: Batch size for training (default: 256).
        train_split: Fraction of data for training (default: 0.9).
        augmentation: Apply data augmentation to train set (default: True).
        num_workers: Number of DataLoader workers (default: 4).
        image_size: Expected image size (default: 96).
        seed: Random seed for reproducible splits (default: 42).

    Returns:
        Tuple of (train_loader, val_loader, full_dataset).
    """
    # Create full dataset (no augmentation for val)
    full_dataset = PushTDataset(
        h5_path=h5_path,
        augmentation=False,
        image_size=image_size,
    )
    n = len(full_dataset)
    n_train = int(n * train_split)

    # Generate reproducible random split
    rng = np.random.RandomState(seed)
    indices = rng.permutation(n)
    train_indices = indices[:n_train].tolist()
    val_indices = indices[n_train:].tolist()

    # Create train dataset with augmentation
    train_dataset = PushTDataset(
        h5_path=h5_path,
        augmentation=augmentation,
        image_size=image_size,
    )

    train_subset = Subset(train_dataset, train_indices)
    val_subset = Subset(full_dataset, val_indices)

    use_cuda = torch.cuda.is_available()

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=use_cuda,
        drop_last=True,
        prefetch_factor=4,
        persistent_workers=num_workers > 0,
    )

    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_cuda,
        drop_last=False,
        prefetch_factor=4,
        persistent_workers=num_workers > 0,
    )

    print(f"Dataset loaded: {n} total samples")
    print(f"  Train: {len(train_indices)} samples ({n_train} / {n})")
    print(f"  Val:   {len(val_indices)} samples ({n - n_train} / {n})")
    print(f"  Batch size: {batch_size}")
    print(f"  Augmentation: {augmentation}")

    return train_loader, val_loader, full_dataset
