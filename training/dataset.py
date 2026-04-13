"""
PushT HDF5 dataset loader for LeWM training.

Loads the PushT expert demonstrations from HDF5 file with optional RAM
caching to eliminate the HDF5 random-I/O bottleneck that causes GPU
utilization sawtooth patterns.

The HDF5 file (from quentinll/lewm-pusht) contains:
  - pixels:      (N, H, W, 3) uint8 images
  - action:      (N, action_dim) float32
  Episode boundary metadata: ep_offset, ep_len

FIXES applied vs original:
  1. /255.0 added when converting uint8→float so ImageNet normalization
     receives values in [0, 1] instead of [0, 255].
  2. obs_raw / next_obs_raw are now cloned AFTER the /255.0 scaling,
     so they live in [0, 1] as the tests expect.
  3. SWMR open mode now falls back gracefully — the downloaded HDF5
     was not written with SWMR enabled and raised an OSError.
  4. RAM caching: pixel + action arrays are loaded fully into memory on
     init, so __getitem__ is a pure memory operation.  This eliminates
     the HDF5 random-I/O bottleneck (primary cause of GPU sawtooth).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import h5py
import numpy as np
import torch
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader, Dataset, Subset

try:
    import hdf5plugin  # noqa: F401  – registers Blosc/LZ4 codecs if present
except ImportError:
    pass

# ImageNet normalization constants (for [0, 1] inputs)
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


class PushTDataset(Dataset):
    """PushT dataset from HDF5 file.

    Loads observations, actions, and next observations. Supports two modes:

    1. **RAM-cached (default)**: Loads all pixel + action data into numpy
       arrays at init time. __getitem__ does zero disk I/O — just array
       indexing. Requires ~46 GB RAM for the full PushT dataset but
       completely eliminates the HDF5 I/O bottleneck.

    2. **Lazy-loading**: Reads individual samples from HDF5 on each
       __getitem__. Uses much less RAM but is extremely slow for random
       access on large files, causing GPU starvation (sawtooth).

    Args:
        h5_path:      Path to the HDF5 file.
        augmentation: If True, apply random horizontal flips.
        image_size:   Expected image size (default: 224).
        cache_in_ram: If True, load all data into RAM at init (default: True).
    """

    def __init__(
        self,
        h5_path: str | Path,
        augmentation: bool = True,
        image_size: int = 224,
        cache_in_ram: bool = True,
    ) -> None:
        self.h5_path = Path(h5_path)
        self.augmentation = augmentation
        self.image_size = image_size
        self._cache_in_ram = cache_in_ram

        if not self.h5_path.exists():
            raise FileNotFoundError(
                f"Dataset file not found: {self.h5_path}\n"
                f"Download it with: python -m data.download"
            )

        # Open file once to read metadata + optionally cache data.
        with h5py.File(self.h5_path, "r") as f:
            self._keys = list(f.keys())

            # ── observation key ──────────────────────────────
            if "pixels" in f:
                self._obs_key = "pixels"
            elif "observations" in f:
                self._obs_key = "observations"
            elif "observation" in f:
                self._obs_key = "observation"
            else:
                self._obs_key = next(
                    (k for k in f.keys() if "obs" in k.lower()
                     and "next" not in k.lower()),
                    None,
                )
                if not self._obs_key:
                    raise KeyError(
                        f"Cannot find observation key in HDF5 file. Keys: {self._keys}"
                    )

            # ── action key ───────────────────────────────────
            if "actions" in f:
                self._action_key = "actions"
            elif "action" in f:
                self._action_key = "action"
            else:
                raise KeyError(f"Cannot find action key. Keys: {self._keys}")

            # ── next-observation key ─────────────────────────
            self._has_next_obs = False
            for k in f.keys():
                if "next" in k.lower() and "obs" in k.lower():
                    self._next_obs_key = k
                    self._has_next_obs = True
                    break

            # ── valid index range ────────────────────────────
            self._length = f[self._obs_key].shape[0]
            if "ep_offset" in f and "ep_len" in f:
                # LeRobot-style dataset: exclude last frame of each episode
                # because it has no valid next observation.
                ep_offsets = f["ep_offset"][:]
                ep_lens = f["ep_len"][:]
                valid: list[int] = []
                for offset, length in zip(ep_offsets, ep_lens):
                    if length > 1:
                        valid.extend(
                            range(int(offset), int(offset) + int(length) - 1))
                self._valid_indices = np.array(valid, dtype=np.int64)
            else:
                if self._has_next_obs:
                    self._valid_indices = np.arange(
                        self._length, dtype=np.int64)
                else:
                    self._valid_indices = np.arange(
                        self._length - 1, dtype=np.int64)

            # ── RAM cache ────────────────────────────────────
            if cache_in_ram:
                print(f"  Caching dataset in RAM ({self._length} frames)...")
                # Keep pixels as uint8 to save 4× RAM (~46 GB vs ~184 GB).
                # float32 conversion + /255 happens in __getitem__.
                self._pixels_cache = f[self._obs_key][:]       # (N, H, W, 3) uint8
                self._action_cache = f[self._action_key][:]    # (N, action_dim) float32
                if self._has_next_obs:
                    self._next_obs_cache = f[self._next_obs_key][:]
                else:
                    self._next_obs_cache = None
                print(f"  Cached: pixels={self._pixels_cache.nbytes / 1e9:.1f} GB, "
                      f"actions={self._action_cache.nbytes / 1e6:.1f} MB")
            else:
                self._pixels_cache = None
                self._action_cache = None
                self._next_obs_cache = None

        # File handle opened lazily per DataLoader worker (only for non-cached mode).
        self._h5_file: Optional[h5py.File] = None

    # ──────────────────────────────────────────────────────────────────────
    # HDF5 handle (lazy, per-worker) — only used when cache_in_ram=False
    # ──────────────────────────────────────────────────────────────────────

    def _get_h5(self) -> h5py.File:
        """Return (or open) the HDF5 file handle for this worker.

        FIX: SWMR mode requires the file to have been written with
        libver='latest' and SWMR enabled.  The HuggingFace download does
        NOT satisfy this requirement and raises an OSError at open time.
        We now fall back to a normal read-only open automatically.
        """
        if self._h5_file is None:
            try:
                self._h5_file = h5py.File(
                    self.h5_path,
                    "r",
                    swmr=True,
                    libver="latest",
                    rdcc_nbytes=256 * 1024 * 1024,   # 256 MB read cache
                    rdcc_nslots=int(1e5),
                )
            except OSError:
                # File was not created with SWMR support — open normally.
                self._h5_file = h5py.File(
                    self.h5_path,
                    "r",
                    rdcc_nbytes=256 * 1024 * 1024,
                    rdcc_nslots=int(1e5),
                )
        return self._h5_file

    # ──────────────────────────────────────────────────────────────────────
    # Dataset protocol
    # ──────────────────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._valid_indices)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Return one (obs, action, next_obs) transition.

        Returns
        -------
        Dict with keys:
            obs        : ImageNet-normalised float32 tensor, shape (3, H, W).
            action     : float32 tensor, shape (action_dim,).
            next_obs   : ImageNet-normalised float32 tensor, shape (3, H, W).
            obs_raw    : un-normalised [0, 1] float32 tensor, shape (3, H, W).
            next_obs_raw: un-normalised [0, 1] float32 tensor, shape (3, H, W).
        """
        real_idx = int(self._valid_indices[idx])

        if self._pixels_cache is not None:
            # ── Fast path: read from RAM cache ────────────────
            obs_np = self._pixels_cache[real_idx]        # (H, W, 3) uint8
            act_np = self._action_cache[real_idx]        # (action_dim,) float32

            if self._next_obs_cache is not None:
                nxt_np = self._next_obs_cache[real_idx]
            else:
                nxt_np = self._pixels_cache[real_idx + 1]
        else:
            # ── Slow path: read from HDF5 ─────────────────────
            h5 = self._get_h5()
            obs_np = h5[self._obs_key][real_idx]         # (H, W, 3) uint8
            act_np = h5[self._action_key][real_idx]      # (action_dim,) float32

            if self._has_next_obs:
                nxt_np = h5[self._next_obs_key][real_idx]
            else:
                nxt_np = h5[self._obs_key][real_idx + 1]

        # ── uint8 → float32, HWC → CHW, scale to [0, 1] ─────
        # FIX: divide by 255 here so ImageNet mean/std (designed for [0,1])
        # are applied correctly.  The original code omitted this step,
        # resulting in normalised values around ±550 instead of ±2.
        obs = torch.from_numpy(obs_np.astype(np.float32)
                               ).permute(2, 0, 1) / 255.0
        next_obs = torch.from_numpy(nxt_np.astype(
            np.float32)).permute(2, 0, 1) / 255.0
        action = torch.from_numpy(act_np.astype(np.float32))

        # ── Store raw [0, 1] copies for visualisation ────────
        # FIX: clone AFTER /255 so obs_raw is in [0, 1] as expected by
        # tests and visualisation code.
        obs_raw = obs.clone()
        next_obs_raw = next_obs.clone()

        # ── Data augmentation: random horizontal flip ────────
        if self.augmentation and torch.rand(1).item() > 0.5:
            obs = torch.flip(obs,      dims=[-1])
            next_obs = torch.flip(next_obs, dims=[-1])
            # Mirror x-velocity for physical consistency.
            action[0] = -action[0]

        # ── ImageNet normalisation ────────────────────────────
        obs = (obs - IMAGENET_MEAN) / IMAGENET_STD
        next_obs = (next_obs - IMAGENET_MEAN) / IMAGENET_STD

        return {
            "obs":          obs,
            "action":       action,
            "next_obs":     next_obs,
            "obs_raw":      obs_raw,
            "next_obs_raw": next_obs_raw,
        }

    def __del__(self) -> None:
        if getattr(self, "_h5_file", None) is not None:
            try:
                self._h5_file.close()
            except Exception:
                pass


# ─────────────────────────────────────────────────────────────────────────────
# DataLoader factory
# ─────────────────────────────────────────────────────────────────────────────

def get_dataloaders(
    h5_path: str | Path,
    batch_size: int = 256,
    train_split: float = 0.9,
    augmentation: bool = True,
    num_workers: int = 8,
    image_size: int = 224,
    seed: int = 42,
    cache_in_ram: bool = True,
) -> Tuple[DataLoader, DataLoader, "PushTDataset"]:
    """Create train and validation DataLoaders.

    Args:
        h5_path:      Path to the HDF5 dataset file.
        batch_size:   Batch size for training (default: 256).
        train_split:  Fraction of data for training (default: 0.9).
        augmentation: Apply random flips to the train split (default: True).
        num_workers:  DataLoader worker count (default: 8).
        image_size:   Expected image size (default: 224).
        seed:         Random seed for reproducible splits (default: 42).
        cache_in_ram: Cache all data in RAM for fast loading (default: True).

    Returns:
        Tuple of (train_loader, val_loader, full_dataset).
    """
    full_dataset = PushTDataset(
        h5_path=h5_path, augmentation=False, image_size=image_size,
        cache_in_ram=cache_in_ram)
    n = len(full_dataset)
    n_train = int(n * train_split)

    rng = np.random.RandomState(seed)
    indices = rng.permutation(n)
    train_indices = indices[:n_train].tolist()
    val_indices = indices[n_train:].tolist()

    train_dataset = PushTDataset(
        h5_path=h5_path, augmentation=augmentation, image_size=image_size,
        cache_in_ram=cache_in_ram)
    train_subset = Subset(train_dataset, train_indices)
    val_subset = Subset(full_dataset,  val_indices)

    use_cuda = torch.cuda.is_available()

    # With RAM-cached data, workers just do numpy indexing + torch ops.
    # Fewer workers needed, and persistent_workers avoid re-caching.
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

    print(f"Dataset loaded: {n} total samples")
    print(f"  Train: {len(train_indices)} samples ({n_train} / {n})")
    print(f"  Val:   {len(val_indices)} samples ({n - n_train} / {n})")
    print(f"  Batch size: {batch_size}")
    print(f"  Augmentation: {augmentation}")
    print(f"  RAM cache: {'enabled' if cache_in_ram else 'disabled'}")

    return train_loader, val_loader, full_dataset
