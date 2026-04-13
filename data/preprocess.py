"""
Pre-process PushT HDF5 into memory-mapped tensor files.

Run ONCE before training:
    python -m data.preprocess --h5_path dataset/pusht_expert_train.h5

This converts:
  HDF5 pixels (N, 224, 224, 3) uint8
  -> memmap obs.npy     (N_valid, 3, 224, 224) uint8   [CHW, no norm]
  -> memmap next_obs.npy (N_valid, 3, 224, 224) uint8
  -> memmap actions.npy  (N_valid, 2)           float32
  -> meta.pt            {n_samples, action_dim, shape info}

At batch time: GPU does uint8->float32/255->normalize in one fast kernel.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import torch
from tqdm import tqdm

try:
    import hdf5plugin  # noqa: F401
except ImportError:
    pass


def build_valid_indices(h5_path: Path) -> np.ndarray:
    """Build array of valid (non-terminal) frame indices."""
    with h5py.File(h5_path, "r") as f:
        if "ep_offset" in f and "ep_len" in f:
            ep_offsets = f["ep_offset"][:]
            ep_lens = f["ep_len"][:]
            valid = []
            for offset, length in zip(ep_offsets, ep_lens):
                if length > 1:
                    valid.extend(range(int(offset), int(offset) + int(length) - 1))
            return np.array(valid, dtype=np.int64)
        n = f["pixels"].shape[0] if "pixels" in f else f["observations"].shape[0]
        return np.arange(n - 1, dtype=np.int64)


def preprocess_dataset(
    h5_path: Path,
    cache_dir: Optional[Path] = None,
    chunk_size: int = 1024,
    force: bool = False,
) -> Path:
    """
    Convert HDF5 -> memory-mapped numpy arrays.

    Layout in cache_dir/:
        obs.npy          uint8   (N, 3, 224, 224)  CHW, [0,255]
        next_obs.npy     uint8   (N, 3, 224, 224)
        actions.npy      float32 (N, action_dim)
        meta.pt          dict with shape and source metadata

    Args:
        h5_path: Source HDF5 file.
        cache_dir: Where to write cache (default: h5_path.parent/cache/).
        chunk_size: Rows to process at once.
        force: Overwrite existing cache.

    Returns:
        Path to cache_dir.
    """
    h5_path = Path(h5_path)
    if cache_dir is None:
        cache_dir = h5_path.parent / "cache"
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    obs_path = cache_dir / "obs.npy"
    nxt_path = cache_dir / "next_obs.npy"
    act_path = cache_dir / "actions.npy"
    meta_path = cache_dir / "meta.pt"

    if all(p.exists() for p in [obs_path, nxt_path, act_path, meta_path]) and not force:
        print(f"✓ Cache already exists at {cache_dir}")
        meta = torch.load(meta_path, weights_only=False)
        print(f"  N={meta['n_samples']:,}  action_dim={meta['action_dim']}")
        return cache_dir

    print("=" * 60)
    print("Pre-processing PushT dataset -> memory-mapped cache")
    print(f"  Source: {h5_path}")
    print(f"  Cache:  {cache_dir}")
    print("=" * 60)

    t0 = time.time()

    with h5py.File(h5_path, "r") as f:
        obs_key = "pixels" if "pixels" in f else "observations"
        act_key = "action" if "action" in f else "actions"
        next_key = next(
            (k for k in f.keys() if "next" in k.lower() and "obs" in k.lower()),
            None,
        )

        total_frames = f[obs_key].shape[0]
        h, w, c = f[obs_key].shape[1], f[obs_key].shape[2], f[obs_key].shape[3]
        action_dim = f[act_key].shape[1]

    print(f"  Total frames in HDF5: {total_frames:,}")
    print(f"  Image shape: ({h}, {w}, {c})")
    print(f"  Action dim:  {action_dim}")

    valid_indices = build_valid_indices(h5_path)
    n_samples = len(valid_indices)
    print(f"  Valid transitions: {n_samples:,}")
    print(f"  Estimated obs cache size: {n_samples * c * h * w / 1e9:.1f} GB (uint8)")

    print("\nAllocating memory-mapped arrays...")
    obs_mm = np.lib.format.open_memmap(
        obs_path, mode="w+", dtype=np.uint8, shape=(n_samples, c, h, w)
    )
    nxt_mm = np.lib.format.open_memmap(
        nxt_path, mode="w+", dtype=np.uint8, shape=(n_samples, c, h, w)
    )
    act_mm = np.lib.format.open_memmap(
        act_path, mode="w+", dtype=np.float32, shape=(n_samples, action_dim)
    )

    print(f"Processing {n_samples:,} transitions in chunks of {chunk_size}...")

    with h5py.File(h5_path, "r") as f:
        obs_ds = f[obs_key]
        act_ds = f[act_key]
        nxt_ds = f[next_key] if next_key is not None else None

        for start in tqdm(range(0, n_samples, chunk_size), desc="Preprocessing"):
            end = min(start + chunk_size, n_samples)
            chunk_idx = valid_indices[start:end]

            sort_order = np.argsort(chunk_idx)
            sorted_idx = chunk_idx[sort_order]
            rev_order = np.argsort(sort_order)

            obs_chunk = obs_ds[sorted_idx.tolist()]
            act_chunk = act_ds[sorted_idx.tolist()]

            if nxt_ds is not None:
                nxt_chunk = nxt_ds[sorted_idx.tolist()]
            else:
                nxt_chunk = obs_ds[(sorted_idx + 1).tolist()]

            obs_chunk = obs_chunk[rev_order]
            nxt_chunk = nxt_chunk[rev_order]
            act_chunk = act_chunk[rev_order]

            obs_mm[start:end] = obs_chunk.transpose(0, 3, 1, 2)
            nxt_mm[start:end] = nxt_chunk.transpose(0, 3, 1, 2)
            act_mm[start:end] = act_chunk.astype(np.float32)

    obs_mm.flush()
    nxt_mm.flush()
    act_mm.flush()
    del obs_mm, nxt_mm, act_mm

    meta = {
        "n_samples": n_samples,
        "action_dim": action_dim,
        "image_shape": (c, h, w),
        "obs_path": str(obs_path),
        "next_obs_path": str(nxt_path),
        "actions_path": str(act_path),
        "h5_source": str(h5_path),
    }
    torch.save(meta, meta_path)

    elapsed = time.time() - t0
    print(f"\n✓ Pre-processing complete in {elapsed / 60:.1f} min")
    print(f"  Cache: {cache_dir}")
    print(f"  obs.npy:      {obs_path.stat().st_size / 1e9:.1f} GB")
    print(f"  next_obs.npy: {nxt_path.stat().st_size / 1e9:.1f} GB")
    print(f"  actions.npy:  {act_path.stat().st_size / 1e9:.3f} GB")

    return cache_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pre-process PushT HDF5 into memmap cache"
    )
    parser.add_argument("--h5_path", type=str, default="dataset/pusht_expert_train.h5")
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--chunk_size", type=int, default=1024)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    preprocess_dataset(
        h5_path=Path(args.h5_path),
        cache_dir=Path(args.cache_dir) if args.cache_dir else None,
        chunk_size=args.chunk_size,
        force=args.force,
    )


if __name__ == "__main__":
    main()
