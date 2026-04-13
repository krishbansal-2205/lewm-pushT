## Pull Request Title
GPU Pipeline Optimization for PushT Training: Memmap Cache + GPU Normalization + AMP/Compile

## Summary
This contribution addresses low GPU utilization during PushT training by removing HDF5/random-read CPU bottlenecks from the critical training path.

### What changed
1. Added one-time preprocessing pipeline:
- New module: data/preprocess.py
- Converts HDF5 transitions into memory-mapped arrays:
  - obs.npy (uint8, CHW)
  - next_obs.npy (uint8, CHW)
  - actions.npy (float32)
  - meta.pt

2. Added cache-backed dataset and GPU batch normalizer:
- Updated: training/dataset.py
- New CachedPushTDataset for fast memmap-backed indexing
- New NormalizeBatch module to run uint8->float/255->ImageNet normalization on GPU
- get_dataloaders now auto-detects cache and falls back to HDF5 when unavailable

3. Updated training loop for throughput:
- Updated: training/train.py
- Uses CUDAPrefetcher + GPU NormalizeBatch path
- Optional torch.compile support
- Optional AMP mixed precision support
- Keeps HDF5 fallback path supported

4. Updated training entrypoint:
- Updated: train.py
- Added CLI options:
  - --cache_dir
  - --preprocess
  - --no_compile
  - --no_amp
- Auto-runs preprocessing when cache is missing

5. Updated default config:
- Updated: configs/pusht.yaml
- Adds cache_dir, use_amp, use_compile, and tuned worker/batch defaults

## Why this should be merged
- Eliminates major CPU bottlenecks from per-sample HDF5 decoding and normalization.
- Keeps GPU fed with faster host->device and in-GPU normalization.
- Preserves compatibility with existing HDF5 workflow via fallback.
- Adds a reproducible one-time preprocessing command for production training runs.

## Repro / Usage
1. Preprocess once:
python -m data.preprocess --h5_path dataset/pusht_expert_train.h5 --cache_dir dataset/cache --chunk_size 2048

2. Train with optimized path:
python train.py --config configs/pusht.yaml --cache_dir dataset/cache

3. Force fallback test (optional):
python train.py --config configs/pusht.yaml --cache_dir nonexistent_dir

## Notes
- I could not run Python validation in this environment because Python is not installed in the current terminal image.
- Please run your CI/tests after pulling this branch.

## Suggested reviewer checklist
- Validate preprocessing output shape and sizes on PushT dataset.
- Confirm train/val loaders use CachedPushTDataset when cache exists.
- Compare GPU utilization and step time before/after.
- Run existing unit tests and training smoke test.
