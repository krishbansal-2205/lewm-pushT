"""
PushT dataset downloader from HuggingFace.

Downloads the official LeWM PushT HDF5 dataset from:
    https://huggingface.co/datasets/quentinll/lewm-pusht

The file is stored as Zstandard-compressed HDF5 (pusht_expert_train.h5.zst).
This script downloads and decompresses it to a local directory.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def get_data_dir() -> Path:
    """Get the data directory, configurable via LEWM_DATA_DIR env var.

    Returns:
        Path to the data directory.
    """
    default_dir = Path.home() / ".lewm_data" / "pusht"
    data_dir = Path(os.environ.get("LEWM_DATA_DIR", str(default_dir)))
    return data_dir


def download_pusht_dataset(data_dir: Path | None = None, force: bool = False) -> Path:
    """Download the PushT dataset from HuggingFace.

    Downloads `pusht_expert_train.h5.zst` from `quentinll/lewm-pusht`,
    decompresses it to `pusht_expert_train.h5`.

    Args:
        data_dir: Directory to save the dataset. Defaults to ~/.lewm_data/pusht.
        force: If True, re-download even if file exists.

    Returns:
        Path to the decompressed HDF5 file.
    """
    if data_dir is None:
        data_dir = get_data_dir()

    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    h5_path = data_dir / "pusht_expert_train.h5"

    # Skip if already downloaded and decompressed
    if h5_path.exists() and h5_path.stat().st_size > 0 and not force:
        print(f"✓ Dataset already exists: {h5_path}")
        print(f"  Size: {h5_path.stat().st_size / 1e9:.2f} GB")
        return h5_path

    print("=" * 60)
    print("Downloading PushT dataset from HuggingFace")
    print("  Repo: quentinll/lewm-pusht")
    print(f"  Target: {data_dir}")
    print("=" * 60)

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("ERROR: huggingface_hub not installed.")
        print("  Install with: pip install huggingface_hub")
        sys.exit(1)

    # Download the compressed file
    zst_filename = "pusht_expert_train.h5.zst"
    print(f"\n→ Downloading {zst_filename} (~12 GB compressed)...")

    zst_path_cached = hf_hub_download(
        repo_id="quentinll/lewm-pusht",
        filename=zst_filename,
        repo_type="dataset",
        local_dir=str(data_dir),
        local_dir_use_symlinks=False,
    )
    zst_path = Path(zst_path_cached)
    print(f"  Downloaded: {zst_path}")

    # Decompress Zstandard → HDF5
    print(f"\n→ Decompressing to {h5_path}...")
    try:
        import zstandard as zstd
    except ImportError:
        print("ERROR: zstandard not installed.")
        print("  Install with: pip install zstandard")
        sys.exit(1)

    decompressor = zstd.ZstdDecompressor()
    with open(zst_path, "rb") as f_in, open(h5_path, "wb") as f_out:
        decompressor.copy_stream(f_in, f_out, read_size=1024 * 1024, write_size=1024 * 1024)

    print(f"  Decompressed: {h5_path}")
    print(f"  Size: {h5_path.stat().st_size / 1e9:.2f} GB")

    # Verify integrity
    if h5_path.stat().st_size == 0:
        raise RuntimeError(f"Decompressed file is empty: {h5_path}")

    try:
        import h5py
        with h5py.File(h5_path, "r") as f:
            keys = list(f.keys())
            print(f"  HDF5 keys: {keys}")
            for key in keys:
                print(f"    {key}: shape={f[key].shape}, dtype={f[key].dtype}")
    except Exception as e:
        print(f"  WARNING: Could not verify HDF5 structure: {e}")

    # Clean up compressed file to save disk space
    if zst_path.exists() and h5_path.exists() and h5_path.stat().st_size > 0:
        zst_path.unlink()
        print(f"  Cleaned up compressed file: {zst_filename}")

    print("\n✓ Download complete!")
    return h5_path


def main() -> None:
    """CLI entry point for downloading the dataset."""
    import argparse

    parser = argparse.ArgumentParser(description="Download PushT dataset for LeWM")
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Directory to save the dataset (default: ~/.lewm_data/pusht)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if file exists",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir) if args.data_dir else None
    download_pusht_dataset(data_dir=data_dir, force=args.force)


if __name__ == "__main__":
    main()
