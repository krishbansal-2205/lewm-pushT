#!/usr/bin/env python3
"""
LeWM Training Entry Point - GPU-optimized.

Workflow:
  1. Pre-process once:
       python -m data.preprocess --h5_path dataset/pusht_expert_train.h5
  2. Train:
       python train.py --config configs/pusht.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from omegaconf import OmegaConf

from data.download import download_pusht_dataset, get_data_dir
from data.preprocess import preprocess_dataset
from models.lewm import LeWM
from training.dataset import get_dataloaders
from training.train import set_seed, train_lewm


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train LeWM on PushT",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=str, default="configs/pusht.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--lambda_reg", type=float, default=None)
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="Pre-processed memmap cache (faster than HDF5)",
    )
    parser.add_argument("--checkpoint_dir", type=str, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--download", action="store_true")
    parser.add_argument(
        "--preprocess",
        action="store_true",
        help="Run preprocessing before training",
    )
    parser.add_argument(
        "--no_compile",
        action="store_true",
        help="Disable torch.compile()",
    )
    parser.add_argument(
        "--no_amp",
        action="store_true",
        help="Disable mixed-precision training",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    config = OmegaConf.load(config_path) if config_path.exists() else OmegaConf.create({})

    cli_overrides = {}
    for key in [
        "epochs",
        "batch_size",
        "lr",
        "lambda_reg",
        "data_dir",
        "cache_dir",
        "checkpoint_dir",
        "num_workers",
        "seed",
        "device",
    ]:
        val = getattr(args, key)
        if val is not None:
            cli_overrides[key] = val
    if args.no_compile:
        cli_overrides["use_compile"] = False
    if args.no_amp:
        cli_overrides["use_amp"] = False
    if cli_overrides:
        config = OmegaConf.merge(config, OmegaConf.create(cli_overrides))

    defaults = {
        "latent_dim": 192,
        "encoder_channels": [32, 64, 128, 256],
        "predictor_hidden": [512, 512, 512],
        "dropout": 0.1,
        "action_dim": 2,
        "image_size": 224,
        "batch_size": 512,
        "epochs": 100,
        "lr": 3e-4,
        "weight_decay": 1e-4,
        "grad_clip": 1.0,
        "lambda_reg": 0.1,
        "early_stopping_patience": 20,
        "checkpoint_dir": "checkpoints",
        "sigreg_num_projections": 64,
        "train_split": 0.9,
        "augmentation": True,
        "num_workers": 8,
        "seed": 42,
        "device": "cuda",
        "log_every": 100,
        "use_amp": True,
        "use_compile": True,
        "data_dir": str(get_data_dir()),
        "cache_dir": None,
    }
    for k, v in defaults.items():
        if k not in config or config[k] is None:
            OmegaConf.update(config, k, v)

    if config.device == "cuda" and not torch.cuda.is_available():
        print("⚠ CUDA not available, falling back to CPU")
        config.device = "cpu"
    device = torch.device(config.device)

    set_seed(config.seed)

    print("\n" + "═" * 60)
    print("LeWM Training Configuration")
    print("═" * 60)
    print(OmegaConf.to_yaml(config))
    print("═" * 60)

    data_dir = Path(config.data_dir).expanduser()
    h5_path = data_dir / "pusht_expert_train.h5"

    if args.download or not h5_path.exists():
        h5_path = download_pusht_dataset(data_dir=data_dir)

    raw_cache = getattr(config, "cache_dir", None)
    if raw_cache is not None:
        cache_dir = Path(raw_cache).expanduser()
    else:
        cache_dir = data_dir / "cache"

    cache_meta = cache_dir / "meta.pt"
    if args.preprocess or not cache_meta.exists():
        if not cache_meta.exists():
            print(f"\n⚠ Cache not found at {cache_dir}")
            print("  Running pre-processing (one-time setup)...")
        preprocess_dataset(h5_path=h5_path, cache_dir=cache_dir)

    print("\nCreating DataLoaders...")
    train_loader, val_loader, normalizer = get_dataloaders(
        h5_path=h5_path,
        cache_dir=cache_dir,
        batch_size=config.batch_size,
        train_split=config.train_split,
        augmentation=config.augmentation,
        num_workers=config.num_workers,
        image_size=config.image_size,
        seed=config.seed,
    )

    print("\nBuilding LeWM model...")
    model = LeWM(config).to(device)
    param_counts = model.count_parameters()
    print("\nModel Parameters:")
    print(f"  Encoder:   {param_counts['encoder']:>10,}")
    print(f"  Predictor: {param_counts['predictor']:>10,}")
    print(f"  Total:     {param_counts['total']:>10,}")

    train_lewm(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device,
        checkpoint_dir=Path(config.checkpoint_dir),
        normalizer=normalizer,
    )

    print("\n✓ Training complete!")
    print(f"  Best model:  {Path(config.checkpoint_dir) / 'best.pt'}")
    print(f"  Final model: {Path(config.checkpoint_dir) / 'final.pt'}")


if __name__ == "__main__":
    main()
