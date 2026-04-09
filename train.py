#!/usr/bin/env python3
"""
LeWM Training Entry Point.

Train LeWorldModel on the PushT dataset.

Usage:
    python train.py --config configs/pusht.yaml
    python train.py --config configs/pusht.yaml --epochs 50 --lr 1e-4
    python train.py --epochs 2 --batch_size 32  # Quick dry-run
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf

from models.lewm import LeWM
from training.train import set_seed, train_lewm
from training.dataset import get_dataloaders
from data.download import download_pusht_dataset, get_data_dir


def main() -> None:
    """Main training entry point."""
    parser = argparse.ArgumentParser(
        description="Train LeWM on PushT",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/pusht.yaml",
        help="Path to config YAML file",
    )
    # Allow CLI overrides for any config parameter
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--lambda_reg", type=float, default=None)
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--checkpoint_dir", type=str, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--download", action="store_true", help="Download dataset first")
    args = parser.parse_args()

    # Load config
    config_path = Path(args.config)
    if config_path.exists():
        config = OmegaConf.load(config_path)
    else:
        print(f"⚠ Config file not found: {config_path}")
        print("  Using default configuration.")
        config = OmegaConf.create({})

    # Apply CLI overrides
    cli_overrides = {}
    for key in ["epochs", "batch_size", "lr", "lambda_reg", "data_dir",
                 "checkpoint_dir", "num_workers", "seed", "device"]:
        val = getattr(args, key)
        if val is not None:
            cli_overrides[key] = val
    if cli_overrides:
        config = OmegaConf.merge(config, OmegaConf.create(cli_overrides))

    # Set defaults for anything not in config
    defaults = {
        "latent_dim": 192,
        "encoder_channels": [32, 64, 128, 256],
        "predictor_hidden": [512, 512, 512],
        "dropout": 0.1,
        "action_dim": 2,
        "image_size": 224,
        "batch_size": 256,
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
        "num_workers": 4,
        "seed": 42,
        "device": "cuda",
        "log_every": 100,
        "data_dir": str(get_data_dir()),
    }
    for k, v in defaults.items():
        if not OmegaConf.is_missing(config, k) and k in config:
            continue
        OmegaConf.update(config, k, v)

    # Device setup
    if config.device == "cuda" and not torch.cuda.is_available():
        print("⚠ CUDA not available, falling back to CPU")
        config.device = "cpu"
    device = torch.device(config.device)

    # Set seed
    set_seed(config.seed)

    # Print config
    print("\n" + "═" * 60)
    print("LeWM Training Configuration")
    print("═" * 60)
    print(OmegaConf.to_yaml(config))
    print("═" * 60)

    # Download dataset if requested or if it doesn't exist
    data_dir = Path(config.data_dir).expanduser()
    h5_path = data_dir / "pusht_expert_train.h5"

    if args.download or not h5_path.exists():
        if not h5_path.exists():
            print(f"\n⚠ Dataset not found at {h5_path}")
            print("  Attempting to download...")
        h5_path = download_pusht_dataset(data_dir=data_dir)
    else:
        print(f"\n✓ Dataset found: {h5_path}")

    # Create dataloaders
    print("\nLoading dataset...")
    train_loader, val_loader, full_dataset = get_dataloaders(
        h5_path=h5_path,
        batch_size=config.batch_size,
        train_split=config.train_split,
        augmentation=config.augmentation,
        num_workers=config.num_workers,
        image_size=config.image_size,
        seed=config.seed,
    )

    # Build model
    print("\nBuilding LeWM model...")
    model = LeWM(config).to(device)

    # Print parameter counts
    param_counts = model.count_parameters()
    print(f"\nModel Parameters:")
    print(f"  Encoder:   {param_counts['encoder']:>10,} params")
    print(f"  Predictor: {param_counts['predictor']:>10,} params")
    print(f"  Total:     {param_counts['total']:>10,} params")

    # Train
    history = train_lewm(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device,
        checkpoint_dir=Path(config.checkpoint_dir),
    )

    print("\n✓ Training complete!")
    print(f"  Best model: {Path(config.checkpoint_dir) / 'best.pt'}")
    print(f"  Final model: {Path(config.checkpoint_dir) / 'final.pt'}")


if __name__ == "__main__":
    main()
