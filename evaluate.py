#!/usr/bin/env python3
"""
LeWM Evaluation Entry Point.

Evaluate a trained LeWM model on PushT goal-reaching tasks using CEM planning.

Usage:
    python evaluate.py --checkpoint checkpoints/best.pt
    python evaluate.py --checkpoint checkpoints/best.pt --n_episodes 50
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf

from models.lewm import LeWM
from training.dataset import PushTDataset
from evaluation.eval import evaluate_model
from data.download import get_data_dir


def main() -> None:
    """Main evaluation entry point."""
    parser = argparse.ArgumentParser(
        description="Evaluate LeWM on PushT goal-reaching",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint (.pt file)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/pusht.yaml",
        help="Path to config YAML file",
    )
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--n_episodes", type=int, default=None)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--success_threshold", type=float, default=None)
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    # Load config
    config_path = Path(args.config)
    if config_path.exists():
        config = OmegaConf.load(config_path)
    else:
        config = OmegaConf.create({})

    # Apply CLI overrides
    overrides = {}
    if args.n_episodes is not None:
        overrides["n_eval_episodes"] = args.n_episodes
    if args.max_steps is not None:
        overrides["max_steps_per_episode"] = args.max_steps
    if args.success_threshold is not None:
        overrides["success_threshold"] = args.success_threshold
    if args.device is not None:
        overrides["device"] = args.device
    if overrides:
        config = OmegaConf.merge(config, OmegaConf.create(overrides))

    # Defaults
    defaults = {
        "cem_n_samples": 512,
        "cem_top_k": 64,
        "cem_n_iters": 5,
        "cem_horizon": 10,
        "action_dim": 2,
        "action_low": -1.0,
        "action_high": 1.0,
        "n_eval_episodes": 100,
        "max_steps_per_episode": 50,
        "success_threshold": 0.15,
        "device": "cuda",
        "data_dir": str(get_data_dir()),
    }
    for k, v in defaults.items():
        if k not in config:
            OmegaConf.update(config, k, v)

    # Device
    if config.device == "cuda" and not torch.cuda.is_available():
        print("⚠ CUDA not available, falling back to CPU")
        config.device = "cpu"
    device = torch.device(config.device)

    # Load model
    checkpoint_path = Path(args.checkpoint)
    print(f"\nLoading model from {checkpoint_path}...")
    model = LeWM.load(checkpoint_path, device=str(device))

    param_counts = model.count_parameters()
    print(f"  Parameters: {param_counts['total']:,}")

    # Load dataset
    data_dir = Path(args.data_dir or config.data_dir).expanduser()
    h5_path = data_dir / "pusht_expert_train.h5"

    if not h5_path.exists():
        print(f"\n✗ Dataset not found: {h5_path}")
        print("  Download with: python -m lewm_pusht.data.download")
        sys.exit(1)

    dataset = PushTDataset(h5_path=h5_path, augmentation=False)
    print(f"  Dataset size: {len(dataset)}")

    # Run evaluation
    metrics = evaluate_model(
        model=model,
        dataset=dataset,
        config=config,
        device=device,
        n_episodes=config.n_eval_episodes,
        max_steps=config.max_steps_per_episode,
        success_threshold=config.success_threshold,
        results_dir=args.results_dir,
    )


if __name__ == "__main__":
    main()
