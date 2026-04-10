"""
Training loop for LeWM on PushT.

Implements the full training pipeline:
- AdamW optimizer with cosine annealing LR schedule
- Gradient clipping
- Collapse detection via latent std monitoring
- Best model checkpointing on validation loss
- Early stopping
- Logging (console + optional WandB)
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.lewm import LeWM


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility.

    Seeds: random, numpy, torch (CPU + CUDA).

    Args:
        seed: Integer seed value.
    """
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def compute_latent_stats(
    model: LeWM,
    loader: DataLoader,
    device: torch.device,
    max_batches: int = 10,
) -> Dict[str, float]:
    """Compute statistics of latent vectors for collapse detection.

    Args:
        model: LeWM model.
        loader: DataLoader to encode.
        device: Compute device.
        max_batches: Max batches to process (for speed).

    Returns:
        Dict with 'mean', 'std', 'min_std', 'max_std' of latent dimensions.
    """
    model.eval()
    all_latents = []
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= max_batches:
                break
            obs = batch["obs"].to(device)
            z = model.encode(obs)
            all_latents.append(z.cpu())

    all_latents = torch.cat(all_latents, dim=0)
    dim_std = all_latents.std(dim=0)  # Std per latent dim

    return {
        "mean": all_latents.mean().item(),
        "std": all_latents.std().item(),
        "min_dim_std": dim_std.min().item(),
        "max_dim_std": dim_std.max().item(),
        "mean_dim_std": dim_std.mean().item(),
    }


def train_lewm(
    model: LeWM,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: Any,
    device: torch.device,
    checkpoint_dir: Path,
) -> Dict[str, list]:
    """Run the full LeWM training loop.

    Args:
        model: LeWM model to train.
        train_loader: Training DataLoader.
        val_loader: Validation DataLoader.
        config: Training hyperparameters.
        device: Compute device.
        checkpoint_dir: Directory to save checkpoints.

    Returns:
        Dict of training history (losses, metrics per epoch).
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Hyperparameters
    lr = getattr(config, "lr", 3e-4)
    weight_decay = getattr(config, "weight_decay", 1e-4)
    grad_clip = getattr(config, "grad_clip", 1.0)
    lambda_reg = getattr(config, "lambda_reg", 0.1)
    epochs = getattr(config, "epochs", 100)
    log_every = getattr(config, "log_every", 100)
    patience = getattr(config, "early_stopping_patience", 20)

    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    total_steps = len(train_loader) * epochs
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-6)

    # Training state
    best_val_loss = float("inf")
    epochs_without_improvement = 0
    global_step = 0

    history: Dict[str, list] = {
        "train_loss": [],
        "train_pred_loss": [],
        "train_reg_loss": [],
        "val_loss": [],
        "val_pred_loss": [],
        "val_reg_loss": [],
        "latent_std": [],
        "lr": [],
    }

    print("\n" + "=" * 60)
    print("Starting LeWM Training")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {train_loader.batch_size}")
    print(f"  LR: {lr}")
    print(f"  Lambda_reg: {lambda_reg}")
    print(f"  Device: {device}")
    print(f"  Checkpoints: {checkpoint_dir}")
    print("=" * 60 + "\n")

    for epoch in range(1, epochs + 1):
        # ─── Training ─────────────────────────────────────────
        model.train()
        epoch_loss = 0.0
        epoch_pred = 0.0
        epoch_reg = 0.0
        n_batches = 0
        t_start = time.time()

        pbar = tqdm(
            train_loader, desc=f"Epoch {epoch:3d}/{epochs}", leave=False)
        for batch in pbar:
            obs = batch["obs"].to(device, non_blocking=True)
            action = batch["action"].to(device, non_blocking=True)
            next_obs = batch["next_obs"].to(device, non_blocking=True)

            optimizer.zero_grad()

            try:
                loss, pred_loss, reg_loss = model.compute_loss(
                    obs, action, next_obs, lambda_reg=lambda_reg
                )
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(
                        "\n⚠ CUDA out of memory! Reduce batch_size in config."
                    )
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    raise
                raise

            loss.backward()

            # Gradient clipping
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            epoch_pred += pred_loss.item()
            epoch_reg += reg_loss.item()
            n_batches += 1
            global_step += 1

            # Logging
            if global_step % log_every == 0:
                current_lr = scheduler.get_last_lr()[0]
                pbar.set_postfix(
                    loss=f"{loss.item():.4f}",
                    pred=f"{pred_loss.item():.4f}",
                    reg=f"{reg_loss.item():.4f}",
                    gnorm=f"{grad_norm:.3f}",
                    lr=f"{current_lr:.2e}",
                )

        # Epoch averages
        epoch_loss /= max(n_batches, 1)
        epoch_pred /= max(n_batches, 1)
        epoch_reg /= max(n_batches, 1)
        epoch_time = time.time() - t_start

        # ─── Validation ──────────────────────────────────────
        model.eval()
        val_loss = 0.0
        val_pred = 0.0
        val_reg = 0.0
        n_val = 0

        with torch.no_grad():
            for batch in val_loader:
                obs = batch["obs"].to(device)
                action = batch["action"].to(device)
                next_obs = batch["next_obs"].to(device)

                loss, pred_loss, reg_loss = model.compute_loss(
                    obs, action, next_obs, lambda_reg=lambda_reg
                )

                val_loss += loss.item()
                val_pred += pred_loss.item()
                val_reg += reg_loss.item()
                n_val += 1

        val_loss /= max(n_val, 1)
        val_pred /= max(n_val, 1)
        val_reg /= max(n_val, 1)

        # ─── Collapse Detection ──────────────────────────────
        latent_stats = compute_latent_stats(model, val_loader, device)

        if latent_stats["mean_dim_std"] < 0.01:
            print(f"\n⚠ WARNING: Possible representation collapse detected!")
            print(
                f"  Latent std = {latent_stats['mean_dim_std']:.6f} (threshold: 0.01)")
            print(
                f"  Consider increasing lambda_reg (currently {lambda_reg})\n")
            # Auto-increase lambda_reg
            lambda_reg *= 2.0
            print(f"  Auto-increased lambda_reg to {lambda_reg}")

        # ─── Record History ───────────────────────────────────
        current_lr = scheduler.get_last_lr()[0]
        history["train_loss"].append(epoch_loss)
        history["train_pred_loss"].append(epoch_pred)
        history["train_reg_loss"].append(epoch_reg)
        history["val_loss"].append(val_loss)
        history["val_pred_loss"].append(val_pred)
        history["val_reg_loss"].append(val_reg)
        history["latent_std"].append(latent_stats["mean_dim_std"])
        history["lr"].append(current_lr)

        # ─── Print Epoch Summary ──────────────────────────────
        print(
            f"Epoch {epoch:3d}/{epochs} | "
            f"Train: {epoch_loss:.4f} (pred={epoch_pred:.4f}, reg={epoch_reg:.4f}) | "
            f"Val: {val_loss:.4f} (pred={val_pred:.4f}) | "
            f"Latent std: {latent_stats['mean_dim_std']:.4f} | "
            f"LR: {current_lr:.2e} | "
            f"Time: {epoch_time:.1f}s"
        )

        # ─── Checkpointing ───────────────────────────────────
        # Save best model based on val prediction loss
        if val_pred < best_val_loss:
            best_val_loss = val_pred
            epochs_without_improvement = 0
            best_path = checkpoint_dir / "best.pt"
            model.save(best_path)
            print(f"  ✓ New best model saved (val_pred={val_pred:.4f})")
        else:
            epochs_without_improvement += 1

        # Save periodic checkpoint
        if epoch % 10 == 0:
            ckpt_path = checkpoint_dir / f"pusht_lewm_epoch{epoch}.pt"
            model.save(ckpt_path)

        # ─── Early Stopping ──────────────────────────────────
        if epochs_without_improvement >= patience:
            print(f"\n⚡ Early stopping at epoch {epoch} "
                  f"(no improvement for {patience} epochs)")
            break

    # Save final model
    final_path = checkpoint_dir / "final.pt"
    model.save(final_path)
    print(f"\n✓ Training complete. Final model saved to {final_path}")
    print(f"  Best val prediction loss: {best_val_loss:.4f}")

    # Save training history
    history_path = checkpoint_dir / "training_history.pt"
    torch.save(history, history_path)
    print(f"  Training history saved to {history_path}")

    return history
