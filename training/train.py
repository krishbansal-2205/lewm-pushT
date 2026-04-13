"""
Training loop for LeWM on PushT.

Implements the full training pipeline:
- CUDAPrefetcher: overlaps H→D transfer with GPU compute (fixes GPU sawtooth)
- AdamW optimizer with cosine annealing LR schedule
- Gradient clipping
- Collapse detection via latent std monitoring
- Best model checkpointing on validation loss
- Early stopping

FIXES applied vs original:
  1. Removed `.float()/255.0` from both training and validation batch
     fetches.  The dataset now returns properly normalised tensors (fixed
     in dataset.py), so dividing by 255 here was a second normalisation
     that pushed all values into a tiny ~0.004 range.
  2. Added `non_blocking=True` to the validation `.to(device)` calls so
     the PCIe transfer overlaps with the previous GPU kernel.
  3. Added CUDAPrefetcher that uses a dedicated CUDA stream to upload the
     next batch while the current batch is being processed — this is the
     primary fix for the GPU utilisation sawtooth pattern.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.lewm import LeWM


# ─────────────────────────────────────────────────────────────────────────────
# CUDA Prefetcher — fixes the GPU utilisation sawtooth
# ─────────────────────────────────────────────────────────────────────────────

class CUDAPrefetcher:
    """Overlaps host→device transfer with GPU computation.

    On a standard DataLoader the sequence is:
        [CPU decode] → [blocking H→D copy] → [GPU compute] → repeat

    The GPU sits idle during the CPU decode and H→D copy phases, which
    shows up as a sawtooth in nvidia-smi.  This prefetcher spawns a
    dedicated CUDA stream that starts uploading batch N+1 while the GPU
    is still computing batch N, so the PCIe transfer is hidden behind
    the compute and utilisation stays near 100 %.

    Usage (drop-in replacement for the DataLoader iterator):
        prefetcher = CUDAPrefetcher(loader, device)
        for batch in prefetcher:
            loss = model(batch["obs"], ...)
    """

    def __init__(self, loader: DataLoader, device: torch.device) -> None:
        self.loader = loader
        self.device = device
        self._stream = torch.cuda.Stream(device=device)
        self._iter: Optional[Iterator] = None
        self._batch: Optional[Dict[str, torch.Tensor]] = None

    def _preload(self) -> None:
        assert self._iter is not None
        try:
            raw = next(self._iter)
        except StopIteration:
            self._batch = None
            return

        with torch.cuda.stream(self._stream):
            self._batch = {
                k: v.to(self.device, non_blocking=True) if isinstance(
                    v, torch.Tensor) else v
                for k, v in raw.items()
            }

    def __iter__(self) -> "CUDAPrefetcher":
        self._iter = iter(self.loader)
        self._preload()
        return self

    def __next__(self) -> Dict[str, torch.Tensor]:
        # Wait for the pre-loaded batch to be ready on the device.
        torch.cuda.current_stream(self.device).wait_stream(self._stream)
        batch = self._batch
        if batch is None:
            raise StopIteration
        # Kick off the next upload while the caller processes this batch.
        self._preload()
        return batch

    def __len__(self) -> int:
        return len(self.loader)


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
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
    """Compute per-dimension std of latent vectors for collapse detection."""
    model.eval()
    all_latents: list[torch.Tensor] = []
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= max_batches:
                break
            obs = batch["obs"].to(device, non_blocking=True)
            z = model.encode(obs)
            all_latents.append(z.cpu())

    all_latents_t = torch.cat(all_latents, dim=0)
    dim_std = all_latents_t.std(dim=0)

    return {
        "mean":         all_latents_t.mean().item(),
        "std":          all_latents_t.std().item(),
        "min_dim_std":  dim_std.min().item(),
        "max_dim_std":  dim_std.max().item(),
        "mean_dim_std": dim_std.mean().item(),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main training loop
# ─────────────────────────────────────────────────────────────────────────────

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
        model:          LeWM model to train.
        train_loader:   Training DataLoader.
        val_loader:     Validation DataLoader.
        config:         Training hyperparameters (OmegaConf or similar).
        device:         Compute device.
        checkpoint_dir: Directory to save checkpoints.

    Returns:
        Dict of training history (losses, metrics per epoch).
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Hyperparameters
    lr = getattr(config, "lr",                    3e-4)
    weight_decay = getattr(config, "weight_decay",          1e-4)
    grad_clip = getattr(config, "grad_clip",              1.0)
    lambda_reg = getattr(config, "lambda_reg",             0.1)
    epochs = getattr(config, "epochs",                 100)
    log_every = getattr(config, "log_every",              100)
    patience = getattr(config, "early_stopping_patience", 20)

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    total_steps = len(train_loader) * epochs
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-6)

    best_val_loss = float("inf")
    epochs_without_improvement = 0
    global_step = 0

    history: Dict[str, list] = {
        "train_loss":     [],
        "train_pred_loss": [],
        "train_reg_loss": [],
        "val_loss":       [],
        "val_pred_loss":  [],
        "val_reg_loss":   [],
        "latent_std":     [],
        "lr":             [],
    }

    use_prefetcher = torch.cuda.is_available()

    print("\n" + "=" * 60)
    print("Starting LeWM Training")
    print(f"  Epochs:      {epochs}")
    print(f"  Batch size:  {train_loader.batch_size}")
    print(f"  LR:          {lr}")
    print(f"  Lambda_reg:  {lambda_reg}")
    print(f"  Device:      {device}")
    print(
        f"  Prefetcher:  {'enabled' if use_prefetcher else 'disabled (no CUDA)'}")
    print(f"  Checkpoints: {checkpoint_dir}")
    print("=" * 60 + "\n")

    for epoch in range(1, epochs + 1):

        # ── Training ──────────────────────────────────────────────────────
        model.train()
        epoch_loss = epoch_pred = epoch_reg = 0.0
        n_batches = 0
        t_start = time.time()

        # Wrap loader in CUDAPrefetcher when CUDA is available to keep
        # the GPU fed without idle gaps between batches (fixes sawtooth).
        train_iter = (
            CUDAPrefetcher(train_loader, device) if use_prefetcher
            else train_loader
        )

        pbar = tqdm(train_iter, desc=f"Epoch {epoch:3d}/{epochs}", leave=False,
                    total=len(train_loader))

        for batch in pbar:
            # FIX: do NOT divide by 255 here — dataset.py already produces
            # correctly ImageNet-normalised float32 tensors.
            if use_prefetcher:
                # Tensors already on device, courtesy of CUDAPrefetcher.
                obs = batch["obs"]
                action = batch["action"]
                next_obs = batch["next_obs"]
            else:
                obs = batch["obs"].to(device,      non_blocking=True)
                action = batch["action"].to(device,   non_blocking=True)
                next_obs = batch["next_obs"].to(device, non_blocking=True)

            optimizer.zero_grad()

            try:
                loss, pred_loss, reg_loss = model.compute_loss(
                    obs, action, next_obs, lambda_reg=lambda_reg
                )
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print("\n⚠ CUDA out of memory! Reduce batch_size in config.")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                raise

            loss.backward()

            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            epoch_pred += pred_loss.item()
            epoch_reg += reg_loss.item()
            n_batches += 1
            global_step += 1

            if global_step % log_every == 0:
                current_lr = scheduler.get_last_lr()[0]
                pbar.set_postfix(
                    loss=f"{loss.item():.4f}",
                    pred=f"{pred_loss.item():.4f}",
                    reg=f"{reg_loss.item():.4f}",
                    gnorm=f"{grad_norm:.3f}",
                    lr=f"{current_lr:.2e}",
                )

        n_batches = max(n_batches, 1)
        epoch_loss /= n_batches
        epoch_pred /= n_batches
        epoch_reg /= n_batches
        epoch_time = time.time() - t_start

        # ── Validation ────────────────────────────────────────────────────
        model.eval()
        val_loss = val_pred = val_reg = 0.0
        n_val = 0

        with torch.no_grad():
            for batch in val_loader:
                # FIX: use non_blocking=True and NO /255.0 division.
                obs = batch["obs"].to(device,      non_blocking=True)
                action = batch["action"].to(device,   non_blocking=True)
                next_obs = batch["next_obs"].to(device, non_blocking=True)

                loss, pred_loss, reg_loss = model.compute_loss(
                    obs, action, next_obs, lambda_reg=lambda_reg
                )

                val_loss += loss.item()
                val_pred += pred_loss.item()
                val_reg += reg_loss.item()
                n_val += 1

        n_val = max(n_val, 1)
        val_loss /= n_val
        val_pred /= n_val
        val_reg /= n_val

        # ── Collapse detection ────────────────────────────────────────────
        latent_stats = compute_latent_stats(model, val_loader, device)

        if latent_stats["mean_dim_std"] < 0.01:
            print(f"\n⚠ WARNING: Possible representation collapse detected!")
            print(
                f"  Latent std = {latent_stats['mean_dim_std']:.6f} (threshold: 0.01)")
            print(
                f"  Consider increasing lambda_reg (currently {lambda_reg})\n")
            lambda_reg *= 2.0
            print(f"  Auto-increased lambda_reg to {lambda_reg}")

        # ── Record history ────────────────────────────────────────────────
        current_lr = scheduler.get_last_lr()[0]
        history["train_loss"].append(epoch_loss)
        history["train_pred_loss"].append(epoch_pred)
        history["train_reg_loss"].append(epoch_reg)
        history["val_loss"].append(val_loss)
        history["val_pred_loss"].append(val_pred)
        history["val_reg_loss"].append(val_reg)
        history["latent_std"].append(latent_stats["mean_dim_std"])
        history["lr"].append(current_lr)

        print(
            f"Epoch {epoch:3d}/{epochs} | "
            f"Train: {epoch_loss:.4f} (pred={epoch_pred:.4f}, reg={epoch_reg:.4f}) | "
            f"Val: {val_loss:.4f} (pred={val_pred:.4f}) | "
            f"Latent std: {latent_stats['mean_dim_std']:.4f} | "
            f"LR: {current_lr:.2e} | "
            f"Time: {epoch_time:.1f}s"
        )

        # ── Checkpointing ─────────────────────────────────────────────────
        if val_pred < best_val_loss:
            best_val_loss = val_pred
            epochs_without_improvement = 0
            best_path = checkpoint_dir / "best.pt"
            model.save(best_path)
            print(f"  ✓ New best model saved (val_pred={val_pred:.4f})")
        else:
            epochs_without_improvement += 1

        if epoch % 10 == 0:
            model.save(checkpoint_dir / f"pusht_lewm_epoch{epoch}.pt")

        # ── Early stopping ────────────────────────────────────────────────
        if epochs_without_improvement >= patience:
            print(f"\n⚡ Early stopping at epoch {epoch} "
                  f"(no improvement for {patience} epochs)")
            break

    # Save final model + history
    final_path = checkpoint_dir / "final.pt"
    model.save(final_path)
    print(f"\n✓ Training complete. Final model saved to {final_path}")
    print(f"  Best val prediction loss: {best_val_loss:.4f}")

    history_path = checkpoint_dir / "training_history.pt"
    torch.save(history, history_path)
    print(f"  Training history saved to {history_path}")

    return history
