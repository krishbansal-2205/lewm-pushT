"""
Training loop for LeWM on PushT - GPU-optimized.

Key features:
  1. NormalizeBatch runs on GPU (uint8->float+normalize)
  2. CUDAPrefetcher overlaps H->D transfer with compute
  3. Optional torch.compile for kernel fusion (PyTorch 2.x)
  4. AMP mixed precision for throughput
  5. Validation via torch.inference_mode
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
from training.dataset import NormalizeBatch


class CUDAPrefetcher:
    """Overlaps host->device transfer with GPU compute via a side stream."""

    def __init__(self, loader: DataLoader, device: torch.device) -> None:
        self.loader = loader
        self.device = device
        self._stream = torch.cuda.Stream(device=device)
        self._iter: Optional[Iterator] = None
        self._batch: Optional[Dict[str, torch.Tensor]] = None

    def _preload(self) -> None:
        try:
            assert self._iter is not None
            raw = next(self._iter)
        except StopIteration:
            self._batch = None
            return

        with torch.cuda.stream(self._stream):
            self._batch = {
                k: v.to(self.device, non_blocking=True) if isinstance(v, torch.Tensor) else v
                for k, v in raw.items()
            }

    def __iter__(self) -> "CUDAPrefetcher":
        self._iter = iter(self.loader)
        self._preload()
        return self

    def __next__(self) -> Dict[str, torch.Tensor]:
        torch.cuda.current_stream(self.device).wait_stream(self._stream)
        batch = self._batch
        if batch is None:
            raise StopIteration
        self._preload()
        return batch

    def __len__(self) -> int:
        return len(self.loader)


def set_seed(seed: int) -> None:
    """Set seeds and enable fast CUDA kernels."""
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


def compute_latent_stats(
    model: LeWM,
    loader: DataLoader,
    device: torch.device,
    normalizer: Optional[NormalizeBatch],
    max_batches: int = 10,
) -> Dict[str, float]:
    """Compute latent stats for collapse monitoring."""
    model.eval()
    all_latents: list[torch.Tensor] = []

    with torch.inference_mode():
        for i, batch in enumerate(loader):
            if i >= max_batches:
                break
            obs = batch["obs"].to(device, non_blocking=True)
            if normalizer is not None:
                obs = normalizer(obs)
            z = model.encode(obs)
            all_latents.append(z.cpu())

    cat = torch.cat(all_latents, dim=0)
    dim_std = cat.std(dim=0)
    return {
        "mean": cat.mean().item(),
        "std": cat.std().item(),
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
    normalizer: Optional[NormalizeBatch] = None,
) -> Dict[str, list]:
    """Run the full LeWM training loop with GPU-optimized data path."""
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    lr = getattr(config, "lr", 3e-4)
    weight_decay = getattr(config, "weight_decay", 1e-4)
    grad_clip = getattr(config, "grad_clip", 1.0)
    lambda_reg = getattr(config, "lambda_reg", 0.1)
    epochs = getattr(config, "epochs", 100)
    log_every = getattr(config, "log_every", 100)
    patience = getattr(config, "early_stopping_patience", 20)
    use_amp = getattr(config, "use_amp", True)
    use_compile = getattr(config, "use_compile", True)

    if normalizer is not None:
        normalizer = normalizer.to(device)
        if use_compile and hasattr(torch, "compile"):
            try:
                print("Compiling NormalizeBatch with torch.compile()...")
                normalizer = torch.compile(normalizer, mode="reduce-overhead")
                print("  ✓ NormalizeBatch compiled")
            except Exception as e:
                print(f"  ⚠ torch.compile failed for normalizer: {e}")

    if use_compile and hasattr(torch, "compile"):
        try:
            print("Compiling LeWM with torch.compile()...")
            model = torch.compile(model, mode="reduce-overhead")
            print("  ✓ LeWM compiled")
        except Exception as e:
            print(f"  ⚠ torch.compile failed for model: {e}")

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    total_steps = len(train_loader) * epochs
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-6)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp and torch.cuda.is_available())

    best_val_loss = float("inf")
    epochs_without_improve = 0
    global_step = 0
    use_prefetcher = device.type == "cuda" and torch.cuda.is_available()

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
    print(f"  Epochs:       {epochs}")
    print(f"  Batch size:   {train_loader.batch_size}")
    print(f"  LR:           {lr}")
    print(f"  Lambda_reg:   {lambda_reg}")
    print(f"  Device:       {device}")
    print(f"  Mixed prec:   {use_amp}")
    print(f"  Prefetcher:   {use_prefetcher}")
    print(f"  Compile:      {use_compile}")
    print(f"  Checkpoints:  {checkpoint_dir}")
    print("=" * 60 + "\n")

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = epoch_pred = epoch_reg = 0.0
        n_batches = 0
        t_start = time.time()

        train_iter = CUDAPrefetcher(train_loader, device) if use_prefetcher else train_loader
        pbar = tqdm(train_iter, desc=f"Epoch {epoch:3d}/{epochs}", leave=False, total=len(train_loader))

        for batch in pbar:
            if use_prefetcher:
                obs_raw = batch["obs"]
                act = batch["action"]
                nxt_raw = batch["next_obs"]
            else:
                obs_raw = batch["obs"].to(device, non_blocking=True)
                act = batch["action"].to(device, non_blocking=True)
                nxt_raw = batch["next_obs"].to(device, non_blocking=True)

            if normalizer is not None:
                obs = normalizer(obs_raw)
                next_obs = normalizer(nxt_raw)
            else:
                obs = obs_raw
                next_obs = nxt_raw

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=use_amp):
                loss, pred_loss, reg_loss = model.compute_loss(
                    obs, act, next_obs, lambda_reg=lambda_reg
                )

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            epoch_loss += loss.item()
            epoch_pred += pred_loss.item()
            epoch_reg += reg_loss.item()
            n_batches += 1
            global_step += 1

            if global_step % log_every == 0:
                pbar.set_postfix(
                    loss=f"{loss.item():.4f}",
                    pred=f"{pred_loss.item():.4f}",
                    reg=f"{reg_loss.item():.4f}",
                    gnorm=f"{grad_norm:.3f}",
                    lr=f"{scheduler.get_last_lr()[0]:.2e}",
                )

        n_batches = max(n_batches, 1)
        epoch_loss /= n_batches
        epoch_pred /= n_batches
        epoch_reg /= n_batches
        epoch_time = time.time() - t_start

        model.eval()
        val_loss = val_pred = val_reg = 0.0
        n_val = 0

        with torch.inference_mode():
            for batch in val_loader:
                obs_raw = batch["obs"].to(device, non_blocking=True)
                act = batch["action"].to(device, non_blocking=True)
                nxt_raw = batch["next_obs"].to(device, non_blocking=True)

                if normalizer is not None:
                    obs = normalizer(obs_raw)
                    next_obs = normalizer(nxt_raw)
                else:
                    obs = obs_raw
                    next_obs = nxt_raw

                with torch.cuda.amp.autocast(enabled=use_amp):
                    loss, pred_loss, reg_loss = model.compute_loss(
                        obs, act, next_obs, lambda_reg=lambda_reg
                    )

                val_loss += loss.item()
                val_pred += pred_loss.item()
                val_reg += reg_loss.item()
                n_val += 1

        n_val = max(n_val, 1)
        val_loss /= n_val
        val_pred /= n_val
        val_reg /= n_val

        latent_stats = compute_latent_stats(model, val_loader, device, normalizer)

        if latent_stats["mean_dim_std"] < 0.01:
            print(f"\n⚠ Collapse detected! Latent std={latent_stats['mean_dim_std']:.6f}")
            lambda_reg *= 2.0
            print(f"  Auto-increased lambda_reg -> {lambda_reg}")

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

        if val_pred < best_val_loss:
            best_val_loss = val_pred
            epochs_without_improve = 0
            raw_model = getattr(model, "_orig_mod", model)
            raw_model.save(checkpoint_dir / "best.pt")
            print(f"  ✓ Best model saved (val_pred={val_pred:.4f})")
        else:
            epochs_without_improve += 1

        if epoch % 10 == 0:
            raw_model = getattr(model, "_orig_mod", model)
            raw_model.save(checkpoint_dir / f"pusht_lewm_epoch{epoch}.pt")

        if epochs_without_improve >= patience:
            print(f"\n⚡ Early stopping at epoch {epoch}")
            break

    raw_model = getattr(model, "_orig_mod", model)
    raw_model.save(checkpoint_dir / "final.pt")

    history_path = checkpoint_dir / "training_history.pt"
    torch.save(history, history_path)
    print(f"\n✓ Training complete | Best val pred: {best_val_loss:.4f}")

    return history
