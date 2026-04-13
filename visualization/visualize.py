"""
Visualization module for LeWM on PushT.

Generates four types of visualizations:
1. Latent space t-SNE — temporal structure of learned representations
2. Planning rollout GIF — CEM planner trajectory visualization
3. Prediction quality plot — latent-space prediction error analysis
4. Training curves — loss and collapse monitoring plots
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
from tqdm import tqdm

from models.lewm import LeWM
from planning.cem import CEMPlanner
from training.dataset import PushTDataset, IMAGENET_MEAN, IMAGENET_STD


def denormalize_image(img: torch.Tensor) -> np.ndarray:
    """Convert a normalized image tensor back to displayable uint8 numpy array.

    Args:
        img: Normalized image, shape (3, H, W).

    Returns:
        RGB image, shape (H, W, 3), uint8.
    """
    img = img.clone()
    img = img * IMAGENET_STD + IMAGENET_MEAN
    img = img.clamp(0, 1)
    img = (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return img


def plot_latent_tsne(
    model: LeWM,
    dataset: PushTDataset,
    device: torch.device,
    output_path: str | Path = "latent_tsne.png",
    n_samples: int = 2000,
) -> None:
    """Generate t-SNE visualization of the learned latent space.

    Encodes observations and projects to 2D via t-SNE, colored by
    dataset index (temporal position). A well-trained model should
    show smooth temporal structure.

    Args:
        model: Trained LeWM model.
        dataset: PushT dataset.
        device: Compute device.
        output_path: Path to save the plot.
        n_samples: Number of samples to encode (default: 2000).
    """
    from sklearn.manifold import TSNE

    model.eval()
    n = min(n_samples, len(dataset))

    # Sample indices uniformly
    indices = np.linspace(0, len(dataset) - 1, n, dtype=int)

    # Encode all samples
    latents = []
    with torch.no_grad():
        for idx in tqdm(indices, desc="Encoding for t-SNE"):
            sample = dataset[idx]
            obs = sample["obs"].unsqueeze(0).to(device)
            z = model.encode(obs).cpu().numpy()
            latents.append(z[0])

    latents = np.array(latents)

    # Run t-SNE
    print("Running t-SNE (this may take a minute)...")
    tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)
    embeddings = tsne.fit_transform(latents)

    # Plot colored by temporal position
    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(
        embeddings[:, 0],
        embeddings[:, 1],
        c=np.arange(n),
        cmap="coolwarm",
        s=8,
        alpha=0.7,
    )
    cbar = plt.colorbar(scatter, ax=ax, label="Dataset Index (temporal position)")
    ax.set_title("LeWM Latent Space — t-SNE", fontsize=14, fontweight="bold")
    ax.set_xlabel("t-SNE dim 1")
    ax.set_ylabel("t-SNE dim 2")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved t-SNE plot to {output_path}")


def create_planning_rollout_gif(
    model: LeWM,
    dataset: PushTDataset,
    config: Any,
    device: torch.device,
    output_dir: str | Path = ".",
    n_rollouts: int = 5,
    n_steps: int = 30,
) -> None:
    """Create GIF animations of CEM planning rollouts.

    For each rollout, shows the start observation with latent distance
    to goal decreasing over planning steps. Goal image shown in corner.

    Args:
        model: Trained LeWM model.
        dataset: PushT dataset.
        config: CEM configuration.
        device: Compute device.
        output_dir: Directory to save GIFs.
        n_rollouts: Number of rollout GIFs to create.
        n_steps: Planning steps per rollout.
    """
    import imageio

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model.eval()
    planner = CEMPlanner(model, config)
    n = len(dataset)
    rng = np.random.RandomState(123)

    for rollout_idx in range(n_rollouts):
        # Pick random start and goal
        start_idx = rng.randint(0, n)
        goal_idx = rng.randint(0, n)
        while goal_idx == start_idx:
            goal_idx = rng.randint(0, n)

        start_sample = dataset[start_idx]
        goal_sample = dataset[goal_idx]

        obs = start_sample["obs"]
        goal_obs = goal_sample["obs"]

        # Get raw images for visualization
        start_img = start_sample.get("obs_raw", obs)
        goal_img = goal_sample.get("obs_raw", goal_obs)

        if isinstance(start_img, torch.Tensor):
            start_img_np = (start_img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        else:
            start_img_np = start_img

        if isinstance(goal_img, torch.Tensor):
            goal_img_np = (goal_img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        else:
            goal_img_np = goal_img

        # Run planning trajectory
        with torch.no_grad():
            s_curr = model.encode(obs.unsqueeze(0).to(device))
            s_goal = model.encode(goal_obs.unsqueeze(0).to(device))

        frames = []
        for step in range(n_steps):
            # Current distance
            with torch.no_grad():
                dist = torch.norm(s_curr - s_goal).item()

            # Create frame
            fig, axes = plt.subplots(1, 2, figsize=(8, 4))

            axes[0].imshow(start_img_np)
            axes[0].set_title(f"Step {step} | dist: {dist:.3f}", fontsize=11)
            axes[0].axis("off")

            axes[1].imshow(goal_img_np)
            axes[1].set_title("Goal", fontsize=11)
            axes[1].axis("off")

            fig.suptitle(f"Planning Rollout #{rollout_idx + 1}", fontsize=13, fontweight="bold")
            plt.tight_layout()

            # Render to image
            fig.canvas.draw()
            # Get the RGBA buffer
            w, h = fig.canvas.get_width_height()
            buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            buf = buf.reshape(h, w, 3)
            frames.append(buf.copy())
            plt.close()

            # FIX: pass the current evolved latent so CEM plans from the
            # correct starting state, not always from encode(original obs).
            action = planner.plan(obs, goal_obs, z_curr=s_curr, z_goal=s_goal)
            action_t = torch.from_numpy(action).float().unsqueeze(0).to(device)
            with torch.no_grad():
                s_curr = model.predictor.predict(s_curr, action_t)

        # Save GIF
        gif_path = output_dir / f"planning_rollout_{rollout_idx + 1}.gif"
        imageio.mimsave(str(gif_path), frames, fps=5, loop=0)
        print(f"✓ Saved {gif_path}")


def plot_prediction_quality(
    model: LeWM,
    dataset: PushTDataset,
    device: torch.device,
    output_path: str | Path = "prediction_quality.png",
    n_samples: int = 10,
) -> None:
    """Plot prediction quality: actual obs, actual next obs, and latent error.

    Since LeWM has no decoder, shows actual images side by side
    with the prediction error in latent space as text overlay.

    Args:
        model: Trained LeWM model.
        dataset: PushT dataset.
        device: Compute device.
        output_path: Path to save the plot.
        n_samples: Number of samples to show (default: 10).
    """
    model.eval()
    rng = np.random.RandomState(42)
    indices = rng.choice(len(dataset), size=n_samples, replace=False)

    fig, axes = plt.subplots(n_samples, 2, figsize=(8, 3 * n_samples))
    if n_samples == 1:
        axes = axes.reshape(1, -1)

    for i, idx in enumerate(indices):
        sample = dataset[idx]
        obs = sample["obs"].unsqueeze(0).to(device)
        action = sample["action"].unsqueeze(0).to(device)
        next_obs = sample["next_obs"].unsqueeze(0).to(device)

        with torch.no_grad():
            z_pred = model.predict(obs, action)
            z_actual = model.encode(next_obs)
            error = torch.norm(z_pred - z_actual).item()

        # Show raw images  
        obs_raw = sample.get("obs_raw", sample["obs"])
        next_obs_raw = sample.get("next_obs_raw", sample["next_obs"])

        if isinstance(obs_raw, torch.Tensor):
            obs_img = (obs_raw.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        else:
            obs_img = obs_raw

        if isinstance(next_obs_raw, torch.Tensor):
            next_img = (next_obs_raw.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        else:
            next_img = next_obs_raw

        axes[i, 0].imshow(obs_img)
        axes[i, 0].set_title(f"Observation t", fontsize=9)
        axes[i, 0].axis("off")

        axes[i, 1].imshow(next_img)
        axes[i, 1].set_title(f"Observation t+1 | Latent Error: {error:.4f}", fontsize=9)
        axes[i, 1].axis("off")

    fig.suptitle("Prediction Quality — Latent Space Error", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved prediction quality plot to {output_path}")


def plot_training_curves(
    history: Dict[str, list],
    output_path: str | Path = "training_curves.png",
) -> None:
    """Plot training curves: losses and latent std over epochs.

    Args:
        history: Training history dict from train_lewm().
        output_path: Path to save the plot.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    epochs = range(1, len(history["train_loss"]) + 1)

    # 1. Total loss
    axes[0, 0].plot(epochs, history["train_loss"], label="Train", color="#2196F3", linewidth=1.5)
    axes[0, 0].plot(epochs, history["val_loss"], label="Val", color="#F44336", linewidth=1.5)
    axes[0, 0].set_title("Total Loss", fontweight="bold")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # 2. Prediction loss
    axes[0, 1].plot(epochs, history["train_pred_loss"], label="Train Pred", color="#4CAF50", linewidth=1.5)
    axes[0, 1].plot(epochs, history["val_pred_loss"], label="Val Pred", color="#FF9800", linewidth=1.5)
    axes[0, 1].set_title("Prediction Loss (MSE)", fontweight="bold")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Loss")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # 3. SIGReg loss
    axes[1, 0].plot(epochs, history["train_reg_loss"], label="Train SIGReg", color="#9C27B0", linewidth=1.5)
    axes[1, 0].plot(epochs, history["val_reg_loss"], label="Val SIGReg", color="#E91E63", linewidth=1.5)
    axes[1, 0].set_title("SIGReg Regularization Loss", fontweight="bold")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Loss")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # 4. Latent std (collapse monitor)
    axes[1, 1].plot(epochs, history["latent_std"], label="Latent Dim Std", color="#00BCD4", linewidth=1.5)
    axes[1, 1].axhline(y=0.01, color="red", linestyle="--", alpha=0.7, label="Collapse threshold (0.01)")
    axes[1, 1].set_title("Latent Std — Collapse Monitor", fontweight="bold")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("Mean Dim Std")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    fig.suptitle("LeWM Training Curves — PushT", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved training curves to {output_path}")


def run_all_visualizations(
    checkpoint_path: str | Path,
    data_path: str | Path,
    config: Any,
    output_dir: str | Path = "visualizations",
) -> None:
    """Run all four visualizations.

    Args:
        checkpoint_path: Path to trained model checkpoint.
        data_path: Path to PushT HDF5 dataset.
        config: Configuration object.
        output_dir: Directory to save visualizations.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    print("Loading model...")
    model = LeWM.load(checkpoint_path, device=str(device))

    # Load dataset (no augmentation for visualization)
    dataset = PushTDataset(h5_path=data_path, augmentation=False)

    print(f"\nRunning visualizations (output: {output_dir})...\n")

    # 1. t-SNE
    plot_latent_tsne(
        model, dataset, device,
        output_path=output_dir / "latent_tsne.png",
    )

    # 2. Planning rollout GIFs
    create_planning_rollout_gif(
        model, dataset, config, device,
        output_dir=output_dir,
    )

    # 3. Prediction quality
    plot_prediction_quality(
        model, dataset, device,
        output_path=output_dir / "prediction_quality.png",
    )

    # 4. Training curves (if history file exists)
    checkpoint_dir = Path(checkpoint_path).parent
    history_path = checkpoint_dir / "training_history.pt"
    if history_path.exists():
        history = torch.load(history_path, weights_only=False)
        plot_training_curves(
            history,
            output_path=output_dir / "training_curves.png",
        )
    else:
        print(f"⚠ Training history not found at {history_path}, skipping training curves.")

    print(f"\n✓ All visualizations saved to {output_dir}")
