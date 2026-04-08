"""
LeWM (LeWorldModel) wrapper module.

Combines the CNN encoder and MLP predictor into a single model with a
unified loss function: MSE prediction loss + λ · SIGReg regularization.

Key novelty: NO EMA target encoder, NO stop-gradient, NO pretrained encoder.
The encoder is trained end-to-end with full gradients on both obs and next_obs.
Representation collapse is prevented solely by SIGReg.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn

from .encoder import Encoder
from .predictor import Predictor
from training.sigreg import SIGReg


class LeWM(nn.Module):
    """LeWorldModel: encoder + predictor with SIGReg regularization.

    This wraps the Encoder and Predictor into a joint-embedding predictive
    architecture (JEPA) that learns latent dynamics from pixels.

    Args:
        config: OmegaConf or dict-like with model hyperparameters.
    """

    def __init__(self, config: Any) -> None:
        super().__init__()
        self.config = config

        # Extract config values with defaults
        latent_dim = getattr(config, "latent_dim", 192)
        action_dim = getattr(config, "action_dim", 2)
        image_size = getattr(config, "image_size", 96)
        dropout = getattr(config, "dropout", 0.1)

        encoder_channels = getattr(config, "encoder_channels", [32, 64, 128, 256])
        if not isinstance(encoder_channels, list):
            encoder_channels = list(encoder_channels)

        predictor_hidden = getattr(config, "predictor_hidden", [512, 512, 512])
        if not isinstance(predictor_hidden, list):
            predictor_hidden = list(predictor_hidden)

        num_projections = getattr(config, "sigreg_num_projections", 64)

        # Build components
        self.encoder = Encoder(
            latent_dim=latent_dim,
            channels=encoder_channels,
            image_size=image_size,
        )
        self.predictor = Predictor(
            latent_dim=latent_dim,
            action_dim=action_dim,
            hidden_dims=predictor_hidden,
            dropout=dropout,
        )
        self.sigreg = SIGReg(num_projections=num_projections)

    def encode(self, obs: torch.Tensor) -> torch.Tensor:
        """Encode observations to latent vectors.

        Args:
            obs: Batch of images, shape (B, 3, H, W).

        Returns:
            Latent vectors, shape (B, latent_dim).
        """
        return self.encoder.encode(obs)

    def predict(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Predict next latent state from observation and action.

        Args:
            obs: Batch of images, shape (B, 3, H, W).
            action: Batch of actions, shape (B, action_dim).

        Returns:
            Predicted next latent state, shape (B, latent_dim).
        """
        latent = self.encode(obs)
        return self.predictor.predict(latent, action)

    def compute_loss(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        next_obs: torch.Tensor,
        lambda_reg: float = 0.1,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute the full LeWM loss.

        loss = MSE(predicted_next_latent, target_next_latent) + λ · SIGReg(latents)

        IMPORTANT: Both encode(obs) and encode(next_obs) use FULL gradients.
        No stop-gradient. No EMA. This is what makes LeWM novel — collapse
        is prevented by SIGReg alone.

        Args:
            obs: Current observations, shape (B, 3, H, W).
            action: Actions taken, shape (B, action_dim).
            next_obs: Next observations, shape (B, 3, H, W).
            lambda_reg: Weight for SIGReg regularization (default: 0.1).

        Returns:
            Tuple of (total_loss, prediction_loss, sigreg_loss), all scalars.
        """
        # Encode current and next observations — FULL GRADIENTS on both
        z_curr = self.encode(obs)           # (B, latent_dim)
        z_next = self.encode(next_obs)      # (B, latent_dim)

        # Predict next latent from current latent + action
        z_pred = self.predictor.predict(z_curr, action)  # (B, latent_dim)

        # Prediction loss: MSE between predicted and actual next latent
        prediction_loss = nn.functional.mse_loss(z_pred, z_next)

        # SIGReg regularization on the combined batch of latent vectors
        z_all = torch.cat([z_curr, z_next], dim=0)  # (2B, latent_dim)
        reg_loss = self.sigreg(z_all)

        # Total loss
        total_loss = prediction_loss + lambda_reg * reg_loss

        return total_loss, prediction_loss, reg_loss

    def save(self, path: str | Path) -> None:
        """Save model checkpoint.

        Args:
            path: File path to save the checkpoint.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Convert config to a serializable dict
        config_dict = {}
        for key in [
            "latent_dim", "action_dim", "image_size", "dropout",
            "encoder_channels", "predictor_hidden", "sigreg_num_projections",
        ]:
            val = getattr(self.config, key, None)
            if val is not None:
                config_dict[key] = list(val) if hasattr(val, "__iter__") and not isinstance(val, str) else val

        torch.save(
            {
                "encoder": self.encoder.state_dict(),
                "predictor": self.predictor.state_dict(),
                "config": config_dict,
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path, device: str = "cpu") -> "LeWM":
        """Load model from checkpoint.

        Args:
            path: File path to the checkpoint.
            device: Device to load the model to.

        Returns:
            Loaded LeWM model.

        Raises:
            FileNotFoundError: If checkpoint file doesn't exist.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {path}\n"
                f"Train a model first with: python train.py --config configs/pusht.yaml"
            )

        checkpoint = torch.load(path, map_location=device, weights_only=False)
        config_dict = checkpoint["config"]

        # Create a simple namespace from config dict
        from omegaconf import OmegaConf
        config = OmegaConf.create(config_dict)

        model = cls(config)
        model.encoder.load_state_dict(checkpoint["encoder"])
        model.predictor.load_state_dict(checkpoint["predictor"])
        model = model.to(device)
        return model

    def count_parameters(self) -> Dict[str, int]:
        """Count trainable parameters.

        Returns:
            Dict with parameter counts for encoder, predictor, and total.
        """
        enc_params = sum(p.numel() for p in self.encoder.parameters() if p.requires_grad)
        pred_params = sum(p.numel() for p in self.predictor.parameters() if p.requires_grad)
        return {
            "encoder": enc_params,
            "predictor": pred_params,
            "total": enc_params + pred_params,
        }
