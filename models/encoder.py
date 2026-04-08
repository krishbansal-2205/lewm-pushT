"""
CNN Encoder for LeWM.

Maps a (3, 96, 96) RGB image to a latent vector of dimension `latent_dim`.
Uses 4 convolutional blocks with BatchNorm and GELU activation,
followed by a linear projection and LayerNorm.

Key design choice: NO EMA target encoder. This encoder is trained end-to-end
with full gradients. SIGReg regularization prevents representation collapse.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from typing import List


class Encoder(nn.Module):
    """CNN encoder: image → latent vector.

    Architecture:
        4 conv blocks (Conv2d → BatchNorm2d → GELU), channels 3→32→64→128→256.
        Kernel 3, stride 2, padding 1 halves spatial dims each layer:
        96 → 48 → 24 → 12 → 6.
        Flatten (256*6*6=9216) → Linear → latent_dim → LayerNorm.

    Args:
        latent_dim: Dimension of the output latent vector (default: 192).
        channels: List of channel sizes for conv blocks (default: [32, 64, 128, 256]).
        image_size: Expected input image size (default: 96).
    """

    def __init__(
        self,
        latent_dim: int = 192,
        channels: List[int] | None = None,
        image_size: int = 96,
    ) -> None:
        super().__init__()
        if channels is None:
            channels = [32, 64, 128, 256]

        self.latent_dim = latent_dim
        self.image_size = image_size

        # Build convolutional blocks
        layers: list[nn.Module] = []
        in_channels = 3
        for out_channels in channels:
            layers.extend([
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.GELU(),
            ])
            in_channels = out_channels

        self.conv_blocks = nn.Sequential(*layers)

        # Compute flattened feature size after conv blocks
        # 96 → 48 → 24 → 12 → 6  (4 stride-2 convolutions)
        spatial = image_size
        for _ in channels:
            spatial = (spatial + 1) // 2  # floor((spatial + 2*1 - 3) / 2 + 1) with pad=1, k=3, s=2
        self.flat_dim = channels[-1] * spatial * spatial

        # Linear projection to latent space
        self.projection = nn.Linear(self.flat_dim, latent_dim)
        self.layer_norm = nn.LayerNorm(latent_dim)

    def encode(self, obs: torch.Tensor) -> torch.Tensor:
        """Encode observations to latent vectors.

        Args:
            obs: Batch of images, shape (B, 3, H, W), float32, normalized.

        Returns:
            Latent vectors, shape (B, latent_dim).
        """
        features = self.conv_blocks(obs)             # (B, 256, 6, 6)
        features = features.reshape(features.size(0), -1)  # (B, 9216)
        latent = self.projection(features)           # (B, latent_dim)
        latent = self.layer_norm(latent)              # (B, latent_dim)
        return latent

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Forward pass (alias for encode)."""
        return self.encode(obs)
