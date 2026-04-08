"""
MLP Predictor for LeWM.

Maps (latent_state, action) → predicted_next_latent using a residual MLP
with GELU activations, LayerNorm, and dropout.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from typing import List


class Predictor(nn.Module):
    """MLP predictor: (latent, action) → predicted next latent.

    Architecture:
        Concatenate latent (latent_dim) + action (action_dim) → input.
        Hidden layers with GELU + LayerNorm + Dropout + residual connections.
        Output: linear → latent_dim, NO activation.

    Args:
        latent_dim: Dimension of the latent vector (default: 192).
        action_dim: Dimension of the action vector (default: 2).
        hidden_dims: List of hidden layer sizes (default: [512, 512, 512]).
        dropout: Dropout probability (default: 0.1).
    """

    def __init__(
        self,
        latent_dim: int = 192,
        action_dim: int = 2,
        hidden_dims: List[int] | None = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [512, 512, 512]

        self.latent_dim = latent_dim
        self.action_dim = action_dim
        input_dim = latent_dim + action_dim

        # Build hidden layers
        self.hidden_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        self.residual_projs = nn.ModuleList()

        prev_dim = input_dim
        for h_dim in hidden_dims:
            self.hidden_layers.append(nn.Linear(prev_dim, h_dim))
            self.norms.append(nn.LayerNorm(h_dim))
            self.dropouts.append(nn.Dropout(dropout))
            # Residual projection when dims don't match
            if prev_dim != h_dim:
                self.residual_projs.append(nn.Linear(prev_dim, h_dim, bias=False))
            else:
                self.residual_projs.append(nn.Identity())
            prev_dim = h_dim

        # Output projection — no activation
        self.output_proj = nn.Linear(prev_dim, latent_dim)
        self.activation = nn.GELU()

    def predict(self, latent: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Predict the next latent state.

        Args:
            latent: Current latent state, shape (B, latent_dim).
            action: Action taken, shape (B, action_dim).

        Returns:
            Predicted next latent state, shape (B, latent_dim).
        """
        x = torch.cat([latent, action], dim=-1)  # (B, latent_dim + action_dim)

        for linear, norm, drop, res_proj in zip(
            self.hidden_layers, self.norms, self.dropouts, self.residual_projs
        ):
            residual = res_proj(x)
            x = linear(x)
            x = self.activation(x)
            x = norm(x)
            x = drop(x)
            x = x + residual  # Residual connection

        x = self.output_proj(x)  # (B, latent_dim)
        return x

    def forward(self, latent: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Forward pass (alias for predict)."""
        return self.predict(latent, action)
