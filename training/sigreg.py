"""
SIGReg (Stable Isotropic Gaussian Regularizer) for LeWM.

This is the key innovation of LeWM. It replaces EMA + stop-gradient by enforcing
that the latent distribution is isotropic Gaussian, preventing representation collapse.

Based on the Cramér-Wold theorem: a multivariate distribution is isotropic Gaussian
if and only if every 1D projection of it is also Gaussian.

The implementation uses random projections and measures non-Gaussianity via
skewness and excess kurtosis of the projected 1D distributions.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


class SIGReg:
    """SIGReg Gaussian regularizer based on the Cramér-Wold theorem.

    Measures how far the latent distribution is from isotropic Gaussian
    by projecting onto random 1D directions and checking normality via
    skewness^2 + (kurtosis - 3)^2.

    Args:
        num_projections: Number of random unit vectors for projection (default: 64).
    """

    def __init__(self, num_projections: int = 64) -> None:
        self.num_projections = num_projections

    def __call__(self, z: torch.Tensor) -> torch.Tensor:
        """Compute the SIGReg regularization loss.

        Args:
            z: Batch of latent vectors, shape (B, D).

        Returns:
            Scalar regularization loss. Lower = more Gaussian.
        """
        return sigreg_loss(z, num_projections=self.num_projections)


def sigreg_loss(z: torch.Tensor, num_projections: int = 64) -> torch.Tensor:
    """Compute SIGReg loss for a batch of latent vectors.

    Algorithm:
        1. Sample M random unit vectors (projections).
        2. Project latent vectors onto each direction.
        3. Standardize each projection (zero-mean, unit-variance).
        4. Measure non-Gaussianity via skewness^2 + kurtosis^2.
        5. Average across all projections.

    A perfectly Gaussian distribution has skewness=0 and excess kurtosis=0,
    so the loss is minimized (≈0) for Gaussian distributions.

    Args:
        z: Batch of latent vectors, shape (B, D).
        num_projections: Number of random 1D projections (default: 64).

    Returns:
        Scalar loss value.
    """
    B, D = z.shape

    # Sample random unit directions on the unit sphere
    directions = torch.randn(num_projections, D, device=z.device, dtype=z.dtype)
    directions = F.normalize(directions, dim=1)  # (M, D)

    # Project latent vectors onto each direction
    projected = z @ directions.T  # (B, M)

    # Standardize each projection to zero-mean, unit-variance
    mean = projected.mean(dim=0, keepdim=True)       # (1, M)
    std = projected.std(dim=0, keepdim=True) + 1e-8  # (1, M)
    projected = (projected - mean) / std             # (B, M)

    # Compute non-Gaussianity measures
    # Skewness: E[x^3] (should be ~0 for Gaussian)
    skewness = (projected ** 3).mean(dim=0)  # (M,)

    # Excess kurtosis: E[x^4] - 3 (should be ~0 for Gaussian)
    kurtosis = (projected ** 4).mean(dim=0) - 3.0  # (M,)

    # Non-Gaussianity score: skewness^2 + kurtosis^2
    loss = (skewness ** 2 + kurtosis ** 2).mean()

    return loss
