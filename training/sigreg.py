"""
SIGReg (Stable Isotropic Gaussian Regularizer) for LeWM.

This is the key innovation of LeWM. It replaces EMA + stop-gradient by enforcing
that the latent distribution is isotropic Gaussian, preventing representation collapse.

Based on the Cramér-Wold theorem: a multivariate distribution is isotropic Gaussian
if and only if every 1D projection of it is also Gaussian.

The implementation uses random projections and measures non-Gaussianity via
skewness and excess kurtosis of the projected 1D distributions.

FIX applied: The original implementation only checked *Gaussianity of shape*
(skewness + kurtosis) but did NOT enforce isotropicity (unit variance per
dimension) or decorrelation.  A near-collapsed encoder (std=0.008) could
produce a perfectly Gaussian-shaped distribution and SIGReg would report
near-zero loss — allowing collapse despite the regulariser.

Now includes three components:
  1. Gaussianity loss  — skewness^2 + excess_kurtosis^2  (original)
  2. Variance loss     — penalises per-dim std deviating from 1
  3. Covariance loss   — decorrelates dimensions (off-diagonal cov → 0)

Also uses fixed random projections (sampled once, stored as a buffer)
instead of re-sampling each call, reducing gradient noise.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SIGReg(nn.Module):
    """SIGReg Gaussian regularizer based on the Cramér-Wold theorem.

    Measures how far the latent distribution is from isotropic Gaussian
    by projecting onto random 1D directions and checking normality via
    skewness^2 + (kurtosis - 3)^2.

    NOW ALSO ENFORCES:
      - Per-dimension variance ≈ 1  (prevents scale collapse)
      - Decorrelation between dimensions (prevents dimensional collapse)

    Args:
        num_projections: Number of random unit vectors for projection (default: 64).
        latent_dim:      Dimension of the latent space (default: 192).
                         Used to pre-allocate fixed projection directions.
        var_weight:      Weight for the variance regularisation term (default: 1.0).
        cov_weight:      Weight for the covariance regularisation term (default: 0.04).
    """

    def __init__(
        self,
        num_projections: int = 64,
        latent_dim: int = 192,
        var_weight: float = 1.0,
        cov_weight: float = 0.04,
    ) -> None:
        super().__init__()
        self.num_projections = num_projections
        self.var_weight = var_weight
        self.cov_weight = cov_weight

        # Fixed random projection directions — sampled once, not every call.
        # Registered as a buffer so they move to the correct device with the model.
        directions = torch.randn(num_projections, latent_dim)
        directions = F.normalize(directions, dim=1)
        self.register_buffer("directions", directions)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Compute the full SIGReg regularization loss.

        Args:
            z: Batch of latent vectors, shape (B, D).

        Returns:
            Scalar regularization loss. Lower = more isotropic Gaussian.
        """
        return sigreg_loss(
            z,
            directions=self.directions,
            var_weight=self.var_weight,
            cov_weight=self.cov_weight,
        )

    # Keep __call__ working as before (nn.Module already routes to forward).


def sigreg_loss(
    z: torch.Tensor,
    directions: torch.Tensor | None = None,
    num_projections: int = 64,
    var_weight: float = 1.0,
    cov_weight: float = 0.04,
) -> torch.Tensor:
    """Compute SIGReg loss for a batch of latent vectors.

    Algorithm:
        1. Sample M random unit vectors (projections) — or use fixed ones.
        2. Project latent vectors onto each direction.
        3. Standardize each projection (zero-mean, unit-variance).
        4. Measure non-Gaussianity via skewness^2 + kurtosis^2.
        5. Average across all projections.
        6. ADD variance loss: hinge on per-dimension std (must be ≥ 1).
        7. ADD covariance loss: off-diagonal elements of the covariance matrix → 0.

    A perfectly isotropic Gaussian has:
      - skewness=0, excess_kurtosis=0  →  gaussianity_loss ≈ 0
      - per-dim std = 1               →  variance_loss = 0
      - uncorrelated dimensions        →  covariance_loss = 0

    Args:
        z:               Batch of latent vectors, shape (B, D).
        directions:      Pre-computed projection directions (M, D). If None,
                         random directions are sampled (legacy behaviour).
        num_projections: Fallback if directions is None (default: 64).
        var_weight:      Weight for variance loss (default: 1.0).
        cov_weight:      Weight for covariance loss (default: 0.04).

    Returns:
        Scalar loss value.
    """
    B, D = z.shape

    # ── 1. Gaussianity loss (original) ────────────────────────────────────
    if directions is None:
        directions = torch.randn(num_projections, D, device=z.device, dtype=z.dtype)
        directions = F.normalize(directions, dim=1)

    # Project latent vectors onto each direction
    projected = z @ directions.T  # (B, M)

    # Standardize each projection to zero-mean, unit-variance
    mean = projected.mean(dim=0, keepdim=True)       # (1, M)
    std = projected.std(dim=0, keepdim=True) + 1e-8  # (1, M)
    projected = (projected - mean) / std             # (B, M)

    # Skewness: E[x^3] (should be ~0 for Gaussian)
    skewness = (projected ** 3).mean(dim=0)  # (M,)

    # Excess kurtosis: E[x^4] - 3 (should be ~0 for Gaussian)
    kurtosis = (projected ** 4).mean(dim=0) - 3.0  # (M,)

    gaussianity_loss = (skewness ** 2 + kurtosis ** 2).mean()

    # ── 2. Variance loss (NEW — prevents scale collapse) ──────────────────
    # Each dimension should have std ≈ 1.  We use a hinge loss so that
    # dimensions with std ≥ 1 are not penalised, but dimensions with
    # std < 1 are strongly pushed toward 1.
    dim_std = z.std(dim=0)  # (D,)
    variance_loss = F.relu(1.0 - dim_std).mean()

    # ── 3. Covariance loss (NEW — prevents dimensional collapse) ──────────
    # The off-diagonal elements of the sample covariance matrix should be 0.
    z_centered = z - z.mean(dim=0, keepdim=True)  # (B, D)
    cov = (z_centered.T @ z_centered) / (B - 1)   # (D, D)

    # Zero out diagonal (we only penalise off-diagonal correlations)
    off_diag = cov.clone()
    off_diag.fill_diagonal_(0.0)
    covariance_loss = (off_diag ** 2).sum() / D

    # ── Total ─────────────────────────────────────────────────────────────
    total = gaussianity_loss + var_weight * variance_loss + cov_weight * covariance_loss

    return total
