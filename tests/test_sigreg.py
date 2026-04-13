"""Tests for the SIGReg regularizer."""

import pytest
import torch

from training.sigreg import SIGReg, sigreg_loss


class TestSIGReg:
    """Test suite for the SIGReg regularizer."""

    def test_returns_scalar(self) -> None:
        """Test that SIGReg returns a scalar loss."""
        z = torch.randn(64, 192)
        loss = sigreg_loss(z)
        assert loss.dim() == 0, f"Expected scalar, got shape {loss.shape}"

    def test_low_for_gaussian_input(self) -> None:
        """Test that SIGReg loss is small for a standard Gaussian distribution."""
        torch.manual_seed(42)
        # Large batch of standard Gaussian samples (mean=0, std=1, uncorrelated)
        z = torch.randn(10000, 192)
        loss = sigreg_loss(z, num_projections=128)
        # For an isotropic Gaussian: gaussianity ≈ 0, variance ≈ 0, covariance ≈ 0
        assert loss.item() < 1.0, f"Loss for Gaussian input should be small, got {loss.item()}"

    def test_high_for_collapsed_input(self) -> None:
        """Test that SIGReg loss is HIGH for a collapsed (near-zero std) distribution.

        This is the critical test — the old SIGReg gave near-zero loss for
        collapsed representations because it only checked Gaussianity of shape,
        not variance.  The fixed version should give high loss.
        """
        # Near-collapsed: all values very close to zero (std ≈ 0.01)
        z = torch.randn(64, 192) * 0.01
        loss = sigreg_loss(z, num_projections=64)
        # Variance loss should dominate: hinge(1 - 0.01) ≈ 0.99 per dim
        assert loss.item() > 0.5, (
            f"Loss for collapsed input should be high (>0.5), got {loss.item()}. "
            f"Variance regularization is not working!"
        )

    def test_high_for_constant_input(self) -> None:
        """Test that SIGReg gives high loss for completely collapsed (constant) input."""
        z = torch.ones(64, 192) * 5.0
        loss = sigreg_loss(z, num_projections=64)
        # All dimensions have zero std → variance_loss ≈ 1.0 per dim
        assert loss.item() > 0.8, f"Loss for constant input should be high, got {loss.item()}"

    def test_class_interface(self) -> None:
        """Test the SIGReg class wrapper (now nn.Module)."""
        reg = SIGReg(num_projections=32, latent_dim=192)
        z = torch.randn(64, 192)
        loss = reg(z)
        assert loss.dim() == 0, "forward() should return scalar"
        assert torch.isfinite(loss), "Loss should be finite"

    def test_fixed_projections(self) -> None:
        """Test that SIGReg class uses fixed projection directions."""
        reg = SIGReg(num_projections=32, latent_dim=64)
        assert hasattr(reg, "directions"), "Should have registered buffer 'directions'"
        assert reg.directions.shape == (32, 64), (
            f"Directions shape should be (32, 64), got {reg.directions.shape}"
        )
        # Verify they are unit vectors
        norms = reg.directions.norm(dim=1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5), \
            "Directions should be unit vectors"

    def test_different_projections(self) -> None:
        """Test with different numbers of projections."""
        z = torch.randn(64, 192)
        for n_proj in [8, 32, 64, 128]:
            loss = sigreg_loss(z, num_projections=n_proj)
            assert torch.isfinite(loss), f"Loss not finite with {n_proj} projections"

    def test_gradient_flow(self) -> None:
        """Test that gradients flow through SIGReg."""
        z = torch.randn(64, 192, requires_grad=True)
        loss = sigreg_loss(z)
        loss.backward()
        assert z.grad is not None, "Gradients should flow through SIGReg"
        assert z.grad.abs().sum() > 0, "Gradients should be non-zero"

    def test_gradient_flow_through_module(self) -> None:
        """Test gradient flow through the nn.Module SIGReg."""
        reg = SIGReg(num_projections=32, latent_dim=192)
        z = torch.randn(64, 192, requires_grad=True)
        loss = reg(z)
        loss.backward()
        assert z.grad is not None, "Gradients should flow through SIGReg module"

    def test_small_batch(self) -> None:
        """Test SIGReg with very small batch sizes."""
        z = torch.randn(4, 192)
        loss = sigreg_loss(z, num_projections=16)
        assert torch.isfinite(loss), "Should handle small batches"

    def test_device_consistency(self) -> None:
        """Test that projections are created on the same device as input."""
        z = torch.randn(32, 192)
        loss = sigreg_loss(z)
        assert loss.device == z.device, "Output device should match input"

    def test_variance_component(self) -> None:
        """Test that variance loss penalizes low-variance dimensions."""
        # Create data with some dimensions at std=1 and some at std=0.01
        z = torch.randn(200, 100)
        z[:, 50:] *= 0.01  # Collapse last 50 dims

        # With var_weight=0, should not penalize
        loss_no_var = sigreg_loss(z, num_projections=32, var_weight=0.0, cov_weight=0.0)
        # With var_weight=1, should penalize
        loss_with_var = sigreg_loss(z, num_projections=32, var_weight=1.0, cov_weight=0.0)

        assert loss_with_var > loss_no_var, (
            f"Variance regularization should increase loss for collapsed dims, "
            f"but got {loss_with_var} <= {loss_no_var}"
        )

    def test_covariance_component(self) -> None:
        """Test that covariance loss penalizes correlated dimensions."""
        torch.manual_seed(42)
        # Independent dimensions
        z_indep = torch.randn(500, 50)
        # Correlated dimensions (all dims are copies of first + noise)
        z_corr = z_indep[:, 0:1].expand(-1, 50) + 0.01 * torch.randn(500, 50)

        loss_indep = sigreg_loss(z_indep, num_projections=32, var_weight=0.0, cov_weight=1.0)
        loss_corr = sigreg_loss(z_corr, num_projections=32, var_weight=0.0, cov_weight=1.0)

        assert loss_corr > loss_indep, (
            f"Correlated dims should have higher loss, "
            f"but got corr={loss_corr} <= indep={loss_indep}"
        )
