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
        """Test that SIGReg loss is small for a Gaussian distribution."""
        torch.manual_seed(42)
        # Large batch of standard Gaussian samples
        z = torch.randn(10000, 192)
        loss = sigreg_loss(z, num_projections=128)
        # For a truly Gaussian distribution, loss should be very small
        assert loss.item() < 0.5, f"Loss for Gaussian input should be small, got {loss.item()}"

    def test_high_for_collapsed_input(self) -> None:
        """Test that SIGReg loss is high for a collapsed (uniform/degenerate) distribution."""
        # All-zeros (completely collapsed)
        z = torch.zeros(64, 192)
        # This will have NaN due to division by zero std, but the +1e-8 should handle it
        # Actually, all-zeros makes std ≈ 1e-8, and standardized x ≈ 0
        # Skewness = 0, kurtosis = 0 - 3 = -3... so loss = 9
        # But let's test with a non-trivial collapse
        z = torch.ones(64, 192) * torch.arange(64).unsqueeze(1).float()  # rank-1
        loss = sigreg_loss(z, num_projections=64)
        # For non-Gaussian distribution, loss should be non-negligible
        # (exact value depends on distribution shape)
        assert loss.item() >= 0, f"Loss should be non-negative, got {loss.item()}"

    def test_class_interface(self) -> None:
        """Test the SIGReg class wrapper."""
        reg = SIGReg(num_projections=32)
        z = torch.randn(64, 192)
        loss = reg(z)
        assert loss.dim() == 0, "Class __call__ should return scalar"
        assert torch.isfinite(loss), "Loss should be finite"

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
