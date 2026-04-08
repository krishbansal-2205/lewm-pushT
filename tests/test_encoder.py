"""Tests for the CNN Encoder."""

import pytest
import torch

from models.encoder import Encoder


class TestEncoder:
    """Test suite for the Encoder module."""

    def test_output_shape(self) -> None:
        """Test encoder produces correct output shape (B, latent_dim)."""
        encoder = Encoder(latent_dim=192, image_size=96)
        x = torch.randn(4, 3, 96, 96)
        z = encoder.encode(x)
        assert z.shape == (4, 192), f"Expected (4, 192), got {z.shape}"

    def test_different_batch_sizes(self) -> None:
        """Test encoder works with different batch sizes."""
        encoder = Encoder(latent_dim=192, image_size=96)
        for bs in [1, 2, 8, 16]:
            x = torch.randn(bs, 3, 96, 96)
            z = encoder.encode(x)
            assert z.shape == (bs, 192), f"Batch size {bs}: Expected ({bs}, 192), got {z.shape}"

    def test_custom_latent_dim(self) -> None:
        """Test encoder with non-default latent dimensions."""
        for dim in [64, 128, 256, 512]:
            encoder = Encoder(latent_dim=dim, image_size=96)
            x = torch.randn(2, 3, 96, 96)
            z = encoder.encode(x)
            assert z.shape == (2, dim), f"Dim {dim}: Expected (2, {dim}), got {z.shape}"

    def test_forward_equals_encode(self) -> None:
        """Test that forward() and encode() produce the same output."""
        encoder = Encoder(latent_dim=192, image_size=96)
        x = torch.randn(2, 3, 96, 96)
        z1 = encoder.encode(x)
        z2 = encoder(x)
        assert torch.allclose(z1, z2), "forward() and encode() should be identical"

    def test_layer_norm_applied(self) -> None:
        """Test that output has approximate zero-mean unit-std properties."""
        encoder = Encoder(latent_dim=192, image_size=96)
        x = torch.randn(32, 3, 96, 96)
        z = encoder.encode(x)
        # LayerNorm normalizes across latent_dim, so each sample should have mean≈0, std≈1
        sample_mean = z[0].mean().item()
        sample_std = z[0].std().item()
        assert abs(sample_mean) < 1.0, f"Mean too large: {sample_mean}"
        assert 0.1 < sample_std < 5.0, f"Std out of range: {sample_std}"

    def test_gradients_flow(self) -> None:
        """Test that gradients flow through the encoder."""
        encoder = Encoder(latent_dim=192, image_size=96)
        x = torch.randn(2, 3, 96, 96, requires_grad=True)
        z = encoder.encode(x)
        loss = z.sum()
        loss.backward()
        assert x.grad is not None, "Gradients should flow to input"
        assert x.grad.abs().sum() > 0, "Gradients should be non-zero"
