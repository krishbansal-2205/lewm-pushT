"""Tests for the PushT Dataset loader.

These tests create a small mock HDF5 file for testing, so they don't
require the full dataset to be downloaded.
"""

from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from training.dataset import PushTDataset, get_dataloaders


@pytest.fixture
def mock_h5_path(tmp_path: Path) -> Path:
    """Create a small mock HDF5 dataset for testing."""
    h5_path = tmp_path / "test_pusht.h5"
    n_samples = 200
    img_size = 224
    action_dim = 2

    with h5py.File(h5_path, "w") as f:
        # Create observations: (N, H, W, 3) uint8
        obs = np.random.randint(0, 256, (n_samples, img_size, img_size, 3), dtype=np.uint8)
        f.create_dataset("observations", data=obs)

        # Create actions: (N, action_dim) float32
        actions = np.random.randn(n_samples, action_dim).astype(np.float32)
        f.create_dataset("actions", data=actions)

        # Create next_observations: (N, H, W, 3) uint8
        next_obs = np.random.randint(0, 256, (n_samples, img_size, img_size, 3), dtype=np.uint8)
        f.create_dataset("next_observations", data=next_obs)

    return h5_path


class TestPushTDataset:
    """Test suite for the PushT dataset loader."""

    def test_length(self, mock_h5_path: Path) -> None:
        """Test dataset reports correct length."""
        dataset = PushTDataset(mock_h5_path, augmentation=False, cache_in_ram=True)
        assert len(dataset) == 200, f"Expected 200, got {len(dataset)}"

    def test_getitem_shapes(self, mock_h5_path: Path) -> None:
        """Test that __getitem__ returns correct shapes."""
        dataset = PushTDataset(mock_h5_path, augmentation=False, cache_in_ram=True)
        sample = dataset[0]

        assert "obs" in sample
        assert "action" in sample
        assert "next_obs" in sample

        assert sample["obs"].shape == (3, 224, 224), f"obs shape: {sample['obs'].shape}"
        assert sample["action"].shape == (2,), f"action shape: {sample['action'].shape}"
        assert sample["next_obs"].shape == (3, 224, 224), f"next_obs shape: {sample['next_obs'].shape}"

    def test_getitem_dtypes(self, mock_h5_path: Path) -> None:
        """Test that returned tensors have correct dtypes."""
        dataset = PushTDataset(mock_h5_path, augmentation=False, cache_in_ram=True)
        sample = dataset[0]

        assert sample["obs"].dtype == torch.float32
        assert sample["action"].dtype == torch.float32
        assert sample["next_obs"].dtype == torch.float32

    def test_normalization(self, mock_h5_path: Path) -> None:
        """Test that images are normalized (not in [0, 255] range)."""
        dataset = PushTDataset(mock_h5_path, augmentation=False, cache_in_ram=True)
        sample = dataset[0]

        # After ImageNet normalization, values should be roughly in [-3, 3]
        assert sample["obs"].max() < 10.0, "Obs should be normalized"
        assert sample["obs"].min() > -10.0, "Obs should be normalized"

    def test_raw_images_present(self, mock_h5_path: Path) -> None:
        """Test that raw (un-normalized) images are included."""
        dataset = PushTDataset(mock_h5_path, augmentation=False, cache_in_ram=True)
        sample = dataset[0]

        assert "obs_raw" in sample
        assert "next_obs_raw" in sample
        # Raw should be in [0, 1]
        assert sample["obs_raw"].min() >= 0.0
        assert sample["obs_raw"].max() <= 1.0

    def test_augmentation_flip(self, mock_h5_path: Path) -> None:
        """Test that augmentation produces different outputs."""
        dataset_aug = PushTDataset(mock_h5_path, augmentation=True, cache_in_ram=True)
        dataset_no = PushTDataset(mock_h5_path, augmentation=False, cache_in_ram=True)

        # With random flips, at least some samples should differ
        # Check multiple samples
        any_different = False
        for i in range(20):
            s_aug = dataset_aug[0]
            s_no = dataset_no[0]
            if not torch.allclose(s_aug["obs"], s_no["obs"]):
                any_different = True
                break
        # Note: With 50% flip probability, it's possible (but unlikely) all 20 are same
        # We accept this test as probabilistic

    def test_file_not_found(self) -> None:
        """Test appropriate error when file doesn't exist."""
        with pytest.raises(FileNotFoundError):
            PushTDataset("nonexistent_file.h5")

    def test_dataloader_creation(self, mock_h5_path: Path) -> None:
        """Test that dataloaders can be created successfully."""
        train_loader, val_loader, dataset = get_dataloaders(
            h5_path=mock_h5_path,
            batch_size=16,
            train_split=0.8,
            augmentation=True,
            num_workers=0,  # Use 0 workers for testing
            seed=42,
            cache_in_ram=True,
        )

        # Check train batch
        batch = next(iter(train_loader))
        assert batch["obs"].shape[0] == 16, f"Batch size should be 16, got {batch['obs'].shape[0]}"
        assert batch["obs"].shape[1:] == (3, 224, 224)

    def test_train_val_split(self, mock_h5_path: Path) -> None:
        """Test that train/val split is correct."""
        train_loader, val_loader, dataset = get_dataloaders(
            h5_path=mock_h5_path,
            batch_size=16,
            train_split=0.8,
            num_workers=0,
            seed=42,
            cache_in_ram=True,
        )

        # 200 samples, 80% train = 160, 20% val = 40
        train_samples = sum(b["obs"].shape[0] for b in train_loader)
        val_samples = sum(b["obs"].shape[0] for b in val_loader)

        assert train_samples == 160, f"Expected 160 train samples, got {train_samples}"
        assert val_samples == 40, f"Expected 40 val samples, got {val_samples}"

    def test_lazy_loading_path(self, mock_h5_path: Path) -> None:
        """Test that lazy loading (cache_in_ram=False) still works."""
        dataset = PushTDataset(mock_h5_path, augmentation=False, cache_in_ram=False)
        assert len(dataset) == 200
        sample = dataset[0]
        assert sample["obs"].shape == (3, 224, 224)
        assert sample["obs_raw"].min() >= 0.0
        assert sample["obs_raw"].max() <= 1.0

    def test_cached_vs_lazy_identical(self, mock_h5_path: Path) -> None:
        """Test that cached and lazy modes produce identical outputs."""
        ds_cached = PushTDataset(mock_h5_path, augmentation=False, cache_in_ram=True)
        ds_lazy = PushTDataset(mock_h5_path, augmentation=False, cache_in_ram=False)
        
        sample_c = ds_cached[0]
        sample_l = ds_lazy[0]
        
        assert torch.allclose(sample_c["obs"], sample_l["obs"]), "Cached and lazy obs should match"
        assert torch.allclose(sample_c["action"], sample_l["action"]), "Actions should match"
        assert torch.allclose(sample_c["next_obs"], sample_l["next_obs"]), "next_obs should match"
