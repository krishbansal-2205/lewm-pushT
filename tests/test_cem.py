"""Tests for the CEM Planner."""

import pytest
import numpy as np
import torch
from omegaconf import OmegaConf

from models.lewm import LeWM
from planning.cem import CEMPlanner


def _make_model_and_config():
    """Create a small model and config for testing."""
    config = OmegaConf.create({
        "latent_dim": 64,
        "encoder_channels": [16, 32, 64, 128],
        "predictor_hidden": [128, 128],
        "dropout": 0.0,
        "action_dim": 2,
        "image_size": 96,
        "sigreg_num_projections": 16,
        "cem_n_samples": 32,
        "cem_top_k": 8,
        "cem_n_iters": 2,
        "cem_horizon": 5,
        "action_low": -1.0,
        "action_high": 1.0,
    })
    model = LeWM(config)
    model.eval()
    return model, config


class TestCEMPlanner:
    """Test suite for the CEM planner."""

    def test_plan_returns_correct_shape(self) -> None:
        """Test that plan() returns action of shape (action_dim,)."""
        model, config = _make_model_and_config()
        planner = CEMPlanner(model, config)

        obs = torch.randn(3, 96, 96)
        goal_obs = torch.randn(3, 96, 96)

        action = planner.plan(obs, goal_obs)
        assert isinstance(action, np.ndarray), "Action should be numpy array"
        assert action.shape == (2,), f"Expected shape (2,), got {action.shape}"

    def test_action_within_bounds(self) -> None:
        """Test that planned actions are within specified bounds."""
        model, config = _make_model_and_config()
        planner = CEMPlanner(model, config)

        obs = torch.randn(3, 96, 96)
        goal_obs = torch.randn(3, 96, 96)

        action = planner.plan(obs, goal_obs)
        assert np.all(action >= -1.0), f"Action below lower bound: {action}"
        assert np.all(action <= 1.0), f"Action above upper bound: {action}"

    def test_plan_trajectory(self) -> None:
        """Test the full planning trajectory loop."""
        model, config = _make_model_and_config()
        planner = CEMPlanner(model, config)

        obs = torch.randn(3, 96, 96)
        goal_obs = torch.randn(3, 96, 96)

        result = planner.plan_trajectory(obs, goal_obs, max_steps=5)
        assert "actions" in result
        assert "latent_distances" in result
        assert "success" in result
        assert "n_steps" in result
        assert len(result["actions"]) <= 5

    def test_deterministic_with_seed(self) -> None:
        """Test that planning is deterministic with fixed seed."""
        model, config = _make_model_and_config()
        planner = CEMPlanner(model, config)

        obs = torch.randn(3, 96, 96)
        goal_obs = torch.randn(3, 96, 96)

        torch.manual_seed(42)
        a1 = planner.plan(obs, goal_obs)

        torch.manual_seed(42)
        a2 = planner.plan(obs, goal_obs)

        np.testing.assert_array_almost_equal(a1, a2, decimal=5)

    def test_finite_outputs(self) -> None:
        """Test that all outputs are finite."""
        model, config = _make_model_and_config()
        planner = CEMPlanner(model, config)

        obs = torch.randn(3, 96, 96)
        goal_obs = torch.randn(3, 96, 96)

        action = planner.plan(obs, goal_obs)
        assert np.all(np.isfinite(action)), f"Action not finite: {action}"
