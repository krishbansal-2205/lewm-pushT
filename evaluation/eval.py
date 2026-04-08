"""
Evaluation module for LeWM on PushT.

Measures planning success rate on goal-reaching tasks by running
the CEM planner in latent space and checking if the final state
is within a distance threshold of the goal.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
from tqdm import tqdm

from models.lewm import LeWM
from planning.cem import CEMPlanner
from training.dataset import PushTDataset


def evaluate_model(
    model: LeWM,
    dataset: PushTDataset,
    config: Any,
    device: torch.device,
    n_episodes: int = 100,
    max_steps: int = 50,
    success_threshold: float = 0.15,
    results_dir: str | Path = "results",
) -> Dict[str, Any]:
    """Evaluate LeWM + CEM planner on PushT goal-reaching tasks.

    Samples (start, goal) pairs from the dataset, runs CEM planning
    in latent space, and measures success rate.

    Args:
        model: Trained LeWM model.
        dataset: PushT dataset for sampling start/goal pairs.
        config: Configuration with CEM hyperparameters.
        device: Compute device.
        n_episodes: Number of evaluation episodes (default: 100).
        max_steps: Max planning steps per episode (default: 50).
        success_threshold: Latent distance threshold for success (default: 0.15).
        results_dir: Directory to save results.

    Returns:
        Dict with evaluation metrics.
    """
    model.eval()
    planner = CEMPlanner(model, config)

    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Sample (start, goal) pairs — use different indices from dataset
    n = len(dataset)
    rng = np.random.RandomState(42)
    start_indices = rng.choice(n, size=n_episodes, replace=False)
    goal_indices = rng.choice(n, size=n_episodes, replace=False)

    # Make sure start != goal
    for i in range(n_episodes):
        while goal_indices[i] == start_indices[i]:
            goal_indices[i] = rng.randint(0, n)

    successes = []
    steps_list = []
    distances = []
    planning_times = []

    print("\n" + "=" * 60)
    print("Evaluating LeWM + CEM Planner on PushT")
    print(f"  Episodes: {n_episodes}")
    print(f"  Max steps: {max_steps}")
    print(f"  Success threshold: {success_threshold}")
    print("=" * 60 + "\n")

    for ep in tqdm(range(n_episodes), desc="Evaluating"):
        start_sample = dataset[start_indices[ep]]
        goal_sample = dataset[goal_indices[ep]]

        obs = start_sample["obs"]        # (3, H, W)
        goal_obs = goal_sample["obs"]    # (3, H, W)

        t_start = time.time()

        result = planner.plan_trajectory(
            obs=obs,
            goal_obs=goal_obs,
            max_steps=max_steps,
            distance_threshold=success_threshold,
        )

        t_elapsed = time.time() - t_start

        successes.append(result["success"])
        steps_list.append(result["n_steps"])
        distances.append(result["final_distance"])
        planning_times.append(t_elapsed)

    # Compute metrics
    success_rate = np.mean(successes) * 100.0
    mean_steps = np.mean([s for s, ok in zip(steps_list, successes) if ok]) if any(successes) else 0.0
    mean_distance = np.mean(distances)
    mean_planning_time = np.mean(planning_times)
    total_planning_time = np.sum(planning_times)

    metrics = {
        "success_rate_pct": round(success_rate, 2),
        "mean_steps_to_success": round(float(mean_steps), 2),
        "mean_final_distance": round(float(mean_distance), 4),
        "mean_planning_time_s": round(float(mean_planning_time), 3),
        "total_planning_time_s": round(float(total_planning_time), 1),
        "n_episodes": n_episodes,
        "n_successes": int(sum(successes)),
        "max_steps": max_steps,
        "success_threshold": success_threshold,
    }

    # Print results
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"  Success Rate:          {metrics['success_rate_pct']:.1f}%")
    print(f"  Successes:             {metrics['n_successes']}/{metrics['n_episodes']}")
    print(f"  Mean Steps (success):  {metrics['mean_steps_to_success']:.1f}")
    print(f"  Mean Final Distance:   {metrics['mean_final_distance']:.4f}")
    print(f"  Mean Planning Time:    {metrics['mean_planning_time_s']:.3f}s / step")
    print(f"  Total Planning Time:   {metrics['total_planning_time_s']:.1f}s")
    print("=" * 60 + "\n")

    # Save results
    results_path = results_dir / "pusht_eval.json"
    with open(results_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Results saved to {results_path}")

    return metrics
