"""
Cross-Entropy Method (CEM) planner for goal-conditioned control with LeWM.

Uses the frozen LeWM as a latent-space simulator to find action sequences
that drive the current state toward a goal state. The planner operates
entirely in latent space — no decoder needed.

Algorithm:
    1. Encode current and goal observations to latent space.
    2. Initialize Gaussian action-sequence distribution.
    3. For each CEM iteration:
       a. Sample N action sequences from the distribution.
       b. Roll out each sequence in latent space using the predictor.
       c. Score each by negative L2 distance of final latent to goal latent.
       d. Select top-K elite sequences, refit distribution.
    4. Return the first action of the best (mean) sequence.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from models.lewm import LeWM


class CEMPlanner:
    """Cross-Entropy Method planner using a frozen LeWM world model.

    Args:
        model: Trained LeWM model (encoder + predictor). Frozen during planning.
        config: Configuration with CEM hyperparameters.
    """

    def __init__(self, model: LeWM, config: Any) -> None:
        self.model = model
        self.model.eval()

        # CEM hyperparameters
        self.n_samples = getattr(config, "cem_n_samples", 512)
        self.top_k = getattr(config, "cem_top_k", 64)
        self.n_iters = getattr(config, "cem_n_iters", 5)
        self.horizon = getattr(config, "cem_horizon", 10)
        self.action_dim = getattr(config, "action_dim", 2)
        self.action_low = getattr(config, "action_low", -1.0)
        self.action_high = getattr(config, "action_high", 1.0)

        # Device
        self.device = next(model.parameters()).device

    @torch.no_grad()
    def plan(self, obs: torch.Tensor, goal_obs: torch.Tensor) -> np.ndarray:
        """Plan the next action using CEM.

        Solves for an action sequence that minimizes latent-space distance
        to the goal, using the LeWM predictor as forward model.

        Args:
            obs: Current observation, shape (3, H, W), normalized.
            goal_obs: Goal observation, shape (3, H, W), normalized.

        Returns:
            First action of the best sequence, shape (action_dim,).
        """
        N = self.n_samples
        H = self.horizon
        K = self.top_k
        device = self.device

        # 1. Encode current and goal observations
        s_curr = self.model.encode(obs.unsqueeze(0).to(device))    # (1, latent_dim)
        s_goal = self.model.encode(goal_obs.unsqueeze(0).to(device))  # (1, latent_dim)

        # 2. Initialize action distribution (zero mean, unit std)
        mu = torch.zeros(H, self.action_dim, device=device)
        std = torch.ones(H, self.action_dim, device=device)

        # 3. CEM optimization loop
        for iteration in range(self.n_iters):
            # Sample N action sequences from current distribution
            noise = torch.randn(N, H, self.action_dim, device=device)
            actions = mu.unsqueeze(0) + std.unsqueeze(0) * noise  # (N, H, action_dim)
            actions = actions.clamp(self.action_low, self.action_high)

            # Roll out in latent space
            s = s_curr.expand(N, -1).clone()  # (N, latent_dim)
            for h in range(H):
                a_h = actions[:, h, :]  # (N, action_dim)
                s = self.model.predictor.predict(s, a_h)  # (N, latent_dim)

            # Score: negative L2 distance to goal in latent space
            scores = -torch.norm(s - s_goal.expand(N, -1), dim=-1)  # (N,)

            # Select elite sequences
            elite_idx = scores.topk(K).indices  # (K,)
            elite = actions[elite_idx]           # (K, H, action_dim)

            # Refit distribution to elite set
            mu = elite.mean(dim=0)               # (H, action_dim)
            std = elite.std(dim=0) + 1e-5        # (H, action_dim)

        # 4. Return first action of the best (mean) sequence
        return mu[0].cpu().numpy()

    @torch.no_grad()
    def plan_trajectory(
        self,
        obs: torch.Tensor,
        goal_obs: torch.Tensor,
        dataset: Any = None,
        max_steps: int = 100,
        distance_threshold: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Run full receding-horizon planning loop.

        At each step: plan → execute first action → simulate next latent state.
        Since we don't have a real environment, we use the LeWM predictor to
        simulate the trajectory entirely in latent space.

        Args:
            obs: Initial observation, shape (3, H, W), normalized.
            goal_obs: Goal observation, shape (3, H, W), normalized.
            dataset: Optional dataset for reference (not used in planning).
            max_steps: Maximum planning steps (default: 100).
            distance_threshold: Stop threshold for latent distance.
                If None, uses 0.1 * ||s_goal||.

        Returns:
            Dict with:
                - 'actions': List of actions taken.
                - 'latent_distances': List of distances to goal.
                - 'success': Whether the goal was reached.
                - 'n_steps': Number of steps taken.
        """
        device = self.device

        # Encode initial and goal
        s_curr = self.model.encode(obs.unsqueeze(0).to(device))
        s_goal = self.model.encode(goal_obs.unsqueeze(0).to(device))

        # Compute distance threshold
        if distance_threshold is None:
            goal_norm = torch.norm(s_goal).item()
            distance_threshold = max(0.1 * goal_norm, 0.5)

        actions_taken: List[np.ndarray] = []
        distances: List[float] = []

        for step in range(max_steps):
            # Compute current distance to goal
            dist = torch.norm(s_curr - s_goal).item()
            distances.append(dist)

            # Check if we've reached the goal
            if dist < distance_threshold:
                return {
                    "actions": actions_taken,
                    "latent_distances": distances,
                    "success": True,
                    "n_steps": step,
                    "final_distance": dist,
                }

            # Plan next action using CEM (from current latent, not re-encoding)
            # For receding horizon, we do full CEM from obs
            action = self.plan(obs, goal_obs)
            actions_taken.append(action)

            # Simulate next state in latent space
            action_t = torch.from_numpy(action).float().unsqueeze(0).to(device)
            s_curr = self.model.predictor.predict(s_curr, action_t)

        # Final distance
        final_dist = torch.norm(s_curr - s_goal).item()
        distances.append(final_dist)

        return {
            "actions": actions_taken,
            "latent_distances": distances,
            "success": final_dist < distance_threshold,
            "n_steps": max_steps,
            "final_distance": final_dist,
        }
