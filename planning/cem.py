"""
Cross-Entropy Method (CEM) planner for goal-conditioned control with LeWM.

Uses the frozen LeWM as a latent-space simulator to find action sequences
that drive the current state toward a goal state.

FIX applied vs original:
  plan() now accepts an optional `z_curr` argument (pre-computed current
  latent). plan_trajectory() passes the evolved s_curr each step instead
  of always re-encoding the original `obs` image.  Without this fix the
  planner optimised actions from the initial latent at every step, making
  the receding-horizon loop degenerate into an open-loop plan.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import torch

from models.lewm import LeWM


class CEMPlanner:
    """Cross-Entropy Method planner using a frozen LeWM world model.

    Args:
        model:  Trained LeWM model (encoder + predictor). Frozen during planning.
        config: Configuration with CEM hyperparameters.
    """

    def __init__(self, model: LeWM, config: Any) -> None:
        self.model = model
        self.model.eval()

        self.n_samples = getattr(config, "cem_n_samples",  512)
        self.top_k = getattr(config, "cem_top_k",       64)
        self.n_iters = getattr(config, "cem_n_iters",      5)
        self.horizon = getattr(config, "cem_horizon",     10)
        self.action_dim = getattr(config, "action_dim",       2)
        self.action_low = getattr(config, "action_low",      -1.0)
        self.action_high = getattr(config, "action_high",      1.0)

        self.device = next(model.parameters()).device

    # ──────────────────────────────────────────────────────────────────────
    # Core CEM optimisation
    # ──────────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def plan(
        self,
        obs: torch.Tensor,
        goal_obs: torch.Tensor,
        z_curr: Optional[torch.Tensor] = None,
        z_goal: Optional[torch.Tensor] = None,
    ) -> np.ndarray:
        """Plan the next action using CEM.

        Args:
            obs:      Current observation image (3, H, W). Used only if
                      z_curr is None.
            goal_obs: Goal observation image (3, H, W). Used only if
                      z_goal is None.
            z_curr:   Pre-encoded current latent (1, D). Avoids a redundant
                      encoder forward pass when called from plan_trajectory.
            z_goal:   Pre-encoded goal latent (1, D).

        Returns:
            First action of the best sequence, shape (action_dim,).
        """
        N, H, K = self.n_samples, self.horizon, self.top_k
        device = self.device

        # Encode only when caller has not already done so.
        if z_curr is None:
            z_curr = self.model.encode(obs.unsqueeze(0).to(device))   # (1, D)
        if z_goal is None:
            z_goal = self.model.encode(
                goal_obs.unsqueeze(0).to(device))  # (1, D)

        mu = torch.zeros(H, self.action_dim, device=device)
        std = torch.ones(H, self.action_dim, device=device)

        for _ in range(self.n_iters):
            noise = torch.randn(N, H, self.action_dim, device=device)
            actions = (mu.unsqueeze(0) + std.unsqueeze(0) * noise).clamp(
                self.action_low, self.action_high
            )  # (N, H, action_dim)

            s = z_curr.expand(N, -1).clone()
            for h in range(H):
                s = self.model.predictor.predict(s, actions[:, h, :])

            scores = -torch.norm(s - z_goal.expand(N, -1), dim=-1)
            elite_idx = scores.topk(K).indices
            elite = actions[elite_idx]

            mu = elite.mean(dim=0)
            std = elite.std(dim=0) + 1e-5

        return mu[0].cpu().numpy()

    # ──────────────────────────────────────────────────────────────────────
    # Receding-horizon planning loop
    # ──────────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def plan_trajectory(
        self,
        obs: torch.Tensor,
        goal_obs: torch.Tensor,
        dataset: Any = None,
        max_steps: int = 100,
        distance_threshold: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Run a full receding-horizon planning loop in latent space.

        FIX: the original implementation called self.plan(obs, goal_obs)
        at every step, which re-encoded the *original* obs image each time.
        This meant CEM always optimised actions from the t=0 latent, making
        the trajectory simulation in latent space irrelevant.  We now pass
        the current evolved latent directly into plan() so each CEM call
        starts from the correct state.

        Args:
            obs:                Initial observation (3, H, W).
            goal_obs:           Goal observation (3, H, W).
            dataset:            Unused — kept for API compatibility.
            max_steps:          Maximum planning steps.
            distance_threshold: Latent-distance success criterion.

        Returns:
            Dict with 'actions', 'latent_distances', 'success', 'n_steps',
            'final_distance'.
        """
        device = self.device

        s_curr = self.model.encode(obs.unsqueeze(0).to(device))       # (1, D)
        s_goal = self.model.encode(goal_obs.unsqueeze(0).to(device))  # (1, D)

        if distance_threshold is None:
            goal_norm = torch.norm(s_goal).item()
            distance_threshold = max(0.1 * goal_norm, 0.5)

        actions_taken: List[np.ndarray] = []
        distances:     List[float] = []

        for step in range(max_steps):
            dist = torch.norm(s_curr - s_goal).item()
            distances.append(dist)

            if dist < distance_threshold:
                return {
                    "actions":          actions_taken,
                    "latent_distances": distances,
                    "success":          True,
                    "n_steps":          step,
                    "final_distance":   dist,
                }

            # FIX: pass the current evolved latent so CEM plans from the
            # correct starting state, not always from encode(original obs).
            action = self.plan(obs, goal_obs, z_curr=s_curr, z_goal=s_goal)
            actions_taken.append(action)

            action_t = torch.from_numpy(action).float().unsqueeze(0).to(device)
            s_curr = self.model.predictor.predict(s_curr, action_t)

        final_dist = torch.norm(s_curr - s_goal).item()
        distances.append(final_dist)

        return {
            "actions":          actions_taken,
            "latent_distances": distances,
            "success":          final_dist < distance_threshold,
            "n_steps":          max_steps,
            "final_distance":   final_dist,
        }
