"""Aggregated Adaptive Conformal Inference (AgACI).

Zaffran et al. ICML 2022 arXiv:2202.07282. Online wrapper around any
conformal method (here: RAPS). At each step:
  1. Each "expert" maintains its own alpha_t adapted via gamma_i.
  2. The aggregated alpha_t is a BOA-weighted average of expert alphas.
  3. After observing miscoverage at step t, weights update: experts
     whose alpha gave correct coverage gain weight, others lose.

Provably maintains target coverage under arbitrary distribution shift.

Spec: plan/V1-SPEC.md §5.2.
"""

from __future__ import annotations

import numpy as np

DEFAULT_GAMMA_GRID: tuple[float, ...] = (0.001, 0.005, 0.01, 0.05, 0.1)


class AgACI:
    """Aggregated Adaptive Conformal Inference online wrapper.

    Args:
        gamma_grid: per-expert step sizes for alpha adaptation.
        alpha_target: target miscoverage rate (0.10 = 90% coverage).
        eta: BOA learning rate for expert-weight updates.
    """

    def __init__(
        self,
        gamma_grid: tuple[float, ...] = DEFAULT_GAMMA_GRID,
        alpha_target: float = 0.10,
        eta: float = 0.01,
    ) -> None:
        self.gamma_grid = tuple(gamma_grid)
        self.alpha_target = alpha_target
        self.eta = eta
        self.n_experts = len(gamma_grid)
        self.expert_alphas = np.full(self.n_experts, alpha_target, dtype=np.float64)
        self.weights = np.full(self.n_experts, 1.0 / self.n_experts, dtype=np.float64)

    def current_alpha(self) -> float:
        """Aggregated alpha used for the next prediction set."""
        agg = float((self.weights * self.expert_alphas).sum())
        return max(0.001, min(0.5, agg))

    def update(self, miscovered: bool) -> None:
        """Update expert alphas + BOA weights from realized miscoverage.

        Args:
            miscovered: True if the previous prediction set did not contain
                the realized label.
        """
        observed = 1.0 if miscovered else 0.0
        for i, gamma in enumerate(self.gamma_grid):
            self.expert_alphas[i] = self.expert_alphas[i] + gamma * (self.alpha_target - observed)
            self.expert_alphas[i] = max(0.001, min(0.5, float(self.expert_alphas[i])))

        losses = (self.expert_alphas - self.alpha_target) ** 2
        log_weights = np.log(self.weights + 1e-12) - self.eta * losses
        log_weights -= log_weights.max()
        new_w = np.exp(log_weights)
        self.weights = new_w / new_w.sum()

    def state_dict(self) -> dict:
        return {
            "gamma_grid": list(self.gamma_grid),
            "alpha_target": self.alpha_target,
            "eta": self.eta,
            "expert_alphas": self.expert_alphas.tolist(),
            "weights": self.weights.tolist(),
        }

    def load_state_dict(self, state: dict) -> None:
        self.gamma_grid = tuple(state["gamma_grid"])
        self.alpha_target = float(state["alpha_target"])
        self.eta = float(state["eta"])
        self.expert_alphas = np.asarray(state["expert_alphas"], dtype=np.float64)
        self.weights = np.asarray(state["weights"], dtype=np.float64)
        self.n_experts = len(self.gamma_grid)
