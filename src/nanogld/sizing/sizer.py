"""V1 top-level Sizer — Stage 1 simple, Stage 2 full F2F machinery.

Stage 1: tanh(Head_B_position) * vol_target_multiplier
Stage 2: friction-Kelly + vol-target + cost + conformal-floor + Kelly-LLLA-mult

Stage 2 must beat Stage 1 by >= 0.2 Sharpe OOS to ship.

Spec: plan/V1-SPEC.md §10.2.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from nanogld.sizing.conformal_floor import (
    DEFAULT_FLOOR_THRESHOLD,
    apply_conformal_floor,
)
from nanogld.sizing.cost_model import CostConfig, cost
from nanogld.sizing.kelly import KellyConfig, kelly_size
from nanogld.sizing.vol_target import (
    DEFAULT_BARS_PER_YEAR,
    DEFAULT_TARGET_VOL,
    DEFAULT_VOL_MULT_CAP,
    vol_target_multiplier,
)


SizerStage = Literal["stage1", "stage2"]


@dataclass(frozen=True)
class SizerConfig:
    """Top-level sizing config."""

    stage: SizerStage = "stage2"
    kelly: KellyConfig = KellyConfig(lambda_kelly=0.4, position_limit=1.0)
    cost: CostConfig = CostConfig(gamma=0.02, k_bps=0.7)
    target_vol: float = DEFAULT_TARGET_VOL
    bars_per_year: int = DEFAULT_BARS_PER_YEAR
    vol_mult_cap: float = DEFAULT_VOL_MULT_CAP
    aps_floor: float = DEFAULT_FLOOR_THRESHOLD
    position_limit: float = 1.0


class Sizer:
    """Stateless per-bar sizing orchestrator.

    Args:
        cfg: SizerConfig.
    """

    def __init__(self, cfg: SizerConfig | None = None) -> None:
        self.cfg = cfg if cfg is not None else SizerConfig()

    def compute(
        self,
        head_b_weight: float | np.ndarray,
        aps_lower_bound: float | np.ndarray,
        posterior_variance: float | np.ndarray,
        realized_var_60: float | np.ndarray,
        prev_position: float | np.ndarray = 0.0,
    ) -> np.ndarray:
        """Compute sized position per bar.

        Args:
            head_b_weight: Head B output (already tanh'd, in [-1, 1]).
            aps_lower_bound: APS lower bound on top-class probability.
            posterior_variance: Laplace LLLA epistemic variance proxy.
            realized_var_60: rolling 60-bar realized variance.
            prev_position: previous bar's position (for cost computation).

        Returns:
            np.ndarray with same shape as `head_b_weight`, clipped to
            position_limit.
        """
        cfg = self.cfg
        edge = np.asarray(head_b_weight, dtype=np.float64)
        prev_pos = np.asarray(prev_position, dtype=np.float64)
        vol_mult = vol_target_multiplier(
            realized_var_60,
            target_vol=cfg.target_vol,
            bars_per_year=cfg.bars_per_year,
            vol_mult_cap=cfg.vol_mult_cap,
        )

        if cfg.stage == "stage1":
            sized = edge * vol_mult
            return np.clip(sized, -cfg.position_limit, cfg.position_limit)

        delta_w_estimate = np.abs(edge - prev_pos)
        cost_per_bar = cost(delta_w_estimate, cfg=cfg.cost)

        total_var = (
            np.asarray(realized_var_60, dtype=np.float64)
            + np.asarray(posterior_variance, dtype=np.float64)
        )

        kelly = kelly_size(
            edge=edge,
            variance=total_var,
            cfg=cfg.kelly,
            cost=cost_per_bar,
        )

        sized = kelly * vol_mult
        sized = apply_conformal_floor(sized, aps_lower_bound, threshold=cfg.aps_floor)
        return np.clip(sized, -cfg.position_limit, cfg.position_limit)
