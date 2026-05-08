"""Sqrt-impact cost model.

cost_t = gamma * sqrt(|delta_w|) + k_bps * |delta_w| / 10000

V1 defaults (per Wright F2F): gamma=0.02, k_bps=0.7.

Spec: plan/V1-SPEC.md §10.1.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

DEFAULT_GAMMA = 0.02
DEFAULT_K_BPS = 0.7


@dataclass(frozen=True)
class CostConfig:
    gamma: float = DEFAULT_GAMMA
    k_bps: float = DEFAULT_K_BPS


_DEFAULT_COST_CFG = CostConfig()


def cost(delta_w: float | np.ndarray, cfg: CostConfig | None = None) -> np.ndarray:
    """Sqrt-impact cost in log-return units."""
    if cfg is None:
        cfg = _DEFAULT_COST_CFG
    dw = np.abs(np.asarray(delta_w, dtype=np.float64))
    return cfg.gamma * np.sqrt(dw) + cfg.k_bps * dw / 10_000.0
