"""Friction-adjusted Kelly sizing (F2F machinery).

Wright et al. 2026: Kelly fraction multiplied by `(1 - cost / |edge|)`
zeroes out trades where cost exceeds edge. lambda=0.4 (V1 default,
half-Kelly-ish).

Math:
    kelly_size = lambda * edge / variance
    effective  = kelly_size * max(0, 1 - cost / max(|edge|, eps))

Spec: plan/V1-SPEC.md §10.1.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class KellyConfig:
    """Kelly sizing config.

    Args:
        lambda_kelly: Kelly fraction multiplier (0.4 V1 default).
        cost_floor_eps: numerical floor on |edge| in the cost ratio.
        position_limit: hard cap on |size|.
    """

    lambda_kelly: float = 0.4
    cost_floor_eps: float = 1e-6
    position_limit: float = 1.0


_DEFAULT_KELLY_CFG = KellyConfig()


def kelly_size(
    edge: float | np.ndarray,
    variance: float | np.ndarray,
    cfg: KellyConfig | None = None,
    cost: float | np.ndarray = 0.0,
) -> np.ndarray:
    """Friction-adjusted Kelly fraction.

    Args:
        edge: per-bar predicted log-return (signed).
        variance: per-bar variance estimate (positive).
        cfg: KellyConfig.
        cost: per-bar round-trip cost in log-return units (e.g. 2bps = 2e-4).

    Returns:
        np.ndarray with same shape as `edge`, clipped to position_limit.
    """
    if cfg is None:
        cfg = _DEFAULT_KELLY_CFG
    edge_arr = np.asarray(edge, dtype=np.float64)
    var_arr = np.asarray(variance, dtype=np.float64)
    cost_arr = np.asarray(cost, dtype=np.float64)

    var_safe = np.where(var_arr > 0, var_arr, np.inf)
    raw_kelly = cfg.lambda_kelly * edge_arr / var_safe

    edge_abs = np.abs(edge_arr)
    cost_ratio = np.where(
        edge_abs > cfg.cost_floor_eps,
        cost_arr / edge_abs,
        np.inf,
    )
    friction_mult = np.maximum(0.0, 1.0 - cost_ratio)

    sized = raw_kelly * friction_mult
    return np.clip(sized, -cfg.position_limit, cfg.position_limit)
