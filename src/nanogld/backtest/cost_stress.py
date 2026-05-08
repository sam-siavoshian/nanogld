"""Cost-stress: run backtest at {0.5x, 1.0x, 1.5x} cost (V1 hard gate).

V1 invariant 19: every reported Sharpe must come with cost-stress at
{0.5x, 1.0x, 1.5x}. V1 promotion gate 2: Sharpe > 0.5 at 1.5x cost.

Spec: plan/V1-SPEC.md §9.2.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from nanogld.backtest.engine import BacktestConfig, vectorized_backtest
from nanogld.backtest.metrics import compute_metrics


@dataclass(frozen=True)
class CostStressResult:
    by_multiplier: dict[float, dict[str, float]]


def cost_stress(
    next_log_returns: np.ndarray,
    positions: np.ndarray,
    base_cost_bps: float = 2.0,
    multipliers: tuple[float, ...] = (0.5, 1.0, 1.5),
) -> CostStressResult:
    """Run backtest at each cost multiplier and bundle metrics.

    Args:
        next_log_returns: (T,).
        positions: (T,).
        base_cost_bps: 1.0x cost in basis points.
        multipliers: cost multipliers to evaluate.

    Returns:
        CostStressResult mapping each multiplier to its metrics dict.
    """
    out: dict[float, dict[str, float]] = {}
    for m in multipliers:
        cfg = BacktestConfig(cost_bps=base_cost_bps * m)
        result = vectorized_backtest(next_log_returns, positions, cfg=cfg)
        out[float(m)] = compute_metrics(result.pnl_per_bar, result.equity_curve)
    return CostStressResult(by_multiplier=out)


def cost_stress_summary(result: CostStressResult) -> dict:
    """Return a JSON-serializable summary."""
    return {f"{m}x": metrics for m, metrics in result.by_multiplier.items()}


def passes_v1_gate(result: CostStressResult, threshold_at_1_5x: float = 0.5) -> bool:
    """V1 gate 2: Sharpe at 1.5x cost > threshold (0.5 default)."""
    metrics_15 = result.by_multiplier.get(1.5)
    if metrics_15 is None:
        return False
    return metrics_15["sharpe"] > threshold_at_1_5x
