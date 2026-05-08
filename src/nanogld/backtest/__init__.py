"""nanoGLD V1 backtest package — public API."""

from nanogld.backtest.cost_stress import (
    CostStressResult,
    cost_stress,
    cost_stress_summary,
    passes_v1_gate,
)
from nanogld.backtest.dsr import deflated_sharpe
from nanogld.backtest.engine import (
    BacktestConfig,
    BacktestResult,
    vectorized_backtest,
)
from nanogld.backtest.metrics import (
    calmar,
    compute_metrics,
    expectancy,
    hit_rate,
    max_drawdown,
    profit_factor,
    sharpe,
    sortino,
)
from nanogld.backtest.per_bucket import per_bucket_metrics

__all__ = [
    "BacktestConfig",
    "BacktestResult",
    "CostStressResult",
    "calmar",
    "compute_metrics",
    "cost_stress",
    "cost_stress_summary",
    "deflated_sharpe",
    "expectancy",
    "hit_rate",
    "max_drawdown",
    "passes_v1_gate",
    "per_bucket_metrics",
    "profit_factor",
    "sharpe",
    "sortino",
    "vectorized_backtest",
]
