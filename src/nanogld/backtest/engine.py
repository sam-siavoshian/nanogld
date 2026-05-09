"""V1 vectorized backtest engine.

Inputs:
    next_log_returns: (T,) per-bar realized log returns
    positions: (T,) per-bar position weight in [-1, +1]
    cost_model: optional callable(delta_w) -> per-bar cost in log-return units

Computes:
    pnl = position * next_log_return - cost(|delta_w|)
    cum_log_equity = cumsum(pnl)
    equity_curve = exp(cum_log_equity)

Spec: plan/06-BACKTEST.md V1 backtest.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class BacktestConfig:
    """Backtest run config."""

    cost_bps: float = 2.0
    initial_equity: float = 1.0
    bars_per_year: int = 3276


@dataclass(frozen=True)
class BacktestResult:
    """Backtest output bundle."""

    pnl_per_bar: np.ndarray = field()
    equity_curve: np.ndarray = field()
    positions: np.ndarray = field()
    realized_returns: np.ndarray = field()


def vectorized_backtest(
    next_log_returns: np.ndarray,
    positions: np.ndarray,
    cfg: BacktestConfig | None = None,
    cost_model: Callable[[np.ndarray], np.ndarray] | None = None,
) -> BacktestResult:
    """Run a vectorized backtest.

    Args:
        next_log_returns: (T,) realized log return for each bar's next-bar window.
        positions: (T,) position weights aligned to next_log_returns.
        cfg: BacktestConfig. Default: 2 bps round-trip, initial_equity 1.0.
        cost_model: optional callable for advanced cost models. If None,
            uses a flat `cost_bps * |delta_w| / 10000` model.

    Returns:
        BacktestResult.
    """
    if cfg is None:
        cfg = BacktestConfig()
    nlr = np.asarray(next_log_returns, dtype=np.float64)
    pos = np.asarray(positions, dtype=np.float64)
    if nlr.shape != pos.shape:
        raise ValueError(f"shape mismatch: returns {nlr.shape} vs positions {pos.shape}")

    delta_w = np.diff(pos, prepend=0.0)
    delta_w_abs = np.abs(delta_w)

    if cost_model is None:
        cost_per_bar = cfg.cost_bps * delta_w_abs / 10_000.0
    else:
        cost_per_bar = np.asarray(cost_model(delta_w_abs), dtype=np.float64)

    pnl = pos * np.nan_to_num(nlr, nan=0.0) - cost_per_bar
    cum_log = np.cumsum(pnl)
    equity = cfg.initial_equity * np.exp(cum_log)
    return BacktestResult(
        pnl_per_bar=pnl,
        equity_curve=equity,
        positions=pos,
        realized_returns=nlr,
    )
