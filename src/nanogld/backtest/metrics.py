"""V1 backtest metrics.

V1 invariant 5: bars_per_year = 3276 (NYSE RTH 30-min). NEVER 17500.

Sortino uses the canonical formula:
    sqrt(mean(min(0, r)^2))
NOT std(r[r<0]) (the wrong common variant).

Spec: plan/06-BACKTEST.md V1 metrics.
"""

from __future__ import annotations

import numpy as np

DEFAULT_BARS_PER_YEAR = 3276


def sharpe(pnl_per_bar: np.ndarray, bars_per_year: int = DEFAULT_BARS_PER_YEAR) -> float:
    """Annualized Sharpe ratio."""
    pnl = np.asarray(pnl_per_bar, dtype=np.float64)
    if pnl.size == 0 or pnl.std(ddof=1) == 0:
        return 0.0
    return float(pnl.mean() / pnl.std(ddof=1) * np.sqrt(bars_per_year))


def sortino(pnl_per_bar: np.ndarray, bars_per_year: int = DEFAULT_BARS_PER_YEAR) -> float:
    """Canonical Sortino: sqrt(mean(min(0, r)^2))."""
    pnl = np.asarray(pnl_per_bar, dtype=np.float64)
    if pnl.size == 0:
        return 0.0
    downside = np.minimum(0.0, pnl)
    downside_dev = float(np.sqrt((downside**2).mean()))
    if downside_dev == 0.0:
        return 0.0
    return float(pnl.mean() / downside_dev * np.sqrt(bars_per_year))


def max_drawdown(equity_curve: np.ndarray) -> float:
    """Maximum drawdown as a positive fraction (0.10 = 10%)."""
    eq = np.asarray(equity_curve, dtype=np.float64)
    if eq.size == 0:
        return 0.0
    peak = np.maximum.accumulate(eq)
    dd = (eq - peak) / np.maximum(peak, 1e-12)
    return float(-dd.min())


def calmar(
    pnl_per_bar: np.ndarray,
    equity_curve: np.ndarray,
    bars_per_year: int = DEFAULT_BARS_PER_YEAR,
) -> float:
    """Annualized return / max-drawdown."""
    mdd = max_drawdown(equity_curve)
    if mdd == 0:
        return 0.0
    pnl = np.asarray(pnl_per_bar, dtype=np.float64)
    annualized_return = float(pnl.mean() * bars_per_year)
    return annualized_return / mdd


def hit_rate(pnl_per_bar: np.ndarray) -> float:
    """Fraction of bars with positive PnL."""
    pnl = np.asarray(pnl_per_bar, dtype=np.float64)
    nz = pnl[pnl != 0]
    if nz.size == 0:
        return 0.0
    return float((nz > 0).mean())


def profit_factor(pnl_per_bar: np.ndarray) -> float:
    """Sum of positive PnL / |sum of negative PnL|."""
    pnl = np.asarray(pnl_per_bar, dtype=np.float64)
    pos = pnl[pnl > 0].sum()
    neg = -pnl[pnl < 0].sum()
    if neg == 0:
        return float("inf") if pos > 0 else 0.0
    return float(pos / neg)


def expectancy(pnl_per_bar: np.ndarray) -> float:
    """Average PnL per non-zero bar."""
    pnl = np.asarray(pnl_per_bar, dtype=np.float64)
    nz = pnl[pnl != 0]
    if nz.size == 0:
        return 0.0
    return float(nz.mean())


def compute_metrics(
    pnl_per_bar: np.ndarray,
    equity_curve: np.ndarray,
    bars_per_year: int = DEFAULT_BARS_PER_YEAR,
) -> dict[str, float]:
    """Bundle all metrics into a dict."""
    return {
        "sharpe": sharpe(pnl_per_bar, bars_per_year=bars_per_year),
        "sortino": sortino(pnl_per_bar, bars_per_year=bars_per_year),
        "calmar": calmar(pnl_per_bar, equity_curve, bars_per_year=bars_per_year),
        "max_drawdown": max_drawdown(equity_curve),
        "hit_rate": hit_rate(pnl_per_bar),
        "profit_factor": profit_factor(pnl_per_bar),
        "expectancy": expectancy(pnl_per_bar),
        "n_bars": int(np.asarray(pnl_per_bar).size),
    }
