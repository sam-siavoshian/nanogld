"""Forecast-to-Fill (F2F) baseline (Wright 2026 arXiv:2511.08571).

Daily gold-futures strategy ported to a 30-min intraday harness via
resample-up to daily, signal-down to intraday by forward-filling the
last daily decision across the bars of that session.

V1-SPEC §0 calls this a **separate scoreboard** — the strategy's
native horizon is daily, and a 1.5x cost stress collapses its reported
2.88 Sharpe to roughly ``-0.03``. The harness includes it for honest
side-by-side reporting; nanoGLD is NOT expected to beat F2F on its
native daily horizon, only to outperform on the intraday scoreboard.

Implementation: trend (20-day EMA crossover) + momentum (5-day return)
two-factor signal, ATR-vol-target sizing, 20-30 day hold windows. We
build daily bars from intraday close samples taken once per session
(via ``is_last_bar_of_day``).
"""

from __future__ import annotations

from typing import Any

import numpy as np


def _daily_indices(is_last_bar: np.ndarray) -> np.ndarray:
    """Return indices into the intraday array where session-close lives."""
    idx = np.flatnonzero(np.asarray(is_last_bar, dtype=bool))
    return idx


def _ema(x: np.ndarray, span: int) -> np.ndarray:
    alpha = 2.0 / (span + 1)
    out = np.empty_like(x, dtype=np.float64)
    if len(x) == 0:
        return out
    out[0] = x[0]
    for i in range(1, len(x)):
        out[i] = alpha * x[i] + (1 - alpha) * out[i - 1]
    return out


def _atr_daily(close: np.ndarray, window: int = 14) -> np.ndarray:
    diff = np.abs(np.diff(close, prepend=close[0]))
    atr = np.empty_like(diff)
    if len(diff) == 0:
        return atr
    atr[0] = diff[0]
    for i in range(1, len(diff)):
        if i < window:
            atr[i] = diff[: i + 1].mean()
        else:
            atr[i] = atr[i - 1] * (window - 1) / window + diff[i] / window
    return atr


def forecast_to_fill_positions(
    ctx: dict[str, Any],
    *,
    trend_fast: int = 5,
    trend_slow: int = 20,
    momentum_lookback: int = 5,
    vol_target_daily: float = 0.01,
    atr_window: int = 14,
    max_position: float = 1.0,
) -> np.ndarray:
    """Daily-F2F signal forward-filled to intraday bars.

    Args:
        ctx: fold context. Must carry ``close`` (intraday closes) and
            ``is_last_bar_of_day`` (bool per intraday bar). Other keys
            are unused.
        trend_fast, trend_slow: EMA spans for the trend factor (daily).
        momentum_lookback: number of sessions for the momentum factor.
        vol_target_daily: target daily vol for ATR sizing.
        atr_window: ATR-14 sample horizon.
        max_position: clamp for the final position weight.
    """
    test_n = len(ctx["next_log_returns"])
    close = np.asarray(ctx.get("close"), dtype=np.float64)
    is_last_bar = np.asarray(ctx.get("is_last_bar_of_day"), dtype=bool)
    if close.size != test_n or is_last_bar.size != test_n:
        return np.zeros(test_n, dtype=np.float64)
    daily_idx = _daily_indices(is_last_bar)
    if daily_idx.size < max(trend_slow, momentum_lookback, atr_window) + 1:
        # Not enough daily samples to drive the F2F signal.
        return np.zeros(test_n, dtype=np.float64)

    daily_close = close[daily_idx]
    ema_fast = _ema(daily_close, trend_fast)
    ema_slow = _ema(daily_close, trend_slow)
    trend_sign = np.sign(ema_fast - ema_slow)
    momentum = np.zeros_like(daily_close)
    momentum[momentum_lookback:] = (
        daily_close[momentum_lookback:] / daily_close[:-momentum_lookback] - 1.0
    )
    momentum_sign = np.sign(momentum)

    raw_signal = np.where(trend_sign == momentum_sign, trend_sign, 0.0)
    atr = _atr_daily(daily_close, window=atr_window)
    atr_rel = np.where(daily_close > 0, atr / daily_close, np.inf)
    size = np.where(atr_rel > 0, vol_target_daily / atr_rel, 0.0)
    size = np.clip(size, 0.0, max_position)
    daily_position = raw_signal * size

    # Forward-fill across intraday bars within each session.
    intraday = np.zeros(test_n, dtype=np.float64)
    session_pos = 0.0
    daily_cursor = 0
    for i in range(test_n):
        intraday[i] = session_pos
        if is_last_bar[i] and daily_cursor < len(daily_position):
            session_pos = float(daily_position[daily_cursor])
            daily_cursor += 1
    return intraday


__all__ = ["forecast_to_fill_positions"]
