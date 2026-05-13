"""Moving-average crossover baseline.

position = +1 when fast EMA > slow EMA, else -1 (or 0 if `flat_below=True`).

Default: 50/200 EMA cross.
"""

from __future__ import annotations

import numpy as np


def ema(x: np.ndarray, span: int) -> np.ndarray:
    """Exponential moving average, alpha = 2/(span+1)."""
    alpha = 2.0 / (span + 1)
    out = np.empty_like(x, dtype=np.float64)
    out[0] = x[0]
    for i in range(1, len(x)):
        out[i] = alpha * x[i] + (1 - alpha) * out[i - 1]
    return out


def ma_cross_positions(
    close: np.ndarray,
    fast_span: int = 50,
    slow_span: int = 200,
    flat_below: bool = False,
    warmup_bars: int | None = None,
) -> np.ndarray:
    """Generate ±1 positions from EMA crossover with explicit warmup zero.

    Args:
        close: (T,) close prices.
        fast_span: fast EMA span.
        slow_span: slow EMA span.
        flat_below: if True, use 0 instead of -1 below cross.
        warmup_bars: number of leading bars to force-zero. Defaults to
            ``slow_span`` (so the slow EMA has had ``slow_span`` samples
            to converge before any position is taken). Set to 0 to keep
            the legacy behavior where bars 1..slow_span-1 carry raw
            signal off a barely-warmed EMA. V1-SPEC §54.

    Returns:
        (T,) position array shifted by 1 (use lagged signal).
    """
    close = np.asarray(close, dtype=np.float64)
    fast = ema(close, fast_span)
    slow = ema(close, slow_span)
    raw = np.where(fast > slow, 1.0, 0.0 if flat_below else -1.0)
    pos = np.empty_like(raw)
    pos[0] = 0.0
    pos[1:] = raw[:-1]
    warm = slow_span if warmup_bars is None else int(warmup_bars)
    if warm > 0:
        pos[: min(warm, len(pos))] = 0.0
    return pos
