"""Donchian breakout baseline.

position = +1 when close breaks N-bar high, -1 when breaks N-bar low.
"""

from __future__ import annotations

import numpy as np


def donchian_positions(
    close: np.ndarray,
    high: np.ndarray | None = None,
    low: np.ndarray | None = None,
    window: int = 20,
) -> np.ndarray:
    """Generate ±1 positions from Donchian breakout.

    Args:
        close: (T,) close prices.
        high: (T,) high prices. Defaults to close.
        low: (T,) low prices. Defaults to close.
        window: lookback bars.

    Returns:
        (T,) position array, shifted by 1.
    """
    close = np.asarray(close, dtype=np.float64)
    high = close if high is None else np.asarray(high, dtype=np.float64)
    low = close if low is None else np.asarray(low, dtype=np.float64)
    n = len(close)

    pos = np.zeros(n, dtype=np.float64)
    cur = 0.0
    for i in range(window, n):
        upper = high[i - window : i].max()
        lower = low[i - window : i].min()
        if close[i] > upper:
            cur = 1.0
        elif close[i] < lower:
            cur = -1.0
        pos[i] = cur
    return pos
