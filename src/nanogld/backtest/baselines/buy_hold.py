"""Buy-and-hold baseline: position = +1 every bar."""

from __future__ import annotations

import numpy as np


def buy_hold_positions(n_bars: int) -> np.ndarray:
    """Return constant +1 positions of length n_bars."""
    return np.ones(n_bars, dtype=np.float64)
