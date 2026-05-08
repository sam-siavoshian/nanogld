"""Gao-Han-Li-Zhou 2014 single-feature half-hour-5 rule on GLD.

position = sign(h5_log_return) * is_high_vol_day
        = +1 if h5 > 0 and high-vol day
        = -1 if h5 < 0 and high-vol day
        =  0 otherwise

Single-feature Sharpe 5.43 in the 2014 sample. V1 promotion floor: if
nanoGLD ties or loses to this, ship the simpler ensemble.

Spec: plan/V1-SPEC.md §0/4.1.
"""

from __future__ import annotations

import numpy as np


def gao_2014_positions(
    h5_log_return: np.ndarray,
    is_high_vol: np.ndarray | None = None,
) -> np.ndarray:
    """Generate positions from the half-hour-5 rule.

    Args:
        h5_log_return: (T,) log return of bar 5 propagated within day.
            NaN before bar 5 in any session → position 0.
        is_high_vol: (T,) bool. If None, fires every day (no gating).

    Returns:
        (T,) position array.
    """
    h5 = np.asarray(h5_log_return, dtype=np.float64)
    sign = np.zeros_like(h5)
    sign[h5 > 0] = 1.0
    sign[h5 < 0] = -1.0
    sign[~np.isfinite(h5)] = 0.0

    if is_high_vol is None:
        return sign
    mask = np.asarray(is_high_vol, dtype=bool)
    return np.where(mask, sign, 0.0)
