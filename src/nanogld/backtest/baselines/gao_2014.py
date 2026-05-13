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
    is_last_bar_of_day: np.ndarray | None = None,
    hold_last_bar_only: bool = True,
) -> np.ndarray:
    """Generate positions from the half-hour-5 rule.

    Args:
        h5_log_return: (T,) log return of bar 5 propagated within day.
            NaN before bar 5 in any session → position 0.
        is_high_vol: (T,) bool. If None, fires every day (no gating).
        is_last_bar_of_day: (T,) bool — True only on the FINAL bar of each
            trading session. Required when ``hold_last_bar_only=True``.
        hold_last_bar_only: per V1-SPEC §54, Gao 2014's signal predicts
            the LAST bar of the day from the 5th half-hour. When True,
            position is held only on the last bar; otherwise the legacy
            "fire every bar after h5" path is taken.

    Returns:
        (T,) position array.
    """
    h5 = np.asarray(h5_log_return, dtype=np.float64)
    sign = np.zeros_like(h5)
    sign[h5 > 0] = 1.0
    sign[h5 < 0] = -1.0
    sign[~np.isfinite(h5)] = 0.0

    if is_high_vol is not None:
        mask = np.asarray(is_high_vol, dtype=bool)
        sign = np.where(mask, sign, 0.0)

    if hold_last_bar_only:
        if is_last_bar_of_day is None:
            raise ValueError(
                "hold_last_bar_only=True requires is_last_bar_of_day; "
                "pass hold_last_bar_only=False to keep the legacy fire-every-bar path"
            )
        last_mask = np.asarray(is_last_bar_of_day, dtype=bool)
        return np.where(last_mask, sign, 0.0)
    return sign
