"""Vol-target multiplier for sizing.

Math:
    realized_vol_annualized = sqrt(realized_var_60bar * bars_per_year)
    multiplier = target_vol / max(realized_vol_annualized, eps)
    multiplier = clip(multiplier, 0, vol_mult_cap)

V1 default: target_vol=0.15 (15% annualized), bars_per_year=3276 (NYSE
RTH 30-min, V1 invariant 5), vol_mult_cap=3.0.

Spec: plan/V1-SPEC.md §10.1.
"""

from __future__ import annotations

import numpy as np

DEFAULT_TARGET_VOL = 0.15
DEFAULT_BARS_PER_YEAR = 3276
DEFAULT_VOL_MULT_CAP = 3.0
EPS = 1e-8


def vol_target_multiplier(
    realized_var_60bar: float | np.ndarray,
    target_vol: float = DEFAULT_TARGET_VOL,
    bars_per_year: int = DEFAULT_BARS_PER_YEAR,
    vol_mult_cap: float = DEFAULT_VOL_MULT_CAP,
) -> np.ndarray:
    """Convert per-bar variance to a sizing multiplier.

    Args:
        realized_var_60bar: variance of log-returns over rolling 60 bars.
        target_vol: annualized vol target (0.15 V1).
        bars_per_year: NYSE RTH 30-min = 3276.
        vol_mult_cap: max allowed multiplier.

    Returns:
        np.ndarray of multipliers in [0, vol_mult_cap].
    """
    var_arr = np.asarray(realized_var_60bar, dtype=np.float64)
    annualized_vol = np.sqrt(np.maximum(var_arr, 0.0) * bars_per_year)
    multiplier = target_vol / np.maximum(annualized_vol, EPS)
    return np.clip(multiplier, 0.0, vol_mult_cap)
