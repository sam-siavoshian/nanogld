"""V1 triple-barrier labeling, López de Prado AFML Ch. 3.

Replaces the V1-draft fixed-5bps thresholding from features/labels.py.

Per bar T:
  next_log_ret = log(close[T+1] / close[T])
  barrier_up   = +1.0 * ATR-14[T]   (ATR computed on close[<=T])
  barrier_down = -1.0 * ATR-14[T]

  if |next_log_ret| < spread_bps[T] / 1e4:
      label = 0          # FLAT (spread-adjusted neutral, TLOB lesson)
  elif next_log_ret >= barrier_up:
      label = +1         # UP
  elif next_log_ret <= -barrier_down:
      label = -1         # DOWN
  else:
      label = 0          # FLAT (within barriers, 1-bar timeout)

CE mapping: -1, 0, +1 → 0, 1, 2.

Spec: plan/04-FEATURE-ENGINEERING.md V1 triple-barrier section.
Spec: plan/V1-SPEC.md §4.5.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from nanogld.data.utils import get_logger

LOG = get_logger("nanogld.features.triple_barrier")

# CE class mapping: int8 {-1, 0, +1} → uint8 {0, 1, 2}
DOWN_CE = 0
FLAT_CE = 1
UP_CE = 2


def triple_barrier_label(
    next_log_return: float | np.ndarray | pd.Series,
    barrier_up: float | np.ndarray | pd.Series,
    barrier_down: float | np.ndarray | pd.Series,
    spread_bps: float | np.ndarray | pd.Series,
) -> np.ndarray:
    """Compute triple-barrier label for one or many bars.

    Args:
        next_log_return: log(close[T+1] / close[T]) per bar.
        barrier_up: positive ATR-scaled threshold per bar (e.g. 1.0 * ATR-14[T]).
        barrier_down: positive ATR-scaled threshold per bar (sign-positive!).
        spread_bps: bid-ask spread in basis points at bar close.

    Returns:
        np.ndarray of int8 in {-1, 0, +1}. NaN inputs map to 0.

    Notes:
        barrier_up and barrier_down should both be passed as positive
        magnitudes. The DOWN comparison is `next_log_return <= -barrier_down`.
        Spread-adjusted neutral fires when |next_log_return| < spread/1e4
        even if a barrier is touched — TLOB Berti & Kasneci 2025 finding
        that label threshold below half-spread is unprofitable noise.
    """
    nlr = np.asarray(next_log_return, dtype=np.float64)
    up = np.asarray(barrier_up, dtype=np.float64)
    dn = np.asarray(barrier_down, dtype=np.float64)
    sp_bps = np.asarray(spread_bps, dtype=np.float64)

    spread_frac = sp_bps / 10_000.0

    nan_mask = np.isnan(nlr) | np.isnan(up) | np.isnan(dn) | np.isnan(sp_bps)
    abs_nlr = np.abs(nlr)

    labels = np.zeros_like(nlr, dtype=np.int8)
    spread_neutral = abs_nlr < spread_frac
    fired_up = (~spread_neutral) & (nlr >= up)
    fired_down = (~spread_neutral) & (nlr <= -dn)

    labels[fired_up] = 1
    labels[fired_down] = -1
    labels[nan_mask] = 0
    return labels


def to_ce_class(triple_barrier: np.ndarray | pd.Series) -> np.ndarray:
    """Map int8 triple-barrier labels {-1, 0, +1} to CE classes {0, 1, 2}."""
    arr = np.asarray(triple_barrier, dtype=np.int8)
    return (arr + 1).astype(np.int8)


def class_distribution(labels: np.ndarray | pd.Series) -> dict[str, float]:
    """Return {DOWN: pct, FLAT: pct, UP: pct} fractions on int8 labels."""
    arr = np.asarray(labels, dtype=np.int8)
    n = arr.size
    if n == 0:
        return {"DOWN": 0.0, "FLAT": 0.0, "UP": 0.0}
    return {
        "DOWN": float(np.sum(arr == -1)) / n,
        "FLAT": float(np.sum(arr == 0)) / n,
        "UP": float(np.sum(arr == 1)) / n,
    }


def add_triple_barrier_columns(
    df: pd.DataFrame,
    *,
    next_log_return_col: str = "next_log_return",
    barrier_up_col: str = "barrier_up",
    barrier_down_col: str = "barrier_down",
    spread_col: str = "gld_spread_bps_t",
    label_col: str = "label_triple_barrier",
    ce_col: str = "label_ce",
) -> pd.DataFrame:
    """Append triple-barrier label + CE-mapped columns to a bar-aligned DataFrame.

    Source columns must already exist (typically built upstream via the
    sidecar generator script). Operates on a copy.
    """
    out = df.copy()
    missing = {next_log_return_col, barrier_up_col, barrier_down_col, spread_col} - set(out.columns)
    if missing:
        raise KeyError(f"missing required columns for triple-barrier labeling: {missing}")

    labels = triple_barrier_label(
        out[next_log_return_col].to_numpy(),
        out[barrier_up_col].to_numpy(),
        out[barrier_down_col].to_numpy(),
        out[spread_col].to_numpy(),
    )
    out[label_col] = labels
    out[ce_col] = to_ce_class(labels)

    dist = class_distribution(labels)
    LOG.info(
        "triple-barrier added: DOWN=%.1f%% FLAT=%.1f%% UP=%.1f%%",
        100 * dist["DOWN"],
        100 * dist["FLAT"],
        100 * dist["UP"],
    )
    return out
