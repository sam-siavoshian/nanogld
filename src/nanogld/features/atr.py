"""V1 ATR-14 (Wilder EMA) + ATR-scaled triple-barrier thresholds.

Average True Range, Wilder smoothing (alpha = 1/period). Used to scale
the up/down barriers for triple-barrier labeling.

PIT-correct: ATR[T] uses only OHLC up to and including bar T. The barriers
written into the feature row at bar T are `close[T] +/- 1.0 * ATR[T]` and
serve as thresholds for next_log_return at T+1.

Spec: plan/04-FEATURE-ENGINEERING.md V1 ATR section.
Spec: plan/V1-SPEC.md §4.5.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from nanogld.data.utils import get_logger

LOG = get_logger("nanogld.features.atr")

DEFAULT_ATR_PERIOD = 14
DEFAULT_BARRIER_MULT = 1.0


def true_range(high: pd.Series, low: pd.Series, close_prev: pd.Series) -> pd.Series:
    """True Range = max(high - low, |high - close_prev|, |low - close_prev|).

    `close_prev` must be the previous bar's close (already shifted by caller).
    Returns NaN where any input is NaN.
    """
    hl = high - low
    hc = (high - close_prev).abs()
    lc = (low - close_prev).abs()
    return pd.concat([hl, hc, lc], axis=1).max(axis=1)


def atr_wilder(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = DEFAULT_ATR_PERIOD,
) -> pd.Series:
    """ATR via Wilder EMA: alpha = 1/period.

    Causal: result at time T uses only data up to T. First (period - 1)
    rows are NaN by design.
    """
    if not (len(high) == len(low) == len(close)):
        raise ValueError("high, low, close must have equal length")
    close_prev = close.shift(1)
    tr = true_range(high, low, close_prev)
    return tr.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()


def add_atr_and_barriers(
    df: pd.DataFrame,
    *,
    high_col: str = "gld_high",
    low_col: str = "gld_low",
    close_col: str = "gld_close",
    period: int = DEFAULT_ATR_PERIOD,
    barrier_mult: float = DEFAULT_BARRIER_MULT,
) -> pd.DataFrame:
    """Append `gld_atr_{period}`, `barrier_up`, `barrier_down` columns.

    Output column units (mixed by design — read carefully):
      gld_atr_{period} : PRICE units (USD per share). Suitable for stop sizing.
      barrier_up       : LOG-RETURN units, POSITIVE magnitude (≈ ATR / close).
      barrier_down     : LOG-RETURN units, POSITIVE magnitude. The caller
                         (triple_barrier_label) does the negative-sign compare.

    Both barriers are POSITIVE magnitudes. The down-barrier is a positive
    threshold the negative log-return must exceed in absolute value:

        DOWN if next_log_return ≤ -barrier_down

    Storing the down-barrier as a positive magnitude (not as a signed
    negative number) makes downstream sizing math symmetric and avoids
    sign-flip bugs. Document any consumer that assumes a signed convention.

    ATR is computed via Wilder EMA (alpha = 1/period), per V1-SPEC §4.5.
    """
    out = df.copy()
    missing = {high_col, low_col, close_col} - set(out.columns)
    if missing:
        raise KeyError(f"missing required OHLC columns: {missing}")

    if "bar_close_utc" in out.columns:
        bc = pd.to_datetime(out["bar_close_utc"], utc=True)
        if not bc.is_monotonic_increasing:
            out = out.sort_values("bar_close_utc").reset_index(drop=True)

    atr_col = f"gld_atr_{period}"
    out[atr_col] = (
        atr_wilder(out[high_col], out[low_col], out[close_col], period=period).astype("float32")
    )
    safe_close = out[close_col].where(out[close_col] > 0)
    n_bad_close = int((~(out[close_col] > 0)).sum())
    if n_bad_close > 0:
        LOG.warning(
            "ATR: %d rows with non-positive close — barriers NaN-masked at those rows",
            n_bad_close,
        )
    atr_log_return = out[atr_col] / safe_close

    out["barrier_up"] = (barrier_mult * atr_log_return).astype("float32")
    out["barrier_down"] = (barrier_mult * atr_log_return).astype("float32")
    n_valid = int(out[atr_col].notna().sum())
    LOG.info(
        "ATR-%d barriers added: %d/%d valid rows, mult=%.2f",
        period,
        n_valid,
        len(out),
        barrier_mult,
    )
    return out
