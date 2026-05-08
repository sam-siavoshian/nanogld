"""V1 half-hour-5 intraday momentum feature (Gao-Han-Li-Zhou 2014).

The 5th half-hour of the NYSE RTH session (~11:30-12:00 ET) carries
predictive signal for the closing half-hour. Reported Sharpe 5.43 on GLD
specifically as a single-feature timing rule, concentrated on high-vol
days.

This module emits two columns:
  gld_h5_log_return : log(close[T] / close[T-1]) when bar T is the 5th
                      RTH bar of the day, propagated forward to all
                      remaining bars in that day. NaN before bar 5.
  gld_h5_x_vol_high : interaction with rolling vol-tercile-high indicator
                      (zero except on high-vol days).

PIT-correct: both columns use only data published by bar T. The forward
propagation within a day uses the value computed at bar 5 of the SAME
session, which is fully observed by the time bar 6+ runs.

Spec: plan/04-FEATURE-ENGINEERING.md V1 half-hour-5 section.
Spec: plan/V1-SPEC.md §4.1.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pandas_market_calendars as mcal

from nanogld.data.utils import ET, get_logger

LOG = get_logger("nanogld.features.h5")

H5_BAR_INDEX = 4  # 0-indexed: bars 0,1,2,3,4 → 5th bar is index 4
DEFAULT_VOL_LOOKBACK = 60
NYSE = mcal.get_calendar("XNYS")


def _rth_bar_index(timestamps_utc: pd.DatetimeIndex) -> pd.Series:
    """Return the within-day RTH bar index (0..12) for each UTC timestamp.

    Bars outside RTH return -1. Uses the bar's NYSE local open time for
    indexing: ET 09:30 → 0, ET 10:00 → 1, ..., ET 15:30 → 12.
    """
    if len(timestamps_utc) == 0:
        return pd.Series([], dtype="int64")
    et = pd.DatetimeIndex(timestamps_utc).tz_convert(ET)
    minutes_since_open = (et.hour - 9) * 60 + (et.minute - 30)
    bar_idx = minutes_since_open // 30
    bar_idx = pd.Series(bar_idx, index=timestamps_utc, dtype="int64")
    bar_idx[bar_idx < 0] = -1
    bar_idx[bar_idx > 12] = -1
    return bar_idx


def fit_h5_vol_threshold(
    train_df: pd.DataFrame,
    *,
    close_col: str = "gld_close",
    bar_open_utc_col: str = "bar_open_utc",
    vol_lookback: int = DEFAULT_VOL_LOOKBACK,
) -> float:
    """Fit the high-vol tercile threshold on a TRAIN-ONLY DataFrame.

    Returns the 2/3 quantile of rolling realized vol over `vol_lookback` bars.
    The returned scalar must be passed to `add_h5_features(..., high_vol_threshold=...)`
    when applied to val/test to avoid leakage.
    """
    if close_col not in train_df.columns:
        return float("nan")
    log_ret = np.log(train_df[close_col] / train_df[close_col].shift(1))
    realized_vol = log_ret.rolling(vol_lookback, min_periods=vol_lookback // 2).std()
    rv = realized_vol.dropna()
    if len(rv) < 3:
        return float("nan")
    return float(rv.quantile(2.0 / 3.0))


def add_h5_features(
    df: pd.DataFrame,
    *,
    high_vol_threshold: float,
    close_col: str = "gld_close",
    bar_open_utc_col: str = "bar_open_utc",
    vol_lookback: int = DEFAULT_VOL_LOOKBACK,
) -> pd.DataFrame:
    """Append `gld_h5_log_return` and `gld_h5_x_vol_high` columns.

    Args:
        df: bar-aligned DataFrame, sorted by bar_open_utc ascending.
        high_vol_threshold: REQUIRED. Frozen 2/3 quantile of train-split RV
            from `fit_h5_vol_threshold(train_df)`. Caller must pass the same
            value at val and test time to prevent leakage.
        close_col: column with bar close prices.
        bar_open_utc_col: column with UTC bar-open timestamps.
        vol_lookback: realized-vol window in bars for the high-vol indicator.
    """
    out = df.copy()
    missing = {close_col, bar_open_utc_col} - set(out.columns)
    if missing:
        raise KeyError(f"missing required columns for h5 feature: {missing}")

    ts = pd.to_datetime(out[bar_open_utc_col], utc=True)
    if not ts.is_monotonic_increasing:
        out = out.sort_values(bar_open_utc_col).reset_index(drop=True)
        ts = pd.to_datetime(out[bar_open_utc_col], utc=True)

    bar_idx_within_day = _rth_bar_index(pd.DatetimeIndex(ts))
    bar_idx_within_day.index = out.index

    log_ret = np.log(out[close_col] / out[close_col].shift(1))

    h5_value_at_bar = log_ret.where(bar_idx_within_day == H5_BAR_INDEX, other=np.nan)

    et_date = pd.DatetimeIndex(ts).tz_convert(ET).normalize().tz_convert("UTC")
    out["_h5_session_date_utc"] = et_date

    h5_per_day = pd.Series(h5_value_at_bar.to_numpy(), index=out["_h5_session_date_utc"])
    h5_per_day = h5_per_day.dropna()
    h5_per_day = h5_per_day[~h5_per_day.index.duplicated(keep="last")]

    out["gld_h5_log_return"] = (
        out["_h5_session_date_utc"].map(h5_per_day).where(bar_idx_within_day >= H5_BAR_INDEX)
    )
    out["gld_h5_log_return"] = out["gld_h5_log_return"].astype("float32")

    realized_vol = log_ret.rolling(vol_lookback, min_periods=vol_lookback // 2).std()
    if np.isnan(high_vol_threshold):
        is_high_vol = pd.Series(np.zeros(len(out), dtype=np.float32), index=out.index)
    else:
        is_high_vol = (realized_vol > high_vol_threshold).astype("float32")
    out["gld_h5_x_vol_high"] = (out["gld_h5_log_return"].fillna(0.0) * is_high_vol).astype(
        "float32"
    )

    out = out.drop(columns=["_h5_session_date_utc"])
    n_valid = int(out["gld_h5_log_return"].notna().sum())
    LOG.info("h5 feature: %d/%d valid rows; vol_lookback=%d", n_valid, len(out), vol_lookback)
    return out
