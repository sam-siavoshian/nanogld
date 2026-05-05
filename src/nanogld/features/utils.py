"""Shared feature-engineering helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from nanogld.data.utils import data_root


def processed_dir() -> Path:
    p = data_root() / "processed"
    p.mkdir(parents=True, exist_ok=True)
    return p


@dataclass(frozen=True)
class FeatureSpec:
    """Documents one feature column on the panel."""

    name: str
    dtype: str = "float64"
    description: str = ""
    source: str = ""  # underlying raw parquet
    nullable: bool = True


def log_returns(s: pd.Series, k: int = 1) -> pd.Series:
    """Log return over k periods. PIT-safe — uses .shift, never future."""
    return np.log(s).diff(k)


def realized_vol(s: pd.Series, window: int) -> pd.Series:
    """Sample stdev of log returns over `window`, annualized factor left to caller."""
    r = np.log(s).diff()
    return r.rolling(window, min_periods=max(2, window // 2)).std()


def garman_klass_vol(
    high: pd.Series, low: pd.Series, open_: pd.Series, close: pd.Series, window: int
) -> pd.Series:
    """Garman-Klass realized vol — more efficient than close-only stdev.
    Spec V1 hard rule: prefer GK over Parkinson (same OHLC, more efficient).
    """
    hl = (np.log(high) - np.log(low)) ** 2
    co = (np.log(close) - np.log(open_)) ** 2
    daily = 0.5 * hl - (2 * np.log(2) - 1) * co
    return daily.rolling(window, min_periods=max(2, window // 2)).mean().pow(0.5)


def rolling_z(s: pd.Series, window: int, min_periods: int | None = None) -> pd.Series:
    """Rolling z-score with explicit min_periods. Returns NaN during warmup."""
    mp = min_periods if min_periods is not None else max(2, window // 2)
    mu = s.rolling(window, min_periods=mp).mean()
    sd = s.rolling(window, min_periods=mp).std()
    return (s - mu) / sd


def daily_index_utc(start: pd.Timestamp, end: pd.Timestamp) -> pd.DatetimeIndex:
    """Calendar-daily UTC index (inclusive). Used as the panel timeline."""
    return pd.date_range(start=start.normalize(), end=end.normalize(), freq="1D", tz="UTC")


def forward_fill_to_daily(
    sparse: pd.DataFrame,
    *,
    on: str,
    cols: list[str],
    daily_idx: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Forward-fill a sparse (weekly/monthly) frame onto a daily UTC grid.

    Strict PIT: a feature value lands on day D only if its `t_visible <= D`.
    Use `t_visible` as the alignment key, not `date`.
    """
    if sparse.empty:
        return pd.DataFrame({c: pd.Series([pd.NA] * len(daily_idx)) for c in cols}, index=daily_idx)

    base = pd.DataFrame({"date_utc": daily_idx})
    sub = sparse[[on, *cols]].sort_values(on).copy()
    sub = sub.rename(columns={on: "t_visible"})
    out = pd.merge_asof(
        base.sort_values("date_utc"),
        sub,
        left_on="date_utc",
        right_on="t_visible",
        direction="backward",
        allow_exact_matches=True,  # data with t_visible == day-end IS visible by next day
    ).drop(columns=["t_visible"])
    return out.set_index("date_utc")[cols]


def assert_no_lookahead(panel: pd.DataFrame, *, day_col: str = "date_utc") -> None:
    """For every row, every t_visible_<source> column must be <= the row's day."""
    days = panel[day_col] if day_col in panel.columns else panel.index.to_series()
    for c in panel.columns:
        if c.startswith("t_visible_"):
            bad = (panel[c].notna() & (panel[c] > days)).sum()
            if bad > 0:
                raise ValueError(f"feature panel {c}: {bad} rows leak future data")
