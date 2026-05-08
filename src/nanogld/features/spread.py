"""V1 GLD bid-ask spread feature in basis points.

Resamples the 5-min mid-spread parquet (built by data/alpaca_quotes.py)
to bar-close timestamps, computes the 5-min trailing average, shifts by
one 5-min step to make it strictly point-in-time at bar close.

Pre-2018 fallback: when a quote parquet is missing for a date, derive a
proxy spread from `(high - low) / mid * 10_000` of the bar itself, scaled
down by 5x so the proxy is a conservative under-estimate of true spread.

Spec: plan/04-FEATURE-ENGINEERING.md V1 spread section.
Spec: plan/V1-SPEC.md §4.6.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from nanogld.data.utils import get_logger, raw_dir

LOG = get_logger("nanogld.features.spread")

DEFAULT_QUOTES_FILENAME = "alpaca_quotes_GLD_5min.parquet"
DEFAULT_TRAILING_MIN = 5
PROXY_SCALE = 0.20  # high-low/mid is ~5x true spread; scale down


def load_quotes_parquet(path: Path | None = None) -> pd.DataFrame | None:
    """Load 5-min mid-spread parquet if it exists, else None.

    Expected schema: {bar_close_utc: timestamp, gld_spread_bps: float32}.
    """
    if path is None:
        path = raw_dir() / DEFAULT_QUOTES_FILENAME
    if not Path(path).exists():
        LOG.warning("%s missing — spread fallback to high-low proxy", path)
        return None
    df = pd.read_parquet(path)
    df["bar_close_utc"] = pd.to_datetime(df["bar_close_utc"], utc=True)
    return df.sort_values("bar_close_utc").reset_index(drop=True)


def add_spread_feature(
    df: pd.DataFrame,
    *,
    quotes_df: pd.DataFrame | None = None,
    bar_close_col: str = "bar_close_utc",
    high_col: str = "gld_high",
    low_col: str = "gld_low",
    close_col: str = "gld_close",
    trailing_min: int = DEFAULT_TRAILING_MIN,
) -> pd.DataFrame:
    """Append `gld_spread_bps_t` column to a bar-aligned DataFrame.

    Args:
        df: bar-aligned DataFrame, sorted by bar_close_utc ascending.
        quotes_df: optional pre-loaded 5-min mid-spread DataFrame. If None,
            attempt to load default parquet; fall back to high-low proxy.
        bar_close_col: column with UTC bar-close timestamps.
        high_col, low_col, close_col: OHLC column names for proxy fallback.
        trailing_min: trailing average window in minutes.
    """
    out = df.copy()
    if bar_close_col not in out.columns:
        raise KeyError(f"{bar_close_col} required for spread feature")

    out[bar_close_col] = pd.to_datetime(out[bar_close_col], utc=True)
    if not out[bar_close_col].is_monotonic_increasing:
        out = out.sort_values(bar_close_col).reset_index(drop=True)

    if quotes_df is None:
        quotes_df = load_quotes_parquet()

    proxy_used = quotes_df is None
    if not proxy_used:
        win_df = (
            quotes_df.set_index("bar_close_utc")["gld_spread_bps"]
            .rolling(f"{trailing_min}min", min_periods=1)
            .mean()
            .rename("gld_spread_bps_t")
            .reset_index()
        )
        merged = pd.merge_asof(
            out[[bar_close_col]].rename(columns={bar_close_col: "bar_close_utc"}),
            win_df,
            on="bar_close_utc",
            direction="backward",
            allow_exact_matches=False,
        )
        out["gld_spread_bps_t"] = merged["gld_spread_bps_t"].astype("float32").to_numpy()

    if proxy_used or out["gld_spread_bps_t"].isna().any():
        proxy = ((out[high_col] - out[low_col]) / out[close_col]) * 10_000.0 * PROXY_SCALE
        proxy = proxy.astype("float32")
        if proxy_used:
            out["gld_spread_bps_t"] = proxy
        else:
            out["gld_spread_bps_t"] = out["gld_spread_bps_t"].where(
                out["gld_spread_bps_t"].notna(), proxy
            )

    out["gld_spread_bps_t"] = out["gld_spread_bps_t"].clip(lower=0.0, upper=200.0).astype("float32")
    n_proxy = int(out["gld_spread_bps_t"].isna().sum())
    LOG.info(
        "spread feature: proxy_used=%s, %d NaN remain (clipped to [0, 200] bps)",
        proxy_used,
        n_proxy,
    )
    return out
