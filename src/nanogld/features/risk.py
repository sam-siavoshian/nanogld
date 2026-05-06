"""GLD bar-frequency risk / volatility features (doc 04 §2, 8 dims).

Input: data/raw/alpaca_bars_GLD_30min.parquet (Source 1) +
       data/raw/calendar_events_v1.parquet (FOMC dates).

Output (one row per 30min bar):
  realized_vol_8, realized_vol_48, realized_vol_240
                    Close-to-close stdev of log returns (8 / 48 / 240 bars).
  vol_ratio_8_48    realized_vol_8 / realized_vol_48 (regime indicator).
  vol_zscore_30d    Z of realized_vol_48 vs its 480-bar (~30d RTH) past.
  garman_klass_8    Garman-Klass realized vol on 8-bar window — uses full
                    OHLC, 7.4× more efficient than close-only stdev.
  days_since_FOMC   /100, capped — days since the most recent FOMC event
                    on or before bar T-1 (uses lagged timestamp).
  is_FOMC_week      1.0 if the bar's date sits within ±3 calendar days of
                    any FOMC date.

V4 leakage rules:
  - All bar features use shift(1) so bar T's value depends only on bars
    [..., T-1]. The lagged timestamp drives the calendar lookup so a bar
    starting at FOMC announcement minute does NOT see itself.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from nanogld.data.utils import get_logger, raw_dir
from nanogld.features.utils import FeatureSpec

LOG = get_logger("nanogld.features.risk")

FEATURES: tuple[FeatureSpec, ...] = (
    FeatureSpec("realized_vol_8", source="alpaca_bars"),
    FeatureSpec("realized_vol_48", source="alpaca_bars"),
    FeatureSpec("realized_vol_240", source="alpaca_bars"),
    FeatureSpec("vol_ratio_8_48", source="alpaca_bars"),
    FeatureSpec("vol_zscore_30d", source="alpaca_bars"),
    FeatureSpec("garman_klass_8", source="alpaca_bars"),
    FeatureSpec("days_since_FOMC", source="calendar"),
    FeatureSpec("is_FOMC_week", source="calendar"),
)


def _load_fomc_dates() -> pd.DatetimeIndex:
    """Return a sorted UTC index of FOMC announcement timestamps. Empty if absent."""
    path = Path(raw_dir()) / "calendar_events_v1.parquet"
    if not path.exists():
        LOG.warning("%s missing — FOMC features will be NaN/0", path)
        return pd.DatetimeIndex([], tz="UTC")
    cal = pd.read_parquet(path)
    fomc = cal[cal["event_type"] == "FOMC"]["event_ts_utc"]
    return pd.DatetimeIndex(pd.to_datetime(fomc, utc=True)).sort_values()


def _days_since_fomc(ts_lag: pd.Series, fomc_idx: pd.DatetimeIndex) -> pd.Series:
    """Days between bar's lagged timestamp and most recent prior FOMC.

    Uses int-ns searchsorted for O(N log M) with no tz pitfalls. Empty
    calendar => NaN; bars before the first FOMC also NaN.
    """
    if len(fomc_idx) == 0 or ts_lag.empty:
        return pd.Series(np.full(len(ts_lag), np.nan), index=ts_lag.index)
    ts = pd.to_datetime(ts_lag, utc=True, errors="coerce")
    ts_ns = ts.astype("int64").to_numpy()
    fomc_ns = fomc_idx.asi8
    # Most recent FOMC AT-OR-BEFORE bar's lagged ts. Use side='right' then -1.
    pos = np.searchsorted(fomc_ns, ts_ns, side="right") - 1
    valid = (pos >= 0) & ts.notna().to_numpy()
    out = np.full(len(ts), np.nan)
    if valid.any():
        idx_safe = np.where(valid, pos, 0)
        prior_ns = fomc_ns[idx_safe]
        deltas_sec = (ts_ns - prior_ns) / 1e9
        out = np.where(valid, deltas_sec / 86_400.0, np.nan)
    return pd.Series(out, index=ts_lag.index)


def _is_fomc_week(ts_lag: pd.Series, fomc_idx: pd.DatetimeIndex) -> pd.Series:
    """1.0 if bar's lagged timestamp lies within ±3 calendar days of any FOMC."""
    if len(fomc_idx) == 0 or ts_lag.empty:
        return pd.Series(np.zeros(len(ts_lag)), index=ts_lag.index)
    ts = pd.to_datetime(ts_lag, utc=True, errors="coerce")
    # Use int64 ns since both sides are UTC-aware; numpy timedelta math
    # requires tz-naive operands.
    ts_ns = ts.astype("int64").to_numpy()
    fomc_ns = fomc_idx.asi8
    window_ns = int(pd.Timedelta(days=3).value)

    pos = np.searchsorted(fomc_ns, ts_ns, side="left")
    pos_clip_left = np.clip(pos - 1, 0, len(fomc_ns) - 1)
    pos_clip_right = np.clip(pos, 0, len(fomc_ns) - 1)
    diff_left = np.abs(ts_ns - fomc_ns[pos_clip_left])
    diff_right = np.abs(fomc_ns[pos_clip_right] - ts_ns)
    abs_min = np.minimum(diff_left, diff_right)
    # NaN bars ts_ns == iNaT (most negative int64). Guard.
    valid = ts.notna().to_numpy()
    out = np.where(valid & (abs_min <= window_ns), 1.0, 0.0)
    return pd.Series(out, index=ts_lag.index)


def build_risk_features() -> pd.DataFrame:
    """Bar-frequency risk features. Empty frame if GLD bars missing."""
    path = Path(raw_dir()) / "alpaca_bars_GLD_30min.parquet"
    if not path.exists():
        LOG.warning("%s missing — skipping risk features", path)
        return pd.DataFrame()

    bars = pd.read_parquet(path).sort_values("timestamp").reset_index(drop=True)
    if bars.empty:
        return pd.DataFrame()

    out = pd.DataFrame(
        {
            "timestamp": bars["timestamp"],
            "t_visible": bars["t_visible"],
        }
    )

    # Log returns from lagged closes — bar T sees only bars ≤ T-1.
    close_lag = bars["close"].shift(1)
    log_returns = np.log(close_lag / close_lag.shift(1))

    out["realized_vol_8"] = log_returns.rolling(8, min_periods=4).std()
    out["realized_vol_48"] = log_returns.rolling(48, min_periods=24).std()
    out["realized_vol_240"] = log_returns.rolling(240, min_periods=120).std()
    out["vol_ratio_8_48"] = out["realized_vol_8"] / out["realized_vol_48"].replace(0, np.nan)

    rv48_mean = out["realized_vol_48"].rolling(480, min_periods=120).mean()
    rv48_std = out["realized_vol_48"].rolling(480, min_periods=120).std()
    out["vol_zscore_30d"] = (out["realized_vol_48"] - rv48_mean) / rv48_std.replace(0, np.nan)

    # Garman-Klass per-bar variance, rolled to 8-bar realized vol.
    high_lag = bars["high"].shift(1)
    low_lag = bars["low"].shift(1)
    open_lag = bars["open"].shift(1)
    log_hl = np.log(high_lag / low_lag)
    log_co = np.log(close_lag / open_lag)
    gk_per_bar = 0.5 * log_hl**2 - (2 * np.log(2) - 1) * log_co**2
    out["garman_klass_8"] = np.sqrt(gk_per_bar.rolling(8, min_periods=4).mean().clip(lower=0))

    # FOMC proximity — uses lagged bar timestamp so bar T can't see itself.
    fomc_idx = _load_fomc_dates()
    ts_lag = bars["timestamp"].shift(1)
    raw_days = _days_since_fomc(ts_lag, fomc_idx)
    # Cap at 100d so the /100 scaling stays bounded (further-out FOMC == 1.0).
    out["days_since_FOMC"] = raw_days.clip(upper=100.0) / 100.0
    out["is_FOMC_week"] = _is_fomc_week(ts_lag, fomc_idx)

    LOG.info("risk features built: %d rows × %d cols", len(out), out.shape[1])
    return out
