"""GLD 30min bar-frequency price features (doc 04 §1, 12 dims).

Input: data/raw/alpaca_bars_GLD_30min.parquet (Source 1, V1 — 10y SIP).
Each input row = one 30min RTH bar (bar_start UTC). The bar's t_visible
= timestamp + 30min (i.e. bar end / release_ts) per V1 hard rule §1.

Output (one row per 30min bar):
  log_return_1, log_return_4, log_return_16, log_return_48
  rsi_14         pandas-ta-classic RSI on close-lagged
  macd_signal    MACDs_12_26_9 (signal line)
  bbands_pct     %B from bbands(20, 2.0)
  high_low_range 8-bar mean of (high - low) — micro-vol proxy
  volume_zscore  20-bar z-score of volume
  close_open_ratio (close/open) - 1 (lagged 1 bar)
  session_phase  4-cat code: 0=open(09:30-11:00ET), 1=mid(11:00-14:00ET),
                 2=afternoon(14:00-15:30ET), 3=close(15:30-16:00ET).
                 Bar timestamps that fall outside RTH (rare extended-hours
                 bars in the parquet) get -1.

V4 leakage rules (doc 04 §V4):
  - Every feature uses df.close.shift(1) etc. so that bar T's feature only
    depends on bars [..., T-1]. The bar's t_visible (= bar end) gates the
    join: only bars with t_visible < downstream timestamp are usable.
  - pandas-ta-classic indicators (rsi/macd/bbands) operate on the lagged
    series; KAMA/Ichimoku-visual/KST/DPO/TRIX/Vortex are FORBIDDEN.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pandas_ta_classic as pta

from nanogld.data.utils import ET, get_logger, raw_dir
from nanogld.features.utils import FeatureSpec

LOG = get_logger("nanogld.features.price")

FEATURES: tuple[FeatureSpec, ...] = (
    FeatureSpec("log_return_1", source="alpaca_bars"),
    FeatureSpec("log_return_4", source="alpaca_bars"),
    FeatureSpec("log_return_16", source="alpaca_bars"),
    FeatureSpec("log_return_48", source="alpaca_bars"),
    FeatureSpec("rsi_14", source="alpaca_bars", description="RSI from pandas-ta-classic"),
    FeatureSpec("macd_signal", source="alpaca_bars", description="MACDs_12_26_9 line"),
    FeatureSpec("bbands_pct", source="alpaca_bars", description="%B from BBANDS(20, 2.0)"),
    FeatureSpec("high_low_range", source="alpaca_bars", description="8-bar mean of high - low"),
    FeatureSpec("volume_zscore", source="alpaca_bars", description="20-bar z of volume"),
    FeatureSpec("close_open_ratio", source="alpaca_bars"),
    FeatureSpec("session_phase", dtype="int64", source="alpaca_bars"),
)


def _session_phase(bar_start_utc: pd.Series) -> pd.Series:
    """4-bucket RTH session phase. -1 if bar falls outside RTH.

    NYSE RTH = 09:30-16:00 ET. 30min bars labelled by start:
      0  open      09:30-11:00 ET  (bars starting 09:30, 10:00, 10:30)
      1  mid       11:00-14:00 ET  (bars starting 11:00..13:30)
      2  afternoon 14:00-15:30 ET  (bars starting 14:00, 14:30, 15:00)
      3  close     15:30-16:00 ET  (bar starting 15:30)
    """
    et = pd.to_datetime(bar_start_utc, utc=True).dt.tz_convert(ET)
    minute_of_day = et.dt.hour * 60 + et.dt.minute
    rth_start = 9 * 60 + 30  # 09:30 ET
    rth_end = 16 * 60  # 16:00 ET
    phase = pd.Series(np.full(len(et), -1, dtype="int64"), index=bar_start_utc.index)
    in_rth = (minute_of_day >= rth_start) & (minute_of_day < rth_end)
    phase.loc[in_rth & (minute_of_day < 11 * 60)] = 0
    phase.loc[in_rth & (minute_of_day >= 11 * 60) & (minute_of_day < 14 * 60)] = 1
    phase.loc[in_rth & (minute_of_day >= 14 * 60) & (minute_of_day < 15 * 60 + 30)] = 2
    phase.loc[in_rth & (minute_of_day >= 15 * 60 + 30)] = 3
    return phase


def build_price_features() -> pd.DataFrame:
    """Bar-frequency price features for GLD. One row per 30min bar.

    Returns empty DataFrame if the GLD bars parquet is missing.
    """
    path = Path(raw_dir()) / "alpaca_bars_GLD_30min.parquet"
    if not path.exists():
        LOG.warning("%s missing — skipping price features", path)
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

    # All features use bar-T-1 data only (no current-bar leakage).
    close_lag = bars["close"].shift(1)
    open_lag = bars["open"].shift(1)
    high_lag = bars["high"].shift(1)
    low_lag = bars["low"].shift(1)
    vol_lag = bars["volume"].shift(1)

    log_close_lag = np.log(close_lag)
    out["log_return_1"] = log_close_lag.diff(1)
    out["log_return_4"] = log_close_lag.diff(4)
    out["log_return_16"] = log_close_lag.diff(16)
    out["log_return_48"] = log_close_lag.diff(48)

    # pandas-ta-classic indicators return all-NaN if the input has a leading
    # NaN (it short-circuits to nan_count == n). Compute on the NaN-stripped
    # close_lag and reindex back to the original axis.
    close_lag_clean = close_lag.dropna()
    rsi_clean = pta.rsi(close_lag_clean, length=14)
    out["rsi_14"] = (
        rsi_clean.reindex(close_lag.index)
        if rsi_clean is not None
        else pd.Series(np.nan, index=close_lag.index)
    )
    macd_df = pta.macd(close_lag_clean, fast=12, slow=26, signal=9)
    if macd_df is not None and not macd_df.empty:
        out["macd_signal"] = macd_df["MACDs_12_26_9"].reindex(close_lag.index)
    else:
        out["macd_signal"] = np.nan
    bb_df = pta.bbands(close_lag_clean, length=20, std=2.0)
    if bb_df is not None and not bb_df.empty:
        out["bbands_pct"] = bb_df["BBP_20_2.0"].reindex(close_lag.index)
    else:
        out["bbands_pct"] = np.nan

    out["high_low_range"] = (high_lag - low_lag).rolling(8, min_periods=4).mean()
    vol_mean = vol_lag.rolling(20, min_periods=10).mean()
    vol_std = vol_lag.rolling(20, min_periods=10).std()
    out["volume_zscore"] = (vol_lag - vol_mean) / vol_std.replace(0, np.nan)
    out["close_open_ratio"] = (close_lag / open_lag) - 1

    out["session_phase"] = _session_phase(bars["timestamp"])

    LOG.info("price features built: %d rows × %d cols", len(out), out.shape[1])
    return out
