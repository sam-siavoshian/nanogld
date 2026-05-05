"""Oil-macro features from Brent (BZ=F) + WTI (CL=F) daily settle.

Inputs: data/raw/brent_daily.parquet, wti_daily.parquet (Source 5).
Output columns (one row per ticker per date):
  log_ret_1d / log_ret_5d / log_ret_20d
  realized_vol_20d (close-to-close)
  gk_vol_20d        (Garman-Klass, more efficient)
  level_close       (raw, for cross-feature spreads)

Plus a derived spread frame:
  brent_wti_spread        absolute USD
  brent_wti_log_spread    log(brent) - log(wti)

Every output row carries `t_visible` propagated from the underlying
settlement timestamp (CL ~14:30 ET, BZ ~15:00 ET; see yfinance_helpers).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from nanogld.data.utils import get_logger, raw_dir
from nanogld.features.utils import (
    FeatureSpec,
    garman_klass_vol,
    log_returns,
    realized_vol,
)

LOG = get_logger("nanogld.features.oil")

PER_TICKER_FEATURES: tuple[FeatureSpec, ...] = (
    FeatureSpec("level_close", source="yfinance", description="daily settle close"),
    FeatureSpec("log_ret_1d", source="yfinance"),
    FeatureSpec("log_ret_5d", source="yfinance"),
    FeatureSpec("log_ret_20d", source="yfinance"),
    FeatureSpec(
        "realized_vol_20d", source="yfinance", description="close-to-close stdev of log ret"
    ),
    FeatureSpec("gk_vol_20d", source="yfinance", description="Garman-Klass realized vol"),
)


def _load(name: str) -> pd.DataFrame:
    path = Path(raw_dir()) / f"{name}_daily.parquet"
    if not path.exists():
        LOG.warning("%s missing — skip oil features for that ticker", path)
        return pd.DataFrame()
    return pd.read_parquet(path).sort_values("date").reset_index(drop=True)


def _per_ticker(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    out = pd.DataFrame({"date": df["date"], "t_visible": df["t_visible"]})
    out[f"{prefix}_level_close"] = df["close"].astype("float64")
    out[f"{prefix}_log_ret_1d"] = log_returns(df["close"], 1)
    out[f"{prefix}_log_ret_5d"] = log_returns(df["close"], 5)
    out[f"{prefix}_log_ret_20d"] = log_returns(df["close"], 20)
    out[f"{prefix}_realized_vol_20d"] = realized_vol(df["close"], 20)
    out[f"{prefix}_gk_vol_20d"] = garman_klass_vol(
        df["high"], df["low"], df["open"], df["close"], 20
    )
    return out


def build_oil_features() -> pd.DataFrame:
    """Wide frame keyed by date with brent_* + wti_* + spread columns.

    Returns empty DataFrame if neither parquet exists yet.
    """
    brent = _load("brent")
    wti = _load("wti")
    if brent.empty and wti.empty:
        return pd.DataFrame()

    bf = _per_ticker(brent, "brent")
    wf = _per_ticker(wti, "wti")

    if bf.empty:
        out = wf
    elif wf.empty:
        out = bf
    else:
        # join on date — same NYSE/NYMEX-business calendar approx; settle times differ.
        # t_visible per side preserved as t_visible_brent / t_visible_wti so the panel
        # joiner keeps the most-recent of the two.
        bf = bf.rename(columns={"t_visible": "t_visible_brent"})
        wf = wf.rename(columns={"t_visible": "t_visible_wti"})
        out = pd.merge(bf, wf, on="date", how="outer").sort_values("date").reset_index(drop=True)
        # Combined visibility = max(brent, wti) — both sides must be settled.
        out["t_visible"] = out[["t_visible_brent", "t_visible_wti"]].max(axis=1)

        out["brent_wti_spread"] = out["brent_level_close"] - out["wti_level_close"]
        out["brent_wti_log_spread"] = np.log(out["brent_level_close"]) - np.log(
            out["wti_level_close"]
        )

    return out
