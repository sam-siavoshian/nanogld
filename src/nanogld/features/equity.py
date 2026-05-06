"""Per-ETF basket features + cross-asset ratios (doc 04 §6 + §7).

Inputs (data/raw/):
  alpaca_bars_<SYM>_30min.parquet for SYM in {SPY, QQQ, IWM, GDX, SLV,
                                              XLF, XLE, XLK, XLU}
  alpaca_bars_GLD_30min.parquet (the primary)

Output (one row per GLD bar — bars not in GLD are dropped):
  per ETF (8 cols × 9 ETFs = 72 cols):
    {sym}_log_ret_1, {sym}_log_ret_4, {sym}_log_ret_16, {sym}_log_ret_48
    {sym}_realized_vol_8, {sym}_realized_vol_48
    {sym}_rs_spy_24      24-bar log-ret spread vs SPY (SPY itself = 0)
    {sym}_corr_gld_30d   1440-bar (≈30d RTH) correlation w/ GLD log-ret
  cross-asset ratios (7 cols):
    gold_silver_ratio              GLD_close / SLV_close (lagged)
    gdx_gld_ratio                  GDX_close / GLD_close (lagged)
    gold_silver_log_ret_5d         5-day diff of log gold-silver ratio
    gdx_gld_log_ret_5d             5-day diff of log GDX/GLD ratio
    spy_gld_corr_30d, qqq_gld_corr_30d, iwm_gld_corr_30d
                                   30-day rolling corr w/ GLD log-ret

Total: 79 cols + (timestamp, t_visible) = 81.

Hard rules:
  - Each ETF parquet ALREADY carries t_visible = bar_close = timestamp+30min.
  - All features use df.shift(1) so bar T sees only data through T-1.
  - As-of merge on timestamp aligns SPY/etc bars to GLD bars; an ETF that
    doesn't have a bar at GLD's exact timestamp gets the most recent prior
    bar (typical when the foreign symbol is halted but GLD trades, rare).
  - 1 day RTH = 13 bars; 30 days ≈ 390 bars; spec asks for 1440 (≈30d
    24/7). We use 390 to match RTH-only data; corr_gld_30d window is set
    here to `WINDOW_30D = 390` (RTH-realistic) — change in one place.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from nanogld.data.utils import get_logger, raw_dir
from nanogld.features.utils import FeatureSpec

LOG = get_logger("nanogld.features.equity")

ETF_BASKET = ("SPY", "QQQ", "IWM", "GDX", "SLV", "XLF", "XLE", "XLK", "XLU")

# 1 RTH day of 30min bars = 13. 30 trading days = 390. The doc 04 spec quotes
# 1440 (≈ 30 calendar days × 48 30min bars) for 24/7 series; for RTH-only
# Alpaca bars we use 390 so we don't roll off 3 months of history.
WINDOW_30D = 390

PER_ETF_FEATURES: tuple[str, ...] = (
    "log_ret_1",
    "log_ret_4",
    "log_ret_16",
    "log_ret_48",
    "realized_vol_8",
    "realized_vol_48",
    "rs_spy_24",
    "corr_gld_30d",
)

RATIO_FEATURES: tuple[str, ...] = (
    "gold_silver_ratio",
    "gdx_gld_ratio",
    "gold_silver_log_ret_5d",
    "gdx_gld_log_ret_5d",
    "spy_gld_corr_30d",
    "qqq_gld_corr_30d",
    "iwm_gld_corr_30d",
)

FEATURES: tuple[FeatureSpec, ...] = (
    *(
        FeatureSpec(f"{sym.lower()}_{feat}", source=f"alpaca_bars_{sym}")
        for sym in ETF_BASKET
        for feat in PER_ETF_FEATURES
    ),
    *(FeatureSpec(name, source="equity_ratios") for name in RATIO_FEATURES),
)


def _load_bars(symbol: str) -> pd.DataFrame:
    path = Path(raw_dir()) / f"alpaca_bars_{symbol}_30min.parquet"
    if not path.exists():
        LOG.warning("%s missing — %s features will be NaN", path, symbol)
        return pd.DataFrame()
    df = pd.read_parquet(path).sort_values("timestamp").reset_index(drop=True)
    return df


def _per_etf(sym: str, df: pd.DataFrame, gld_log_ret: pd.Series) -> pd.DataFrame:
    """Compute the 8 per-ETF features. Index aligns with the input frame."""
    if df.empty:
        return pd.DataFrame()
    out = pd.DataFrame({"timestamp": df["timestamp"]})
    close_lag = df["close"].shift(1)
    log_close_lag = np.log(close_lag)
    log_ret_1 = log_close_lag.diff(1)
    out[f"{sym.lower()}_log_ret_1"] = log_ret_1
    out[f"{sym.lower()}_log_ret_4"] = log_close_lag.diff(4)
    out[f"{sym.lower()}_log_ret_16"] = log_close_lag.diff(16)
    out[f"{sym.lower()}_log_ret_48"] = log_close_lag.diff(48)
    out[f"{sym.lower()}_realized_vol_8"] = log_ret_1.rolling(8, min_periods=4).std()
    out[f"{sym.lower()}_realized_vol_48"] = log_ret_1.rolling(48, min_periods=24).std()
    return out


def build_equity_features() -> pd.DataFrame:
    """One row per GLD bar with 9-ETF features and cross-asset ratios."""
    gld = _load_bars("GLD")
    if gld.empty:
        LOG.warning("GLD bars missing — skipping equity features")
        return pd.DataFrame()

    # Anchor to GLD timestamps. Other ETFs are merged via merge_asof backward
    # on timestamp so we always have the most recent prior bar of each.
    base = pd.DataFrame({"timestamp": gld["timestamp"], "t_visible": gld["t_visible"]})
    base["timestamp"] = pd.to_datetime(base["timestamp"], utc=True).astype("datetime64[ns, UTC]")

    # GLD log returns for cross-correlations (lagged so T uses [..., T-1]).
    gld_close_lag = gld["close"].shift(1)
    gld_log_ret = np.log(gld_close_lag).diff(1)

    spy_log_ret_for_rs: pd.Series | None = None
    per_etf_frames: dict[str, pd.DataFrame] = {}
    closes_lagged: dict[str, pd.Series] = {"GLD": gld_close_lag}

    for sym in ETF_BASKET:
        df = _load_bars(sym)
        if df.empty:
            continue
        # Compute features in the ETF's own frame, then merge_asof onto GLD ts.
        sub = _per_etf(sym, df, gld_log_ret)
        sub["timestamp"] = pd.to_datetime(sub["timestamp"], utc=True).astype("datetime64[ns, UTC]")
        # Carry the lagged log-ret separately so we can do RS-vs-SPY post-merge
        # on the unified GLD timeline.
        sub[f"{sym.lower()}_log_ret_1_for_rs"] = sub[f"{sym.lower()}_log_ret_1"]
        per_etf_frames[sym] = sub
        # Stash close-lagged for ratios.
        closes_lagged[sym] = df["close"].shift(1).set_axis(df["timestamp"])

        if sym == "SPY":
            spy_log_ret_for_rs = sub.set_index("timestamp")["spy_log_ret_1_for_rs"]

    # Merge each ETF frame as-of onto the GLD timeline.
    merged = base.copy()
    for sym in ETF_BASKET:
        if sym not in per_etf_frames:
            for col in PER_ETF_FEATURES:
                merged[f"{sym.lower()}_{col}"] = np.nan
            continue
        sub = per_etf_frames[sym].sort_values("timestamp")
        merged = pd.merge_asof(
            merged.sort_values("timestamp"),
            sub,
            on="timestamp",
            direction="backward",
            allow_exact_matches=True,  # same-bar timestamp = same instant, OK
        )

    # RS vs SPY — bar's lagged log-ret minus SPY's lagged log-ret. SPY = 0 self.
    if spy_log_ret_for_rs is not None:
        # Re-index spy log-ret onto the merged timeline (timestamp index).
        merged_idxed = merged.set_index("timestamp")
        spy_aligned = spy_log_ret_for_rs.reindex(merged_idxed.index, method="ffill")
        for sym in ETF_BASKET:
            col_self = f"{sym.lower()}_log_ret_1_for_rs"
            if col_self not in merged_idxed.columns:
                merged[f"{sym.lower()}_rs_spy_24"] = np.nan
                continue
            sym_ret = merged_idxed[col_self]
            if sym == "SPY":
                rs_24 = pd.Series(np.zeros(len(merged_idxed)), index=merged_idxed.index)
            else:
                # 24-bar rolling sum of log-ret diff (≈ ~2 trading days RTH)
                rs_24 = (sym_ret - spy_aligned).rolling(24, min_periods=12).sum()
            merged[f"{sym.lower()}_rs_spy_24"] = rs_24.reindex(merged["timestamp"]).values
    else:
        for sym in ETF_BASKET:
            merged[f"{sym.lower()}_rs_spy_24"] = np.nan

    # 30-day correlation with GLD log-ret. Use the merged log_ret_1 columns
    # (now aligned to GLD timestamps).
    for sym in ETF_BASKET:
        col_ret = f"{sym.lower()}_log_ret_1"
        if col_ret in merged.columns:
            merged[f"{sym.lower()}_corr_gld_30d"] = (
                merged[col_ret]
                .rolling(WINDOW_30D, min_periods=WINDOW_30D // 4)
                .corr(gld_log_ret.set_axis(merged.index))
            )
        else:
            merged[f"{sym.lower()}_corr_gld_30d"] = np.nan

    # Drop helper RS columns
    drop_cols = [c for c in merged.columns if c.endswith("_for_rs")]
    merged = merged.drop(columns=drop_cols)

    # Ratios — recomputed against GLD timeline. SLV/GDX merged_asof above gave
    # us their lagged closes via the merge; rebuild from `_log_ret_1` is wrong
    # (it's a return, not a level). Use original closes via merge.
    gld_close_series = gld[["timestamp", "close"]].rename(columns={"close": "_gld_close"})
    gld_close_series["timestamp"] = pd.to_datetime(gld_close_series["timestamp"], utc=True).astype(
        "datetime64[ns, UTC]"
    )

    base_for_ratio = merged[["timestamp"]].copy()
    base_for_ratio = base_for_ratio.merge(gld_close_series, on="timestamp", how="left")
    for sym in ("SLV", "GDX", "SPY", "QQQ", "IWM"):
        df = _load_bars(sym)
        if df.empty:
            base_for_ratio[f"_{sym.lower()}_close"] = np.nan
            continue
        sub = df[["timestamp", "close"]].rename(columns={"close": f"_{sym.lower()}_close"})
        sub["timestamp"] = pd.to_datetime(sub["timestamp"], utc=True).astype("datetime64[ns, UTC]")
        base_for_ratio = pd.merge_asof(
            base_for_ratio.sort_values("timestamp"),
            sub.sort_values("timestamp"),
            on="timestamp",
            direction="backward",
            allow_exact_matches=True,
        )

    # Lagged closes
    gld_close_lag2 = base_for_ratio["_gld_close"].shift(1)

    if "_slv_close" in base_for_ratio.columns:
        slv_lag = base_for_ratio["_slv_close"].shift(1)
        gsr = gld_close_lag2 / slv_lag.replace(0, np.nan)
        merged["gold_silver_ratio"] = gsr
        merged["gold_silver_log_ret_5d"] = np.log(gsr).diff(5 * 13)  # 5d RTH = 65 bars
    else:
        merged["gold_silver_ratio"] = np.nan
        merged["gold_silver_log_ret_5d"] = np.nan

    if "_gdx_close" in base_for_ratio.columns:
        gdx_lag = base_for_ratio["_gdx_close"].shift(1)
        ggr = gdx_lag / gld_close_lag2.replace(0, np.nan)
        merged["gdx_gld_ratio"] = ggr
        merged["gdx_gld_log_ret_5d"] = np.log(ggr).diff(5 * 13)
    else:
        merged["gdx_gld_ratio"] = np.nan
        merged["gdx_gld_log_ret_5d"] = np.nan

    # Stocks-vs-gold cross-correlations (separate from per-ETF since we want
    # the explicit named outputs the spec lists).
    merged["spy_gld_corr_30d"] = merged.get(
        "spy_corr_gld_30d", pd.Series(np.nan, index=merged.index)
    )
    merged["qqq_gld_corr_30d"] = merged.get(
        "qqq_corr_gld_30d", pd.Series(np.nan, index=merged.index)
    )
    merged["iwm_gld_corr_30d"] = merged.get(
        "iwm_corr_gld_30d", pd.Series(np.nan, index=merged.index)
    )

    # Reorder
    feat_cols = [f.name for f in FEATURES]
    keep = ["timestamp", "t_visible", *feat_cols]
    out = merged.loc[:, [c for c in keep if c in merged.columns]].copy()
    LOG.info("equity features built: %d rows × %d cols", len(out), out.shape[1])
    return out
