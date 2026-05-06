"""Short-window macro features (doc 04 §3, 6 dims).

Inputs (FRED ALFRED parquets in data/raw/):
  fred_dtwexbgs_all_releases.parquet   broad USD index
  fred_dgs10_all_releases.parquet      10y nominal yield
  fred_dgs2_all_releases.parquet       2y nominal yield
  fred_t10yie_all_releases.parquet     10y breakeven inflation
  fred_vixcls_all_releases.parquet     VIX close

Output (one row per ALFRED release that lands a fresh observation; the
panel join then forward-fills to daily/30min):
  dxy_log_return_5d   log(DTWEXBGS) - log(DTWEXBGS lagged 5 obs)
  dgs10_level         DGS10 / 10  (rough scaling to keep z-score sane)
  dgs2_level          DGS2 / 10
  term_spread_10y_2y  DGS10 - DGS2
  real_rate_10y       DGS10 - T10YIE  (10y real rate via breakeven proxy)
  vix_level           VIXCLS / 30

Vintage discipline: every series is keyed by its `t_visible` (release_ts +
ET tod from FRED_RELEASE_TOD_ET). The output's t_visible is the MAX across
the joined series for that observation date — i.e., the row is only fully
visible once every component has cleared its release time.

V4 leakage rule: real-rate uses T10YIE not T5YIE (T5YIE = 5y breakeven,
T5YIFR = 5y forward — both available, but matching DGS10 to T10YIE keeps
horizon consistent).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from nanogld.data.utils import get_logger, raw_dir
from nanogld.features.utils import FeatureSpec

LOG = get_logger("nanogld.features.macro")

FEATURES: tuple[FeatureSpec, ...] = (
    FeatureSpec("dxy_log_return_5d", source="fred_DTWEXBGS"),
    FeatureSpec("dgs10_level", source="fred_DGS10"),
    FeatureSpec("dgs2_level", source="fred_DGS2"),
    FeatureSpec("term_spread_10y_2y", source="fred"),
    FeatureSpec("real_rate_10y", source="fred"),
    FeatureSpec("vix_level", source="fred_VIXCLS"),
)


def _latest_per_date(series_id: str) -> pd.DataFrame:
    """Latest revision per observation `date` from a FRED ALFRED parquet.

    Returns a frame with columns [date, value, t_visible]. Empty if missing.
    """
    path = Path(raw_dir()) / f"fred_{series_id.lower()}_all_releases.parquet"
    if not path.exists():
        LOG.warning("%s missing — %s features will be skipped", path, series_id)
        return pd.DataFrame(columns=["date", "value", "t_visible"])
    df = pd.read_parquet(path)
    if df.empty:
        return pd.DataFrame(columns=["date", "value", "t_visible"])
    # Latest revision per date — take the row with the largest realtime_start
    # (= last revision before "now"). For PIT, downstream join uses t_visible.
    latest = (
        df.sort_values(["date", "realtime_start"]).groupby("date").tail(1).reset_index(drop=True)
    )
    return latest[["date", "value", "t_visible"]].sort_values("date").reset_index(drop=True)


SERIES = ("DTWEXBGS", "DGS10", "DGS2", "T10YIE", "VIXCLS")


def build_per_series() -> dict[str, pd.DataFrame]:
    """Per-series view used by build.py's per-series attach path.

    Returns one frame per series with its level / its 5d log return / its
    derived columns. Each frame carries its own t_visible so the panel
    attach happens per-column without cross-series alignment loss.
    """
    out: dict[str, pd.DataFrame] = {}
    for sid in SERIES:
        f = _latest_per_date(sid)
        if f.empty:
            continue
        out[sid] = f.rename(columns={"value": f"{sid.lower()}_raw"})
    return out


def build_features_per_series() -> dict[str, pd.DataFrame]:
    """Same as build_per_series() but each frame already has the spec's
    derived feature columns (level / 5d log return for DXY).

    Each frame columns: {date, t_visible, <feature cols owned by this series>}.
    """
    raw = build_per_series()
    derived: dict[str, pd.DataFrame] = {}

    if "DTWEXBGS" in raw:
        f = raw["DTWEXBGS"].copy()
        dxy = f["dtwexbgs_raw"].astype("float64")
        f["dxy_log_return_5d"] = np.log(dxy / dxy.shift(5))
        derived["DTWEXBGS"] = f[["date", "t_visible", "dxy_log_return_5d"]]
    if "DGS10" in raw:
        f = raw["DGS10"].copy()
        f["dgs10_level"] = f["dgs10_raw"].astype("float64") / 10.0
        derived["DGS10"] = f[["date", "t_visible", "dgs10_level"]]
    if "DGS2" in raw:
        f = raw["DGS2"].copy()
        f["dgs2_level"] = f["dgs2_raw"].astype("float64") / 10.0
        derived["DGS2"] = f[["date", "t_visible", "dgs2_level"]]
    if "VIXCLS" in raw:
        f = raw["VIXCLS"].copy()
        f["vix_level"] = f["vixcls_raw"].astype("float64") / 30.0
        derived["VIXCLS"] = f[["date", "t_visible", "vix_level"]]

    # Cross-series derivations: term_spread (DGS10 - DGS2) and real_rate
    # (DGS10 - T10YIE). These need both series, so we attach them on the
    # later release's t_visible. asof-merge backward.
    if "DGS10" in raw and "DGS2" in raw:
        a = raw["DGS10"][["date", "t_visible", "dgs10_raw"]].sort_values("t_visible")
        b = (
            raw["DGS2"][["date", "t_visible", "dgs2_raw"]]
            .sort_values("t_visible")
            .rename(columns={"t_visible": "_b_t_visible"})
        )
        merged = pd.merge_asof(
            a,
            b[["date", "_b_t_visible", "dgs2_raw"]],
            on="date",
            direction="backward",
            allow_exact_matches=True,
        )
        # PIT: pick the LATER of the two visibilities so we don't claim a value
        # before BOTH series have published.
        merged["t_visible"] = merged[["t_visible", "_b_t_visible"]].max(axis=1)
        merged["term_spread_10y_2y"] = merged["dgs10_raw"].astype("float64") - merged[
            "dgs2_raw"
        ].astype("float64")
        derived["term_spread_10y_2y"] = merged.dropna(subset=["term_spread_10y_2y", "t_visible"])[
            ["date", "t_visible", "term_spread_10y_2y"]
        ]
    if "DGS10" in raw and "T10YIE" in raw:
        a = raw["DGS10"][["date", "t_visible", "dgs10_raw"]].sort_values("t_visible")
        b = (
            raw["T10YIE"][["date", "t_visible", "t10yie_raw"]]
            .sort_values("t_visible")
            .rename(columns={"t_visible": "_b_t_visible"})
        )
        merged = pd.merge_asof(
            a,
            b[["date", "_b_t_visible", "t10yie_raw"]],
            on="date",
            direction="backward",
            allow_exact_matches=True,
        )
        merged["t_visible"] = merged[["t_visible", "_b_t_visible"]].max(axis=1)
        merged["real_rate_10y"] = merged["dgs10_raw"].astype("float64") - merged[
            "t10yie_raw"
        ].astype("float64")
        derived["real_rate_10y"] = merged.dropna(subset=["real_rate_10y", "t_visible"])[
            ["date", "t_visible", "real_rate_10y"]
        ]

    return derived


def build_macro_features() -> pd.DataFrame:
    """Long-form view (rows = union of per-series releases). Kept so
    `_attach_group` can still use macro as a single frame for diagnostics
    and tests; build_panel() prefers `build_features_per_series()` for the
    panel attach because cross-series cadences differ.
    """
    per_series = build_features_per_series()
    if not per_series:
        LOG.warning("no FRED parquets found — skipping macro features")
        return pd.DataFrame()
    long = (
        pd.concat(per_series.values(), ignore_index=True)
        .sort_values("t_visible")
        .reset_index(drop=True)
    )
    LOG.info("macro features built: %d rows × %d cols", len(long), long.shape[1])
    return long
