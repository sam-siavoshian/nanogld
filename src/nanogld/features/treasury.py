"""Treasury curve + TIPS + breakevens features (doc 04 §8, ~30 dims).

Inputs (FRED ALFRED parquets in data/raw/, all vintage-correct):
  Nominal:    DGS3MO, DGS6MO, DGS2, DGS5, DGS10, DGS30
  TIPS:       DFII5, DFII10
  Breakevens: T5YIE, T10YIE, T5YIFR

Output (one row per observation date, joined across all 11 series):
  11 levels:   {sid}_level                   raw value (no rescale)
  11 1d-changes: {sid}_change_1d             diff(1) on observation date
  4 spreads:   spread_10y_2y, spread_30y_10y, spread_5y_2y, spread_10y_3m
  1 butterfly: butterfly_2_5_10              2*DGS5 - DGS2 - DGS10
  2 real-rate: real_rate_10y_direct (DFII10)
               real_rate_10y_breakeven (DGS10 - T10YIE)

t_visible = max across the per-series releases for the observation date.
The panel forward-fills with strict-< asof per V1 hard rule §1.

Real rates are the #1 gold price driver; we keep both proxies for redundancy.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from nanogld.data.utils import get_logger, raw_dir
from nanogld.features.utils import FeatureSpec

LOG = get_logger("nanogld.features.treasury")

TREASURY_NOMINAL = ("DGS3MO", "DGS6MO", "DGS2", "DGS5", "DGS10", "DGS30")
TREASURY_TIPS = ("DFII5", "DFII10")
BREAKEVENS = ("T5YIE", "T10YIE", "T5YIFR")
ALL_SERIES = TREASURY_NOMINAL + TREASURY_TIPS + BREAKEVENS  # 11

LEVEL_COLS: tuple[str, ...] = tuple(f"{s.lower()}_level" for s in ALL_SERIES)
CHANGE_COLS: tuple[str, ...] = tuple(f"{s.lower()}_change_1d" for s in ALL_SERIES)
SPREAD_COLS: tuple[str, ...] = (
    "spread_10y_2y",
    "spread_30y_10y",
    "spread_5y_2y",
    "spread_10y_3m",
)
DERIVED_COLS: tuple[str, ...] = (
    "butterfly_2_5_10",
    "real_rate_10y_direct",
    "real_rate_10y_breakeven",
)

FEATURES: tuple[FeatureSpec, ...] = (
    *(FeatureSpec(c, source="fred") for c in LEVEL_COLS),
    *(FeatureSpec(c, source="fred") for c in CHANGE_COLS),
    *(FeatureSpec(c, source="fred") for c in SPREAD_COLS),
    *(FeatureSpec(c, source="fred") for c in DERIVED_COLS),
)


def _latest_per_date(series_id: str) -> pd.DataFrame:
    """Latest revision per observation date — one t_visible per row."""
    path = Path(raw_dir()) / f"fred_{series_id.lower()}_all_releases.parquet"
    if not path.exists():
        LOG.warning("%s missing — %s features will be NaN", path, series_id)
        return pd.DataFrame(columns=["date", "value", "t_visible"])
    df = pd.read_parquet(path)
    if df.empty:
        return pd.DataFrame(columns=["date", "value", "t_visible"])
    latest = (
        df.sort_values(["date", "realtime_start"]).groupby("date").tail(1).reset_index(drop=True)
    )
    return latest[["date", "value", "t_visible"]].sort_values("date").reset_index(drop=True)


def build_per_series() -> dict[str, pd.DataFrame]:
    """Per-series view used by build.py's per-series attach path.

    Returns {sid: frame}. Each frame columns:
      date, t_visible, {sid}_level, {sid}_change_1d
    Cross-series derived features (spreads, butterfly, real-rate) attach
    via separate frames keyed by the LATER of the two source visibilities.
    """
    out: dict[str, pd.DataFrame] = {}
    raws = {sid: _latest_per_date(sid) for sid in ALL_SERIES}
    for sid in ALL_SERIES:
        f = raws[sid]
        if f.empty:
            continue
        level = f["value"].astype("float64").reset_index(drop=True)
        out[sid] = pd.DataFrame(
            {
                "date": f["date"].reset_index(drop=True),
                "t_visible": f["t_visible"].reset_index(drop=True),
                f"{sid.lower()}_level": level,
                f"{sid.lower()}_change_1d": level.diff(1),
            }
        )
    return out


def build_derived_features(per_series: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Cross-series derived features (spreads, butterfly, real rate)."""
    out: dict[str, pd.DataFrame] = {}

    def _pair(a: str, b: str, name: str, formula) -> pd.DataFrame | None:
        if a not in per_series or b not in per_series:
            return None
        fa = per_series[a].rename(columns={"t_visible": "_a_t"})
        fb = per_series[b].rename(columns={"t_visible": "_b_t"})
        ka = f"{a.lower()}_level"
        kb = f"{b.lower()}_level"
        fa = fa[["date", "_a_t", ka]].sort_values("date")
        fb = fb[["date", "_b_t", kb]].sort_values("date")
        merged = fa.merge(fb, on="date", how="inner")
        if merged.empty:
            return None
        merged["t_visible"] = merged[["_a_t", "_b_t"]].max(axis=1)
        merged[name] = formula(merged[ka], merged[kb])
        return merged.dropna(subset=[name, "t_visible"])[["date", "t_visible", name]]

    # spreads
    s = _pair("DGS10", "DGS2", "spread_10y_2y", lambda x, y: x - y)
    if s is not None:
        out["spread_10y_2y"] = s
    s = _pair("DGS30", "DGS10", "spread_30y_10y", lambda x, y: x - y)
    if s is not None:
        out["spread_30y_10y"] = s
    s = _pair("DGS5", "DGS2", "spread_5y_2y", lambda x, y: x - y)
    if s is not None:
        out["spread_5y_2y"] = s
    s = _pair("DGS10", "DGS3MO", "spread_10y_3m", lambda x, y: x - y)
    if s is not None:
        out["spread_10y_3m"] = s

    # butterfly_2_5_10 = 2*DGS5 - DGS2 - DGS10 — needs all three.
    if all(s in per_series for s in ("DGS2", "DGS5", "DGS10")):
        f2 = per_series["DGS2"].rename(columns={"t_visible": "_t2"})
        f5 = per_series["DGS5"].rename(columns={"t_visible": "_t5"})
        f10 = per_series["DGS10"].rename(columns={"t_visible": "_t10"})
        m = (
            f2[["date", "_t2", "dgs2_level"]]
            .merge(f5[["date", "_t5", "dgs5_level"]], on="date", how="inner")
            .merge(f10[["date", "_t10", "dgs10_level"]], on="date", how="inner")
        )
        if not m.empty:
            m["t_visible"] = m[["_t2", "_t5", "_t10"]].max(axis=1)
            m["butterfly_2_5_10"] = 2 * m["dgs5_level"] - m["dgs2_level"] - m["dgs10_level"]
            out["butterfly_2_5_10"] = m.dropna(subset=["butterfly_2_5_10", "t_visible"])[
                ["date", "t_visible", "butterfly_2_5_10"]
            ]

    # Real rates
    if "DFII10" in per_series:
        f = per_series["DFII10"][["date", "t_visible", "dfii10_level"]].copy()
        f["real_rate_10y_direct"] = f["dfii10_level"]
        out["real_rate_10y_direct"] = f.dropna(subset=["real_rate_10y_direct", "t_visible"])[
            ["date", "t_visible", "real_rate_10y_direct"]
        ]
    if "DGS10" in per_series and "T10YIE" in per_series:
        rr = _pair(
            "DGS10",
            "T10YIE",
            "real_rate_10y_breakeven",
            lambda a, b: a - b,
        )
        if rr is not None:
            out["real_rate_10y_breakeven"] = rr

    return out


def build_treasury_features() -> pd.DataFrame:
    """Long-form view (rows = union of per-series releases + derived rows).
    Kept for diagnostics / tests. The panel attach path prefers the per-series
    helpers above to avoid cross-series alignment loss.
    """
    per_series = build_per_series()
    if not per_series:
        LOG.warning("no treasury FRED parquets found — skipping")
        return pd.DataFrame()
    parts: list[pd.DataFrame] = list(per_series.values())
    derived = build_derived_features(per_series)
    parts.extend(derived.values())
    long = pd.concat(parts, ignore_index=True).sort_values("t_visible").reset_index(drop=True)
    LOG.info("treasury features built: %d rows × %d cols", len(long), long.shape[1])
    return long
