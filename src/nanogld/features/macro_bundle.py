"""Full FRED macro bundle (doc 04 §9, ~60 dims).

Inputs (FRED ALFRED parquets in data/raw/):
  Labor:      UNRATE, PAYEMS, ICSA, CCSA, JTSJOL
  Inflation:  CPIAUCSL, CPILFESL, PCEPI, PCEPILFE
  Growth:     GDPC1, INDPRO, RSAFS, HOUST, UMCSENT
  Money/Fed:  M2SL, WALCL, RRPONTSYD, DFF, FEDFUNDS

19 series. Per series produced:
  {series}_level
  {series}_yoy_change       (level / level.shift(periods)) - 1
  {series}_mom_change       (level / level.shift(1)) - 1
                              "1 obs" — for monthly series this is MoM,
                              for daily/weekly it's previous-period change.

Plus 3 derived signals:
  real_fedfunds              DFF - CPILFESL_yoy_change*100  (Bernanke shadow rate)
  m2_yoy                     M2SL_yoy_change
  icsa_4w_ma                 4-period MA of ICSA (4 weekly observations)

Skip: SOFR per spec (DFF replaces FEDFUNDS for daily; FEDFUNDS kept for
monthly aggregates).

Vintage discipline: each series uses ALFRED `realtime_start`-based latest
revision per observation date. The output here is a COMBINED frame with
ALL columns; rows are unioned per-release across series. Critically each
column is independently forward-fillable along its t_visible — i.e., when
the daily joiner sees a panel-day, it can find each column's latest visible
value via per-column merge_asof, NOT via single-row alignment.

build.py implements this per-column merge inline since the global pattern
of one t_visible per group breaks for unsynchronized series.

V4 hard rule §9: CPI/PCE annual seasonal revisions silently rewrite 5y of
history every Feb (CPI) / Aug (PCE). The per-row latest-revision pull
captures the LATEST revision; live PIT lookups must filter realtime_start
<= now via vintage_lookup() — this builder produces the daily panel that
the joiner then strict-< asof's via t_visible.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from nanogld.data.utils import get_logger, raw_dir
from nanogld.features.utils import FeatureSpec

LOG = get_logger("nanogld.features.macro_bundle")

MACRO_LABOR = ("UNRATE", "PAYEMS", "ICSA", "CCSA", "JTSJOL")
MACRO_INFLATION = ("CPIAUCSL", "CPILFESL", "PCEPI", "PCEPILFE")
MACRO_GROWTH = ("GDPC1", "INDPRO", "RSAFS", "HOUST", "UMCSENT")
MACRO_FED = ("M2SL", "WALCL", "RRPONTSYD", "DFF", "FEDFUNDS")
ALL_SERIES = MACRO_LABOR + MACRO_INFLATION + MACRO_GROWTH + MACRO_FED  # 19

# Periods for YoY: monthly=12, weekly≈52, quarterly=4, daily≈260. Per-series.
YOY_PERIODS: dict[str, int] = {
    "UNRATE": 12,
    "PAYEMS": 12,
    "ICSA": 52,
    "CCSA": 52,
    "JTSJOL": 12,
    "CPIAUCSL": 12,
    "CPILFESL": 12,
    "PCEPI": 12,
    "PCEPILFE": 12,
    "GDPC1": 4,
    "INDPRO": 12,
    "RSAFS": 12,
    "HOUST": 12,
    "UMCSENT": 12,
    "M2SL": 12,
    "WALCL": 52,
    "RRPONTSYD": 260,
    "DFF": 260,
    "FEDFUNDS": 12,
}


def _level_yoy_mom_cols(sid: str) -> tuple[str, str, str]:
    base = sid.lower()
    return f"{base}_level", f"{base}_yoy_change", f"{base}_mom_change"


FEATURES: tuple[FeatureSpec, ...] = (
    *(FeatureSpec(c, source=f"fred_{sid}") for sid in ALL_SERIES for c in _level_yoy_mom_cols(sid)),
    FeatureSpec("real_fedfunds", source="fred"),
    FeatureSpec("m2_yoy", source="fred"),
    FeatureSpec("icsa_4w_ma", source="fred"),
)


def build_per_series() -> dict[str, pd.DataFrame]:
    """Per-series frames keyed by series_id. Each frame has columns:
    {date, t_visible, {sid}_level, {sid}_yoy_change, {sid}_mom_change}.

    Use this when you need per-series PIT-correct forward-fill (e.g., the
    daily panel builder). Series cadences differ — UNRATE monthly, ICSA
    weekly, DFF daily — so combining them at the row level loses values.
    """
    out: dict[str, pd.DataFrame] = {}
    for sid in ALL_SERIES:
        path = Path(raw_dir()) / f"fred_{sid.lower()}_all_releases.parquet"
        if not path.exists():
            LOG.warning("%s missing — %s features will be NaN", path, sid)
            continue
        df = pd.read_parquet(path)
        if df.empty:
            continue
        # PIT-correct: take FIRST realtime_start per observation date (initial
        # release). Taking .tail(1) returns the LATEST revision which has a
        # much later realtime_start → introduces look-ahead bias and drops
        # everything before the most recent revision date.
        latest = (
            df.sort_values(["date", "realtime_start"])
            .groupby("date")
            .head(1)
            .reset_index(drop=True)
            .sort_values("date")
            .reset_index(drop=True)
        )
        frame = pd.DataFrame(
            {
                "date": latest["date"],
                "t_visible": latest["t_visible"],
            }
        )
        level = latest["value"].astype("float64")
        frame[f"{sid.lower()}_level"] = level
        frame[f"{sid.lower()}_yoy_change"] = (level / level.shift(YOY_PERIODS[sid])) - 1.0
        frame[f"{sid.lower()}_mom_change"] = (level / level.shift(1)) - 1.0
        out[sid] = frame
    return out


def build_macro_bundle_features() -> pd.DataFrame:
    """Concatenated long-form view (kept for diagnostics + tests).

    Returns the union of all per-series frames sorted by t_visible. Each row
    has columns for ONE series + NaN elsewhere. The downstream `build.py`
    uses `build_per_series()` directly to attach each series independently
    — DO NOT use this output as input to `forward_fill_to_daily`, you'll
    get all-NaN rows for any series whose t_visible isn't the latest.
    """
    frames = build_per_series()
    derived = build_derived_features(frames)

    if not frames and derived.empty:
        LOG.warning("no macro FRED parquets found — skipping")
        return pd.DataFrame()

    parts = list(frames.values())
    if not derived.empty:
        parts.append(derived)
    long = pd.concat(parts, ignore_index=True).sort_values("t_visible").reset_index(drop=True)
    LOG.info("macro_bundle features built: %d rows × %d cols", len(long), long.shape[1])
    return long


def build_derived_features(per_series: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Compute the 3 derived signals on the per-series ICSA, M2, DFF, CPI streams.

    Each derived row is keyed by the timestamp of the SOURCE release event:
      - icsa_4w_ma is per ICSA release (weekly)
      - m2_yoy mirrors the M2SL release (monthly)
      - real_fedfunds is per DFF release (daily) joined backward to CPILFESL.
    """
    out_frames: list[pd.DataFrame] = []

    if "ICSA" in per_series:
        icsa = per_series["ICSA"].copy()
        icsa["icsa_4w_ma"] = icsa["icsa_level"].rolling(4, min_periods=2).mean()
        out_frames.append(icsa[["date", "t_visible", "icsa_4w_ma"]])

    if "M2SL" in per_series:
        m2 = per_series["M2SL"].copy()
        m2_alias = m2[["date", "t_visible", "m2sl_yoy_change"]].rename(
            columns={"m2sl_yoy_change": "m2_yoy"}
        )
        out_frames.append(m2_alias)

    if "DFF" in per_series and "CPILFESL" in per_series:
        dff = per_series["DFF"][["date", "t_visible", "dff_level"]].sort_values("t_visible")
        cpi = (
            per_series["CPILFESL"][["t_visible", "cpilfesl_yoy_change"]]
            .dropna(subset=["cpilfesl_yoy_change"])
            .sort_values("t_visible")
        )
        if not dff.empty and not cpi.empty:
            merged = pd.merge_asof(
                dff,
                cpi.rename(columns={"cpilfesl_yoy_change": "_cpi_yoy"}),
                on="t_visible",
                direction="backward",
                allow_exact_matches=True,
            )
            merged["real_fedfunds"] = (
                merged["dff_level"].astype("float64") - merged["_cpi_yoy"].astype("float64") * 100.0
            )
            out_frames.append(merged[["date", "t_visible", "real_fedfunds"]])

    if not out_frames:
        return pd.DataFrame()
    return pd.concat(out_frames, ignore_index=True).sort_values("t_visible").reset_index(drop=True)
