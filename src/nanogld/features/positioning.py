"""COT positioning features for COMEX gold (contract 088691).

Input: data/raw/cftc_cot_gold_weekly.parquet (Source 9, V4 — disaggregated).
Output columns (per weekly row):
  mm_net               managed money long - short
  mm_net_pct_oi        mm_net / open_interest
  mm_z_52w             52-week rolling z-score of mm_net
  mm_wow_change        mm_net - mm_net.shift(1)
  comm_net             producer/merchant long - short (commercial hedgers)
  comm_net_pct_oi
  nonrept_net          small spec long - short
  oi_z_52w             52-week z of open interest

t_visible carried through from cot.py (Friday 16:00 ET, holiday-adjusted).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from nanogld.data.utils import get_logger, raw_dir
from nanogld.features.utils import FeatureSpec, rolling_z

LOG = get_logger("nanogld.features.positioning")

FEATURES: tuple[FeatureSpec, ...] = (
    FeatureSpec("cot_mm_net", source="cot"),
    FeatureSpec("cot_mm_net_pct_oi", source="cot"),
    FeatureSpec("cot_mm_z_52w", source="cot"),
    FeatureSpec("cot_mm_wow_change", source="cot"),
    FeatureSpec("cot_comm_net", source="cot"),
    FeatureSpec("cot_comm_net_pct_oi", source="cot"),
    FeatureSpec("cot_nonrept_net", source="cot"),
    FeatureSpec("cot_oi_z_52w", source="cot"),
    FeatureSpec("cot_irregular_release", dtype="bool", source="cot"),
)


def build_cot_features() -> pd.DataFrame:
    path = Path(raw_dir()) / "cftc_cot_gold_weekly.parquet"
    if not path.exists():
        LOG.warning("%s missing — skipping COT features", path)
        return pd.DataFrame()
    df = pd.read_parquet(path).sort_values("report_date").reset_index(drop=True)

    out = pd.DataFrame()
    out["report_date"] = df["report_date"]
    out["t_visible"] = df["t_visible"]

    out["cot_mm_net"] = df["mm_long"] - df["mm_short"]
    out["cot_mm_net_pct_oi"] = out["cot_mm_net"] / df["oi_open_interest"]
    out["cot_mm_z_52w"] = rolling_z(out["cot_mm_net"], window=52, min_periods=20)
    out["cot_mm_wow_change"] = out["cot_mm_net"].diff(1)

    out["cot_comm_net"] = df["comm_long"] - df["comm_short"]
    out["cot_comm_net_pct_oi"] = out["cot_comm_net"] / df["oi_open_interest"]

    out["cot_nonrept_net"] = df["nonrept_long"] - df["nonrept_short"]
    out["cot_oi_z_52w"] = rolling_z(df["oi_open_interest"], window=52, min_periods=20)

    out["cot_irregular_release"] = df["irregular_release"].astype("bool")

    return out
