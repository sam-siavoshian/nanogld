"""GPR (geopolitical risk) features.

Input: data/raw/gpr_combined.parquet (Source 6, Caldara-Iacoviello).
The raw file has 116 series — the V1 monthly canonical is the `GPR` series;
AI-GPR daily lives under `AIGPR_DAILY_*`. Each row carries fetch_ts (vintage
key) and t_visible.

Output columns (per row of date / monthly):
  gpr_level
  gpr_mom              MoM diff
  gpr_yoy              YoY diff (12m)
  gpr_z_60m            60-month rolling z-score
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from nanogld.data.utils import get_logger, raw_dir
from nanogld.features.utils import FeatureSpec, rolling_z

LOG = get_logger("nanogld.features.geopolitical")

FEATURES: tuple[FeatureSpec, ...] = (
    FeatureSpec("gpr_level", source="gpr"),
    FeatureSpec("gpr_mom", source="gpr"),
    FeatureSpec("gpr_yoy", source="gpr"),
    FeatureSpec("gpr_z_60m", source="gpr"),
)


def build_gpr_features() -> pd.DataFrame:
    path = Path(raw_dir()) / "gpr_combined.parquet"
    if not path.exists():
        LOG.warning("%s missing — skipping GPR features", path)
        return pd.DataFrame()

    raw = pd.read_parquet(path)
    monthly = raw[raw["series"] == "GPR"].copy()
    if monthly.empty:
        LOG.warning("no 'GPR' canonical series in raw GPR parquet — skipping")
        return pd.DataFrame()

    # Latest fetch per (date) — vintage discipline (we self-snapshot weekly;
    # a backtest at time T uses the latest fetch_ts <= T).
    latest = (
        monthly.sort_values(["date", "fetch_ts"]).groupby("date").tail(1).reset_index(drop=True)
    )
    out = pd.DataFrame()
    out["date"] = latest["date"]
    out["t_visible"] = latest["t_visible"]
    out["gpr_level"] = latest["value"].astype("float64")
    out["gpr_mom"] = out["gpr_level"].diff(1)
    out["gpr_yoy"] = out["gpr_level"].diff(12)
    out["gpr_z_60m"] = rolling_z(out["gpr_level"], window=60, min_periods=24)

    return out.sort_values("date").reset_index(drop=True)
