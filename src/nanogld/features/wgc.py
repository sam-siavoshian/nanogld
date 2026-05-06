"""WGC (World Gold Council) central-bank flow features (doc 04 §11, 3 dims).

Input: data/raw/wgc_central_bank_quarterly.parquet (Source 11).
The raw parquet is a long form table — one row per (country, period). The
`country == 'Total above'` row is the world aggregate per WGC's report.

Output (one row per quarterly period):
  wgc_total_net_purchase_tonnes_q   World "Total above" net_purchases_tonnes
                                    for the quarter (positive = buying).
  wgc_total_net_purchase_yoy        Same value diffed against 4 quarters ago
                                    (year-over-year change).
  wgc_is_net_buyer_q                1.0 if the quarterly total > 0 else 0.0.

Frequency: quarterly. The downstream panel forward-fills onto the daily
grid via t_visible (release_ts is ~2 months after quarter-end per WGC's
cadence). V4 hard rule §V11: WGC self-snapshot weekly; no public vintage
archive — the parquet's t_visible already reflects the snapshot fetch_ts.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from nanogld.data.utils import get_logger, raw_dir
from nanogld.features.utils import FeatureSpec

LOG = get_logger("nanogld.features.wgc")

# WGC's world aggregate row label. Verified against
# data/raw/wgc_central_bank_quarterly.parquet on 2026-05-05.
WORLD_AGGREGATE_KEY = "Total above"

FEATURES: tuple[FeatureSpec, ...] = (
    FeatureSpec("wgc_total_net_purchase_tonnes_q", source="wgc"),
    FeatureSpec("wgc_total_net_purchase_yoy", source="wgc"),
    FeatureSpec("wgc_is_net_buyer_q", source="wgc"),
)


def build_wgc_features() -> pd.DataFrame:
    """Quarterly WGC central-bank flow panel. Empty if parquet missing.

    The parquet has both 'quarterly' and 'monthly' rows on some countries;
    we hold to quarterly for V1 to match the spec's cadence.
    """
    path = Path(raw_dir()) / "wgc_central_bank_quarterly.parquet"
    if not path.exists():
        LOG.warning("%s missing — skipping WGC features", path)
        return pd.DataFrame()

    raw = pd.read_parquet(path)
    if raw.empty:
        return pd.DataFrame()

    world = raw[
        (raw["country"] == WORLD_AGGREGATE_KEY)
        & (raw["frequency"].fillna("quarterly").str.lower() == "quarterly")
    ].copy()
    if world.empty:
        LOG.warning("WGC parquet has no '%s' quarterly rows — skipping", WORLD_AGGREGATE_KEY)
        return pd.DataFrame()

    # One row per quarter. If for any reason the parquet has multiple snapshots
    # per (country, period), keep the latest fetch_ts.
    if "fetch_ts" in world.columns:
        world = (
            world.sort_values(["period", "fetch_ts"])
            .groupby("period")
            .tail(1)
            .reset_index(drop=True)
        )
    world = world.sort_values("period").reset_index(drop=True)

    out = pd.DataFrame(
        {
            "period": world["period"],
            "t_visible": world["t_visible"],
            "release_ts": world["release_ts"],
        }
    )
    # Prefer the parquet's `net_purchases_tonnes` if populated. Fall back to
    # diffing `holdings_tonnes` quarter-over-quarter — the WGC "Total above"
    # row in our snapshot only ships holdings, so derive the flow from the
    # stock change.
    net_purch_raw = world["net_purchases_tonnes"].astype("float64")
    if net_purch_raw.notna().sum() == 0 and "holdings_tonnes" in world.columns:
        holdings = world["holdings_tonnes"].astype("float64")
        net_purch = holdings.diff(1)
        LOG.info(
            "WGC net_purchases_tonnes empty for '%s' — derived from holdings.diff(1)",
            WORLD_AGGREGATE_KEY,
        )
    else:
        net_purch = net_purch_raw
    out["wgc_total_net_purchase_tonnes_q"] = net_purch
    out["wgc_total_net_purchase_yoy"] = net_purch - net_purch.shift(4)
    out["wgc_is_net_buyer_q"] = (net_purch > 0).astype("float64").where(net_purch.notna(), np.nan)

    LOG.info("wgc features built: %d rows × %d cols", len(out), out.shape[1])
    return out
