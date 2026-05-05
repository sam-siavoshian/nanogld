"""Calendar event-proximity binaries.

V1 hard rule §14: binary windows ONLY. NO `minutes_until_event` raw features
(calendar memorization risk). Each row gets True/False flags for whether
an event of each type sits within ±24 h (default) of the row's day.

Input: data/raw/calendar_events_v1.parquet.
Output (per daily row):
  event_within_24h_FOMC
  event_within_24h_CPI
  event_within_24h_NFP
  event_within_24h_GDP
  event_within_24h_JOLTS
  event_within_24h_PCE
  event_within_24h_FOMC_minutes
  event_within_24h_any_tier1     any tier-1 event in the window
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from nanogld.data.utils import get_logger, raw_dir
from nanogld.features.utils import FeatureSpec

LOG = get_logger("nanogld.features.calendar")

EVENT_TYPES = ("FOMC", "CPI", "NFP", "GDP", "JOLTS", "PCE", "FOMC_minutes")
WINDOW = pd.Timedelta(hours=24)

FEATURES: tuple[FeatureSpec, ...] = tuple(
    FeatureSpec(f"event_within_24h_{e}", dtype="bool", source="calendar") for e in EVENT_TYPES
) + (FeatureSpec("event_within_24h_any_tier1", dtype="bool", source="calendar"),)


def build_calendar_features(daily_idx: pd.DatetimeIndex) -> pd.DataFrame:
    """Return one row per UTC day with binary event-proximity flags.

    `daily_idx` must be tz=UTC. Output is keyed by `date_utc`.
    """
    path = Path(raw_dir()) / "calendar_events_v1.parquet"
    if not path.exists():
        LOG.warning("%s missing — skipping calendar features", path)
        return pd.DataFrame(index=daily_idx)

    cal = pd.read_parquet(path)
    cal["event_ts_utc"] = pd.to_datetime(cal["event_ts_utc"], utc=True)
    by_type: dict[str, pd.DatetimeIndex] = {
        et: pd.DatetimeIndex(cal.loc[cal["event_type"] == et, "event_ts_utc"]).sort_values()
        for et in EVENT_TYPES
    }
    tier1 = pd.DatetimeIndex(cal.loc[cal["tier"] == 1, "event_ts_utc"]).sort_values()

    out = pd.DataFrame(index=daily_idx)

    def in_window(idx: pd.DatetimeIndex, day: pd.Timestamp) -> bool:
        if len(idx) == 0:
            return False
        lo = day - WINDOW
        hi = day + WINDOW
        i = idx.searchsorted(lo, side="left")
        return bool(i < len(idx) and idx[i] <= hi)

    for et, idx in by_type.items():
        out[f"event_within_24h_{et}"] = [in_window(idx, d) for d in daily_idx]
    out["event_within_24h_any_tier1"] = [in_window(tier1, d) for d in daily_idx]

    out.index.name = "date_utc"
    return out
