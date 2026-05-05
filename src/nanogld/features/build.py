"""Build the daily macro feature panel from existing raw parquets.

Output: data/processed/macro_features_daily.parquet (+ _meta.json sidecar).

Schema:
  index           DatetimeIndex name='date_utc' (UTC midnight)
  oil_*           brent + wti levels, log returns 1/5/20d, vol, spreads
  cot_*           managed money + commercial + non-rep positioning, %OI, z-scores
  gpr_*           level / MoM / YoY / 60m z-score
  event_within_24h_*    binary calendar windows
  t_visible_max   the latest t_visible across joined sources for that day

PIT discipline: each feature group's `t_visible` is forward-filled onto the
daily UTC grid via merge_asof(direction='backward'). The output's
`t_visible_max` is what the eventual 30min joiner will compare against
each bar_close (strict-< per V1 hard rule).

This panel is daily-frequency. When Alpaca GLD 30min bars land, the join.py
in nanogld/data forward-fills these features onto each 30min bar via
strict-< asof on `t_visible_max`.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from nanogld.data.utils import get_logger
from nanogld.features import calendar_features, geopolitical, oil, positioning
from nanogld.features.utils import (
    daily_index_utc,
    forward_fill_to_daily,
    processed_dir,
)

LOG = get_logger("nanogld.features.build")

DEFAULT_START = pd.Timestamp("2021-04-24", tz="UTC")
DEFAULT_END = pd.Timestamp("2026-04-24", tz="UTC")


def _split_features_t_visible(
    df: pd.DataFrame, on: str = "t_visible"
) -> tuple[list[str], pd.Series]:
    """Return (feature columns ex t_visible/keys, the t_visible series)."""
    skip = {on, "date", "report_date", "period", "event_ts_utc", "fetch_ts"}
    feat_cols = [c for c in df.columns if c not in skip]
    return feat_cols, df[on] if on in df.columns else pd.Series(dtype="datetime64[ns, UTC]")


def build_panel(
    *,
    start: pd.Timestamp = DEFAULT_START,
    end: pd.Timestamp = DEFAULT_END,
) -> pd.DataFrame:
    """Build the daily macro panel."""
    daily = daily_index_utc(start, end)
    LOG.info("daily panel: %d rows from %s to %s", len(daily), start, end)
    panel = pd.DataFrame(index=daily)
    panel.index.name = "date_utc"

    # Track the latest visible t_visible per source group for the global gate
    visible_per_group: dict[str, pd.Series] = {}

    # Oil
    oil_df = oil.build_oil_features()
    if not oil_df.empty:
        feat_cols, _ = _split_features_t_visible(oil_df, on="t_visible")
        sub = oil_df[["t_visible", *feat_cols]].dropna(subset=["t_visible"])
        ff = forward_fill_to_daily(sub, on="t_visible", cols=feat_cols, daily_idx=daily)
        panel = panel.join(ff)
        # Group t_visible
        ff_t = forward_fill_to_daily(
            sub[["t_visible"]].assign(_t=sub["t_visible"]),
            on="t_visible",
            cols=["_t"],
            daily_idx=daily,
        )
        visible_per_group["oil"] = ff_t["_t"]
        LOG.info("oil features attached: %d cols", len(feat_cols))

    # COT
    cot_df = positioning.build_cot_features()
    if not cot_df.empty:
        feat_cols, _ = _split_features_t_visible(cot_df, on="t_visible")
        sub = cot_df[["t_visible", *feat_cols]].dropna(subset=["t_visible"])
        ff = forward_fill_to_daily(sub, on="t_visible", cols=feat_cols, daily_idx=daily)
        panel = panel.join(ff)
        ff_t = forward_fill_to_daily(
            sub[["t_visible"]].assign(_t=sub["t_visible"]),
            on="t_visible",
            cols=["_t"],
            daily_idx=daily,
        )
        visible_per_group["cot"] = ff_t["_t"]
        LOG.info("cot features attached: %d cols", len(feat_cols))

    # GPR
    gpr_df = geopolitical.build_gpr_features()
    if not gpr_df.empty:
        feat_cols, _ = _split_features_t_visible(gpr_df, on="t_visible")
        sub = gpr_df[["t_visible", *feat_cols]].dropna(subset=["t_visible"])
        ff = forward_fill_to_daily(sub, on="t_visible", cols=feat_cols, daily_idx=daily)
        panel = panel.join(ff)
        ff_t = forward_fill_to_daily(
            sub[["t_visible"]].assign(_t=sub["t_visible"]),
            on="t_visible",
            cols=["_t"],
            daily_idx=daily,
        )
        visible_per_group["gpr"] = ff_t["_t"]
        LOG.info("gpr features attached: %d cols", len(feat_cols))

    # Calendar — already daily, just join
    cal = calendar_features.build_calendar_features(daily)
    panel = panel.join(cal)
    LOG.info("calendar features attached: %d cols", cal.shape[1])

    # Global t_visible_max — used by the 30min joiner via strict-< asof
    if visible_per_group:
        t_df = pd.DataFrame(visible_per_group)
        panel["t_visible_max"] = t_df.max(axis=1)
    else:
        panel["t_visible_max"] = pd.NaT

    return panel


def write_panel(
    *,
    start: pd.Timestamp = DEFAULT_START,
    end: pd.Timestamp = DEFAULT_END,
) -> tuple[pd.DataFrame, Path, Path]:
    panel = build_panel(start=start, end=end)
    out_path = processed_dir() / "macro_features_daily.parquet"
    panel.to_parquet(out_path, compression="zstd")

    meta = {
        "version": "v1-pre-gld",
        "built_utc": datetime.now(tz=UTC).isoformat(),
        "start": start.isoformat(),
        "end": end.isoformat(),
        "rows": int(len(panel)),
        "cols": int(panel.shape[1]),
        "feature_columns": [c for c in panel.columns if c != "t_visible_max"],
        "missing_sources": [
            s
            for s in ("alpaca_bars", "alpaca_etfs", "alpaca_news", "fred", "gdelt", "fnspid")
            if not (Path("data/raw") / f"{s}_GLD_30min.parquet").exists()
            and s in {"alpaca_bars"}  # only flag the single-file outputs
        ],
        "non_null_pct_per_col": {
            c: float(panel[c].notna().mean()) for c in panel.columns if c != "t_visible_max"
        },
    }
    meta_path = processed_dir() / "macro_features_daily_meta.json"
    with meta_path.open("w") as f:
        json.dump(meta, f, indent=2, default=str)
    LOG.info("wrote %s (%d rows × %d cols)", out_path, *panel.shape)
    LOG.info("meta -> %s", meta_path)
    return panel, out_path, meta_path


if __name__ == "__main__":
    panel, parquet, meta = write_panel()
    print(f"\n{parquet} ({panel.shape[0]} rows × {panel.shape[1]} cols)")
    print(panel.tail(3).to_string(max_colwidth=20))
