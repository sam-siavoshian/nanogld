"""Build the daily macro feature panel from existing raw parquets.

Output: data/processed/macro_features_daily.parquet (+ _meta.json sidecar).

Schema:
  index           DatetimeIndex name='date_utc' (UTC midnight)
  oil_*           brent + wti levels, log returns 1/5/20d, vol, spreads
  cot_*           managed money + commercial + non-rep positioning, %OI, z-scores
  gpr_*           level / MoM / YoY / 60m z-score
  event_within_24h_*    binary calendar windows
  macro_*         short-window macro: DXY 5d, DGS10/2 levels, term spread, real rate, VIX
  treasury_*      full curve (11 levels + 11 1d-changes + 4 spreads + butterfly + 2 real-rate)
  macro_bundle_*  19-series macro (level + yoy + mom each) + 3 derived
  wgc_*           WGC central-bank flow features (quarterly)
  t_visible_max   the latest t_visible across joined sources for that day

PIT discipline: each feature group's `t_visible` is forward-filled onto the
daily UTC grid via merge_asof(direction='backward'). The output's
`t_visible_max` is what the eventual 30min joiner will compare against
each bar_close (strict-< per V1 hard rule).

Bar-frequency feature groups (price, risk, equity, sentiment) are NOT
absorbed into this daily panel. They feed the joiner directly via
nanogld.data.join, which forward-fills them onto each 30min bar. This
module's responsibility is to make sure those builders are importable
and can be invoked by the joiner; it does NOT call them itself.

Per-series attach path (NEW): some FRED-derived groups (macro, macro_bundle)
have series with mismatched release cadences (VIXCLS daily vs JTSJOL monthly
vs ICSA weekly). A single t_visible-asof picks the latest row across the
outer-joined frame and zeros out columns that didn't observe on that day.
For those groups we attach EACH SERIES INDEPENDENTLY using its own per-row
t_visible — so each column's last-known-good value carries forward.

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
from nanogld.features import (
    calendar_features,
    geopolitical,
    macro,
    macro_bundle,
    oil,
    positioning,
    treasury,
    wgc,
)
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
    skip = {on, "date", "report_date", "period", "event_ts_utc", "fetch_ts", "release_ts"}
    feat_cols = [c for c in df.columns if c not in skip]
    return feat_cols, df[on] if on in df.columns else pd.Series(dtype="datetime64[ns, UTC]")


def _attach_group(
    panel: pd.DataFrame,
    daily: pd.DatetimeIndex,
    df: pd.DataFrame,
    *,
    visible_per_group: dict[str, pd.Series],
    group_name: str,
) -> pd.DataFrame:
    """Forward-fill a daily-frequency feature group onto the panel.

    Columns already on the panel (from a prior group) are skipped so we don't
    blow up on overlapping FRED-derived names. macro.py and treasury.py both
    expose dgs10_level / dgs2_level — first-attached wins, second-attached
    skips. macro.py loads first so it claims the simpler scaled values.

    Each column is fwd-filled INDEPENDENTLY against rows where that column
    is non-null — preserves carry-forward across FRED holiday gaps.
    """
    if df.empty:
        return panel
    feat_cols, _ = _split_features_t_visible(df, on="t_visible")
    fresh_cols = [c for c in feat_cols if c not in panel.columns]
    if not fresh_cols:
        LOG.info("%s features: all %d cols already on panel — skipping", group_name, len(feat_cols))
        return panel
    if len(fresh_cols) < len(feat_cols):
        LOG.info(
            "%s features: %d duplicate cols dropped (already attached by prior group)",
            group_name,
            len(feat_cols) - len(fresh_cols),
        )
    max_t_visible: pd.Series | None = None
    cols_attached = 0
    for col in fresh_cols:
        sub = df[["t_visible", col]].dropna(subset=["t_visible", col]).sort_values("t_visible")
        if sub.empty:
            continue
        ff = forward_fill_to_daily(sub, on="t_visible", cols=[col], daily_idx=daily)
        panel = panel.join(ff)
        ff_t = forward_fill_to_daily(
            sub[["t_visible"]].assign(_t=sub["t_visible"]),
            on="t_visible",
            cols=["_t"],
            daily_idx=daily,
        )
        if max_t_visible is None:
            max_t_visible = ff_t["_t"]
        else:
            max_t_visible = pd.concat([max_t_visible, ff_t["_t"]], axis=1).max(axis=1)
        cols_attached += 1
    if max_t_visible is not None:
        visible_per_group[group_name] = max_t_visible
    LOG.info("%s features attached: %d cols", group_name, cols_attached)
    return panel


def _attach_per_series(
    panel: pd.DataFrame,
    daily: pd.DatetimeIndex,
    per_series: dict[str, pd.DataFrame],
    *,
    visible_per_group: dict[str, pd.Series],
    group_name: str,
) -> pd.DataFrame:
    """Attach a dict of per-series frames. Each frame's columns flow into
    the panel via its OWN t_visible-asof — so a column missing on day D
    falls back to its most recent prior release.

    For each per-series frame, each output column is forward-filled
    INDEPENDENTLY using only rows where THAT column is non-null. This
    matters for FRED holiday days where the release_ts row exists but the
    `value` is NaN; without per-column dropna the merge_asof picks the
    NaN row and "freezes" the output across the holiday.
    """
    total_cols = 0
    max_t_visible: pd.Series | None = None
    for _sid, frame in per_series.items():
        if frame.empty:
            continue
        feat_cols, _ = _split_features_t_visible(frame, on="t_visible")
        if not feat_cols:
            continue
        for col in feat_cols:
            if col in panel.columns:
                continue
            sub = (
                frame[["t_visible", col]].dropna(subset=["t_visible", col]).sort_values("t_visible")
            )
            if sub.empty:
                continue
            ff = forward_fill_to_daily(sub, on="t_visible", cols=[col], daily_idx=daily)
            panel = panel.join(ff)
            ff_t = forward_fill_to_daily(
                sub[["t_visible"]].assign(_t=sub["t_visible"]),
                on="t_visible",
                cols=["_t"],
                daily_idx=daily,
            )
            if max_t_visible is None:
                max_t_visible = ff_t["_t"]
            else:
                max_t_visible = pd.concat([max_t_visible, ff_t["_t"]], axis=1).max(axis=1)
            total_cols += 1
    if max_t_visible is not None:
        visible_per_group[group_name] = max_t_visible
    LOG.info("%s features attached: %d cols (per-series path)", group_name, total_cols)
    return panel


def _attach_derived(
    panel: pd.DataFrame,
    daily: pd.DatetimeIndex,
    derived_df: pd.DataFrame,
    *,
    visible_per_group: dict[str, pd.Series],
    group_name: str,
) -> pd.DataFrame:
    """Attach derived multi-column frame where each row has its own t_visible
    but columns may be sparse. Each column attached via its own t_visible.
    """
    if derived_df.empty:
        return panel
    feat_cols, _ = _split_features_t_visible(derived_df, on="t_visible")
    fresh_cols = [c for c in feat_cols if c not in panel.columns]
    if not fresh_cols:
        return panel
    max_t_visible: pd.Series | None = None
    cols_attached = 0
    for col in fresh_cols:
        sub = (
            derived_df[["t_visible", col]]
            .dropna(subset=["t_visible", col])
            .sort_values("t_visible")
        )
        if sub.empty:
            continue
        ff = forward_fill_to_daily(sub, on="t_visible", cols=[col], daily_idx=daily)
        panel = panel.join(ff)
        ff_t = forward_fill_to_daily(
            sub[["t_visible"]].assign(_t=sub["t_visible"]),
            on="t_visible",
            cols=["_t"],
            daily_idx=daily,
        )
        if max_t_visible is None:
            max_t_visible = ff_t["_t"]
        else:
            max_t_visible = pd.concat([max_t_visible, ff_t["_t"]], axis=1).max(axis=1)
        cols_attached += 1
    if max_t_visible is not None:
        # Merge into existing visible if any.
        visible_per_group[f"{group_name}_derived"] = max_t_visible
    LOG.info("%s derived features attached: %d cols", group_name, cols_attached)
    return panel


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

    # Existing daily groups
    panel = _attach_group(
        panel,
        daily,
        oil.build_oil_features(),
        visible_per_group=visible_per_group,
        group_name="oil",
    )
    panel = _attach_group(
        panel,
        daily,
        positioning.build_cot_features(),
        visible_per_group=visible_per_group,
        group_name="cot",
    )
    panel = _attach_group(
        panel,
        daily,
        geopolitical.build_gpr_features(),
        visible_per_group=visible_per_group,
        group_name="gpr",
    )

    # V1 expansion daily groups — these are FRED-derived with mismatched
    # cadences across series (DTWEXBGS daily vs T10YIE daily vs VIXCLS daily
    # but with different release-tods; UNRATE monthly vs DFF daily, etc.).
    # Use the per-series attach path so each column carries its own visibility.
    macro_features = macro.build_features_per_series()
    panel = _attach_per_series(
        panel,
        daily,
        macro_features,
        visible_per_group=visible_per_group,
        group_name="macro",
    )

    treasury_per_series = treasury.build_per_series()
    treasury_derived = treasury.build_derived_features(treasury_per_series)
    panel = _attach_per_series(
        panel,
        daily,
        treasury_per_series,
        visible_per_group=visible_per_group,
        group_name="treasury",
    )
    panel = _attach_per_series(
        panel,
        daily,
        treasury_derived,
        visible_per_group=visible_per_group,
        group_name="treasury_derived",
    )

    # Macro bundle is the same shape — 19 series with wildly different cadences.
    macro_bundle_per_series = macro_bundle.build_per_series()
    panel = _attach_per_series(
        panel,
        daily,
        macro_bundle_per_series,
        visible_per_group=visible_per_group,
        group_name="macro_bundle",
    )
    derived = macro_bundle.build_derived_features(macro_bundle_per_series)
    panel = _attach_derived(
        panel,
        daily,
        derived,
        visible_per_group=visible_per_group,
        group_name="macro_bundle",
    )

    panel = _attach_group(
        panel,
        daily,
        wgc.build_wgc_features(),
        visible_per_group=visible_per_group,
        group_name="wgc",
    )

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
