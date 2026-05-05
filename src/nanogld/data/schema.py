"""Column manifests + validators for every data-pipeline source.

Each source emits a tidy DataFrame with `release_ts` and `t_visible`
columns. `validate(df, manifest)` checks columns + dtypes and asserts
the V1 hard-rule PIT invariant: `release_ts <= t_visible`.

We use simple dict-based manifests rather than pandera/pydantic frames
to keep the dep surface minimal. Tests in `tests/test_join_schema.py`
re-use the same manifests.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from nanogld.data.utils import assert_t_visible_invariant


@dataclass(frozen=True)
class ColumnSpec:
    name: str
    dtype: str  # pandas dtype string ("float64", "int64", "string", "datetime64[ns, UTC]", "bool")
    nullable: bool = False
    description: str = ""


@dataclass(frozen=True)
class Manifest:
    """A source's column contract."""

    name: str
    columns: tuple[ColumnSpec, ...]
    primary_key: tuple[str, ...] = field(default_factory=tuple)


def _is_dt_utc(s: pd.Series) -> bool:
    return isinstance(s.dtype, pd.DatetimeTZDtype) and str(s.dtype.tz) == "UTC"


def _check_dtype(s: pd.Series, want: str) -> tuple[bool, str]:
    if want.startswith("datetime64[ns, UTC]"):
        return _is_dt_utc(s), str(s.dtype)
    if want == "string":
        return s.dtype.kind in {"O", "U"} or pd.api.types.is_string_dtype(s), str(s.dtype)
    return str(s.dtype) == want or s.dtype == np.dtype(want), str(s.dtype)


def validate(df: pd.DataFrame, m: Manifest, *, allow_extra: bool = True) -> None:
    """Check df against `m`. Raises ValueError on mismatch.

    Always enforces:
      - every column in manifest is present
      - dtype matches (UTC tz-aware checked specially)
      - non-nullable columns have no NaN
      - if both `release_ts` + `t_visible` present, release_ts <= t_visible
    """
    missing = [c.name for c in m.columns if c.name not in df.columns]
    if missing:
        raise ValueError(f"[{m.name}] missing columns: {missing}")

    if not allow_extra:
        extra = [c for c in df.columns if c not in {col.name for col in m.columns}]
        if extra:
            raise ValueError(f"[{m.name}] unexpected columns: {extra}")

    for spec in m.columns:
        ok, got = _check_dtype(df[spec.name], spec.dtype)
        if not ok:
            raise ValueError(f"[{m.name}] {spec.name!r} dtype {got!r} != expected {spec.dtype!r}")
        if not spec.nullable and df[spec.name].isna().any():
            raise ValueError(f"[{m.name}] {spec.name!r} has NaN but is non-nullable")

    if "release_ts" in df.columns and "t_visible" in df.columns:
        assert_t_visible_invariant(df)


# ────────────────────────────────────────────────────────────────────────────
# Per-source manifests
# ────────────────────────────────────────────────────────────────────────────


BARS_MANIFEST = Manifest(
    name="bars",
    columns=(
        ColumnSpec("symbol", "string", description="Ticker (GLD, SPY, ...)"),
        ColumnSpec("timestamp", "datetime64[ns, UTC]", description="Bar START (Alpaca semantics)"),
        ColumnSpec("open", "float64"),
        ColumnSpec("high", "float64"),
        ColumnSpec("low", "float64"),
        ColumnSpec("close", "float64"),
        ColumnSpec("volume", "float64"),
        ColumnSpec("trade_count", "float64", nullable=True),
        ColumnSpec("vwap", "float64", nullable=True),
        ColumnSpec("release_ts", "datetime64[ns, UTC]", description="Bar END = timestamp + 30min"),
        ColumnSpec("t_visible", "datetime64[ns, UTC]"),
    ),
    primary_key=("symbol", "timestamp"),
)


NEWS_MANIFEST = Manifest(
    name="news",
    columns=(
        ColumnSpec("article_id", "string"),
        ColumnSpec("source", "string", description="alpaca_benzinga, kitco, gdelt, fnspid, ..."),
        ColumnSpec(
            "created_at",
            "datetime64[ns, UTC]",
            description="Alpaca uses created_at, NOT published_at",
        ),
        ColumnSpec("title", "string"),
        ColumnSpec("body", "string", nullable=True),
        ColumnSpec("url", "string", nullable=True),
        ColumnSpec("symbols", "string", nullable=True, description="Pipe-delimited tickers"),
        ColumnSpec("bias_tier", "string", description="See doc 03 SOURCE_REGISTRY"),
        ColumnSpec("release_ts", "datetime64[ns, UTC]"),
        ColumnSpec("t_visible", "datetime64[ns, UTC]"),
    ),
    primary_key=("source", "article_id"),
)


FRED_MANIFEST = Manifest(
    name="fred_alfred",
    columns=(
        ColumnSpec("series_id", "string"),
        ColumnSpec("date", "datetime64[ns, UTC]", description="Observation date (00:00 UTC)"),
        ColumnSpec("value", "float64", nullable=True),
        ColumnSpec("realtime_start", "datetime64[ns, UTC]", description="ALFRED first-public date"),
        ColumnSpec("realtime_end", "datetime64[ns, UTC]", nullable=True),
        ColumnSpec(
            "release_ts", "datetime64[ns, UTC]", description="Date + series-specific ET tod"
        ),
        ColumnSpec("t_visible", "datetime64[ns, UTC]"),
    ),
    primary_key=("series_id", "date", "realtime_start"),
)


GDELT_MANIFEST = Manifest(
    name="gdelt_gkg",
    columns=(
        ColumnSpec("pub_ts_utc", "datetime64[ns, UTC]", description="Article publication ts"),
        ColumnSpec(
            "partition_ts_utc", "datetime64[ns, UTC]", description="BigQuery _PARTITIONTIME"
        ),
        ColumnSpec("url", "string"),
        ColumnSpec("v2_themes", "string", nullable=True),
        ColumnSpec("v2_tone", "string", nullable=True),
        ColumnSpec("v2_locations", "string", nullable=True),
        ColumnSpec("release_ts", "datetime64[ns, UTC]"),
        ColumnSpec("t_visible", "datetime64[ns, UTC]"),
    ),
    primary_key=("url", "pub_ts_utc"),
)


YFINANCE_DAILY_MANIFEST = Manifest(
    name="yfinance_daily",
    columns=(
        ColumnSpec("ticker", "string"),
        ColumnSpec("date", "datetime64[ns, UTC]"),
        ColumnSpec("open", "float64"),
        ColumnSpec("high", "float64"),
        ColumnSpec("low", "float64"),
        ColumnSpec("close", "float64"),
        ColumnSpec("volume", "float64", nullable=True),
        ColumnSpec("settlement_ts_utc", "datetime64[ns, UTC]"),
        ColumnSpec("release_ts", "datetime64[ns, UTC]"),
        ColumnSpec("t_visible", "datetime64[ns, UTC]"),
    ),
    primary_key=("ticker", "date"),
)


GPR_MANIFEST = Manifest(
    name="gpr",
    columns=(
        ColumnSpec("series", "string", description="GPR / GPRD / AIGPR_DAILY / ..."),
        ColumnSpec("date", "datetime64[ns, UTC]"),
        ColumnSpec("value", "float64", nullable=True),
        ColumnSpec(
            "fetch_ts", "datetime64[ns, UTC]", description="When we self-snapshotted (vintage key)"
        ),
        ColumnSpec("source_sha", "string"),
        ColumnSpec("release_ts", "datetime64[ns, UTC]"),
        ColumnSpec("t_visible", "datetime64[ns, UTC]"),
    ),
    primary_key=("series", "date", "fetch_ts"),
)


COT_MANIFEST = Manifest(
    name="cftc_cot",
    columns=(
        ColumnSpec("contract_code", "string"),
        ColumnSpec("contract_name", "string"),
        ColumnSpec("report_date", "datetime64[ns, UTC]", description="Tuesday 4 PM ET reference"),
        ColumnSpec("oi_open_interest", "float64", nullable=True),
        ColumnSpec("mm_long", "float64", nullable=True),
        ColumnSpec("mm_short", "float64", nullable=True),
        ColumnSpec("mm_spread", "float64", nullable=True),
        ColumnSpec("comm_long", "float64", nullable=True),
        ColumnSpec("comm_short", "float64", nullable=True),
        ColumnSpec("nonrept_long", "float64", nullable=True),
        ColumnSpec("nonrept_short", "float64", nullable=True),
        ColumnSpec(
            "irregular_release", "bool", description="True if 2025 shutdown gap or holiday-shift"
        ),
        ColumnSpec("release_ts", "datetime64[ns, UTC]"),
        ColumnSpec("t_visible", "datetime64[ns, UTC]"),
    ),
    primary_key=("contract_code", "report_date"),
)


WGC_MANIFEST = Manifest(
    name="wgc",
    columns=(
        ColumnSpec("country", "string"),
        ColumnSpec("period", "datetime64[ns, UTC]", description="Reporting period start"),
        ColumnSpec("frequency", "string", description="monthly | quarterly"),
        ColumnSpec("holdings_tonnes", "float64", nullable=True),
        ColumnSpec("net_purchases_tonnes", "float64", nullable=True),
        ColumnSpec("pct_total_reserves", "float64", nullable=True),
        ColumnSpec("fetch_ts", "datetime64[ns, UTC]"),
        ColumnSpec("source_sha", "string"),
        ColumnSpec("release_ts", "datetime64[ns, UTC]"),
        ColumnSpec("t_visible", "datetime64[ns, UTC]"),
    ),
    primary_key=("country", "period", "fetch_ts"),
)


CALENDAR_MANIFEST = Manifest(
    name="calendar_events",
    columns=(
        ColumnSpec("event_type", "string", description="FOMC|CPI|NFP|GDP|JOLTS|PCE|FOMC_minutes"),
        ColumnSpec("event_ts_utc", "datetime64[ns, UTC]"),
        ColumnSpec("tier", "int64", description="1 = market-moving, 2 = secondary"),
        ColumnSpec("release_ts", "datetime64[ns, UTC]"),
        ColumnSpec("t_visible", "datetime64[ns, UTC]"),
    ),
    primary_key=("event_type", "event_ts_utc"),
)


# Joined snapshot — partial spec (doc 04 expands feature columns).
SNAPSHOT_KEY_COLUMNS: tuple[ColumnSpec, ...] = (
    ColumnSpec(
        "bar_close_utc", "datetime64[ns, UTC]", description="Bar END = bar.timestamp + 30min"
    ),
    ColumnSpec("symbol", "string"),
    ColumnSpec("close", "float64"),
)


ALL_MANIFESTS: dict[str, Manifest] = {
    m.name: m
    for m in (
        BARS_MANIFEST,
        NEWS_MANIFEST,
        FRED_MANIFEST,
        GDELT_MANIFEST,
        YFINANCE_DAILY_MANIFEST,
        GPR_MANIFEST,
        COT_MANIFEST,
        WGC_MANIFEST,
        CALENDAR_MANIFEST,
    )
}
