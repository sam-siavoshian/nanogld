"""Schema validation tests for every source manifest.

Each source must:
- Have a Manifest in nanogld.data.schema.
- Validate its own emitted DataFrame via nanogld.data.schema.validate.
- Carry release_ts + t_visible columns satisfying release_ts <= t_visible.
"""

from __future__ import annotations

import pandas as pd
import pytest

from nanogld.data.schema import (
    ALL_MANIFESTS,
    BARS_MANIFEST,
    CALENDAR_MANIFEST,
    NEWS_MANIFEST,
    ColumnSpec,
    Manifest,
    validate,
)
from nanogld.data.utils import UTC


def _now() -> pd.Timestamp:
    return pd.Timestamp("2024-01-15 14:00:00", tz=UTC)


def test_all_manifests_contain_release_and_visible() -> None:
    for name, m in ALL_MANIFESTS.items():
        cols = {c.name for c in m.columns}
        assert "t_visible" in cols, f"manifest {name!r} missing t_visible"
        assert "release_ts" in cols, f"manifest {name!r} missing release_ts"


def test_validate_rejects_missing_column() -> None:
    df = pd.DataFrame({"event_ts_utc": [_now()], "tier": [1]})
    with pytest.raises(ValueError, match="missing columns"):
        validate(df, CALENDAR_MANIFEST)


def test_validate_rejects_bad_dtype() -> None:
    spec = Manifest(
        name="x",
        columns=(
            ColumnSpec("v", "float64"),
            ColumnSpec("release_ts", "datetime64[ns, UTC]"),
            ColumnSpec("t_visible", "datetime64[ns, UTC]"),
        ),
    )
    df = pd.DataFrame({"v": ["string-not-float"], "release_ts": [_now()], "t_visible": [_now()]})
    with pytest.raises(ValueError, match="dtype"):
        validate(df, spec)


def test_validate_rejects_null_in_non_nullable() -> None:
    spec = Manifest(
        name="x",
        columns=(
            ColumnSpec("v", "float64"),
            ColumnSpec("release_ts", "datetime64[ns, UTC]"),
            ColumnSpec("t_visible", "datetime64[ns, UTC]"),
        ),
    )
    df = pd.DataFrame({"v": [float("nan")], "release_ts": [_now()], "t_visible": [_now()]})
    with pytest.raises(ValueError, match="non-nullable"):
        validate(df, spec)


def test_validate_rejects_pit_violation() -> None:
    spec = Manifest(
        name="x",
        columns=(
            ColumnSpec("v", "float64"),
            ColumnSpec("release_ts", "datetime64[ns, UTC]"),
            ColumnSpec("t_visible", "datetime64[ns, UTC]"),
        ),
    )
    df = pd.DataFrame(
        {
            "v": [1.0],
            "release_ts": [_now()],
            "t_visible": [_now() - pd.Timedelta(seconds=1)],
        }
    )
    with pytest.raises(ValueError, match="release_ts <= t_visible"):
        validate(df, spec)


def test_news_manifest_accepts_minimal_clean_frame() -> None:
    df = pd.DataFrame(
        {
            "article_id": ["a"],
            "source": ["alpaca_benzinga"],
            "created_at": [_now()],
            "title": ["t"],
            "body": ["b"],
            "url": ["http://x"],
            "symbols": ["GLD"],
            "bias_tier": ["mainstream_neutral"],
            "release_ts": [_now()],
            "t_visible": [_now() + pd.Timedelta(seconds=60)],
        }
    )
    for c in ("article_id", "source", "title", "body", "url", "symbols", "bias_tier"):
        df[c] = df[c].astype("string")
    validate(df, NEWS_MANIFEST)


def test_bars_manifest_accepts_minimal_clean_frame() -> None:
    df = pd.DataFrame(
        {
            "symbol": ["GLD"],
            "timestamp": [_now()],
            "open": [100.0],
            "high": [100.5],
            "low": [99.8],
            "close": [100.0],
            "volume": [1000.0],
            "trade_count": [100.0],
            "vwap": [100.05],
            "release_ts": [_now() + pd.Timedelta(minutes=30)],
            "t_visible": [_now() + pd.Timedelta(minutes=30)],
        }
    )
    df["symbol"] = df["symbol"].astype("string")
    validate(df, BARS_MANIFEST)
