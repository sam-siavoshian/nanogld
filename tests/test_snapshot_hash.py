"""Determinism + content-addressing tests for snapshot hashing.

Spec: plan/02-DATA-PIPELINE.md "Snapshot Hashing" + plan/01 Critical Corrections.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from nanogld.data.snapshot import build_meta, snapshot_hash, write_snapshot


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "bar_close_utc": pd.to_datetime(
                ["2024-01-15 14:30:00", "2024-01-15 15:00:00"], utc=True
            ),
            "gld_close": [100.0, 100.5],
            "feature_a": [1, 2],
        }
    )


def test_snapshot_hash_deterministic() -> None:
    df1 = _frame()
    df2 = _frame()
    assert snapshot_hash(df1) == snapshot_hash(df2)


def test_snapshot_hash_changes_on_value_change() -> None:
    df1 = _frame()
    df2 = _frame()
    df2.loc[0, "gld_close"] = 999.0
    assert snapshot_hash(df1) != snapshot_hash(df2)


def test_snapshot_hash_changes_on_column_rename() -> None:
    """Renamed column = different artifact even if values match (incident:
    hash_pandas_object alone misses this; we additionally hash columns).
    """
    df1 = _frame()
    df2 = _frame().rename(columns={"gld_close": "GLD_close"})
    assert snapshot_hash(df1) != snapshot_hash(df2)


def test_snapshot_hash_changes_on_dtype_change() -> None:
    df1 = _frame()
    df2 = _frame()
    df2["feature_a"] = df2["feature_a"].astype("float64")
    assert snapshot_hash(df1) != snapshot_hash(df2)


def test_meta_roundtrip(tmp_path: Path) -> None:
    df = _frame()
    parquet, meta_path, meta = write_snapshot(df, sources=[{"name": "test"}], target_dir=tmp_path)
    assert parquet.exists()
    assert meta_path.exists()
    loaded = json.loads(meta_path.read_text())
    assert loaded["snapshot_hash"] == meta["snapshot_hash"]
    assert loaded["row_count"] == 2
    # parquet content reproduces the same hash
    rt = pd.read_parquet(parquet)
    assert snapshot_hash(rt) == loaded["snapshot_hash"]


def test_build_meta_records_time_range() -> None:
    df = _frame()
    meta = build_meta(df)
    assert meta["row_count"] == 2
    assert meta["column_count"] == 3
    assert meta["time_range_utc"][0].startswith("2024-01-15")
