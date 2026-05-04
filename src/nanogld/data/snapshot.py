"""Snapshot hashing + meta.json writer.

Per doc 01 Critical Corrections: hash via `pd.util.hash_pandas_object` (10-100×
faster + handles float repr / NaN / locale correctly) instead of `df.to_csv()`.
The hash incorporates row hashes + column names + dtypes so a renamed column
or changed dtype produces a different hash even if values match.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from nanogld.data.utils import UTC, snapshots_dir


def snapshot_hash(df: pd.DataFrame) -> str:
    """SHA256 of dataframe content + schema. Returns first 16 hex chars."""
    row_hash = pd.util.hash_pandas_object(df, index=True).values.tobytes()
    cols = str(tuple(df.columns)).encode()
    dtypes = str(df.dtypes.to_dict()).encode()
    h = hashlib.sha256()
    h.update(row_hash)
    h.update(cols)
    h.update(dtypes)
    return h.hexdigest()[:16]


def _git_commit() -> str | None:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=Path(__file__).resolve().parents[3],
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
    except Exception:
        return None


def build_meta(
    df: pd.DataFrame,
    *,
    snapshot_version: str = "v1",
    schema_version: str = "v1.0.0",
    sources: list[dict[str, Any]] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Construct a meta dict matching plan/02-DATA-PIPELINE.md."""
    idx = df["bar_close_utc"] if "bar_close_utc" in df.columns else df.index
    if hasattr(idx, "min") and hasattr(idx, "max"):
        time_min = pd.Timestamp(idx.min())
        time_max = pd.Timestamp(idx.max())
        if time_min.tzinfo is None:
            time_min = time_min.tz_localize("UTC")
        if time_max.tzinfo is None:
            time_max = time_max.tz_localize("UTC")
        time_range = [time_min.isoformat(), time_max.isoformat()]
    else:
        time_range = [None, None]

    meta: dict[str, Any] = {
        "snapshot_version": snapshot_version,
        "schema_version": schema_version,
        "snapshot_hash": snapshot_hash(df),
        "created_utc": datetime.now(tz=UTC).isoformat(),
        "row_count": int(len(df)),
        "column_count": int(df.shape[1]),
        "time_range_utc": time_range,
        "git_commit": _git_commit(),
        "data_sources": sources or [],
    }
    if extra:
        meta["extra"] = extra
    return meta


def write_snapshot(
    df: pd.DataFrame,
    *,
    snapshot_version: str = "v1",
    schema_version: str = "v1.0.0",
    sources: list[dict[str, Any]] | None = None,
    extra: dict[str, Any] | None = None,
    target_dir: Path | None = None,
    overwrite: bool = False,
) -> tuple[Path, Path, dict[str, Any]]:
    """Write `data/snapshots/<version>_<hash>.parquet` + sidecar meta.json.

    Returns (parquet_path, meta_path, meta_dict).
    """
    target = target_dir or snapshots_dir()
    target.mkdir(parents=True, exist_ok=True)

    meta = build_meta(
        df,
        snapshot_version=snapshot_version,
        schema_version=schema_version,
        sources=sources,
        extra=extra,
    )
    base = f"{snapshot_version}_{meta['snapshot_hash']}"
    parquet_path = target / f"{base}.parquet"
    meta_path = target / f"{base}_meta.json"

    if parquet_path.exists() and not overwrite:
        raise FileExistsError(f"{parquet_path} exists; pass overwrite=True to replace")

    df.to_parquet(parquet_path, compression="zstd", index=False)
    with meta_path.open("w") as f:
        json.dump(meta, f, indent=2, default=str)
    return parquet_path, meta_path, meta
