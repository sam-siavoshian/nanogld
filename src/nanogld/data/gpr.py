"""Caldara-Iacoviello Geopolitical Risk Index — Source 6.

Two files, both self-snapshotted weekly (no public vintage archive):
  - GPR monthly time series (Excel) — 1900+, ~3-7 day lag into new month.
  - AI-GPR daily (CSV) — claims daily but ~30-day lag in practice.

Spec: plan/02-DATA-PIPELINE.md "Source 6 — V4 CORRECTED" + Open Question
about AI-GPR lag.

Hard rule: methodology revisions silently rewrite history. Self-snapshot
weekly with fetch-date keying so backtests use the vintage that existed
at decision time, not today's revised history.
"""

from __future__ import annotations

import hashlib
from datetime import date, datetime
from pathlib import Path

import pandas as pd

from nanogld.data.schema import GPR_MANIFEST, validate
from nanogld.data.utils import UTC, get_logger, http_get_bytes, raw_dir

LOG = get_logger("nanogld.data.gpr")

GPR_MONTHLY_URL = "https://www.matteoiacoviello.com/gpr_files/data_gpr_export.xls"
AIGPR_DAILY_URL = "https://www.matteoiacoviello.com/ai_gpr_files/ai_gpr_data_daily.csv"

AIGPR_LAG_DAYS = 30  # V4 verified empirical lag


def _gpr_cache_dir() -> Path:
    p = raw_dir() / "gpr"
    p.mkdir(parents=True, exist_ok=True)
    return p


def fetch_and_snapshot() -> dict[str, dict[str, str]]:
    """Pull both files, store raw bytes keyed by fetch_ts + sha."""
    fetch_ts = datetime.now(tz=UTC).strftime("%Y-%m-%dT%H-%M-%SZ")
    out: dict[str, dict[str, str]] = {}
    for name, url in [("monthly", GPR_MONTHLY_URL), ("aigpr_daily", AIGPR_DAILY_URL)]:
        try:
            body = http_get_bytes(url)
        except Exception as e:  # noqa: BLE001
            LOG.warning("GPR fetch failed (%s): %s", name, e)
            continue
        sha = hashlib.sha256(body).hexdigest()[:16]
        suffix = ".xls" if name == "monthly" else ".csv"
        path = _gpr_cache_dir() / f"{name}_{fetch_ts}_{sha}{suffix}"
        path.write_bytes(body)
        out[name] = {
            "path": str(path),
            "sha": sha,
            "fetch_ts": fetch_ts,
            "size_kb": str(len(body) // 1024),
        }
        LOG.info(
            "snapshotted GPR %s -> %s (%s KB, sha=%s)", name, path.name, len(body) // 1024, sha
        )
    return out


def _monthly_release_ts_utc(month_start: pd.Timestamp) -> pd.Timestamp:
    """7th of the following month, 12:00 UTC."""
    next_month = month_start + pd.DateOffset(months=1)
    target = date(next_month.year, next_month.month, 7)
    return pd.Timestamp(datetime.combine(target, datetime.min.time()), tz=UTC) + pd.Timedelta(
        hours=12
    )


def _aigpr_release_ts_utc(observation_date: pd.Timestamp) -> pd.Timestamp:
    """AI-GPR has ~30-day lag. Visibility = observation_date + 30 days at 12:00 UTC."""
    rel = observation_date.tz_convert(UTC) + pd.Timedelta(days=AIGPR_LAG_DAYS)
    return rel.normalize() + pd.Timedelta(hours=12)


def parse_monthly_xls(path: Path) -> pd.DataFrame:
    """Parse Caldara-Iacoviello GPR monthly Excel.

    Layout: Sheet 0 has a date column ('month' or first column) and 100+
    series columns. We extract (series, date, value) long form.
    """
    try:
        df = pd.read_excel(path, sheet_name=0, engine="xlrd")
    except Exception as e:  # noqa: BLE001
        LOG.warning("GPR monthly parse failed (%s): %s", path.name, e)
        return pd.DataFrame()

    # First column is the date (label varies: "month", "Month", "DATE", ...)
    date_col = df.columns[0]
    df = df.rename(columns={date_col: "date"})
    df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True)
    df = df.dropna(subset=["date"])

    long = df.melt(id_vars=["date"], var_name="series", value_name="value")
    long = long.dropna(subset=["value"])
    long["value"] = pd.to_numeric(long["value"], errors="coerce")
    long = long.dropna(subset=["value"])
    long["series"] = long["series"].astype(str)
    return long


def parse_aigpr_daily_csv(path: Path) -> pd.DataFrame:
    """Parse AI-GPR daily CSV. Layout: date, gpr_value (and optional companions)."""
    try:
        df = pd.read_csv(path)
    except Exception as e:  # noqa: BLE001
        LOG.warning("AI-GPR daily parse failed (%s): %s", path.name, e)
        return pd.DataFrame()

    date_col = next((c for c in df.columns if c.lower() in {"date", "day"}), df.columns[0])
    df = df.rename(columns={date_col: "date"})
    df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True)
    df = df.dropna(subset=["date"])

    long = df.melt(id_vars=["date"], var_name="series", value_name="value")
    long = long.dropna(subset=["value"])
    long["value"] = pd.to_numeric(long["value"], errors="coerce")
    long = long.dropna(subset=["value"])
    long["series"] = "AIGPR_DAILY_" + long["series"].astype(str)
    return long


def build_gpr_dataframe(snap: dict[str, dict[str, str]]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []

    if "monthly" in snap:
        info = snap["monthly"]
        m = parse_monthly_xls(Path(info["path"]))
        if not m.empty:
            m["fetch_ts"] = pd.Timestamp(info["fetch_ts"], tz=UTC)
            m["source_sha"] = info["sha"]
            m["release_ts"] = m["date"].apply(_monthly_release_ts_utc)
            m["t_visible"] = m["release_ts"]
            frames.append(m)

    if "aigpr_daily" in snap:
        info = snap["aigpr_daily"]
        a = parse_aigpr_daily_csv(Path(info["path"]))
        if not a.empty:
            a["fetch_ts"] = pd.Timestamp(info["fetch_ts"], tz=UTC)
            a["source_sha"] = info["sha"]
            a["release_ts"] = a["date"].apply(_aigpr_release_ts_utc)
            a["t_visible"] = a["release_ts"]
            frames.append(a)

    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    df["series"] = df["series"].astype("string")
    df["source_sha"] = df["source_sha"].astype("string")
    df["value"] = df["value"].astype("float64")
    return df[[c.name for c in GPR_MANIFEST.columns]]


def write_gpr_parquet() -> tuple[pd.DataFrame, str]:
    snap = fetch_and_snapshot()
    df = build_gpr_dataframe(snap)
    if df.empty:
        LOG.warning("GPR parse produced 0 rows; raw snapshots retained")
        return df, ""
    validate(df, GPR_MANIFEST)
    out_path = raw_dir() / "gpr_combined.parquet"
    df.to_parquet(out_path, compression="zstd", index=False)
    LOG.info("wrote %d GPR rows to %s (series: %s)", len(df), out_path, df["series"].nunique())
    return df, str(out_path)


if __name__ == "__main__":
    df, path = write_gpr_parquet()
    if path:
        print(f"wrote {len(df)} rows to {path}")
        print(df.groupby("series").size().head(10))
    else:
        print("GPR parse incomplete — raw snapshots saved under data/raw/gpr/")
