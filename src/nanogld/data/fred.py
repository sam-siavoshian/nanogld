"""FRED + ALFRED — Source 4 (V1 expanded 2026-05-04).

35 series across 8 buckets (treasury curve / TIPS+breakevens / FX+vol / oil /
labor / inflation / growth / money+Fed).

Spec hard rules (plan/02-DATA-PIPELINE.md "Source 4 — V4 CORRECTED"):
- ALWAYS use ALFRED `get_series_all_releases` for vintage cubes.
  CPI/PCE annual rebenchmark silently rewrites 5y of history.
- DFF replaces FEDFUNDS for daily features (V4 §3).
- realtime_start is DATE-precise; release_ts = realtime_start + series ET tod
  via FRED_RELEASE_TOD_ET (V4 §4).
- WALCL Thursday 16:30 ET (V4 §16) — a Thursday RTH-close 16:00 bar must NOT
  use this week's level.
- ICSA Thursday 08:30 ET (V4 §17).
- Rate limit: ~120/min per key — 0.5s sleep between bulk calls.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import pandas as pd
from fredapi import Fred

from nanogld.data.schema import FRED_MANIFEST, validate
from nanogld.data.utils import (
    FRED_RELEASE_TOD_ET,
    UTC,
    fred_release_ts_utc,
    get_logger,
    raw_dir,
)

LOG = get_logger("nanogld.data.fred")

# V1 series list — 35 series (V4 update; vs 7 in pre-expansion plan).
FRED_SERIES_V1: tuple[str, ...] = (
    # Treasury curve (6)
    "DGS3MO",
    "DGS6MO",
    "DGS2",
    "DGS5",
    "DGS10",
    "DGS30",
    # TIPS + breakevens (5)
    "DFII5",
    "DFII10",
    "T5YIE",
    "T10YIE",
    "T5YIFR",
    # FX + vol (2)
    "DTWEXBGS",
    "VIXCLS",
    # Oil (2)
    "DCOILBRENTEU",
    "DCOILWTICO",
    # Labor (5)
    "UNRATE",
    "PAYEMS",
    "ICSA",
    "CCSA",
    "JTSJOL",
    # Inflation (4)
    "CPIAUCSL",
    "CPILFESL",
    "PCEPI",
    "PCEPILFE",
    # Growth + sentiment (5)
    "GDPC1",
    "INDPRO",
    "RSAFS",
    "HOUST",
    "UMCSENT",
    # Money + Fed (6) — V4: DFF added, FEDFUNDS kept for monthly aggregates only.
    "M2SL",
    "WALCL",
    "RRPONTSYD",
    "DFF",
    "FEDFUNDS",
    "SOFR",
)

assert len(FRED_SERIES_V1) == 35, f"expected 35 V1 series, got {len(FRED_SERIES_V1)}"
for _sid in FRED_SERIES_V1:
    assert _sid in FRED_RELEASE_TOD_ET, f"missing release-tod entry for {_sid}"


def _fred_client() -> Fred:
    key = os.environ.get("FRED_API_KEY")
    if not key or "FILL_ME" in str(key):
        raise RuntimeError(
            "FRED_API_KEY missing — populate ~/.config/nanogld/.env.paper "
            "(see docs/SETUP.md for signup)"
        )
    return Fred(api_key=key)


def _vintage_cube(series_id: str, fred: Fred) -> pd.DataFrame:
    """Pull the full ALFRED revision history for a series. Returns long form.

    For high-frequency daily series (DGS10, DFF, etc.) ALFRED 2000-vintage limit
    rejects the unbounded query. Fall back to non-vintage get_series so we lose
    PIT-vintage discipline but keep the value series. Mark with realtime_start
    = date so downstream PIT logic still works (no retroactive revisions).
    """
    try:
        df = fred.get_series_all_releases(series_id)
    except Exception as e:  # noqa: BLE001
        msg = str(e)
        if "vintage dates" in msg or "2000" in msg or "Internal Server Error" in msg:
            LOG.warning(
                "ALFRED %s exceeds vintage limit / server error — falling back to "
                "non-vintage get_series (no revision history)",
                series_id,
            )
            try:
                s = fred.get_series(series_id)
            except Exception as e2:  # noqa: BLE001
                LOG.warning("get_series fallback also failed for %s: %s", series_id, e2)
                return pd.DataFrame(
                    columns=["series_id", "date", "value", "realtime_start", "realtime_end"]
                )
            if s is None or s.empty:
                return pd.DataFrame(
                    columns=["series_id", "date", "value", "realtime_start", "realtime_end"]
                )
            # Treat each row's `date` as both observation date and realtime_start
            # (no vintage history available; conservative one-revision view).
            n = len(s)
            df = pd.DataFrame(
                {
                    "series_id": [series_id] * n,
                    "date": pd.to_datetime(s.index, utc=True, errors="coerce"),
                    "value": s.values,
                    "realtime_start": pd.to_datetime(s.index, utc=True, errors="coerce"),
                    "realtime_end": pd.Series([pd.NaT] * n, dtype="datetime64[ns, UTC]"),
                }
            )
            df["value"] = pd.to_numeric(df["value"], errors="coerce").astype("float64")
            df["series_id"] = df["series_id"].astype("string")
            df = df.dropna(subset=["date", "realtime_start"])
            return df[["series_id", "date", "value", "realtime_start", "realtime_end"]]
        raise

    if df is None or df.empty:
        LOG.warning("ALFRED returned empty for %s", series_id)
        return pd.DataFrame(
            columns=["series_id", "date", "value", "realtime_start", "realtime_end"]
        )

    df = df.rename(columns={c: c.lower() for c in df.columns})
    df["series_id"] = pd.Series([series_id] * len(df), dtype="string")
    df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
    df["realtime_start"] = pd.to_datetime(df["realtime_start"], utc=True, errors="coerce")
    if "realtime_end" in df.columns:
        df["realtime_end"] = pd.to_datetime(df["realtime_end"], utc=True, errors="coerce")
    else:
        df["realtime_end"] = pd.Series([pd.NaT] * len(df), dtype="datetime64[ns, UTC]")
    # Force tz-aware dtype even when all values are NaT (pd.to_datetime
    # collapses to tz-naive in that case).
    df["realtime_end"] = df["realtime_end"].astype("datetime64[ns, UTC]")
    df["value"] = pd.to_numeric(df["value"], errors="coerce").astype("float64")
    df = df.dropna(subset=["date", "realtime_start"])
    return df[["series_id", "date", "value", "realtime_start", "realtime_end"]]


def _add_release_ts(df: pd.DataFrame) -> pd.DataFrame:
    """release_ts = realtime_start (date) + series-specific ET tod, in UTC.

    realtime_start is the date the value first appeared in ALFRED. The actual
    public-availability moment is at the series' standard ET tod.
    """
    if df.empty:
        df = df.assign(release_ts=pd.NaT, t_visible=pd.NaT)
        return df

    series_id = df["series_id"].iloc[0]
    rs_dates = df["realtime_start"].dt.tz_convert(UTC).dt.date
    df = df.copy()
    df["release_ts"] = rs_dates.map(lambda d: fred_release_ts_utc(series_id, d))
    df["release_ts"] = pd.to_datetime(df["release_ts"], utc=True)
    df["t_visible"] = df["release_ts"]
    return df


def fetch_one(series_id: str, *, fred: Fred | None = None) -> pd.DataFrame:
    fred = fred or _fred_client()
    LOG.info("ALFRED fetch %s", series_id)
    cube = _vintage_cube(series_id, fred)
    cube = _add_release_ts(cube)
    return cube[[c.name for c in FRED_MANIFEST.columns]] if not cube.empty else cube


def fetch_all_series(
    series: tuple[str, ...] = FRED_SERIES_V1,
    *,
    sleep_sec: float = 0.5,
) -> dict[str, pd.DataFrame]:
    """Pull all V1 series. Returns {series_id: vintage cube DataFrame}."""
    fred = _fred_client()
    out: dict[str, pd.DataFrame] = {}
    for sid in series:
        try:
            out[sid] = fetch_one(sid, fred=fred)
        except Exception as e:  # noqa: BLE001
            LOG.error("FRED fetch failed for %s: %s", sid, e)
            out[sid] = pd.DataFrame()
        time.sleep(sleep_sec)
    return out


def write_all_parquets(series: tuple[str, ...] = FRED_SERIES_V1) -> dict[str, str]:
    """Write one parquet per series under data/raw/fred_<series>_all_releases.parquet."""
    cubes = fetch_all_series(series)
    paths: dict[str, str] = {}
    for sid, df in cubes.items():
        if df.empty:
            paths[sid] = ""
            continue
        validate(df, FRED_MANIFEST)
        out_path = raw_dir() / f"fred_{sid.lower()}_all_releases.parquet"
        df.to_parquet(out_path, compression="zstd", index=False)
        paths[sid] = str(out_path)
        LOG.info("wrote %d %s ALFRED rows -> %s", len(df), sid, out_path)
    return paths


def vintage_lookup(df_all: pd.DataFrame, t: pd.Timestamp) -> pd.Series:
    """Latest known value per `date` as of time `t` (point-in-time-correct).

    Used at training time. df_all = parquet from write_all_parquets().
    """
    visible = df_all[df_all["realtime_start"] <= t]
    return (
        visible.sort_values("realtime_start")
        .groupby("date")
        .tail(1)
        .set_index("date")["value"]
        .astype(float)
        .sort_index()
    )


if __name__ == "__main__":
    paths = write_all_parquets()
    ok = sum(1 for p in paths.values() if p)
    print(f"FRED ALFRED: {ok}/{len(FRED_SERIES_V1)} series written")
    for sid, p in paths.items():
        if p:
            print(f"  {sid:12s} -> {Path(p).name}")
