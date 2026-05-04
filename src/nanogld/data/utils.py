"""Shared helpers for the data pipeline.

Point-in-time discipline: every feature row carries `t_visible: pd.Timestamp`
(earliest moment publicly available). Joins use strict `<` against the bar
prediction time. The `t_visible_invariant` helper asserts the contract.

Spec: plan/02-DATA-PIPELINE.md "Hard Rule (V1)".
"""

from __future__ import annotations

import logging
import logging.handlers
import os
import time
from datetime import UTC, date, datetime, timedelta
from datetime import time as dt_time
from pathlib import Path
from typing import Final
from zoneinfo import ZoneInfo

import pandas as pd
import pandas_market_calendars as mcal
import requests

# ────────────────────────────────────────────────────────────────────────────
# Project-wide constants
# ────────────────────────────────────────────────────────────────────────────

ET: Final = ZoneInfo("America/New_York")
LONDON: Final = ZoneInfo("Europe/London")

# 5y window (matches plan/02-DATA-PIPELINE.md "Source 1" defaults)
START_DATE_UTC: Final = datetime(2021, 4, 24, tzinfo=UTC)
END_DATE_UTC: Final = datetime(2026, 4, 24, tzinfo=UTC)

# Naive variants for SDKs (Alpaca/yfinance) that want tz-unaware defaults.
START_DATE_NAIVE: Final = START_DATE_UTC.replace(tzinfo=None)
END_DATE_NAIVE: Final = END_DATE_UTC.replace(tzinfo=None)

# Bar interval = 30 minutes RTH. 13 bars per RTH session; ~3276 per year.
BAR_INTERVAL_MIN: Final = 30

# Publication latency buffers (per V4 verification round)
NEWS_LATENCY_SEC_ALPACA: Final = 60  # Alpaca News → wire-clock skew safety
NEWS_LATENCY_MIN_GDELT: Final = 30  # GDELT slot ingestion lag (V4 update from 15 → 30)

# Project paths (caller may override DATA_ROOT_ENV)
DATA_ROOT_ENV: Final = "NANOGLD_DATA_ROOT"
_PROJECT_ROOT: Final = Path(__file__).resolve().parents[3]
DEFAULT_DATA_ROOT: Final = _PROJECT_ROOT / "data"


def data_root() -> Path:
    """Return the data root, honoring the env override."""
    return Path(os.environ.get(DATA_ROOT_ENV, str(DEFAULT_DATA_ROOT)))


def raw_dir() -> Path:
    p = data_root() / "raw"
    p.mkdir(parents=True, exist_ok=True)
    return p


def snapshots_dir() -> Path:
    p = data_root() / "snapshots"
    p.mkdir(parents=True, exist_ok=True)
    return p


# ────────────────────────────────────────────────────────────────────────────
# FRED release-time-of-day table (V4 hard rule §4)
#
# realtime_start in ALFRED is DATE-PRECISE. Need static release-time-of-day
# lookup so a bar at 07:00 ET on date D does NOT use a series that publishes
# at 08:30 ET on date D. Times below are sourced from official BLS / BEA /
# Fed / NY Fed / EIA release schedules.
# ────────────────────────────────────────────────────────────────────────────

FRED_RELEASE_TOD_ET: Final[dict[str, dt_time]] = {
    # Treasury curve + TIPS + breakevens — Fed H.15 daily ~4:15 PM ET
    "DGS3MO": dt_time(16, 15),
    "DGS6MO": dt_time(16, 15),
    "DGS2": dt_time(16, 15),
    "DGS5": dt_time(16, 15),
    "DGS10": dt_time(16, 15),
    "DGS30": dt_time(16, 15),
    "DFII5": dt_time(16, 15),
    "DFII10": dt_time(16, 15),
    "T5YIE": dt_time(16, 15),
    "T10YIE": dt_time(16, 15),
    "T5YIFR": dt_time(16, 15),
    # FX + vol
    "DTWEXBGS": dt_time(16, 15),  # H.10 daily
    "VIXCLS": dt_time(8, 37),  # CBOE close ingested next morning by FRED
    # Oil
    "DCOILBRENTEU": dt_time(16, 0),  # EIA daily spot
    "DCOILWTICO": dt_time(16, 0),
    # Labor
    "UNRATE": dt_time(8, 30),  # BLS Employment Situation (1st Friday)
    "PAYEMS": dt_time(8, 30),
    "ICSA": dt_time(8, 30),  # DOL weekly Thursday
    "CCSA": dt_time(8, 30),
    "JTSJOL": dt_time(10, 0),  # BLS JOLTS
    # Inflation
    "CPIAUCSL": dt_time(8, 30),  # BLS CPI mid-month
    "CPILFESL": dt_time(8, 30),
    "PCEPI": dt_time(8, 30),  # BEA Personal Income end-of-month
    "PCEPILFE": dt_time(8, 30),
    # Growth + sentiment
    "GDPC1": dt_time(8, 30),  # BEA GDP, quarterly
    "INDPRO": dt_time(9, 15),  # Fed G.17 mid-month
    "RSAFS": dt_time(8, 30),  # Census mid-month
    "HOUST": dt_time(8, 30),
    "UMCSENT": dt_time(10, 0),  # UMich
    # Money + Fed
    "M2SL": dt_time(13, 0),  # Fed H.6 ~1:00 PM ET, 4th Tuesday
    "WALCL": dt_time(16, 30),  # Fed H.4.1, Thursday — critical for Thursday close bars
    "RRPONTSYD": dt_time(13, 30),  # NY Fed TOMO
    "DFF": dt_time(9, 0),  # H.15 next-business-day
    "FEDFUNDS": dt_time(9, 0),  # Monthly aggregate, next 1st BD
    "SOFR": dt_time(8, 0),  # NY Fed previous-BD
}


def fred_release_ts_utc(series_id: str, observation_date: date) -> pd.Timestamp:
    """Return UTC release timestamp for a FRED observation.

    realtime_start is the date the value first appeared in ALFRED. The
    actual public-availability moment is `realtime_start` at the series'
    standard ET time-of-day, converted to UTC.
    """
    if series_id not in FRED_RELEASE_TOD_ET:
        raise KeyError(f"FRED series {series_id!r} missing release-time entry")
    ts_et = datetime.combine(observation_date, FRED_RELEASE_TOD_ET[series_id], tzinfo=ET)
    return pd.Timestamp(ts_et).tz_convert(UTC)


# ────────────────────────────────────────────────────────────────────────────
# NYSE calendar helpers
# ────────────────────────────────────────────────────────────────────────────


def nyse_calendar() -> mcal.MarketCalendar:
    return mcal.get_calendar("NYSE")


def nyse_rth_index(start: pd.Timestamp, end: pd.Timestamp, freq: str = "30min") -> pd.DatetimeIndex:
    """RTH bar-end timestamps in UTC. 13 bars/day for 30min."""
    cal = nyse_calendar()
    schedule = cal.schedule(start_date=start.date(), end_date=end.date())
    rng = mcal.date_range(schedule, frequency=freq, force_close=True)
    return rng.tz_convert(UTC) if rng.tz is not None else rng.tz_localize(UTC)


def is_nyse_session_day(d: date) -> bool:
    cal = nyse_calendar()
    sched = cal.schedule(start_date=d, end_date=d)
    return not sched.empty


def next_nyse_session_day(d: date) -> date:
    cal = nyse_calendar()
    sched = cal.valid_days(start_date=d, end_date=d + timedelta(days=10))
    if sched.empty:
        raise RuntimeError(f"No NYSE session within 10 days of {d}")
    return sched[0].date()


def cot_release_ts_utc(report_date_tuesday: date) -> pd.Timestamp:
    """COT report observed on Tuesday is published the following Friday 3:30 PM ET.

    Holiday-Friday rolls to the next NYSE session day. We add a 30-min safety
    buffer (so 16:00 ET = 20:00/21:00 UTC depending on DST). Returns 16:00 ET
    on the release session, in UTC.
    """
    target_friday = report_date_tuesday + timedelta(days=3)
    if not is_nyse_session_day(target_friday):
        target_friday = next_nyse_session_day(target_friday)
    ts_et = datetime.combine(target_friday, dt_time(16, 0), tzinfo=ET)  # 15:30 + 30min buffer
    return pd.Timestamp(ts_et).tz_convert(UTC)


# ────────────────────────────────────────────────────────────────────────────
# Point-in-time invariant
# ────────────────────────────────────────────────────────────────────────────


def assert_t_visible_invariant(df: pd.DataFrame, *, release_col: str = "release_ts") -> None:
    """Every row's `release_ts <= t_visible`. Raises if any row violates."""
    if "t_visible" not in df.columns:
        raise ValueError("dataframe missing 't_visible' column")
    if release_col not in df.columns:
        raise ValueError(f"dataframe missing {release_col!r} column")
    bad = df.loc[df[release_col].notna() & (df[release_col] > df["t_visible"])]
    if not bad.empty:
        raise ValueError(
            f"{len(bad)} rows violate release_ts <= t_visible. First offender:\n{bad.head(1)}"
        )


def merge_asof_pit(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    left_on: str,
    right_on: str = "t_visible",
    by: str | list[str] | None = None,
    suffixes: tuple[str, str] = ("", "_r"),
) -> pd.DataFrame:
    """Strict-< asof merge: a feature is only joined into bar T if its
    `t_visible < T`. No exact matches (closes the on-the-edge leak).
    """
    left_sorted = left.sort_values(left_on)
    right_sorted = right.sort_values(right_on)
    return pd.merge_asof(
        left_sorted,
        right_sorted,
        left_on=left_on,
        right_on=right_on,
        by=by,
        direction="backward",
        allow_exact_matches=False,
        suffixes=suffixes,
    )


# ────────────────────────────────────────────────────────────────────────────
# HTTP helpers (vintage snapshotting)
# ────────────────────────────────────────────────────────────────────────────


class FetchError(RuntimeError):
    """Network fetch failed after retries."""


def http_get_bytes(url: str, *, timeout: int = 60, max_retries: int = 3) -> bytes:
    """GET with retry + exponential backoff. Returns response body."""
    last: Exception | None = None
    for attempt in range(max_retries):
        try:
            resp = requests.get(url, timeout=timeout, headers={"User-Agent": "nanoGLD/0.1"})
            resp.raise_for_status()
            return resp.content
        except Exception as e:
            last = e
            time.sleep(2**attempt)
    raise FetchError(f"GET {url} failed after {max_retries} retries: {last}")


# ────────────────────────────────────────────────────────────────────────────
# Logging
# ────────────────────────────────────────────────────────────────────────────


def get_logger(name: str = "nanogld.data") -> logging.Logger:
    """Rotating in-process logger (10 MB × 14 backups). Avoids the macOS
    newsyslog root-permission trap that the spec calls out.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(name)s | %(levelname)s | %(message)s")

    log_dir = data_root().parent / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    fh = logging.handlers.RotatingFileHandler(
        log_dir / f"{name.replace('.', '_')}.log",
        maxBytes=10_000_000,
        backupCount=14,
    )
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    return logger
