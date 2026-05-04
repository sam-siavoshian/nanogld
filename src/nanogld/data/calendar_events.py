"""Deterministic macro release calendar — Source 11 of plan/02-DATA-PIPELINE.md.

NO API. Builds the schedule of high-impact macro releases (FOMC, CPI, NFP,
GDP, JOLTS, PCE, FOMC minutes) for 2021-2026 from official Fed / BLS / BEA
calendars. Doc 04 consumes this to build event-proximity features (V1 hard
rule §14: binary windows ONLY, no `minutes_until_event`).

Verification: spec line 1054 — owner should /browse-verify FOMC against
federalreserve.gov before training. Dates below are best-effort against
the published schedule; emergency meetings (e.g. March 2020 cut) are NOT
included.
"""

from __future__ import annotations

import calendar
from datetime import date, datetime, timedelta
from datetime import time as dt_time

import pandas as pd

from nanogld.data.schema import CALENDAR_MANIFEST, validate
from nanogld.data.utils import ET, UTC, raw_dir

# ────────────────────────────────────────────────────────────────────────────
# FOMC scheduled meetings — federalreserve.gov/monetarypolicy/fomccalendars.htm
#
# Each tuple is (year, month, day_of_decision). Decision is announced 14:00 ET,
# press conference ~14:30 ET. We use 14:00 ET single event for V1.
# ────────────────────────────────────────────────────────────────────────────

_FOMC_SCHEDULE: dict[int, list[tuple[int, int]]] = {
    2021: [(1, 27), (3, 17), (4, 28), (6, 16), (7, 28), (9, 22), (11, 3), (12, 15)],
    2022: [(1, 26), (3, 16), (5, 4), (6, 15), (7, 27), (9, 21), (11, 2), (12, 14)],
    2023: [(2, 1), (3, 22), (5, 3), (6, 14), (7, 26), (9, 20), (11, 1), (12, 13)],
    2024: [(1, 31), (3, 20), (5, 1), (6, 12), (7, 31), (9, 18), (11, 7), (12, 18)],
    2025: [(1, 29), (3, 19), (5, 7), (6, 18), (7, 30), (9, 17), (10, 29), (12, 10)],
    2026: [(1, 28), (3, 18), (4, 29), (6, 17), (7, 29), (9, 16), (10, 28), (12, 16)],
}


def _et_to_utc(d: date, tod: dt_time) -> pd.Timestamp:
    return pd.Timestamp(datetime.combine(d, tod, tzinfo=ET)).tz_convert(UTC)


def _first_friday(year: int, month: int) -> date:
    cal = calendar.monthcalendar(year, month)
    for week in cal:
        if week[calendar.FRIDAY]:
            return date(year, month, week[calendar.FRIDAY])
    raise RuntimeError(f"no Friday in {year}-{month:02d}")


def _last_business_day(year: int, month: int) -> date:
    last = date(year, month, calendar.monthrange(year, month)[1])
    while last.weekday() >= 5:
        last -= timedelta(days=1)
    return last


def _last_thursday(year: int, month: int) -> date:
    last = date(year, month, calendar.monthrange(year, month)[1])
    delta = (last.weekday() - calendar.THURSDAY) % 7
    return last - timedelta(days=delta)


def _gdp_advance_release_dates(year_start: int, year_end: int) -> list[date]:
    """GDP advance lands ~last Thursday of the month after each quarter end."""
    out: list[date] = []
    for y in range(year_start, year_end + 1):
        for m in (1, 4, 7, 10):
            out.append(_last_thursday(y, m))
    return out


def _fomc_minutes_dates(year_start: int, year_end: int) -> list[date]:
    """Minutes published ~3 weeks after each FOMC meeting at 14:00 ET."""
    out: list[date] = []
    for y, schedule in _FOMC_SCHEDULE.items():
        if not (year_start <= y <= year_end):
            continue
        for m, d in schedule:
            out.append(date(y, m, d) + timedelta(weeks=3))
    return out


def _approx_dates(year_start: int, year_end: int, *, day_of_month: int) -> list[date]:
    out: list[date] = []
    for y in range(year_start, year_end + 1):
        for m in range(1, 13):
            d = date(y, m, day_of_month)
            while d.weekday() >= 5:
                d += timedelta(days=1)
            out.append(d)
    return out


# ────────────────────────────────────────────────────────────────────────────
# Event-builder API
# ────────────────────────────────────────────────────────────────────────────


def build_calendar(year_start: int = 2021, year_end: int = 2026) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    # FOMC decisions — 14:00 ET, tier 1
    for y, sched in _FOMC_SCHEDULE.items():
        if not (year_start <= y <= year_end):
            continue
        for m, d in sched:
            rows.append(
                {
                    "event_type": "FOMC",
                    "event_ts_utc": _et_to_utc(date(y, m, d), dt_time(14, 0)),
                    "tier": 1,
                }
            )

    # FOMC minutes — 3w after each meeting, 14:00 ET, tier 2
    for d in _fomc_minutes_dates(year_start, year_end):
        rows.append(
            {
                "event_type": "FOMC_minutes",
                "event_ts_utc": _et_to_utc(d, dt_time(14, 0)),
                "tier": 2,
            }
        )

    # NFP — 1st Friday, 08:30 ET, tier 1
    for y in range(year_start, year_end + 1):
        for m in range(1, 13):
            rows.append(
                {
                    "event_type": "NFP",
                    "event_ts_utc": _et_to_utc(_first_friday(y, m), dt_time(8, 30)),
                    "tier": 1,
                }
            )

    # CPI — ~12th of month (BLS variable Tue 10-14), 08:30 ET, tier 1
    # Owner should /browse-verify exact CPI dates before training; doc 02 line 1054.
    for d in _approx_dates(year_start, year_end, day_of_month=12):
        rows.append(
            {
                "event_type": "CPI",
                "event_ts_utc": _et_to_utc(d, dt_time(8, 30)),
                "tier": 1,
            }
        )

    # GDP advance — last Thu of Apr/Jul/Oct/Jan, 08:30 ET, tier 1
    for d in _gdp_advance_release_dates(year_start, year_end):
        rows.append(
            {
                "event_type": "GDP",
                "event_ts_utc": _et_to_utc(d, dt_time(8, 30)),
                "tier": 1,
            }
        )

    # JOLTS — ~9th of month, 10:00 ET, tier 2
    for d in _approx_dates(year_start, year_end, day_of_month=9):
        rows.append(
            {
                "event_type": "JOLTS",
                "event_ts_utc": _et_to_utc(d, dt_time(10, 0)),
                "tier": 2,
            }
        )

    # PCE — last BD of month, 08:30 ET, tier 1 (Fed's preferred inflation gauge)
    for y in range(year_start, year_end + 1):
        for m in range(1, 13):
            rows.append(
                {
                    "event_type": "PCE",
                    "event_ts_utc": _et_to_utc(_last_business_day(y, m), dt_time(8, 30)),
                    "tier": 1,
                }
            )

    df = pd.DataFrame(rows).sort_values("event_ts_utc").reset_index(drop=True)
    df["event_type"] = df["event_type"].astype("string")
    df["tier"] = df["tier"].astype("int64")
    df["event_ts_utc"] = pd.to_datetime(df["event_ts_utc"], utc=True)

    # Calendar SCHEDULES are public ~12 months ahead (Fed posts annual FOMC
    # schedule ~year out; BLS/BEA the same). t_visible is the announce-time,
    # NOT the event-time. Conservative: event_ts - 365 days.
    df["release_ts"] = df["event_ts_utc"] - pd.Timedelta(days=365)
    df["t_visible"] = df["release_ts"]
    return df


def write_calendar_parquet(
    year_start: int = 2021, year_end: int = 2026
) -> tuple[pd.DataFrame, str]:
    """Build + persist the deterministic calendar. Returns (df, parquet_path)."""
    df = build_calendar(year_start, year_end)
    validate(df, CALENDAR_MANIFEST)
    out_path = raw_dir() / "calendar_events_v1.parquet"
    df.to_parquet(out_path, compression="zstd", index=False)
    return df, str(out_path)


if __name__ == "__main__":
    df, path = write_calendar_parquet()
    print(f"wrote {len(df)} events to {path}")
    print(df.groupby("event_type").size())
