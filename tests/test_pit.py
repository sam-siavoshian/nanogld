"""Golden-fixture tests for point-in-time correctness.

Spec: plan/02-DATA-PIPELINE.md "Golden Fixture Test (NON-NEGOTIABLE)".

The single most important test in the project. Hand-crafted dataset where
the right answer is known up-front. If this fails, the joiner is broken
and every downstream metric is suspect.
"""

from __future__ import annotations

import pandas as pd
import pytest

from nanogld.data.join import join_snapshot
from nanogld.data.utils import (
    NEWS_LATENCY_SEC_ALPACA,
    UTC,
    assert_t_visible_invariant,
    cot_release_ts_utc,
    fred_release_ts_utc,
    merge_asof_pit,
)


def _ts(s: str) -> pd.Timestamp:
    return pd.Timestamp(s, tz=UTC)


def _bars_fixture() -> pd.DataFrame:
    """3 GLD bars: 14:00, 14:30, 15:00 ET (= 18:00, 18:30, 19:00 UTC for non-DST)."""
    rows = [
        {
            "symbol": "GLD",
            "timestamp": _ts("2024-01-15 14:00:00"),
            "open": 100.0,
            "high": 100.5,
            "low": 99.8,
            "close": 100.0,
            "volume": 1000.0,
            "trade_count": 100.0,
            "vwap": 100.05,
        },
        {
            "symbol": "GLD",
            "timestamp": _ts("2024-01-15 14:30:00"),
            "open": 100.0,
            "high": 100.7,
            "low": 100.0,
            "close": 100.5,
            "volume": 1100.0,
            "trade_count": 110.0,
            "vwap": 100.4,
        },
        {
            "symbol": "GLD",
            "timestamp": _ts("2024-01-15 15:00:00"),
            "open": 100.5,
            "high": 100.6,
            "low": 100.2,
            "close": 100.3,
            "volume": 1200.0,
            "trade_count": 120.0,
            "vwap": 100.45,
        },
    ]
    df = pd.DataFrame(rows)
    df["symbol"] = df["symbol"].astype("string")
    df["release_ts"] = df["timestamp"] + pd.Timedelta(minutes=30)
    df["t_visible"] = df["release_ts"]
    return df


# ────────────────────────────────────────────────────────────────────────────
# Hard-rule sanity
# ────────────────────────────────────────────────────────────────────────────


def test_assert_t_visible_invariant_passes_on_clean_data() -> None:
    df = _bars_fixture()
    assert_t_visible_invariant(df)  # should not raise


def test_assert_t_visible_invariant_raises_on_violation() -> None:
    df = _bars_fixture().copy()
    df.loc[0, "release_ts"] = df.loc[0, "t_visible"] + pd.Timedelta(seconds=1)
    with pytest.raises(ValueError, match="release_ts <= t_visible"):
        assert_t_visible_invariant(df)


def test_merge_asof_pit_strict_lt() -> None:
    bars = _bars_fixture()
    sig = pd.DataFrame(
        {
            "t_visible": [_ts("2024-01-15 14:30:00")],
            "value": [42.0],
        }
    )
    out = merge_asof_pit(bars, sig, left_on="t_visible", right_on="t_visible")
    bar2_close = _ts("2024-01-15 15:00:00")
    bar1_close = _ts("2024-01-15 14:30:00")
    bar2_value = out.loc[out["timestamp"] == _ts("2024-01-15 14:30:00"), "value"].iloc[0]
    bar1_value = out.loc[out["timestamp"] == _ts("2024-01-15 14:00:00"), "value"].iloc[0]
    # bar with timestamp=14:00 has t_visible=14:30; signal at t_visible=14:30 is NOT
    # joined (strict <), so bar1 must be NaN and bar2 (t_visible=15:00) should see 42.
    _ = bar1_close, bar2_close
    assert pd.isna(bar1_value)
    assert bar2_value == 42.0


# ────────────────────────────────────────────────────────────────────────────
# Source-specific release-ts builders
# ────────────────────────────────────────────────────────────────────────────


def test_cot_release_ts_friday_after_tuesday() -> None:
    """COT for Tuesday 2024-01-09 publishes Friday 2024-01-12 (NYSE session)."""
    from datetime import date

    ts = cot_release_ts_utc(date(2024, 1, 9))
    # Friday 16:00 ET = 21:00 UTC in winter (EST = UTC-5)
    assert ts.weekday() == 4
    assert ts.tz_convert("America/New_York").hour == 16


def test_fred_release_ts_dgs10_4pm_et() -> None:
    """DGS10 publishes 16:15 ET — the spec table value."""
    from datetime import date

    ts = fred_release_ts_utc("DGS10", date(2024, 1, 15))
    et = ts.tz_convert("America/New_York")
    assert et.hour == 16 and et.minute == 15


def test_fred_release_ts_dff_replaces_fedfunds_for_daily() -> None:
    """V4 §3: DFF is the daily series. FEDFUNDS is monthly, posted next BD ~9 AM ET."""
    from datetime import date

    dff = fred_release_ts_utc("DFF", date(2024, 1, 15))
    ff = fred_release_ts_utc("FEDFUNDS", date(2024, 1, 15))
    # Both have entries — distinct ROLE not distinct time. Test that table is complete.
    assert dff.tz_convert("America/New_York").hour == 9
    assert ff.tz_convert("America/New_York").hour == 9


# ────────────────────────────────────────────────────────────────────────────
# End-to-end: joiner respects PIT
# ────────────────────────────────────────────────────────────────────────────


def test_joiner_no_self_visibility() -> None:
    """The bar at timestamp T (release_ts T+30min) MUST NOT appear in the
    joined row at bar_close_utc T+30min — bar visibility = bar END (strict <).
    """
    bars = _bars_fixture()
    out = join_snapshot({"bars": bars})
    # bar_close 14:30 is the close of the FIRST bar (timestamp 14:00).
    # gld_lag1_close at this row should be NaN (no bars with t_visible < 14:30).
    row = out.iloc[0]
    assert row["bar_close_utc"] == _ts("2024-01-15 14:30:00")
    assert pd.isna(row.get("gld_lag1_close"))


def test_joiner_lagged_bar_visible_one_step_later() -> None:
    """At bar_close 15:00, the bar with timestamp 14:00 (visible at 14:30) IS in scope."""
    bars = _bars_fixture()
    out = join_snapshot({"bars": bars})
    row_15_00 = out[out["bar_close_utc"] == _ts("2024-01-15 15:00:00")].iloc[0]
    assert row_15_00["gld_lag1_close"] == 100.0


def test_news_count_window() -> None:
    """News with t_visible in (prev_close, close] is counted in this bar."""
    bars = _bars_fixture()
    news = pd.DataFrame(
        {
            "article_id": ["a", "b"],
            "source": ["alpaca_benzinga", "alpaca_benzinga"],
            "created_at": [
                _ts("2024-01-15 14:25:00"),  # before any bar close — counted in 14:30 bar
                _ts("2024-01-15 14:40:00"),  # in (14:30, 15:00] — counted in 15:00 bar
            ],
            "title": ["EARLY", "MID"],
            "body": [None, None],
            "url": [None, None],
            "symbols": ["GLD", "GLD"],
            "bias_tier": ["mainstream_neutral", "mainstream_neutral"],
        }
    )
    news["release_ts"] = news["created_at"]
    news["t_visible"] = news["created_at"] + pd.Timedelta(seconds=NEWS_LATENCY_SEC_ALPACA)
    for c in ("article_id", "source", "title", "body", "url", "symbols", "bias_tier"):
        news[c] = news[c].astype("string")

    out = join_snapshot({"bars": bars, "alpaca_news": news})
    row_14_30 = out[out["bar_close_utc"] == _ts("2024-01-15 14:30:00")].iloc[0]
    row_15_00 = out[out["bar_close_utc"] == _ts("2024-01-15 15:00:00")].iloc[0]
    # 14:25 + 60s = 14:26:00 < 14:30 → counted at 14:30 close
    assert row_14_30["alpaca_news_count"] == 1
    # 14:40 + 60s = 14:41:00 falls in (14:30, 15:00] → counted at 15:00 close
    assert row_15_00["alpaca_news_count"] == 1


def test_calendar_event_proximity_binary() -> None:
    """V1 hard rule §14: binary windows ONLY, no minutes_until_event."""
    bars = _bars_fixture()
    cal = pd.DataFrame(
        {
            "event_type": ["FOMC"],
            "event_ts_utc": [_ts("2024-01-15 14:30:00")],
            "tier": [1],
            "release_ts": [_ts("2024-01-15 14:30:00")],
            "t_visible": [_ts("2024-01-15 14:30:00")],
        }
    )
    cal["event_type"] = cal["event_type"].astype("string")
    cal["tier"] = cal["tier"].astype("int64")

    out = join_snapshot({"bars": bars, "calendar": cal})
    # All 3 bars (14:30, 15:00, 15:30) are within the [event-30min, event+60min] window
    # for the 14:30 event, so all should flag True.
    assert out["event_within_60min"].all()
