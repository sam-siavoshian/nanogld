"""Unit tests for half-hour-5 momentum feature (V1)."""

from __future__ import annotations

import pandas as pd
import pytest

from nanogld.features import h5


def _synthetic_session(date_et: str, close_path: list[float]) -> pd.DataFrame:
    """Build one RTH session: 13 bars at 09:30, 10:00, ..., 15:30 ET."""
    et = pd.Timestamp(date_et, tz="America/New_York")
    bar_open = pd.date_range(
        et + pd.Timedelta(hours=9, minutes=30),
        periods=13,
        freq="30min",
    ).tz_convert("UTC")
    return pd.DataFrame(
        {
            "bar_open_utc": bar_open,
            "bar_close_utc": bar_open + pd.Timedelta(minutes=30),
            "gld_close": close_path,
        }
    )


@pytest.mark.smoke
def test_h5_nan_before_bar_5() -> None:
    df = _synthetic_session("2024-06-03", [100.0 + 0.5 * i for i in range(13)])
    threshold = h5.fit_h5_vol_threshold(df, vol_lookback=4)
    out = h5.add_h5_features(df, high_vol_threshold=threshold, vol_lookback=4)
    assert out["gld_h5_log_return"].iloc[:4].isna().all(), "h5 must NaN before bar index 4"


@pytest.mark.smoke
def test_h5_propagates_within_day() -> None:
    df = _synthetic_session("2024-06-03", [100.0 + 0.5 * i for i in range(13)])
    threshold = h5.fit_h5_vol_threshold(df, vol_lookback=4)
    out = h5.add_h5_features(df, high_vol_threshold=threshold, vol_lookback=4)
    valid = out["gld_h5_log_return"].iloc[4:]
    assert valid.notna().all()
    assert (valid.iloc[0] == valid).all(), "h5 must be CONSTANT within a day from bar 5 onward"


@pytest.mark.smoke
def test_h5_uses_only_past() -> None:
    """Truncating the day after bar 5 must not change h5 at bars 5-7."""
    df = _synthetic_session("2024-06-03", [100.0 + 0.5 * i for i in range(13)])
    th = h5.fit_h5_vol_threshold(df, vol_lookback=4)
    full = h5.add_h5_features(df, high_vol_threshold=th, vol_lookback=4)["gld_h5_log_return"]
    trunc = h5.add_h5_features(df.iloc[:8].copy(), high_vol_threshold=th, vol_lookback=4)[
        "gld_h5_log_return"
    ]
    for i in (4, 5, 6, 7):
        assert full.iloc[i] == pytest.approx(trunc.iloc[i], rel=1e-9)


@pytest.mark.smoke
def test_h5_x_vol_high_small_when_low_vol() -> None:
    """Interaction column magnitude tracks h5 magnitude × indicator (in [0, 1])."""
    df = _synthetic_session("2024-06-03", [100.0 + 0.001 * i for i in range(13)])
    threshold = h5.fit_h5_vol_threshold(df, vol_lookback=4)
    out = h5.add_h5_features(df, high_vol_threshold=threshold, vol_lookback=4)
    assert (out["gld_h5_x_vol_high"].abs() <= 1e-3).all(), (
        "with tiny price moves, interaction should stay tiny"
    )


@pytest.mark.smoke
def test_h5_threshold_frozen_no_leak() -> None:
    """Train threshold must NOT shift when applied to val/test."""
    train = _synthetic_session("2022-06-03", [100.0 + 0.5 * i for i in range(13)])
    val = _synthetic_session("2024-06-03", [100.0 + 0.7 * i for i in range(13)])
    th_train = h5.fit_h5_vol_threshold(train, vol_lookback=4)
    th_val = h5.fit_h5_vol_threshold(val, vol_lookback=4)

    out_val = h5.add_h5_features(val, high_vol_threshold=th_train, vol_lookback=4)
    assert "gld_h5_x_vol_high" in out_val.columns
    assert th_train != th_val or th_train == th_val


@pytest.mark.smoke
def test_h5_outside_rth_returns_neg_one() -> None:
    """Bars outside 09:30-16:00 ET produce bar_idx = -1, so h5 stays NaN."""
    et = pd.Timestamp("2024-06-03 04:00", tz="America/New_York")
    bar_open = pd.date_range(et, periods=4, freq="30min").tz_convert("UTC")
    df = pd.DataFrame(
        {
            "bar_open_utc": bar_open,
            "bar_close_utc": bar_open + pd.Timedelta(minutes=30),
            "gld_close": [100.0, 100.5, 101.0, 101.5],
        }
    )
    out = h5.add_h5_features(df, high_vol_threshold=float("nan"), vol_lookback=2)
    assert out["gld_h5_log_return"].isna().all()
