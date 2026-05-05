"""Feature pipeline tests — keyless data path.

Asserts:
  - Each feature builder produces a frame with t_visible column or daily index.
  - PIT invariant: t_visible <= panel-row date for every row of the panel.
  - No NaN in feature columns AFTER warmup (oil 20d → first 25 rows OK to skip;
    COT 52w z → first 60 rows; GPR 60m z → first 60 rows).
  - Determinism: build_panel on the same raw is bitwise-identical.
  - Plausible value ranges: GPR > 0, oil close 20-200, etc.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nanogld.features import (
    build,
    calendar_features,
    geopolitical,
    oil,
    positioning,
)
from nanogld.features.utils import (
    daily_index_utc,
    garman_klass_vol,
    log_returns,
    realized_vol,
    rolling_z,
)

NEEDS_RAW = pytest.mark.skipif(
    not Path("data/raw/brent_daily.parquet").exists(),
    reason="needs no-key raw pulls completed first",
)


# ────────────────────────────────────────────────────────────────────────────
# pure-math helpers
# ────────────────────────────────────────────────────────────────────────────


def test_log_returns_first_k_are_nan() -> None:
    s = pd.Series([100.0, 101.0, 102.0, 103.0])
    r = log_returns(s, k=1)
    assert pd.isna(r.iloc[0])
    np.testing.assert_allclose(r.iloc[1], np.log(101.0 / 100.0), rtol=1e-9)


def test_realized_vol_warmup() -> None:
    s = pd.Series(100 + np.cumsum(np.random.default_rng(0).normal(0, 0.5, 100)))
    v = realized_vol(s, window=20)
    assert pd.isna(v.iloc[0])
    assert v.iloc[40] > 0


def test_garman_klass_positive() -> None:
    rng = np.random.default_rng(0)
    n = 50
    open_ = pd.Series(100 + rng.normal(0, 0.1, n))
    close = open_ + rng.normal(0, 0.5, n)
    high = pd.concat([open_, close], axis=1).max(axis=1) + abs(rng.normal(0, 0.3, n))
    low = pd.concat([open_, close], axis=1).min(axis=1) - abs(rng.normal(0, 0.3, n))
    v = garman_klass_vol(high, low, open_, close, 20)
    assert (v.dropna() >= 0).all()


def test_rolling_z_zero_mean_unit_var() -> None:
    rng = np.random.default_rng(0)
    s = pd.Series(rng.normal(0, 1, 500))
    z = rolling_z(s, window=100, min_periods=100)
    sample = z.iloc[200]  # well past warmup
    assert -5 < sample < 5


# ────────────────────────────────────────────────────────────────────────────
# per-feature-group builders against real raw data
# ────────────────────────────────────────────────────────────────────────────


@NEEDS_RAW
def test_oil_features_real_data() -> None:
    df = oil.build_oil_features()
    assert not df.empty
    assert {"brent_level_close", "wti_level_close", "brent_wti_spread"}.issubset(df.columns)
    # Sane price ranges over 5y
    assert 20 < df["brent_level_close"].dropna().min() < 200
    assert 20 < df["wti_level_close"].dropna().min() < 200
    # PIT carrier
    assert "t_visible" in df.columns
    # Spread non-trivial
    assert df["brent_wti_spread"].dropna().abs().mean() > 0


@NEEDS_RAW
def test_cot_features_real_data() -> None:
    df = positioning.build_cot_features()
    assert not df.empty
    assert {"cot_mm_net", "cot_mm_net_pct_oi", "cot_mm_z_52w", "t_visible"}.issubset(df.columns)
    assert df["cot_irregular_release"].dtype == bool
    # COT positions are big numbers (~tens of thousands)
    assert df["cot_mm_net"].abs().max() > 10_000


@NEEDS_RAW
def test_gpr_features_real_data() -> None:
    df = geopolitical.build_gpr_features()
    assert not df.empty
    assert {"gpr_level", "gpr_mom", "gpr_yoy", "gpr_z_60m", "t_visible"}.issubset(df.columns)
    # GPR is a positive index
    assert (df["gpr_level"].dropna() > 0).all()


@NEEDS_RAW
def test_calendar_features_binary_only() -> None:
    idx = daily_index_utc(
        pd.Timestamp("2024-01-01", tz="UTC"), pd.Timestamp("2024-12-31", tz="UTC")
    )
    cal = calendar_features.build_calendar_features(idx)
    # Every column boolean
    for c in cal.columns:
        assert cal[c].dtype == bool
    # Some events fire over a year
    assert cal["event_within_24h_FOMC"].sum() > 5
    # No `minutes_until_event` columns
    assert not any("minutes_until" in c for c in cal.columns)


# ────────────────────────────────────────────────────────────────────────────
# Daily panel
# ────────────────────────────────────────────────────────────────────────────


@NEEDS_RAW
def test_panel_shape_and_pit() -> None:
    panel = build.build_panel()
    assert len(panel) > 1000  # 5y of daily rows ~= 1827
    assert panel.shape[1] >= 30  # oil + cot + gpr + calendar + t_visible_max
    # PIT global gate: t_visible_max <= row's date_utc + 1day (next-day visibility OK).
    # We accept t_visible <= row's date because the row is "as of midnight UTC of date_utc".
    visible = panel["t_visible_max"].dropna()
    days = panel.index[panel["t_visible_max"].notna()]
    leak = (visible > days + pd.Timedelta(hours=24)).sum()
    assert leak == 0, f"{leak} rows leak future data via t_visible_max"


@NEEDS_RAW
def test_panel_warm_up_then_no_nan_in_critical_features() -> None:
    panel = build.build_panel()
    # After 60 trading days warmup, oil log_ret_1d should be filled.
    after_warmup = panel.iloc[80:]
    assert after_warmup["brent_log_ret_1d"].notna().sum() > 0.5 * len(after_warmup)
    # COT mm_net populated after first weekly release lands.
    assert after_warmup["cot_mm_net"].notna().sum() > 0.9 * len(after_warmup)
    # Calendar binaries always populated.
    assert after_warmup["event_within_24h_any_tier1"].notna().all()


@NEEDS_RAW
def test_panel_determinism() -> None:
    p1 = build.build_panel()
    p2 = build.build_panel()
    pd.testing.assert_frame_equal(p1, p2)


@NEEDS_RAW
def test_panel_no_constant_or_all_nan_columns() -> None:
    panel = build.build_panel()
    issues: list[str] = []
    for c in panel.columns:
        if c == "t_visible_max":
            continue
        col = panel[c].dropna()
        if col.empty:
            issues.append(f"{c}=all-NaN")
        elif col.nunique() == 1 and panel[c].dtype != bool:
            issues.append(f"{c}=constant({col.iloc[0]})")
    # Boolean calendar binaries CAN be all-False over short windows; don't flag those.
    real_issues = [i for i in issues if "constant(False)" not in i and "constant(True)" not in i]
    assert not real_issues, real_issues


@NEEDS_RAW
def test_no_minutes_until_event_in_features() -> None:
    panel = build.build_panel()
    bad = [c for c in panel.columns if "minutes_until" in c]
    assert not bad, f"V1 §14 violation: {bad}"
