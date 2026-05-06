"""V1 expansion feature module tests (price, risk, macro, equity, treasury,
macro_bundle, wgc, sentiment).

Each module is exercised against the real raw parquets (skipped if missing).
Asserts the basic V1 contract:
  - build_<name>_features() returns a non-empty DataFrame (sentiment is the
    documented exception — empty stub).
  - The frame carries `t_visible: pd.Timestamp` (UTC-aware).
  - `t_visible` is monotonic non-decreasing once sorted; `release_ts <=
    t_visible` for sources that ship a release_ts column.
  - No all-NaN feature columns AFTER the warmup band (first ~50-3300 rows
    depending on rolling window).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nanogld.features import (
    equity,
    macro,
    macro_bundle,
    price,
    risk,
    sentiment,
    treasury,
    wgc,
)

# ────────────────────────────────────────────────────────────────────────────
# Skip-if-missing fixtures
# ────────────────────────────────────────────────────────────────────────────


def _have(path: str) -> bool:
    return Path(path).exists()


NEEDS_GLD = pytest.mark.skipif(
    not _have("data/raw/alpaca_bars_GLD_30min.parquet"),
    reason="needs Alpaca GLD 30min parquet",
)
NEEDS_FRED = pytest.mark.skipif(
    not _have("data/raw/fred_dgs10_all_releases.parquet"),
    reason="needs FRED ALFRED parquets",
)
NEEDS_WGC = pytest.mark.skipif(
    not _have("data/raw/wgc_central_bank_quarterly.parquet"),
    reason="needs WGC central-bank parquet",
)
NEEDS_ETFS = pytest.mark.skipif(
    not _have("data/raw/alpaca_bars_SPY_30min.parquet"),
    reason="needs Alpaca SPY 30min parquet",
)


# ────────────────────────────────────────────────────────────────────────────
# Smoke: every module imports + has the expected build entry-point
# ────────────────────────────────────────────────────────────────────────────


def test_imports_all_v1_modules() -> None:
    for mod in (price, risk, macro, equity, treasury, macro_bundle, wgc, sentiment):
        assert hasattr(mod, "FEATURES"), f"{mod.__name__} missing FEATURES tuple"
        assert isinstance(mod.FEATURES, tuple)
    assert callable(price.build_price_features)
    assert callable(risk.build_risk_features)
    assert callable(macro.build_macro_features)
    assert callable(equity.build_equity_features)
    assert callable(treasury.build_treasury_features)
    assert callable(macro_bundle.build_macro_bundle_features)
    assert callable(wgc.build_wgc_features)
    assert callable(sentiment.build_sentiment_features)


# ────────────────────────────────────────────────────────────────────────────
# Common V1 contract helpers
# ────────────────────────────────────────────────────────────────────────────


def _assert_basic_contract(df: pd.DataFrame, *, allow_empty_for_stub: bool = False) -> None:
    """Frame must be non-empty (unless stub) and carry tz-aware t_visible."""
    if allow_empty_for_stub and df.empty:
        # stub case (sentiment) — still must declare a t_visible column.
        assert "t_visible" in df.columns
        return
    assert not df.empty, "feature builder returned empty frame"
    assert "t_visible" in df.columns, "feature frame missing t_visible"
    # t_visible must be tz-aware (UTC).
    tv = df["t_visible"].dropna()
    if not tv.empty:
        assert tv.dt.tz is not None, "t_visible should be tz-aware"
    # release_ts <= t_visible if both columns present.
    if "release_ts" in df.columns:
        bad = df.loc[
            df["release_ts"].notna() & df["t_visible"].notna(),
            ["release_ts", "t_visible"],
        ]
        if not bad.empty:
            violations = (bad["release_ts"] > bad["t_visible"]).sum()
            assert violations == 0, f"{violations} rows violate release_ts <= t_visible"


def _assert_monotonic_or_sortable(df: pd.DataFrame) -> None:
    if "t_visible" not in df.columns or df.empty:
        return
    tv = df["t_visible"].dropna()
    if tv.empty:
        return
    sorted_tv = tv.sort_values().reset_index(drop=True)
    diffs = sorted_tv.diff().dropna()
    # Non-decreasing once sorted (trivially true after sort, but verify no
    # negative values which would mean NaT propagation issues).
    assert (diffs >= pd.Timedelta(0)).all()


def _no_all_nan_features(df: pd.DataFrame, *, ignore_cols: set[str] | None = None) -> None:
    """Every feature column has at least some non-null value."""
    ignore = {"timestamp", "t_visible", "release_ts", "date", "period"} | (ignore_cols or set())
    feat_cols = [c for c in df.columns if c not in ignore]
    bad = [c for c in feat_cols if df[c].dropna().empty]
    assert not bad, f"all-NaN feature columns: {bad}"


# ────────────────────────────────────────────────────────────────────────────
# price.py
# ────────────────────────────────────────────────────────────────────────────


@NEEDS_GLD
def test_price_features_real_data() -> None:
    df = price.build_price_features()
    _assert_basic_contract(df)
    _assert_monotonic_or_sortable(df.sort_values("t_visible"))
    expected = {
        "log_return_1",
        "log_return_4",
        "log_return_16",
        "log_return_48",
        "rsi_14",
        "macd_signal",
        "bbands_pct",
        "high_low_range",
        "volume_zscore",
        "close_open_ratio",
        "session_phase",
    }
    assert expected.issubset(df.columns), f"missing: {expected - set(df.columns)}"
    # After warmup (~50 bars enough for MACD slow=26 + signal=9), every numeric
    # feature should be populated for the bulk of rows.
    after = df.iloc[80:]
    for c in ("log_return_1", "rsi_14", "macd_signal", "bbands_pct"):
        nn = after[c].notna().mean()
        assert nn > 0.95, f"{c} only {nn:.2%} populated after warmup"
    # Session phase is integer in {-1, 0, 1, 2, 3}
    assert set(df["session_phase"].dropna().unique()).issubset({-1, 0, 1, 2, 3})
    # Sanity: log_return_1 magnitudes look like 30min returns (rarely > 5%).
    lr = df["log_return_1"].dropna()
    assert lr.abs().quantile(0.999) < 0.10


# ────────────────────────────────────────────────────────────────────────────
# risk.py
# ────────────────────────────────────────────────────────────────────────────


@NEEDS_GLD
def test_risk_features_real_data() -> None:
    df = risk.build_risk_features()
    _assert_basic_contract(df)
    expected = {
        "realized_vol_8",
        "realized_vol_48",
        "realized_vol_240",
        "vol_ratio_8_48",
        "vol_zscore_30d",
        "garman_klass_8",
        "days_since_FOMC",
        "is_FOMC_week",
    }
    assert expected.issubset(df.columns), f"missing: {expected - set(df.columns)}"
    after = df.iloc[1000:]
    for c in ("realized_vol_8", "realized_vol_48", "realized_vol_240"):
        nn = after[c].notna().mean()
        assert nn > 0.95, f"{c} only {nn:.2%} populated after warmup"
    # Vol must be non-negative.
    assert (df["realized_vol_8"].dropna() >= 0).all()
    assert (df["garman_klass_8"].dropna() >= 0).all()
    # is_FOMC_week is {0, 1}
    assert set(df["is_FOMC_week"].dropna().unique()).issubset({0.0, 1.0})


# ────────────────────────────────────────────────────────────────────────────
# macro.py
# ────────────────────────────────────────────────────────────────────────────


@NEEDS_FRED
def test_macro_features_real_data() -> None:
    df = macro.build_macro_features()
    _assert_basic_contract(df)
    _assert_monotonic_or_sortable(df.sort_values("t_visible"))
    expected = {
        "dxy_log_return_5d",
        "dgs10_level",
        "dgs2_level",
        "term_spread_10y_2y",
        "real_rate_10y",
        "vix_level",
    }
    assert expected.issubset(df.columns), f"missing: {expected - set(df.columns)}"
    _no_all_nan_features(df)
    # Sanity: DGS10 raw / 10 should land roughly in [0, 1] (i.e. 0-10%).
    dgs10 = df["dgs10_level"].dropna()
    assert dgs10.between(0, 1.5).mean() > 0.9


# ────────────────────────────────────────────────────────────────────────────
# equity.py
# ────────────────────────────────────────────────────────────────────────────


@NEEDS_ETFS
def test_equity_features_real_data() -> None:
    df = equity.build_equity_features()
    _assert_basic_contract(df)
    # 9 ETFs × 8 features + 7 ratios = 79 feature cols
    feat_cols = [c for c in df.columns if c not in ("timestamp", "t_visible")]
    assert len(feat_cols) >= 70, f"only {len(feat_cols)} feature cols"
    # spot check named ratios
    for c in (
        "gold_silver_ratio",
        "gdx_gld_ratio",
        "spy_gld_corr_30d",
        "qqq_gld_corr_30d",
        "iwm_gld_corr_30d",
    ):
        assert c in df.columns, f"missing {c}"
    # After 1500 bars, gold/silver ratio + gdx/gld should be populated and
    # land in plausible bands.
    after = df.iloc[1500:]
    gsr = after["gold_silver_ratio"].dropna()
    if not gsr.empty:
        # GLD ≈ 1/10 oz of gold price; SLV ≈ 1 oz of silver price. The price
        # ratio is around 5-12 depending on era (NOT the gold-silver ounce
        # ratio of 50-100). Just assert positivity + finite.
        assert (gsr > 0).all()
        assert np.isfinite(gsr).all()


# ────────────────────────────────────────────────────────────────────────────
# treasury.py
# ────────────────────────────────────────────────────────────────────────────


@NEEDS_FRED
def test_treasury_features_real_data() -> None:
    df = treasury.build_treasury_features()
    _assert_basic_contract(df)
    _assert_monotonic_or_sortable(df.sort_values("t_visible"))
    # 11 levels + 11 changes + 4 spreads + 1 butterfly + 2 real = 29
    expected_min = {
        "dgs10_level",
        "dgs2_level",
        "dgs3mo_level",
        "spread_10y_2y",
        "spread_10y_3m",
        "butterfly_2_5_10",
        "real_rate_10y_direct",
        "real_rate_10y_breakeven",
    }
    assert expected_min.issubset(df.columns)
    _no_all_nan_features(df)
    # Treasury 10y - 2y spread mostly in [-2, 4] %.
    spread = df["spread_10y_2y"].dropna()
    assert spread.between(-3, 5).mean() > 0.95


# ────────────────────────────────────────────────────────────────────────────
# macro_bundle.py
# ────────────────────────────────────────────────────────────────────────────


@NEEDS_FRED
def test_macro_bundle_features_real_data() -> None:
    df = macro_bundle.build_macro_bundle_features()
    _assert_basic_contract(df)
    # spot check 5 critical series
    for sid in ("UNRATE", "PAYEMS", "CPIAUCSL", "M2SL", "DFF"):
        for suf in ("level", "yoy_change", "mom_change"):
            assert f"{sid.lower()}_{suf}" in df.columns
    # Derived
    for c in ("real_fedfunds", "m2_yoy", "icsa_4w_ma"):
        assert c in df.columns


@NEEDS_FRED
def test_macro_bundle_per_series_api() -> None:
    """The build_per_series() helper drives the panel attach path."""
    per_series = macro_bundle.build_per_series()
    assert "UNRATE" in per_series
    assert "DFF" in per_series
    f = per_series["UNRATE"]
    assert {"date", "t_visible", "unrate_level"}.issubset(f.columns)
    assert not f.empty
    # YoY shifts 12 obs (monthly), so first 12 are NaN.
    assert pd.isna(f["unrate_yoy_change"].iloc[0])
    assert f["unrate_yoy_change"].notna().sum() > 0


# ────────────────────────────────────────────────────────────────────────────
# wgc.py
# ────────────────────────────────────────────────────────────────────────────


@NEEDS_WGC
def test_wgc_features_real_data() -> None:
    df = wgc.build_wgc_features()
    _assert_basic_contract(df)
    expected = {
        "wgc_total_net_purchase_tonnes_q",
        "wgc_total_net_purchase_yoy",
        "wgc_is_net_buyer_q",
    }
    assert expected.issubset(df.columns)
    # is_net_buyer is binary float
    isb = df["wgc_is_net_buyer_q"].dropna()
    assert set(isb.unique()).issubset({0.0, 1.0})
    # Quarterly cadence — should have approx 90-day spacing on `period`.
    if "period" in df.columns and len(df) > 5:
        diffs = df["period"].sort_values().diff().dropna()
        median_days = diffs.dt.days.median()
        assert 80 <= median_days <= 100, f"period diff median={median_days}d, expected ~91"


# ────────────────────────────────────────────────────────────────────────────
# sentiment.py — stub
# ────────────────────────────────────────────────────────────────────────────


def test_sentiment_features_stub() -> None:
    df = sentiment.build_sentiment_features()
    _assert_basic_contract(df, allow_empty_for_stub=True)
    # Must declare the future column slots so callers know what's coming.
    assert any("polarity" in f.name for f in sentiment.FEATURES)


# ────────────────────────────────────────────────────────────────────────────
# Cross-module: total feature dim + panel build
# ────────────────────────────────────────────────────────────────────────────


@NEEDS_FRED
@NEEDS_GLD
@NEEDS_ETFS
def test_combined_feature_dim_count() -> None:
    """V1 spec target: ~232 numeric dims across the 8 V1 modules."""
    counts: dict[str, int] = {}

    def _n(df: pd.DataFrame) -> int:
        return sum(
            1
            for c in df.columns
            if c not in ("timestamp", "t_visible", "date", "release_ts", "period")
        )

    counts["price"] = _n(price.build_price_features())
    counts["risk"] = _n(risk.build_risk_features())
    counts["macro"] = _n(macro.build_macro_features())
    counts["equity"] = _n(equity.build_equity_features())
    counts["treasury"] = _n(treasury.build_treasury_features())
    counts["macro_bundle"] = _n(macro_bundle.build_macro_bundle_features())
    counts["wgc"] = _n(wgc.build_wgc_features())
    counts["sentiment"] = _n(sentiment.build_sentiment_features())
    total = sum(counts.values())
    print(f"\nfeature counts: {counts}")
    print(f"total V1 dims: {total}")
    # 11 price + 8 risk + 6 macro + ~79 equity + 29 treasury + ~60 macro_bundle + 3 wgc
    # = ~196 minimum. Slack for derived rows in macro_bundle.
    assert total > 150, f"expected >150 V1 dims, got {total}"
