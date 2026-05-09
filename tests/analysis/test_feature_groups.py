"""Unit tests for feature_groups (pure-function module)."""

from __future__ import annotations

import pytest

from nanogld.analysis.feature_groups import (
    CATEGORIES,
    classify_features,
    rollup_by_group,
)

pytestmark = pytest.mark.smoke


def test_classify_known_categories() -> None:
    names = [
        "gld_close",
        "gld_log_return",
        "gld_h5_log_return",
        "gld_atr_14",
        "gld_rv_60",
        "vix_close",
        "fred_dff",
        "is_fomc_week",
        "regime_hmm_p_high_vol",
        "regime_vix_tercile_low",
        "news_count_60min",
        "spy_log_return",
        "gldvslv_ratio",
        "us_2y_yield",
        "real_yield_10y",
        "completely_unknown_feature_xyz",
    ]
    classified = classify_features(names)
    assert classified["gld_close"] == "price"
    assert classified["gld_log_return"] == "price"
    assert classified["gld_h5_log_return"] == "price"
    assert classified["gld_atr_14"] == "volatility"
    assert classified["gld_rv_60"] == "volatility"
    assert classified["vix_close"] == "macro"
    assert classified["fred_dff"] == "macro"
    assert classified["is_fomc_week"] == "calendar"
    assert classified["regime_hmm_p_high_vol"] == "regime"
    assert classified["regime_vix_tercile_low"] == "regime"
    assert classified["news_count_60min"] == "news"
    assert classified["spy_log_return"] == "flow"
    assert classified["gldvslv_ratio"] == "flow"
    assert classified["us_2y_yield"] == "rates"
    assert classified["real_yield_10y"] == "rates"
    assert classified["completely_unknown_feature_xyz"] == "other"


def test_categories_complete() -> None:
    expected = {
        "price",
        "volatility",
        "macro",
        "calendar",
        "regime",
        "news",
        "flow",
        "rates",
        "other",
    }
    assert set(CATEGORIES) == expected


def test_rollup_orders_descending_by_sum_abs() -> None:
    names = ["gld_close", "vix_close", "fred_dff", "gld_atr_14"]
    importance = [0.4, 0.1, 0.2, 0.5]
    rollups = rollup_by_group(names, importance)
    assert len(rollups) == 3
    sums = [r.sum_abs_importance for r in rollups]
    assert sums == sorted(sums, reverse=True)


def test_rollup_validates_lengths() -> None:
    with pytest.raises(ValueError):
        rollup_by_group(["a", "b"], [1.0])


def test_rollup_picks_top_feature_by_abs() -> None:
    names = ["fred_dff", "vix_close", "fred_dgs10"]
    importance = [-0.8, 0.3, 0.5]
    rollups = rollup_by_group(names, importance)
    macro_roll = next(r for r in rollups if r.category == "macro")
    assert macro_roll.top_feature == "fred_dff"
    assert pytest.approx(macro_roll.top_value) == -0.8
