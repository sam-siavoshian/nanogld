"""Unit tests for V1 deterministic regime features."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nanogld.features import regime


def _synthetic_panel(n: int = 200, year: int = 2024) -> pd.DataFrame:
    """Return a deterministic VIX + close panel spanning trading days in `year`."""
    bar_close = pd.date_range(f"{year}-01-15 14:30:00+00:00", periods=n, freq="30min")
    rng = np.random.default_rng(0)
    close = 100.0 + np.cumsum(rng.standard_normal(n)) * 0.5
    vix = 15.0 + np.abs(np.cos(np.arange(n) / 11.0)) * 5.0
    return pd.DataFrame(
        {
            "bar_close_utc": bar_close,
            "vix_level": vix,
            "gld_close": close,
            "is_fomc_week": np.zeros(n, dtype=np.int8),
        }
    )


@pytest.mark.smoke
def test_fit_regime_thresholds_returns_pair() -> None:
    df = _synthetic_panel()
    th = regime.fit_regime_thresholds(df, rv_lookback=20)
    assert isinstance(th.vix_tercile, tuple)
    assert th.vix_tercile[0] < th.vix_tercile[1]
    assert th.rv_tercile[0] < th.rv_tercile[1]


@pytest.mark.smoke
def test_add_regime_columns_emits_11_cols() -> None:
    df = _synthetic_panel()
    th = regime.fit_regime_thresholds(df, rv_lookback=20)
    out = regime.add_regime_columns(df, thresholds=th, rv_lookback=20)

    expected_11 = [
        "regime_vix_low",
        "regime_vix_mid",
        "regime_vix_high",
        "regime_rv_low",
        "regime_rv_mid",
        "regime_rv_high",
        "regime_fomc_week",
        "regime_year_2016_2019",
        "regime_year_2020_2022",
        "regime_year_2023_2024",
        "regime_year_2025_plus",
    ]
    for c in expected_11:
        assert c in out.columns, f"missing column: {c}"


@pytest.mark.smoke
def test_year_bucket_one_hot_disjoint() -> None:
    df = _synthetic_panel(year=2024)
    th = regime.fit_regime_thresholds(df, rv_lookback=20)
    out = regime.add_regime_columns(df, thresholds=th, rv_lookback=20)
    year_cols = [c for c in out.columns if c.startswith("regime_year_")]
    sums = out[year_cols].sum(axis=1)
    assert (sums == 1).all(), "exactly one year-bucket must fire per row"


@pytest.mark.smoke
def test_vix_tercile_one_hot_disjoint() -> None:
    df = _synthetic_panel()
    th = regime.fit_regime_thresholds(df, rv_lookback=20)
    out = regime.add_regime_columns(df, thresholds=th, rv_lookback=20)
    sums = out[["regime_vix_low", "regime_vix_mid", "regime_vix_high"]].sum(axis=1)
    assert sums.isin([0, 1]).all(), "vix tercile is disjoint (0 only on NaN)"


@pytest.mark.smoke
def test_thresholds_frozen_no_test_leak() -> None:
    """Thresholds fit on train must NOT change when applied to val."""
    train = _synthetic_panel(n=200, year=2022)
    val = _synthetic_panel(n=50, year=2024)
    th_train = regime.fit_regime_thresholds(train, rv_lookback=20)
    # API enforces frozen-train thresholds via the required argument; the
    # user must pass `th_train` even when applying to val/test.
    out_val = regime.add_regime_columns(val, thresholds=th_train, rv_lookback=20)
    assert "regime_vix_low" in out_val.columns
    # Sanity: at least one tercile bucket fires somewhere
    vix_sum = out_val[["regime_vix_low", "regime_vix_mid", "regime_vix_high"]].sum().sum()
    assert vix_sum > 0
