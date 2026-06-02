"""Unit tests for ATR-14 + barriers (V1)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nanogld.features import atr


@pytest.mark.smoke
def test_true_range_basic() -> None:
    high = pd.Series([10.0, 11.0, 11.5])
    low = pd.Series([9.5, 10.2, 10.8])
    close_prev = pd.Series([np.nan, 10.0, 10.5])

    tr = atr.true_range(high, low, close_prev)
    assert tr.iloc[0] == pytest.approx(0.5)
    assert tr.iloc[1] == pytest.approx(1.0)
    assert tr.iloc[2] == pytest.approx(1.0)


@pytest.mark.smoke
def test_atr_pit_no_future_leak() -> None:
    """ATR[T] uses only data through bar T; first 13 rows are NaN."""
    n = 30
    rng = np.random.default_rng(42)
    close = pd.Series(np.cumsum(rng.standard_normal(n)) + 100.0)
    high = close + 0.5
    low = close - 0.5

    series = atr.atr_wilder(high, low, close, period=14)

    assert series.iloc[:13].isna().all(), "ATR-14 must NaN-out first 13 bars"
    assert series.iloc[13:].notna().all(), "ATR-14 must be defined from bar 14 onward"


@pytest.mark.smoke
def test_atr_only_uses_past() -> None:
    n = 30
    rng = np.random.default_rng(0)
    close = pd.Series(np.cumsum(rng.standard_normal(n)) + 100.0)
    high = close + 0.5
    low = close - 0.5

    full = atr.atr_wilder(high, low, close, period=14)

    truncated = atr.atr_wilder(high.iloc[:20], low.iloc[:20], close.iloc[:20], period=14)
    assert truncated.iloc[19] == pytest.approx(full.iloc[19], rel=1e-9)


@pytest.mark.smoke
def test_add_atr_and_barriers_columns() -> None:
    n = 30
    rng = np.random.default_rng(0)
    close = pd.Series(np.cumsum(rng.standard_normal(n)) + 100.0)
    df = pd.DataFrame({"gld_high": close + 0.5, "gld_low": close - 0.5, "gld_close": close})

    out = atr.add_atr_and_barriers(df, period=14, barrier_mult=1.0)
    assert "gld_atr_14" in out.columns
    assert "barrier_up" in out.columns
    assert "barrier_down" in out.columns
    valid = out.dropna(subset=["barrier_up", "barrier_down"])
    assert (valid["barrier_up"] >= 0).all()
    assert (valid["barrier_down"] >= 0).all()
    assert (valid["barrier_up"] == valid["barrier_down"]).all()
