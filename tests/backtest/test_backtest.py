"""Unit tests for V1 backtest engine + metrics + cost-stress."""

from __future__ import annotations

import numpy as np
import pytest

from nanogld.backtest import (
    BacktestConfig,
    compute_metrics,
    cost_stress,
    deflated_sharpe,
    max_drawdown,
    passes_v1_gate,
    per_bucket_metrics,
    sharpe,
    sortino,
    vectorized_backtest,
)
from nanogld.backtest.baselines import (
    buy_hold_positions,
    donchian_positions,
    gao_2014_positions,
    ma_cross_positions,
)


@pytest.mark.smoke
def test_vectorized_backtest_basic() -> None:
    nlr = np.array([0.001, 0.002, -0.001, 0.003])
    pos = np.array([1.0, 1.0, 1.0, 1.0])
    out = vectorized_backtest(nlr, pos, cfg=BacktestConfig(cost_bps=0.0))
    assert out.pnl_per_bar.shape == (4,)
    assert out.equity_curve.shape == (4,)


@pytest.mark.smoke
def test_buy_hold_returns_market() -> None:
    nlr = np.array([0.001, -0.002, 0.003])
    pos = buy_hold_positions(3)
    out = vectorized_backtest(nlr, pos, cfg=BacktestConfig(cost_bps=0.0))
    expected_cum = np.cumsum(nlr)
    np.testing.assert_array_almost_equal(np.log(out.equity_curve), expected_cum)


@pytest.mark.smoke
def test_max_drawdown_simple() -> None:
    eq = np.array([1.0, 1.1, 1.0, 0.9, 1.05])
    mdd = max_drawdown(eq)
    assert mdd == pytest.approx((1.1 - 0.9) / 1.1)


@pytest.mark.smoke
def test_sharpe_annualization() -> None:
    rng = np.random.default_rng(0)
    pnl = rng.normal(0.001, 0.01, size=1000)
    s = sharpe(pnl, bars_per_year=252)
    assert s > 0


@pytest.mark.smoke
def test_sortino_canonical_formula() -> None:
    pnl = np.array([0.01, -0.01, 0.02, -0.02, 0.03])
    s_calc = sortino(pnl, bars_per_year=252)
    downside = np.minimum(0.0, pnl)
    expected = pnl.mean() / np.sqrt((downside**2).mean()) * np.sqrt(252)
    assert s_calc == pytest.approx(expected, rel=1e-6)


@pytest.mark.smoke
def test_compute_metrics_keys() -> None:
    pnl = np.array([0.001, -0.0005, 0.002, -0.0003])
    eq = np.exp(np.cumsum(pnl))
    m = compute_metrics(pnl, eq)
    assert "sharpe" in m and "sortino" in m and "max_drawdown" in m


@pytest.mark.smoke
def test_cost_stress_three_levels() -> None:
    nlr = np.array([0.001, 0.002, -0.001, 0.003, 0.0005])
    pos = np.array([1.0, 0.5, 0.5, 0.0, 1.0])
    result = cost_stress(nlr, pos, base_cost_bps=2.0)
    assert set(result.by_multiplier.keys()) == {0.5, 1.0, 1.5}


@pytest.mark.smoke
def test_passes_v1_gate_thresholding() -> None:
    nlr = np.full(500, 0.001)
    pos = np.ones(500)
    result = cost_stress(nlr, pos, base_cost_bps=0.0)
    assert passes_v1_gate(result, threshold_at_1_5x=0.5) is True


@pytest.mark.smoke
def test_per_bucket_metrics_keys() -> None:
    pnl = np.array([0.001, 0.002, -0.001, 0.003])
    eq = np.exp(np.cumsum(pnl))
    mask = np.array([True, False, True, False])
    out = per_bucket_metrics(pnl, eq, mask)
    assert set(out.keys()) == {"present", "absent", "both"}


@pytest.mark.smoke
def test_dsr_returns_pair() -> None:
    p, dsr = deflated_sharpe(sharpe_observed=2.0, n_trials=10, n_obs=1000)
    assert 0.0 <= p <= 1.0
    assert isinstance(dsr, float)


@pytest.mark.smoke
def test_ma_cross_positions_lagged() -> None:
    close = np.linspace(100, 200, 300)
    pos = ma_cross_positions(close, fast_span=10, slow_span=20)
    assert pos.shape == (300,)
    assert pos[0] == 0.0


@pytest.mark.smoke
def test_donchian_basic() -> None:
    close = np.array([100.0] * 25 + [105.0] * 25)
    pos = donchian_positions(close, window=20)
    assert pos.shape == (50,)


@pytest.mark.smoke
def test_gao_2014_high_vol_only() -> None:
    h5 = np.array([0.001, -0.001, 0.001, -0.001])
    high_vol = np.array([True, True, False, False])
    pos = gao_2014_positions(h5, is_high_vol=high_vol)
    assert pos[0] == 1.0
    assert pos[1] == -1.0
    assert pos[2] == 0.0
    assert pos[3] == 0.0
