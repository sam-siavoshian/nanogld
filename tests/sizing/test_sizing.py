"""Unit tests for V1 sizing stack."""

from __future__ import annotations

import numpy as np
import pytest

from nanogld.sizing import (
    ATRStop,
    CostConfig,
    DrawdownCircuitBreaker,
    KellyConfig,
    Sizer,
    SizerConfig,
    TimeoutExit,
    apply_conformal_floor,
    cost,
    kelly_size,
    vol_target_multiplier,
)


@pytest.mark.smoke
def test_kelly_size_zero_cost() -> None:
    out = kelly_size(edge=0.001, variance=0.0001, cfg=KellyConfig(), cost=0.0)
    assert out > 0.0
    assert out <= 1.0


@pytest.mark.smoke
def test_kelly_size_cost_above_edge_zeros() -> None:
    out = kelly_size(edge=0.001, variance=0.0001, cfg=KellyConfig(), cost=0.01)
    assert out == 0.0


@pytest.mark.smoke
def test_kelly_size_clipped_to_position_limit() -> None:
    out = kelly_size(edge=10.0, variance=1e-8, cfg=KellyConfig(position_limit=1.0), cost=0.0)
    assert abs(out) <= 1.0


@pytest.mark.smoke
def test_vol_target_multiplier_15pct_synthetic() -> None:
    var = (0.15 / np.sqrt(3276)) ** 2
    m = vol_target_multiplier(var)
    assert m == pytest.approx(1.0, rel=1e-3)


@pytest.mark.smoke
def test_vol_target_capped() -> None:
    out = vol_target_multiplier(realized_var_60bar=1e-12, vol_mult_cap=3.0)
    assert out == 3.0


@pytest.mark.smoke
def test_cost_zero_delta_returns_zero() -> None:
    assert cost(0.0) == 0.0


@pytest.mark.smoke
def test_cost_monotonic_in_delta() -> None:
    a = cost(0.01)
    b = cost(0.05)
    assert b > a


@pytest.mark.smoke
def test_conformal_floor_zeros_below_threshold() -> None:
    sized = apply_conformal_floor(0.5, aps_lower_bound=0.30, threshold=0.40)
    assert sized == 0.0


@pytest.mark.smoke
def test_conformal_floor_passes_above() -> None:
    sized = apply_conformal_floor(0.5, aps_lower_bound=0.50, threshold=0.40)
    assert sized == 0.5


@pytest.mark.smoke
def test_sizer_stage1_pass_through_x_vol_mult() -> None:
    cfg = SizerConfig(stage="stage1")
    s = Sizer(cfg)
    var60 = (0.15 / np.sqrt(3276)) ** 2
    out = s.compute(
        head_b_weight=0.6,
        aps_lower_bound=0.99,
        posterior_variance=0.0,
        realized_var_60=var60,
        prev_position=0.0,
    )
    assert abs(out - 0.6) < 1e-3


@pytest.mark.smoke
def test_sizer_stage2_zeros_when_floor_kicks() -> None:
    cfg = SizerConfig(stage="stage2", aps_floor=0.40)
    s = Sizer(cfg)
    out = s.compute(
        head_b_weight=0.5,
        aps_lower_bound=0.30,
        posterior_variance=0.0,
        realized_var_60=1e-4,
        prev_position=0.0,
    )
    assert out == 0.0


@pytest.mark.smoke
def test_atr_stop_long_hard_fires() -> None:
    s = ATRStop(entry_price=100.0, entry_atr=1.0, side=1, hard_mult=2.0)
    assert s.update(99.0) == "hold"
    assert s.update(98.0) == "hold"
    assert s.update(97.0) == "exit"


@pytest.mark.smoke
def test_atr_stop_short_hard_fires() -> None:
    s = ATRStop(entry_price=100.0, entry_atr=1.0, side=-1, hard_mult=2.0)
    assert s.update(101.0) == "hold"
    assert s.update(103.0) == "exit"


@pytest.mark.smoke
def test_timeout_fires_at_max_bars() -> None:
    t = TimeoutExit(max_bars=5)
    for _ in range(4):
        assert not t.step()
    assert t.step() is True


@pytest.mark.smoke
def test_drawdown_breaker_states() -> None:
    cb = DrawdownCircuitBreaker()
    assert cb.step(1.0) == 1.0
    assert cb.step(0.95) <= 0.5
    assert cb.step(0.85) == 0.0
