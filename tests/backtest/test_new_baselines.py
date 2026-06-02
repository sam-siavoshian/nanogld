"""Shape + determinism locks for the 7 new V1 baselines.

These don't try to verify any specific Sharpe — too few synthetic bars
to make that meaningful. The asserts are:

1. Output shape matches the test slice length.
2. Position values land in ``[-1, +1]`` (tanh-bounded for the ML
   baselines; ±1/0 for XGBoost; clamped for F2F).
3. Deterministic given the seed (re-run produces identical output).
4. Dry-run / no-train-data path returns zeros (smoke contract).

Note on XGBoost: xgboost 3.2 on darwin/arm64 segfaults inside its
``data.py:_from_numpy_array`` when called from inside a pytest worker.
The direct Python path is stable. The XGBoost tests below are guarded
with ``@pytest.mark.skipif(sys.platform == 'darwin')`` so the rest of
the suite still runs; linux production runs (GTX Spark x86_64) use the
linux-x86_64 wheel where this segfault does not occur.
"""

from __future__ import annotations

import sys
from typing import Any

import numpy as np
import pytest

from nanogld.backtest.baselines import (
    dlinear_positions,
    forecast_to_fill_positions,
    timemixer_positions,
    tsmixer_positions,
    vlstm_positions,
    xgboost_positions,
    xlstm_time_positions,
)

_xgb_skip = pytest.mark.skipif(
    sys.platform == "darwin",
    reason="xgboost 3.2 darwin/arm64 segfault inside pytest; verified OK on linux",
)


def _toy_ctx(n_train: int = 80, n_test: int = 40, t: int = 8, f: int = 6, seed: int = 0) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    train_features_window = rng.normal(0.0, 1.0, size=(n_train, t, f)).astype(np.float32)
    test_features_window = rng.normal(0.0, 1.0, size=(n_test, t, f)).astype(np.float32)
    train_next = rng.normal(0.0, 0.001, size=n_train).astype(np.float32)
    next_log_returns = rng.normal(0.0, 0.001, size=n_test).astype(np.float64)
    close = 100.0 * np.exp(np.cumsum(next_log_returns))
    h5 = rng.normal(0.0, 0.001, size=n_test).astype(np.float64)
    is_last_bar = np.zeros(n_test, dtype=bool)
    is_last_bar[::5] = True
    is_high_vol = rng.random(n_test) > 0.5
    train_features = rng.normal(0.0, 1.0, size=(n_train, f)).astype(np.float32)
    train_labels = rng.integers(0, 3, size=n_train).astype(np.int64)
    test_features = rng.normal(0.0, 1.0, size=(n_test, f)).astype(np.float32)
    return {
        "next_log_returns": next_log_returns,
        "is_news_present": np.zeros(n_test, dtype=bool),
        "close": close,
        "h5_log_return": h5,
        "is_high_vol": is_high_vol,
        "is_last_bar_of_day": is_last_bar,
        "train_features_window": train_features_window,
        "test_features_window": test_features_window,
        "train_next_log_return": train_next,
        "train_features": train_features,
        "train_labels": train_labels,
        "test_features": test_features,
    }


@_xgb_skip
def test_xgboost_positions_shape_and_classes() -> None:
    ctx = _toy_ctx()
    positions = xgboost_positions(ctx)
    assert positions.shape == (len(ctx["next_log_returns"]),)
    assert set(np.unique(positions.astype(int))).issubset({-1, 0, 1})


def test_xgboost_dry_run_returns_zeros() -> None:
    """Dry-run path skips the xgboost call entirely; safe on darwin."""
    ctx = _toy_ctx()
    ctx.pop("train_features")
    positions = xgboost_positions(ctx)
    assert positions.shape == (len(ctx["next_log_returns"]),)
    assert np.all(positions == 0.0)


def test_dlinear_positions_shape_and_bounded() -> None:
    ctx = _toy_ctx()
    positions = dlinear_positions(ctx, epochs=2, seed=42)
    assert positions.shape == (len(ctx["next_log_returns"]),)
    assert ((positions >= -1.0 - 1e-6) & (positions <= 1.0 + 1e-6)).all()


def test_dlinear_deterministic_with_seed() -> None:
    ctx = _toy_ctx()
    a = dlinear_positions(ctx, epochs=2, seed=42)
    b = dlinear_positions(ctx, epochs=2, seed=42)
    assert np.allclose(a, b)


def test_tsmixer_positions_shape_and_bounded() -> None:
    ctx = _toy_ctx()
    positions = tsmixer_positions(ctx, epochs=2, seed=42)
    assert positions.shape == (len(ctx["next_log_returns"]),)
    assert ((positions >= -1.0 - 1e-6) & (positions <= 1.0 + 1e-6)).all()


def test_timemixer_positions_shape_and_bounded() -> None:
    ctx = _toy_ctx()
    positions = timemixer_positions(ctx, epochs=2, seed=42)
    assert positions.shape == (len(ctx["next_log_returns"]),)
    assert ((positions >= -1.0 - 1e-6) & (positions <= 1.0 + 1e-6)).all()


def test_xlstm_time_positions_shape_and_bounded() -> None:
    ctx = _toy_ctx()
    positions = xlstm_time_positions(ctx, epochs=2, seed=42)
    assert positions.shape == (len(ctx["next_log_returns"]),)
    assert ((positions >= -1.0 - 1e-6) & (positions <= 1.0 + 1e-6)).all()


def test_vlstm_positions_shape_and_bounded() -> None:
    ctx = _toy_ctx()
    positions = vlstm_positions(ctx, epochs=2, seed=42)
    assert positions.shape == (len(ctx["next_log_returns"]),)
    assert ((positions >= -1.0 - 1e-6) & (positions <= 1.0 + 1e-6)).all()


def test_forecast_to_fill_shape_and_bounded() -> None:
    ctx = _toy_ctx()
    positions = forecast_to_fill_positions(ctx)
    assert positions.shape == (len(ctx["next_log_returns"]),)
    # Within the configured max_position.
    assert ((positions >= -1.0 - 1e-6) & (positions <= 1.0 + 1e-6)).all()


def test_forecast_to_fill_dry_run_no_close() -> None:
    ctx = _toy_ctx()
    ctx.pop("close")
    positions = forecast_to_fill_positions(ctx)
    assert np.all(positions == 0.0)


def test_all_ml_baselines_zero_when_missing_train() -> None:
    """When ``train_features_window`` is absent every ML baseline returns zeros."""
    ctx = _toy_ctx()
    ctx.pop("train_features_window")
    for fn in (dlinear_positions, tsmixer_positions, timemixer_positions,
               xlstm_time_positions, vlstm_positions):
        positions = fn(ctx)
        assert np.all(positions == 0.0)
