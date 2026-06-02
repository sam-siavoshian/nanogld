"""Unit tests for V1 HMM regime model."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("hmmlearn", reason="hmmlearn required")

from nanogld.features import hmm_regime


def _synthetic_two_regime_panel(n: int = 800, seed: int = 7) -> pd.DataFrame:
    """Half low-vol (sigma=0.001), half high-vol (sigma=0.01)."""
    rng = np.random.default_rng(seed)
    half = n // 2
    low_returns = rng.normal(0.0, 0.001, half)
    high_returns = rng.normal(0.0, 0.01, half)
    log_returns = np.concatenate([low_returns, high_returns])
    close = 100.0 * np.exp(np.cumsum(log_returns))
    bar_close = pd.date_range("2022-01-03 14:30:00+00:00", periods=n, freq="30min")
    return pd.DataFrame({"bar_close_utc": bar_close, "gld_close": close})


@pytest.mark.smoke
def test_fit_hmm_short_input_raises() -> None:
    df = pd.DataFrame({"gld_close": [100.0, 101.0]})
    with pytest.raises(ValueError):
        hmm_regime.fit_hmm(df, rv_lookback=10)


@pytest.mark.smoke
def test_fit_hmm_returns_two_states() -> None:
    df = _synthetic_two_regime_panel()
    model = hmm_regime.fit_hmm(df, rv_lookback=20)
    assert model.n_components == 2


@pytest.mark.smoke
def test_predict_proba_in_zero_one_range() -> None:
    df = _synthetic_two_regime_panel()
    model = hmm_regime.fit_hmm(df, rv_lookback=20)
    p = hmm_regime.predict_proba_high_vol(model, df, rv_lookback=20)
    valid = p[p != 0.5]
    assert (valid >= 0.0).all() and (valid <= 1.0).all()


@pytest.mark.smoke
def test_high_vol_state_concentrates_in_high_half() -> None:
    df = _synthetic_two_regime_panel(n=800)
    model = hmm_regime.fit_hmm(df, rv_lookback=20)
    p = hmm_regime.predict_proba_high_vol(model, df, rv_lookback=20)
    half = len(df) // 2
    avg_high_p_first_half = p[20:half].mean()
    avg_high_p_second_half = p[half:].mean()
    assert avg_high_p_second_half > avg_high_p_first_half


@pytest.mark.smoke
def test_save_load_round_trip(tmp_path) -> None:
    df = _synthetic_two_regime_panel()
    model = hmm_regime.fit_hmm(df, rv_lookback=20)
    path = tmp_path / "hmm.joblib"
    hmm_regime.save_hmm(model, path)
    reloaded = hmm_regime.load_hmm(path)
    assert reloaded.n_components == 2
    p_orig = hmm_regime.predict_proba_high_vol(model, df, rv_lookback=20)
    p_reloaded = hmm_regime.predict_proba_high_vol(reloaded, df, rv_lookback=20)
    np.testing.assert_array_almost_equal(p_orig, p_reloaded)


@pytest.mark.smoke
def test_add_hmm_column_appends_correct_name() -> None:
    df = _synthetic_two_regime_panel()
    model = hmm_regime.fit_hmm(df, rv_lookback=20)
    out = hmm_regime.add_hmm_column(df, model=model, rv_lookback=20)
    assert "regime_hmm_p_high_vol" in out.columns
    assert out["regime_hmm_p_high_vol"].dtype == np.float32


@pytest.mark.smoke
def test_predict_no_close_column_returns_neutral() -> None:
    df = pd.DataFrame({"x": [1, 2, 3]})

    class FakeModel:
        means_ = np.array([[0, 0.001], [0, 0.01]])

        def predict_proba(self, X):  # noqa: ANN001
            return np.full((len(X), 2), 0.5)

    p = hmm_regime.predict_proba_high_vol(FakeModel(), df)
    assert (p == 0.5).all()
