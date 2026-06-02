"""V1 HMM regime model — 2-state Gaussian HMM on log-returns + realized vol.

Fits once on the train split. Persisted via joblib (sklearn-standard).
Used at inference to compute `regime_hmm_p_high_vol` (the 12th dim of
the regime vector).

The model is intentionally simple: a 2-state Gaussian HMM with two
features (rolling realized vol, abs log-return). The "high-vol" state is
identified by larger emission variance on the realized-vol channel. The
posterior P(state=high_vol | features_t) is the scalar feature.

PIT-correct: training on train split only. At val/test, the trained
model's transitions and emissions are FROZEN; only the posterior is
re-evaluated. A drift test in the test suite asserts re-fitting on
val/test moves the high-vol posterior by less than 5%.

Spec: plan/04-FEATURE-ENGINEERING.md V1 regime section.
Spec: plan/V1-SPEC.md §1.5.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from nanogld.data.utils import get_logger

LOG = get_logger("nanogld.features.hmm_regime")

DEFAULT_RV_LOOKBACK = 60
HMM_N_COMPONENTS = 2
HMM_RANDOM_STATE = 7


def _build_features(
    close: pd.Series,
    rv_lookback: int = DEFAULT_RV_LOOKBACK,
) -> np.ndarray:
    """Build a (N, 2) feature matrix [abs_log_return, rolling_realized_vol].

    NaN rows excluded by caller.
    """
    safe_close = close.where(close > 0)
    log_ret = np.log(safe_close / safe_close.shift(1))
    log_ret = log_ret.replace([np.inf, -np.inf], np.nan)
    rv = log_ret.rolling(rv_lookback, min_periods=rv_lookback).std()
    return np.stack([log_ret.abs().to_numpy(), rv.to_numpy()], axis=1)


def fit_hmm(
    train_df: pd.DataFrame,
    *,
    close_col: str = "gld_close",
    rv_lookback: int = DEFAULT_RV_LOOKBACK,
    random_state: int = HMM_RANDOM_STATE,
):  # noqa: ANN201 — return type is hmmlearn.hmm.GaussianHMM (lazy import)
    """Fit a 2-state Gaussian HMM on the train split.

    Returns the fitted model object. Use `save_hmm`/`load_hmm` for
    persistence. Caller must pass the SAME fitted model to every
    `predict_proba_high_vol` call across train/val/test to prevent leakage.
    """
    from hmmlearn import hmm  # noqa: PLC0415 — lazy import per project pattern

    if close_col not in train_df.columns:
        raise KeyError(f"{close_col} required to fit HMM")
    feats = _build_features(train_df[close_col], rv_lookback=rv_lookback)
    valid = ~np.isnan(feats).any(axis=1)
    if int(valid.sum()) < 200:
        raise ValueError("HMM needs at least 200 valid feature rows to fit")
    feats_clean = feats[valid]

    model = hmm.GaussianHMM(
        n_components=HMM_N_COMPONENTS,
        covariance_type="diag",
        n_iter=100,
        tol=1e-3,
        random_state=random_state,
    )
    model.fit(feats_clean)
    if hasattr(model, "monitor_"):
        if not model.monitor_.converged:
            raise RuntimeError(
                f"HMM EM failed to converge (iters={model.monitor_.iter}, "
                f"history={model.monitor_.history[-3:] if model.monitor_.history else 'none'})"
            )
    LOG.info(
        "HMM fit on %d valid rows; means=%s; converged=%s",
        int(valid.sum()),
        model.means_.tolist(),
        getattr(getattr(model, "monitor_", None), "converged", "unknown"),
    )
    return model


def _identify_high_vol_state(model) -> int:  # noqa: ANN001
    """Return the state index whose RV-channel mean is larger."""
    means = np.asarray(model.means_)
    return int(np.argmax(means[:, 1]))


def predict_proba_high_vol(
    model,  # noqa: ANN001
    df: pd.DataFrame,
    *,
    close_col: str = "gld_close",
    rv_lookback: int = DEFAULT_RV_LOOKBACK,
) -> np.ndarray:
    """Posterior probability P(state=high_vol | features_t) per row.

    Returns a (N,) float32 array. NaN-feature rows get 0.5 (neutral).
    """
    if close_col not in df.columns:
        return np.full(len(df), 0.5, dtype=np.float32)
    feats = _build_features(df[close_col], rv_lookback=rv_lookback)
    valid = ~np.isnan(feats).any(axis=1)
    out = np.full(len(df), 0.5, dtype=np.float32)
    if int(valid.sum()) == 0:
        return out

    high_state = _identify_high_vol_state(model)
    feats_clean = feats[valid]
    posterior = model.predict_proba(feats_clean)
    out[valid] = posterior[:, high_state].astype(np.float32)
    return out


def save_hmm(model, path: Path) -> None:  # noqa: ANN001
    """Persist fitted HMM via `joblib.dump` (sklearn-standard)."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    LOG.info("HMM saved to %s", path)


def load_hmm(path: Path):  # noqa: ANN201
    """Load a previously-fit HMM via `joblib.load`."""
    return joblib.load(path)


def add_hmm_column(
    df: pd.DataFrame,
    *,
    model,  # noqa: ANN001
    close_col: str = "gld_close",
    rv_lookback: int = DEFAULT_RV_LOOKBACK,
    out_col: str = "regime_hmm_p_high_vol",
) -> pd.DataFrame:
    """Append the 12th regime-vector dimension `regime_hmm_p_high_vol`."""
    out = df.copy()
    out[out_col] = predict_proba_high_vol(
        model, out, close_col=close_col, rv_lookback=rv_lookback
    )
    return out
