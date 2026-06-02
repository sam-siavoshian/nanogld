"""XGBoost 3-class baseline.

Train on flattened per-bar features, predict ``argmax(prob) - 1`` mapped
to position weight in ``{-1, 0, +1}``. Per V1-SPEC §0 the simpler
ensemble of Gao 2014 + XGBoost is the must-beat bar — if nanoGLD does
not beat this by >= 0.2 Sharpe, ship the simpler ensemble instead.

Training inputs are pulled from the walk-forward fold context:

    ctx["train_features"]: (N_train, F) float
    ctx["train_labels"]:   (N_train,)   int in {0, 1, 2}   (DOWN, FLAT, UP)
    ctx["test_features"]:  (N_test, F)  float

Returns ``(N_test,)`` position weights.

V1 config defaults per V1-SPEC §3.4 (XGBoost as the simple-ensemble bar):

    n_estimators=500, max_depth=6, lr=0.05, subsample=0.8,
    colsample_bytree=0.8, multi:softprob.

**Platform note (darwin/arm64):** xgboost 3.2 segfaults inside
``data.py:_from_numpy_array`` when invoked from a pytest worker — the
direct Python REPL is stable, but pytest is not. The corresponding test
is skipped on darwin via ``sys.platform`` guard. Linux production runs
(GTX Spark x86_64) use the linux-x86_64 wheel where this issue does
not manifest. Verify on the target box before submitting a real run.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def xgboost_positions(
    ctx: dict[str, Any],
    *,
    n_estimators: int = 500,
    max_depth: int = 6,
    learning_rate: float = 0.05,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    early_stopping_rounds: int = 50,
    random_state: int = 0,
    val_frac: float = 0.1,
) -> np.ndarray:
    """Fit XGBoost on train slice; emit ±1 / 0 positions for test slice.

    Args:
        ctx: fold context with ``train_features``, ``train_labels``,
            ``test_features``. If the train arrays are missing or the
            label set has fewer than two classes, returns a zero
            position vector for the test slice.

    Returns:
        ``(N_test,)`` numpy array of position weights.
    """
    import xgboost as xgb  # noqa: PLC0415

    test_n = len(ctx["next_log_returns"])
    train_x = ctx.get("train_features")
    train_y = ctx.get("train_labels")
    test_x = ctx.get("test_features")
    if train_x is None or train_y is None or test_x is None:
        # Smoke / dry-run path: no train data available — return zeros so
        # the harness exercises the shape contract without falsely
        # claiming an edge.
        return np.zeros(test_n, dtype=np.float64)

    train_x = np.asarray(train_x, dtype=np.float64)
    train_y = np.asarray(train_y, dtype=np.int64)
    test_x = np.asarray(test_x, dtype=np.float64)
    if len(np.unique(train_y)) < 2:
        return np.zeros(test_n, dtype=np.float64)

    # XGBoost 3.x sklearn wrapper segfaults on darwin/arm64 inside its
    # numpy meta-conversion (see xgboost/data.py:_meta_from_numpy). The
    # low-level Booster + DMatrix path is stable. Use it.
    dtrain = xgb.DMatrix(np.ascontiguousarray(train_x), label=np.ascontiguousarray(train_y.astype(np.float32)))
    dtest = xgb.DMatrix(np.ascontiguousarray(test_x))
    params: dict[str, Any] = {
        "objective": "multi:softprob",
        "num_class": 3,
        "max_depth": max_depth,
        "learning_rate": learning_rate,
        "subsample": subsample,
        "colsample_bytree": colsample_bytree,
        "verbosity": 0,
        "nthread": 1,
        "seed": random_state,
    }
    booster = xgb.train(params, dtrain, num_boost_round=n_estimators)
    probs = booster.predict(dtest)  # (N_test, 3)
    preds = np.argmax(probs, axis=-1).astype(np.int64)
    positions = (preds - 1).astype(np.float64)
    if positions.shape[0] != test_n:
        raise RuntimeError(
            f"xgboost_positions: predicted {positions.shape[0]} positions, "
            f"expected {test_n}"
        )
    return positions


__all__ = ["xgboost_positions"]
