"""Regression locks for walk-forward fold boundaries (V1-SPEC §9.5 / §32)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nanogld.data.walk_forward_splits import (
    FoldBoundary,
    assert_folds_disjoint,
    compute_fold_boundaries,
)


def _synthetic_ts(years: float = 10.5, freq_min: int = 30) -> np.ndarray:
    start = pd.Timestamp("2016-01-01T00:00:00Z")
    n = int(years * 365.25 * 24 * 60 / freq_min)
    rng = pd.date_range(start, periods=n, freq=f"{freq_min}min")
    return rng.view("int64")  # ns since epoch


def test_default_returns_four_folds_for_10y_span() -> None:
    ts = _synthetic_ts(years=10.5)
    folds = compute_fold_boundaries(ts)
    assert len(folds) == 4
    for n, fb in enumerate(folds):
        assert fb.fold_idx == n


def test_fold_windows_have_correct_relative_widths() -> None:
    ts = _synthetic_ts(years=10.5)
    folds = compute_fold_boundaries(ts)
    # Train should be ~6x val and ~6x test (3y vs 6mo vs 6mo).
    fb = folds[0]
    assert fb.n_train() > 5 * fb.n_val()
    assert fb.n_train() > 5 * fb.n_test()


def test_embargo_present_between_train_val_and_val_test() -> None:
    ts = _synthetic_ts(years=10.5)
    folds = compute_fold_boundaries(ts)
    fb = folds[0]
    # 1-week embargo at 30-min RTH cadence => non-zero gap.
    assert fb.val_start > fb.train_end
    assert fb.test_start > fb.val_end


def test_assert_folds_disjoint_passes_for_default() -> None:
    ts = _synthetic_ts(years=10.5)
    folds = compute_fold_boundaries(ts)
    assert_folds_disjoint(folds)


def test_assert_folds_disjoint_rejects_malformed() -> None:
    bad = [
        FoldBoundary(
            fold_idx=0,
            train_start=0,
            train_end=10,
            val_start=5,  # overlaps train
            val_end=20,
            test_start=25,
            test_end=30,
        )
    ]
    with pytest.raises(AssertionError, match="malformed"):
        assert_folds_disjoint(bad)


def test_step_months_advances_train_start() -> None:
    ts = _synthetic_ts(years=10.5)
    folds = compute_fold_boundaries(ts)
    for prev, nxt in zip(folds, folds[1:]):
        assert nxt.train_start > prev.train_start
        assert nxt.test_start > prev.test_start


def test_empty_input_rejected() -> None:
    with pytest.raises(ValueError, match="at least 2"):
        compute_fold_boundaries(np.array([], dtype=np.int64))


def test_non_monotonic_rejected() -> None:
    arr = np.array([3, 2, 1], dtype=np.int64)
    with pytest.raises(ValueError, match="monotonically"):
        compute_fold_boundaries(arr)


def test_short_span_returns_fewer_folds() -> None:
    """Dataset shorter than 4 fold-spans => correspondingly fewer folds."""
    ts = _synthetic_ts(years=3.8)  # barely enough for fold 0
    folds = compute_fold_boundaries(ts)
    assert len(folds) <= 2  # fold 0 fits; fold 1 may or may not


def test_train_mask_shape() -> None:
    ts = _synthetic_ts(years=10.5)
    folds = compute_fold_boundaries(ts)
    mask = folds[0].train_mask(n_total=len(ts))
    assert mask.shape == (len(ts),)
    assert mask.sum() == folds[0].n_train()
