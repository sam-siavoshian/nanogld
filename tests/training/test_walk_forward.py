"""Unit tests for walk-forward CV."""

from __future__ import annotations

import pandas as pd
import pytest

from nanogld.training.walk_forward import (
    DEFAULT_EMBARGO_DAYS,
    Fold,
    split_within_fold,
    walk_forward_folds,
)


@pytest.mark.smoke
def test_yields_4_folds_on_5y() -> None:
    """5y span with 3y train + 6mo val + 6mo test, 3mo step → 4 folds."""
    ts = pd.date_range("2019-01-01", "2024-01-01", freq="30min", tz="UTC").to_series()
    folds = list(walk_forward_folds(ts))
    assert len(folds) == 4


@pytest.mark.smoke
def test_no_train_val_overlap() -> None:
    ts = pd.date_range("2019-01-01", "2025-01-01", freq="30min", tz="UTC").to_series()
    for f in walk_forward_folds(ts):
        assert f.train_end < f.val_start
        assert f.val_end < f.test_start


@pytest.mark.smoke
def test_embargo_at_least_a_week() -> None:
    ts = pd.date_range("2019-01-01", "2025-01-01", freq="30min", tz="UTC").to_series()
    expected_gap = pd.Timedelta(days=DEFAULT_EMBARGO_DAYS)
    for f in walk_forward_folds(ts):
        assert (f.val_start - f.train_end) >= expected_gap
        assert (f.test_start - f.val_end) >= expected_gap


@pytest.mark.smoke
def test_split_within_fold_sums_to_full() -> None:
    s = split_within_fold(val_window_size=100)
    assert s.val_a_slice.start == 0
    assert s.val_a_slice.stop == 50
    assert s.val_b_slice.start == 50
    assert s.val_c_slice.stop == 100


@pytest.mark.smoke
def test_empty_timestamps_yields_nothing() -> None:
    ts = pd.Series([], dtype="datetime64[ns, UTC]")
    folds = list(walk_forward_folds(ts))
    assert folds == []


@pytest.mark.smoke
def test_fold_dataclass_frozen() -> None:
    f = Fold(
        fold_idx=0,
        train_start=pd.Timestamp("2019-01-01", tz="UTC"),
        train_end=pd.Timestamp("2022-01-01", tz="UTC"),
        val_start=pd.Timestamp("2022-01-08", tz="UTC"),
        val_end=pd.Timestamp("2022-07-08", tz="UTC"),
        test_start=pd.Timestamp("2022-07-15", tz="UTC"),
        test_end=pd.Timestamp("2023-01-15", tz="UTC"),
    )
    from dataclasses import FrozenInstanceError

    with pytest.raises(FrozenInstanceError):
        f.fold_idx = 99  # type: ignore[misc]
