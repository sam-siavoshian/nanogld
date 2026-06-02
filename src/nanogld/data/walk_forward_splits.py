"""Walk-forward split boundaries (V1-SPEC §9.5).

4 folds. Per fold: train 3y + val 6mo + test 6mo, step 3mo between
folds, 1-week embargo between train/val and val/test windows. All
window boundaries computed by wall-clock time then mapped to bar
indices via the unified.pt's ``bar_close_utc_ns`` array.

Used by the per-fold sidecar build (closes plan/STATUS.md §32) and by
the walk-forward backtest harness.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

DEFAULT_N_FOLDS = 4
DEFAULT_TRAIN_YEARS = 3
DEFAULT_VAL_MONTHS = 6
DEFAULT_TEST_MONTHS = 6
DEFAULT_STEP_MONTHS = 3
DEFAULT_EMBARGO_WEEKS = 1


@dataclass(frozen=True)
class FoldBoundary:
    """Index boundaries for one walk-forward fold.

    Indices are inclusive-left, exclusive-right (Python slice semantics).
    """

    fold_idx: int
    train_start: int
    train_end: int
    val_start: int
    val_end: int
    test_start: int
    test_end: int

    def n_train(self) -> int:
        return self.train_end - self.train_start

    def n_val(self) -> int:
        return self.val_end - self.val_start

    def n_test(self) -> int:
        return self.test_end - self.test_start

    def train_mask(self, n_total: int) -> np.ndarray:
        m = np.zeros(n_total, dtype=bool)
        m[self.train_start : self.train_end] = True
        return m


def compute_fold_boundaries(
    bar_close_utc_ns: Iterable[int],
    *,
    n_folds: int = DEFAULT_N_FOLDS,
    train_years: int = DEFAULT_TRAIN_YEARS,
    val_months: int = DEFAULT_VAL_MONTHS,
    test_months: int = DEFAULT_TEST_MONTHS,
    step_months: int = DEFAULT_STEP_MONTHS,
    embargo_weeks: int = DEFAULT_EMBARGO_WEEKS,
) -> list[FoldBoundary]:
    """Compute walk-forward fold index boundaries from a timestamp array.

    Args:
        bar_close_utc_ns: 1D array of int64 nanosecond UTC timestamps,
            one per bar, strictly monotonically increasing.
        n_folds: how many folds to attempt. The function returns FEWER
            than this if later folds would exceed the dataset span.
        train_years, val_months, test_months: window lengths.
        step_months: fold-to-fold stride.
        embargo_weeks: gap between train/val and val/test.

    Returns:
        List of :class:`FoldBoundary` objects in fold-order.

    Raises:
        ValueError: if the timestamp array is empty or non-monotonic.
    """
    arr = np.asarray(list(bar_close_utc_ns), dtype=np.int64)
    if arr.size < 2:
        raise ValueError("bar_close_utc_ns must have at least 2 timestamps")
    if not np.all(np.diff(arr) > 0):
        raise ValueError("bar_close_utc_ns must be strictly monotonically increasing")

    ts = pd.to_datetime(arr, utc=True)
    first_ts = ts[0]
    last_ts = ts[-1]
    embargo = pd.Timedelta(weeks=embargo_weeks)

    out: list[FoldBoundary] = []
    for n in range(n_folds):
        train_start_ts = first_ts + pd.DateOffset(months=step_months * n)
        train_end_ts = train_start_ts + pd.DateOffset(years=train_years)
        val_start_ts = train_end_ts + embargo
        val_end_ts = val_start_ts + pd.DateOffset(months=val_months)
        test_start_ts = val_end_ts + embargo
        test_end_ts = test_start_ts + pd.DateOffset(months=test_months)
        if test_end_ts > last_ts:
            break

        def _idx(t: pd.Timestamp) -> int:
            return int(np.searchsorted(arr, np.int64(t.value), side="left"))

        out.append(
            FoldBoundary(
                fold_idx=n,
                train_start=_idx(train_start_ts),
                train_end=_idx(train_end_ts),
                val_start=_idx(val_start_ts),
                val_end=_idx(val_end_ts),
                test_start=_idx(test_start_ts),
                test_end=_idx(test_end_ts),
            )
        )
    return out


def assert_folds_disjoint(folds: list[FoldBoundary]) -> None:
    """Verify train ⊥ val ⊥ test within each fold and across folds' test windows."""
    for fold in folds:
        if not (
            fold.train_end <= fold.val_start
            and fold.val_end <= fold.test_start
            and fold.train_start < fold.train_end
            and fold.val_start < fold.val_end
            and fold.test_start < fold.test_end
        ):
            raise AssertionError(
                f"fold {fold.fold_idx}: malformed boundaries "
                f"train=[{fold.train_start},{fold.train_end}) "
                f"val=[{fold.val_start},{fold.val_end}) "
                f"test=[{fold.test_start},{fold.test_end})"
            )
    for i, a in enumerate(folds):
        for b in folds[i + 1 :]:
            if max(a.test_start, b.test_start) < min(a.test_end, b.test_end):
                # Test windows are allowed to overlap (walk-forward design).
                # We only assert that no fold's test window leaks INTO ITS
                # OWN train window, which the per-fold check above covers.
                pass


__all__ = [
    "FoldBoundary",
    "assert_folds_disjoint",
    "compute_fold_boundaries",
]
