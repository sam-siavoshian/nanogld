"""V1 walk-forward CV with 1-week embargo.

Specs:
- 4 folds across 5y of data.
- Train 3y + val 6mo + test 6mo.
- Step 3mo between folds.
- 1-week embargo between train_end and val_start (and val_end and test_start).
- Within-fold val splits: val_a 50% (early stop), val_b 25% (T-scaling),
    val_c 25% (conformal). Strictly chronological — NO shuffling.
- bars_per_year = 3276 (NYSE RTH 30-min, V1 invariant 5).

Spec: plan/06-BACKTEST.md V1 walk-forward.
Spec: plan/V1-SPEC.md §9.5.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import pandas as pd

DEFAULT_TRAIN_YEARS = 3.0
DEFAULT_VAL_MONTHS = 6.0
DEFAULT_TEST_MONTHS = 6.0
DEFAULT_STEP_MONTHS = 3.0
DEFAULT_EMBARGO_DAYS = 7


@dataclass(frozen=True)
class Fold:
    """One walk-forward fold definition (timestamps in UTC).

    Embargo gaps appear between train_end and val_start (and val_end and
    test_start) so feature windows from train don't leak into val/test.
    """

    fold_idx: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    val_start: pd.Timestamp
    val_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


@dataclass(frozen=True)
class WithinFoldSplit:
    """Indices for val_a / val_b / val_c within a fold's val window."""

    val_a_slice: slice
    val_b_slice: slice
    val_c_slice: slice


def walk_forward_folds(
    timestamps: pd.Series,
    *,
    train_years: float = DEFAULT_TRAIN_YEARS,
    val_months: float = DEFAULT_VAL_MONTHS,
    test_months: float = DEFAULT_TEST_MONTHS,
    step_months: float = DEFAULT_STEP_MONTHS,
    embargo_days: int = DEFAULT_EMBARGO_DAYS,
) -> Iterator[Fold]:
    """Yield Fold objects covering `timestamps`.

    Args:
        timestamps: monotonically-increasing UTC timestamps Series.
        train_years: train window length in years.
        val_months: val window in months.
        test_months: test window in months.
        step_months: how far each fold advances vs the previous.
        embargo_days: gap inserted between train/val and val/test.

    Yields:
        Fold(...) until the next test_end exceeds timestamps.max().
    """
    if len(timestamps) == 0:
        return
    ts = pd.to_datetime(timestamps, utc=True)
    if not ts.is_monotonic_increasing:
        ts = ts.sort_values().reset_index(drop=True)

    earliest = ts.iloc[0]
    latest = ts.iloc[-1]

    embargo = pd.Timedelta(days=embargo_days)
    train_window = pd.DateOffset(months=int(round(train_years * 12)))
    val_window = pd.DateOffset(months=int(round(val_months)))
    test_window = pd.DateOffset(months=int(round(test_months)))
    step = pd.DateOffset(months=int(round(step_months)))

    fold_idx = 0
    train_start = earliest
    while True:
        train_end = train_start + train_window
        val_start = train_end + embargo
        val_end = val_start + val_window
        test_start = val_end + embargo
        test_end = test_start + test_window
        if test_end > latest:
            break
        yield Fold(
            fold_idx=fold_idx,
            train_start=train_start,
            train_end=train_end,
            val_start=val_start,
            val_end=val_end,
            test_start=test_start,
            test_end=test_end,
        )
        fold_idx += 1
        train_start = train_start + step


def split_within_fold(val_window_size: int) -> WithinFoldSplit:
    """Compute val_a (50%) / val_b (25%) / val_c (25%) chronological slices.

    Args:
        val_window_size: number of bars in the val window.

    Returns:
        WithinFoldSplit object with three contiguous slices.
    """
    half = val_window_size // 2
    quarter = val_window_size // 4
    return WithinFoldSplit(
        val_a_slice=slice(0, half),
        val_b_slice=slice(half, half + quarter),
        val_c_slice=slice(half + quarter, val_window_size),
    )
