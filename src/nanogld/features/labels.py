"""3-class direction labels — doc 04 §labels (lines 631-654).

For bar T, label is based on bar T+1's return:
  next_log_ret = log(close[T+1] / close[T])
  UP   (2) if next_log_ret > +threshold
  DOWN (0) if next_log_ret < -threshold
  FLAT (1) otherwise

Default threshold = 5bps (0.0005 in log return units).
Spec target class distribution at 5bps on 30min GLD: ~28/44/28 (DOWN/FLAT/UP).

Last bar of dataset has no T+1 → label = NaN, drop at training time.
Last bar of trading day → next bar is overnight gap (or AM premarket).
We compute label anyway; downstream training can choose to mask overnight
predictions if desired.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from nanogld.data.utils import get_logger

LOG = get_logger("nanogld.features.labels")

DEFAULT_THRESHOLD_BPS = 5
DOWN, FLAT, UP = 0, 1, 2
DEFAULT_TRAIN_END = pd.Timestamp("2023-12-31", tz="UTC")
DEFAULT_VAL_END = pd.Timestamp("2024-12-31", tz="UTC")


def make_labels(close: pd.Series, threshold_bps: int = DEFAULT_THRESHOLD_BPS) -> pd.Series:
    """3-class label per bar based on next bar's log return.

    Args:
        close: bar close price series, sorted ascending by time.
        threshold_bps: ±threshold in basis points (1bps = 0.01%). Default 5.

    Returns:
        int-typed Series with values {0=DOWN, 1=FLAT, 2=UP}, NaN for last bar.
    """
    if len(close) < 2:
        return pd.Series([], dtype="float64")
    next_close = close.shift(-1)
    next_log_return = np.log(next_close / close)
    threshold = threshold_bps / 10_000.0

    labels = pd.Series(np.full(len(close), FLAT, dtype="float64"), index=close.index, name="label")
    labels[next_log_return > threshold] = UP
    labels[next_log_return < -threshold] = DOWN
    labels[next_log_return.isna()] = np.nan
    return labels


def class_distribution(labels: pd.Series) -> dict[str, float]:
    """Return {DOWN: pct, FLAT: pct, UP: pct} excluding NaN."""
    valid = labels.dropna()
    if valid.empty:
        return {"DOWN": 0.0, "FLAT": 0.0, "UP": 0.0}
    n = len(valid)
    return {
        "DOWN": float((valid == DOWN).sum()) / n,
        "FLAT": float((valid == FLAT).sum()) / n,
        "UP": float((valid == UP).sum()) / n,
    }


def class_weights(labels: pd.Series) -> dict[int, float]:
    """Inverse-frequency weights for cross-entropy: len/(K*count_per_class)."""
    valid = labels.dropna()
    if valid.empty:
        return {DOWN: 1.0, FLAT: 1.0, UP: 1.0}
    n = len(valid)
    out: dict[int, float] = {}
    for k in (DOWN, FLAT, UP):
        cnt = int((valid == k).sum())
        out[k] = float(n) / (3.0 * max(cnt, 1))
    return out


def add_labels_and_splits(
    df: pd.DataFrame,
    *,
    close_col: str = "gld_close",
    threshold_bps: int = DEFAULT_THRESHOLD_BPS,
    train_end: pd.Timestamp = DEFAULT_TRAIN_END,
    val_end: pd.Timestamp = DEFAULT_VAL_END,
) -> pd.DataFrame:
    """Append label + split columns to a bar-aligned snapshot.

    label:        int {0,1,2} per doc 04 spec, NaN for last row
    label_split:  string in {'train', 'val', 'test'} based on bar_close_utc
    next_log_return: float64 — useful for triple-barrier or vol-scaled labels later

    Defaults split: train ≤ 2023-12-31, val 2024 calendar year, test 2025+.
    For 10y window 2016-2026 this is ~80/10/10 by years.
    """
    out = df.copy()
    if close_col not in out.columns:
        LOG.warning("close_col %r missing — labels skipped", close_col)
        out["label"] = np.nan
        out["next_log_return"] = np.nan
        out["label_split"] = "test"
        return out

    out = out.sort_values("bar_close_utc").reset_index(drop=True)
    next_close = out[close_col].shift(-1)
    out["next_log_return"] = np.log(next_close / out[close_col])
    out["label"] = make_labels(out[close_col], threshold_bps=threshold_bps)

    # Time-series split: leakage-free, no random shuffling.
    bc = pd.to_datetime(out["bar_close_utc"], utc=True)
    split = pd.Series("train", index=out.index, dtype="string")
    split[bc > train_end] = "val"
    split[bc > val_end] = "test"
    out["label_split"] = split

    dist = class_distribution(out["label"])
    LOG.info(
        "labels added at %dbps: DOWN=%.1f%% FLAT=%.1f%% UP=%.1f%% (target 28/44/28)",
        threshold_bps,
        100 * dist["DOWN"],
        100 * dist["FLAT"],
        100 * dist["UP"],
    )
    train_n = int((out["label_split"] == "train").sum())
    val_n = int((out["label_split"] == "val").sum())
    test_n = int((out["label_split"] == "test").sum())
    LOG.info("split: train=%d val=%d test=%d", train_n, val_n, test_n)
    return out
