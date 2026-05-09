"""Per-bucket eval — {news-present, news-absent, both}.

V1 invariant 18 (HARD): every reported metric must be split by news
presence. Without this we fly blind on the 51% no-news bars.

Spec: plan/V1-SPEC.md §9.1.
"""

from __future__ import annotations

import numpy as np

from nanogld.backtest.metrics import compute_metrics


def per_bucket_metrics(
    pnl_per_bar: np.ndarray,
    equity_curve: np.ndarray,
    is_news_present: np.ndarray,
) -> dict[str, dict[str, float]]:
    """Compute metrics for {present, absent, both} buckets.

    Args:
        pnl_per_bar: (T,).
        equity_curve: (T,).
        is_news_present: (T,) bool or 0/1.

    Returns:
        Dict mapping bucket name to its metrics dict.
    """
    pnl = np.asarray(pnl_per_bar, dtype=np.float64)
    eq = np.asarray(equity_curve, dtype=np.float64)
    mask = np.asarray(is_news_present, dtype=bool)

    out: dict[str, dict[str, float]] = {}
    out["both"] = compute_metrics(pnl, eq)

    if mask.any():
        idx = np.where(mask)[0]
        bucket_pnl = pnl[idx]
        bucket_eq = np.exp(np.cumsum(bucket_pnl))
        out["present"] = compute_metrics(bucket_pnl, bucket_eq)
    else:
        out["present"] = compute_metrics(np.array([]), np.array([1.0]))

    if (~mask).any():
        idx = np.where(~mask)[0]
        bucket_pnl = pnl[idx]
        bucket_eq = np.exp(np.cumsum(bucket_pnl))
        out["absent"] = compute_metrics(bucket_pnl, bucket_eq)
    else:
        out["absent"] = compute_metrics(np.array([]), np.array([1.0]))
    return out
