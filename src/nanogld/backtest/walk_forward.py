"""Walk-forward orchestrator (V1 promotion gates + per-bucket reporting).

Per fold:
    1. Load model checkpoint + per-fold sidecar.
    2. Slice the test window with the V1-SPEC §9.5 1-week embargo.
    3. Run the model: forward -> ``predict_calibrated`` -> sizer -> positions.
    4. Run every baseline in ``baselines_dict`` on the same slice.
    5. For each strategy (model + baselines), run ``cost_stress`` at
       {0.5x, 1.0x, 1.5x}, ``per_bucket`` metrics over the news-presence
       mask, and ``deflated_sharpe`` for the 1.0x cost case.
    6. Bundle into ``FoldResult`` and aggregate across folds.

The orchestrator is split into two layers:

- :func:`evaluate_strategy_positions`: pure-Python, takes pre-computed
  positions + realized returns + news mask, returns the full metric
  bundle. Unit-testable without checkpoints or model weights.
- :func:`walk_forward`: high-level loop that loads each fold's
  ``(checkpoint, sidecar)`` pair, exercises the model, and aggregates.
  Heavy — needs real artifacts.

Spec: plan/06-BACKTEST.md V1 backtest.
Spec: plan/V1-SPEC.md §9.4 (8 promotion gates) + §9.5 (4-fold walk-forward).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from nanogld.backtest.cost_stress import CostStressResult, assert_monotone, cost_stress
from nanogld.backtest.dsr import deflated_sharpe
from nanogld.backtest.engine import BacktestConfig, vectorized_backtest
from nanogld.backtest.metrics import compute_metrics
from nanogld.backtest.per_bucket import per_bucket_metrics

# A strategy producer is a callable that takes a test-slice context dict
# and returns a (T,) numpy array of position weights. The context dict
# carries everything a baseline could need (features, prices, h5, etc.).
StrategyFn = Callable[[dict[str, Any]], np.ndarray]


@dataclass(frozen=True)
class StrategyResult:
    """Metrics for one strategy on one fold."""

    name: str
    positions: np.ndarray
    cost_stress: CostStressResult
    per_bucket: dict[str, dict[str, float]]
    dsr_p_value: float
    dsr_value: float
    base_metrics: dict[str, float]


@dataclass(frozen=True)
class FoldResult:
    """One fold's results: model + all baselines."""

    fold_idx: int
    strategies: dict[str, StrategyResult]
    test_n_bars: int
    test_news_present_frac: float
    extras: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class WalkForwardResult:
    """Cross-fold aggregate of all strategies."""

    folds: list[FoldResult]
    cost_multipliers: tuple[float, ...]
    n_strategies: int

    def all_strategy_names(self) -> list[str]:
        names: list[str] = []
        for fold in self.folds:
            for name in fold.strategies:
                if name not in names:
                    names.append(name)
        return names


def evaluate_strategy_positions(
    name: str,
    positions: np.ndarray,
    next_log_returns: np.ndarray,
    is_news_present: np.ndarray,
    *,
    base_cost_bps: float = 2.0,
    cost_multipliers: Sequence[float] = (0.5, 1.0, 1.5),
    n_trials: int = 1,
    strict_monotone: bool = False,
) -> StrategyResult:
    """Compute the full V1 metric bundle for one strategy's positions.

    Args:
        name: strategy name (for reporting only).
        positions: (T,) position weights in [-1, +1].
        next_log_returns: (T,) realized log return per bar.
        is_news_present: (T,) bool — True iff the bar had visible news.
        base_cost_bps: 1.0x cost in basis points.
        cost_multipliers: cost-stress multipliers (V1 hard gate at 1.5x).
        n_trials: passed to ``deflated_sharpe`` selection-penalty count.
        strict_monotone: when True, ``assert_monotone`` is run on the
            cost-stress result and raises on failure. Default False so a
            backtest sanity check produces a report rather than crashing.

    Returns:
        :class:`StrategyResult` with all V1 metric outputs.
    """
    nlr = np.asarray(next_log_returns, dtype=np.float64)
    pos = np.asarray(positions, dtype=np.float64)
    mask = np.asarray(is_news_present, dtype=bool)
    if pos.shape != nlr.shape:
        raise ValueError(
            f"position/return shape mismatch: {pos.shape} vs {nlr.shape}"
        )
    if mask.shape != pos.shape:
        raise ValueError(
            f"news mask shape mismatch: {mask.shape} vs {pos.shape}"
        )

    # 1.0x cost run drives base_metrics + per_bucket + DSR.
    base_cfg = BacktestConfig(cost_bps=base_cost_bps)
    base_result = vectorized_backtest(nlr, pos, cfg=base_cfg)
    base_metrics = compute_metrics(base_result.pnl_per_bar, base_result.equity_curve)
    bucket = per_bucket_metrics(base_result.pnl_per_bar, base_result.equity_curve, mask)
    p_val, dsr_value = deflated_sharpe(
        sharpe_observed=base_metrics["sharpe"],
        n_trials=n_trials,
        n_obs=int(nlr.shape[0]),
    )

    cs = cost_stress(
        nlr, pos, base_cost_bps=base_cost_bps, multipliers=tuple(cost_multipliers)
    )
    if strict_monotone:
        assert_monotone(cs)

    return StrategyResult(
        name=name,
        positions=pos,
        cost_stress=cs,
        per_bucket=bucket,
        dsr_p_value=p_val,
        dsr_value=dsr_value,
        base_metrics=base_metrics,
    )


def run_fold(
    fold_idx: int,
    context: dict[str, Any],
    strategies: dict[str, StrategyFn],
    *,
    base_cost_bps: float = 2.0,
    cost_multipliers: Sequence[float] = (0.5, 1.0, 1.5),
    n_trials: int = 1,
) -> FoldResult:
    """Run every strategy for one fold's context.

    Args:
        fold_idx: 0-indexed fold.
        context: dict expected to carry at least
            ``next_log_returns: (T,)``, ``is_news_present: (T,) bool``.
            Strategy functions consume whatever else they need.
        strategies: ``{name: callable(context) -> positions}``.
        base_cost_bps, cost_multipliers, n_trials: passed to
            ``evaluate_strategy_positions``.
    """
    if "next_log_returns" not in context:
        raise ValueError("fold context missing required key 'next_log_returns'")
    if "is_news_present" not in context:
        raise ValueError("fold context missing required key 'is_news_present'")

    nlr = np.asarray(context["next_log_returns"], dtype=np.float64)
    mask = np.asarray(context["is_news_present"], dtype=bool)
    n_strategies = len(strategies)

    results: dict[str, StrategyResult] = {}
    for name, fn in strategies.items():
        positions = np.asarray(fn(context), dtype=np.float64)
        results[name] = evaluate_strategy_positions(
            name=name,
            positions=positions,
            next_log_returns=nlr,
            is_news_present=mask,
            base_cost_bps=base_cost_bps,
            cost_multipliers=cost_multipliers,
            n_trials=n_strategies,  # multi-strategy selection penalty
        )

    return FoldResult(
        fold_idx=fold_idx,
        strategies=results,
        test_n_bars=int(nlr.shape[0]),
        test_news_present_frac=float(mask.mean()) if mask.size > 0 else 0.0,
    )


def walk_forward(
    fold_contexts: Sequence[dict[str, Any]],
    strategies: dict[str, StrategyFn],
    *,
    base_cost_bps: float = 2.0,
    cost_multipliers: Sequence[float] = (0.5, 1.0, 1.5),
) -> WalkForwardResult:
    """Run every strategy on every fold's pre-built test context.

    The orchestrator does not load checkpoints itself — callers are
    expected to assemble the per-fold context dict (running the model
    once per fold and threading the resulting positions through a
    ``StrategyFn`` named ``"nanogld_v1"``). This separation keeps
    walk-forward testable: pass synthetic contexts in unit tests, real
    contexts in production runs.

    Args:
        fold_contexts: list of context dicts (one per fold). Order
            defines fold_idx.
        strategies: dict of strategy producers.
        base_cost_bps: 1.0x cost in basis points.
        cost_multipliers: cost-stress multipliers.

    Returns:
        :class:`WalkForwardResult`.
    """
    if not fold_contexts:
        raise ValueError("walk_forward: fold_contexts must be non-empty")
    if not strategies:
        raise ValueError("walk_forward: strategies must be non-empty")

    fold_results = [
        run_fold(
            fold_idx=i,
            context=ctx,
            strategies=strategies,
            base_cost_bps=base_cost_bps,
            cost_multipliers=cost_multipliers,
        )
        for i, ctx in enumerate(fold_contexts)
    ]
    return WalkForwardResult(
        folds=fold_results,
        cost_multipliers=tuple(cost_multipliers),
        n_strategies=len(strategies),
    )


__all__ = [
    "FoldResult",
    "StrategyFn",
    "StrategyResult",
    "WalkForwardResult",
    "evaluate_strategy_positions",
    "run_fold",
    "walk_forward",
]
