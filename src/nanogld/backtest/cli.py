"""CLI entrypoint for the V1 backtest.

Usage::

    uv run python -m nanogld.backtest run \\
        --config configs/v1_main.yaml \\
        --checkpoints fold_0/llrd/llrd_final.pt,fold_1/llrd/llrd_final.pt,... \\
        --sidecars sidecar_fold_0.pt,sidecar_fold_1.pt,... \\
        --out reports/backtest/

For early dry runs (no real checkpoints yet), pass ``--dry-run`` to skip
the model-inference step and exercise the harness against synthetic
positions (used by smoke tests and the integration CI).

The CLI is intentionally thin: it delegates work to
:func:`nanogld.backtest.walk_forward.walk_forward` and
:func:`nanogld.backtest.report.render_report`. Model inference plumbing
(``checkpoint -> predict_calibrated -> sizer -> positions``) lives in
``_run_model_strategy``.

Spec: plan/06-BACKTEST.md V1 backtest.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from nanogld.backtest.baselines import (
    buy_hold_positions,
    dlinear_positions,
    donchian_positions,
    forecast_to_fill_positions,
    gao_2014_positions,
    ma_cross_positions,
    timemixer_positions,
    tsmixer_positions,
    vlstm_positions,
    xgboost_positions,
    xlstm_time_positions,
)
from nanogld.backtest.production import build_production_contexts
from nanogld.backtest.report import MODEL_NAME, render_report
from nanogld.backtest.walk_forward import StrategyFn, walk_forward

LOG = logging.getLogger("nanogld.backtest.cli")


def _coerce_path_list(s: str) -> list[Path]:
    parts = [p.strip() for p in s.split(",") if p.strip()]
    return [Path(p) for p in parts]


def _dry_run_contexts(n_folds: int = 4, t: int = 200, seed: int = 0) -> list[dict[str, Any]]:
    """Synthetic per-fold contexts for harness smoke tests."""
    rng = np.random.default_rng(seed)
    contexts: list[dict[str, Any]] = []
    for f in range(n_folds):
        nlr = rng.normal(0.0, 0.001, size=t)
        close = 100.0 * np.exp(np.cumsum(nlr))
        h5 = rng.normal(0.0, 0.001, size=t)
        is_news_present = rng.random(t) > 0.5
        is_last_bar = np.zeros(t, dtype=bool)
        is_last_bar[::13] = True
        is_high_vol = rng.random(t) > 0.5
        # The "model" strategy under dry-run: random tiny positions, so
        # baselines stay strictly more interesting.
        contexts.append(
            {
                "fold_idx": f,
                "next_log_returns": nlr,
                "is_news_present": is_news_present,
                "close": close,
                "h5_log_return": h5,
                "is_high_vol": is_high_vol,
                "is_last_bar_of_day": is_last_bar,
                "_dry_run_model_positions": rng.normal(0.0, 0.3, size=t).clip(-1.0, 1.0),
            }
        )
    return contexts


def _default_strategies(*, hold_last_bar_only: bool = True) -> dict[str, StrategyFn]:
    """The 4 baselines currently in-tree + a model-positions slot.

    The model slot is filled at runtime: either by a real
    ``_run_model_strategy`` for production runs, or by the synthetic
    dry-run path which pulls from ``ctx["_dry_run_model_positions"]``.
    """

    def _buy_hold(ctx: dict[str, Any]) -> np.ndarray:
        return buy_hold_positions(n_bars=len(ctx["next_log_returns"]))

    def _ma_cross(ctx: dict[str, Any]) -> np.ndarray:
        return ma_cross_positions(ctx["close"], fast_span=10, slow_span=20)

    def _donchian(ctx: dict[str, Any]) -> np.ndarray:
        return donchian_positions(ctx["close"], window=20)

    def _gao(ctx: dict[str, Any]) -> np.ndarray:
        return gao_2014_positions(
            ctx["h5_log_return"],
            is_high_vol=ctx.get("is_high_vol"),
            is_last_bar_of_day=ctx.get("is_last_bar_of_day"),
            hold_last_bar_only=hold_last_bar_only,
        )

    def _model(ctx: dict[str, Any]) -> np.ndarray:
        # Production runs precompute model positions per fold and stash
        # them under '_production_model_positions'; the dry-run path
        # uses '_dry_run_model_positions'. Whichever is present, use.
        if "_production_model_positions" in ctx:
            return np.asarray(ctx["_production_model_positions"])
        return np.asarray(ctx["_dry_run_model_positions"])

    return {
        MODEL_NAME: _model,
        "buy_hold": _buy_hold,
        "ma_cross": _ma_cross,
        "donchian": _donchian,
        "gao_2014": _gao,
        "xgboost": xgboost_positions,
        "dlinear": dlinear_positions,
        "tsmixer": tsmixer_positions,
        "timemixer": timemixer_positions,
        "xlstm_time": xlstm_time_positions,
        "vlstm": vlstm_positions,
        "forecast_to_fill": forecast_to_fill_positions,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="nanogld.backtest", description="V1 backtest harness"
    )
    sub = parser.add_subparsers(dest="cmd", required=True)
    run_p = sub.add_parser("run", help="run V1 walk-forward backtest")
    run_p.add_argument("--config", type=Path, default=None)
    run_p.add_argument(
        "--checkpoints",
        type=_coerce_path_list,
        default=[],
        help="comma-separated per-fold llrd_final.pt paths",
    )
    run_p.add_argument(
        "--sidecars",
        type=_coerce_path_list,
        default=[],
        help="comma-separated per-fold sidecar.pt paths (post-#32 refactor)",
    )
    run_p.add_argument(
        "--out", type=Path, default=Path("reports/backtest"), help="report output dir"
    )
    run_p.add_argument("--bars-per-year", type=int, default=3276)
    run_p.add_argument("--base-cost-bps", type=float, default=2.0)
    run_p.add_argument(
        "--calibration-dirs",
        type=_coerce_path_list,
        default=[],
        help="comma-separated per-fold calibration directories (optional)",
    )
    run_p.add_argument("--device", type=str, default="cpu")
    run_p.add_argument("--batch-size", type=int, default=32)
    run_p.add_argument(
        "--conformal-floor",
        type=float,
        default=0.40,
        help="APS lower-bound cutoff for the sizer's conformal floor (V1-SPEC §10.1)",
    )
    run_p.add_argument(
        "--dry-run",
        action="store_true",
        help="skip model inference, use synthetic positions (smoke path)",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(name)s | %(message)s")

    if args.cmd != "run":
        parser.print_help()
        return 1

    if args.dry_run:
        LOG.info("dry-run mode: synthetic contexts, no checkpoint inference")
        contexts = _dry_run_contexts()
        strategies = _default_strategies()
    else:
        if not args.checkpoints:
            LOG.error("--checkpoints required when --dry-run not set")
            return 2
        if not args.sidecars or len(args.sidecars) != len(args.checkpoints):
            LOG.error(
                "--sidecars must be supplied with same count as --checkpoints "
                f"(got {len(args.sidecars)} sidecars vs {len(args.checkpoints)} checkpoints)"
            )
            return 2
        if args.config is None:
            LOG.error("--config required in production mode (used for model construction)")
            return 2
        unified_path = Path("data/processed/training_v1_unified.pt")
        if not unified_path.exists():
            LOG.error("unified.pt not found at %s; cannot run production", unified_path)
            return 2
        cal_dirs: list[Path | None] = (
            list(args.calibration_dirs) if args.calibration_dirs else [None] * len(args.checkpoints)
        )
        if cal_dirs and len(cal_dirs) != len(args.checkpoints):
            LOG.error(
                "--calibration-dirs count (%d) must match --checkpoints (%d)",
                len(cal_dirs),
                len(args.checkpoints),
            )
            return 2
        LOG.info(
            "production mode: %d folds, device=%s, batch_size=%d, conformal_floor=%.2f",
            len(args.checkpoints),
            args.device,
            args.batch_size,
            args.conformal_floor,
        )
        contexts = build_production_contexts(
            config_path=args.config,
            unified_path=unified_path,
            checkpoints=args.checkpoints,
            sidecars=args.sidecars,
            calibration_dirs=cal_dirs,
            device=args.device,
            batch_size=args.batch_size,
            conformal_floor=args.conformal_floor,
        )
        # Production baselines need REAL train data; the harness in
        # production mode runs only the model strategy + price-only
        # baselines that survive zero-train. ML baselines that depend
        # on ctx["train_features_*"] correctly fall back to zeros.
        strategies = _default_strategies()

    wf = walk_forward(
        contexts,
        strategies,
        base_cost_bps=args.base_cost_bps,
    )
    md_path = render_report(wf, args.out, base_cost_bps=args.base_cost_bps)
    LOG.info("report written: %s", md_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
