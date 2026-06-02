"""Gao 2014 + XGBoost ensemble — the V1-SPEC §0 fallback ship candidate.

The rule: if nanoGLD does not beat this ensemble by ≥ 0.2 Sharpe on
4-fold walk-forward OOS net of 2 bps round-trip cost, ship the
ensemble instead.

This script trains/evaluates the ensemble on the existing per-fold
sidecar splits and writes a backtest-shaped report under
``reports/ensemble_fallback/`` so it can be compared head-to-head with
nanoGLD's reports.

Steps per fold:
  1. Load unified.pt + sidecar_fold_<N>.pt.
  2. Pull train/test slices from splits.
  3. Train XGBoost multi:softprob on (features_train, labels_train).
  4. Predict test → argmax(prob) → position in {-1, 0, +1}.
  5. Compute Gao 2014 position from h5_log_return on test bars.
  6. Ensemble: simple-average the two position arrays.
  7. Vectorized backtest at cost {0.5x, 1.0x, 1.5x} × 2 bps.

Total: ~4 folds × ~30s XGBoost fit + ~10s Gao + ~5s backtest ≈ 3 min
wall-clock on Mac mini CPU.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from nanogld.backtest.baselines.gao_2014 import gao_2014_positions
from nanogld.backtest.cost_stress import cost_stress
from nanogld.backtest.engine import BacktestConfig, vectorized_backtest
from nanogld.backtest.metrics import compute_metrics

LOG = logging.getLogger("ensemble_fallback")


def _train_xgboost(
    train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray
) -> np.ndarray:
    """Train XGBoost multi:softprob; return position in {-1, 0, +1}."""
    try:
        from xgboost import XGBClassifier  # noqa: PLC0415
    except ImportError as exc:
        raise RuntimeError(
            "xgboost required: `uv add xgboost>=2.1`"
        ) from exc

    clf = XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softprob",
        num_class=3,
        tree_method="hist",
        device="cpu",
        verbosity=1,
        n_jobs=1,
    )
    LOG.info(
        "xgb fit train=%d test=%d features=%d", train_x.shape[0], test_x.shape[0], train_x.shape[1]
    )
    t0 = time.time()
    clf.fit(train_x, train_y)
    LOG.info("xgb fit done in %.1fs", time.time() - t0)
    pred = clf.predict(test_x)  # {0, 1, 2}
    # 0=DOWN, 1=FLAT, 2=UP → position {-1, 0, +1}.
    return (pred.astype(np.float64) - 1.0)


def _gao_positions(
    sidecar: dict[str, np.ndarray],
    test_indices: np.ndarray,
) -> np.ndarray:
    h5 = sidecar.get("gld_h5_log_return")
    if h5 is None:
        LOG.warning("sidecar missing gld_h5_log_return; gao positions = 0")
        return np.zeros(len(test_indices), dtype=np.float64)
    h5 = np.asarray(h5)[test_indices]
    return gao_2014_positions(h5, is_high_vol=None, hold_last_bar_only=False)


def _flatten_features(features: np.ndarray) -> np.ndarray:
    """If features are (N, F) keep; if (N, T, F) flatten time×features."""
    if features.ndim == 2:
        return features
    if features.ndim == 3:
        return features.reshape(features.shape[0], -1)
    raise ValueError(f"unsupported feature shape {features.shape}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ensemble_fallback")
    parser.add_argument("--unified", type=Path, default=Path("data/processed/training_v1_unified.pt"))
    parser.add_argument("--sidecar-template", type=str, default="data/processed/training_v1_sidecar_fold_{fold}.pt")
    parser.add_argument("--n-folds", type=int, default=4)
    parser.add_argument("--out", type=Path, default=Path("reports/ensemble_fallback"))
    parser.add_argument("--base-cost-bps", type=float, default=2.0)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

    if not args.unified.exists():
        raise FileNotFoundError(args.unified)
    LOG.info("loading unified.pt …")
    unified = torch.load(args.unified, weights_only=False, map_location="cpu")
    features = np.asarray(unified["features"])  # (N, F) float
    labels = np.asarray(unified["labels"])  # (N,) int
    splits = np.asarray(unified["splits"])  # (N,) str
    next_log_return_all_from_sidecar = None  # filled per fold

    args.out.mkdir(parents=True, exist_ok=True)
    fold_results: list[dict] = []

    for fold in range(args.n_folds):
        sidecar_path = Path(args.sidecar_template.format(fold=fold))
        if not sidecar_path.exists():
            LOG.warning("fold %d sidecar missing: %s — skipping", fold, sidecar_path)
            continue
        sidecar = torch.load(sidecar_path, weights_only=False, map_location="cpu")
        sidecar = {k: np.asarray(v) if hasattr(v, "shape") else v for k, v in sidecar.items()}

        train_mask = splits == "train"
        test_mask = splits == "test"
        if not test_mask.any():
            LOG.warning("fold %d no test bars; skipping", fold)
            continue

        train_idx = np.where(train_mask)[0]
        test_idx = np.where(test_mask)[0]
        train_x = _flatten_features(features[train_idx])
        train_y = labels[train_idx].astype(np.int64)
        test_x = _flatten_features(features[test_idx])

        if next_log_return_all_from_sidecar is None and "next_log_return" in sidecar:
            next_log_return_all_from_sidecar = sidecar["next_log_return"]
        nlr_test = sidecar["next_log_return"][test_idx]

        # XGBoost positions.
        xgb_pos = _train_xgboost(train_x, train_y, test_x)
        # Gao 2014 positions.
        gao_pos = _gao_positions(sidecar, test_idx)
        # Ensemble: simple average + clip to [-1, +1].
        ens_pos = 0.5 * (xgb_pos + gao_pos)
        ens_pos = np.clip(ens_pos, -1.0, 1.0)

        # Cost-stress backtest.
        cs = cost_stress(
            next_log_returns=nlr_test,
            positions=ens_pos,
            base_cost_bps=args.base_cost_bps,
        )
        # Headline metrics at 1.0x cost.
        cfg10 = BacktestConfig(cost_bps=args.base_cost_bps)
        bt = vectorized_backtest(nlr_test, ens_pos, cfg=cfg10)
        m = compute_metrics(bt.pnl_per_bar, bt.equity_curve)

        per_fold = {
            "fold": fold,
            "n_train": int(train_x.shape[0]),
            "n_test": int(test_x.shape[0]),
            "sharpe_1x": float(m.get("sharpe", 0.0)),
            "sortino_1x": float(m.get("sortino", 0.0)),
            "mdd": float(m.get("max_drawdown", 0.0)),
            "hit_rate": float(m.get("hit_rate", 0.0)),
            "cost_stress": {
                f"{mlt}x": cs.by_multiplier[mlt] for mlt in (0.5, 1.0, 1.5)
            },
            "xgb_position_mean": float(xgb_pos.mean()),
            "gao_position_mean": float(gao_pos.mean()),
            "ensemble_position_mean": float(ens_pos.mean()),
        }
        fold_results.append(per_fold)
        LOG.info(
            "fold %d ensemble Sharpe 1x=%.3f, cost-stress 1.5x=%.3f",
            fold,
            per_fold["sharpe_1x"],
            cs.by_multiplier[1.5]["sharpe"],
        )

    if not fold_results:
        LOG.error("no folds completed")
        return 1

    # Aggregate across folds.
    sharpes_15 = [fr["cost_stress"]["1.5x"]["sharpe"] for fr in fold_results]
    summary = {
        "ensemble": "gao_2014 + xgboost (V1-SPEC §0 fallback)",
        "n_folds": len(fold_results),
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "headline_sharpe_1x_mean": float(np.mean([fr["sharpe_1x"] for fr in fold_results])),
        "headline_sharpe_15x_mean": float(np.mean(sharpes_15)),
        "promotion_floor": (
            "nanogld must beat this Sharpe by >= 0.2 net 2bp on walk-forward 4-fold "
            "(plan/V1-SPEC §0)"
        ),
        "per_fold": fold_results,
    }
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_json = args.out / f"ensemble_fallback_{ts}.json"
    out_md = args.out / f"ensemble_fallback_{ts}.md"
    out_json.write_text(json.dumps(summary, indent=2, default=str))

    # Render a short markdown report.
    lines = [
        "# Gao 2014 + XGBoost ensemble — V1-SPEC §0 fallback",
        "",
        f"- generated: `{summary['generated_utc']}`",
        f"- folds: {summary['n_folds']}",
        f"- mean Sharpe @ 1.0× cost: **{summary['headline_sharpe_1x_mean']:.3f}**",
        f"- mean Sharpe @ 1.5× cost: **{summary['headline_sharpe_15x_mean']:.3f}**",
        "",
        "## Per-fold",
        "",
        "| Fold | n_train | n_test | Sharpe | Sortino | MDD | S 0.5× | S 1.0× | S 1.5× |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for fr in fold_results:
        cs = fr["cost_stress"]
        s05 = cs["0.5x"]["sharpe"]
        s10 = cs["1.0x"]["sharpe"]
        s15 = cs["1.5x"]["sharpe"]
        lines.append(
            f"| {fr['fold']} | {fr['n_train']} | {fr['n_test']} "
            f"| {fr['sharpe_1x']:.3f} | {fr['sortino_1x']:.3f} | {fr['mdd']:.2%} "
            f"| {s05:.3f} | {s10:.3f} | {s15:.3f} |"
        )
    out_md.write_text("\n".join(lines) + "\n")
    LOG.info("ensemble report: %s", out_md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
