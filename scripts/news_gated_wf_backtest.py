"""News-gated ensemble backtest, 4-fold walk-forward, baselines only.

Hypothesis: active strategies post positive Sharpe on news-present bars
and negative Sharpe on news-absent bars (V4 fold 0 result, paper §6.7).
A trivial hybrid that trades only when news is present should aggregate
to positive Sharpe.

Strategies:
  - news_gated_ma_cross:  ma_cross_position(close, 10, 20) * is_news_present
  - news_gated_gao_2014:  gao_2014_position(...) * is_news_present
  - news_gated_donchian:  donchian_position(close, 20) * is_news_present
  - news_gated_ensemble:  mean(above three) * is_news_present

For each fold, compute positions on the fold's test slice (per
compute_fold_boundaries), score Sharpe net 1.0× and 1.5× cost.
Report per-fold + aggregate (mean and median across folds) Sharpe.

CPU-only, no GPU, no model inference. ~minutes total.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from nanogld.backtest.baselines import (  # noqa: E402
    buy_hold_positions,
    donchian_positions,
    gao_2014_positions,
    ma_cross_positions,
)
from nanogld.data.walk_forward_splits import compute_fold_boundaries  # noqa: E402

BARS_PER_YEAR = 3276
BASE_COST_BPS = 2.0
GLD_CLOSE_FEATURE_IDX = 3


def _arr(x: Any) -> np.ndarray:
    return x.numpy() if hasattr(x, "numpy") else np.asarray(x)


def _derive_is_last_bar_of_day(bcn: np.ndarray) -> np.ndarray:
    day_key = bcn // 86_400_000_000_000
    last = np.empty(bcn.shape[0], dtype=bool)
    last[:-1] = day_key[1:] != day_key[:-1]
    last[-1] = True
    return last


def _propagate_is_high_vol(bcn: np.ndarray, h5x: np.ndarray) -> np.ndarray:
    """Mark every bar of a day with any nonzero h5x as high-vol."""
    day_key = bcn // 86_400_000_000_000
    nz = np.abs(h5x) > 1e-10
    hv_days = set(day_key[nz].tolist())
    return np.array([int(d) in hv_days for d in day_key], dtype=bool)


def _ffill_per_day(bcn: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Forward-fill `x` within each UTC day, NaN-safe."""
    day_key = bcn // 86_400_000_000_000
    out = x.astype(np.float64).copy()
    last_day = None
    last_val = float("nan")
    for i in range(len(out)):
        d = int(day_key[i])
        if d != last_day:
            last_val = float("nan")
            last_day = d
        v = out[i]
        if np.isfinite(v):
            last_val = float(v)
        else:
            out[i] = last_val
    return out


def _sharpe(returns: np.ndarray, eps: float = 1e-12) -> float:
    r = returns[np.isfinite(returns)]
    if len(r) < 2:
        return 0.0
    mu, sigma = r.mean(), r.std(ddof=1)
    if sigma < eps:
        return 0.0
    return float(mu / sigma * np.sqrt(BARS_PER_YEAR))


def _pnl_with_cost(positions: np.ndarray, next_log_returns: np.ndarray, cost_mult: float = 1.0) -> np.ndarray:
    """PnL per bar with linear turnover cost."""
    p = np.nan_to_num(positions, nan=0.0)
    r = np.nan_to_num(next_log_returns, nan=0.0)
    cost_frac = (BASE_COST_BPS * cost_mult) / 10_000.0
    prev = np.concatenate([[0.0], p[:-1]])
    turnover = np.abs(p - prev)
    return p * r - cost_frac * turnover


def main() -> int:
    unified = torch.load(REPO_ROOT / "data" / "processed" / "training_v1_unified.pt", weights_only=False)
    features = _arr(unified["features"])
    bcn = _arr(unified["bar_close_utc_ns"]).astype(np.int64)
    close_full = features[:, GLD_CLOSE_FEATURE_IDX].astype(np.float64)
    last_bar_full = _derive_is_last_bar_of_day(bcn)
    fold_boundaries = compute_fold_boundaries(bcn)
    print(f"folds discovered: {len(fold_boundaries)}")

    # is_news_present derived from bar_news_offsets (a bar has news iff offsets[i+1] > offsets[i])
    bn_off = _arr(unified["bar_news_offsets"]).astype(np.int64)
    n_news_per_bar = np.diff(bn_off)
    is_news_present_full = (n_news_per_bar > 0).astype(bool)
    assert len(is_news_present_full) == len(bcn), f"{len(is_news_present_full)} vs {len(bcn)}"

    cols_header = ["fold", "n_bars", "news_frac",
                   "buy_hold_S", "buy_hold_S15",
                   "gated_ma_S", "gated_ma_S15",
                   "gated_gao_S", "gated_gao_S15",
                   "gated_donch_S", "gated_donch_S15",
                   "gated_ens_S", "gated_ens_S15"]
    print("\n" + " | ".join(f"{c:>11s}" for c in cols_header))

    per_fold_metrics: list[dict[str, float]] = []
    for fb in fold_boundaries[:4]:
        # Build per-fold sidecar load. fold_idx encoded in fb.
        side_path = REPO_ROOT / "data" / "processed" / f"training_v1_sidecar_fold_{fb.fold_idx}.pt"
        side = torch.load(side_path, weights_only=False)
        h5_raw = _arr(side["gld_h5_log_return"])
        h5x = _arr(side["gld_h5_x_vol_high"])
        nlr_full = _arr(side["next_log_return"]).astype(np.float64)
        h5_filled = _ffill_per_day(bcn, h5_raw)
        is_high_vol_full = _propagate_is_high_vol(bcn, h5x)

        test_slice = slice(fb.test_start, fb.test_end)
        close_t = close_full[test_slice]
        h5_t = h5_filled[test_slice]
        hv_t = is_high_vol_full[test_slice]
        last_t = last_bar_full[test_slice]
        news_t = is_news_present_full[test_slice].astype(np.float64)
        nlr_t = nlr_full[test_slice]
        n_bars = len(close_t)

        bh = buy_hold_positions(n_bars=n_bars).astype(np.float64)
        ma = ma_cross_positions(close_t, fast_span=10, slow_span=20).astype(np.float64)
        dn = donchian_positions(close_t, window=20).astype(np.float64)
        ga = gao_2014_positions(h5_t, is_high_vol=hv_t, is_last_bar_of_day=last_t, hold_last_bar_only=True).astype(np.float64)

        gated_ma = ma * news_t
        gated_dn = dn * news_t
        gated_ga = ga * news_t
        gated_ens = ((gated_ma + gated_dn + gated_ga) / 3.0)

        def s(pos: np.ndarray, mult: float = 1.0) -> float:
            return _sharpe(_pnl_with_cost(pos, nlr_t, mult))

        row = [
            f"{fb.fold_idx:>11d}", f"{n_bars:>11d}", f"{news_t.mean():>10.4f}",
            f"{s(bh):>11.4f}", f"{s(bh, 1.5):>11.4f}",
            f"{s(gated_ma):>11.4f}", f"{s(gated_ma, 1.5):>11.4f}",
            f"{s(gated_ga):>11.4f}", f"{s(gated_ga, 1.5):>11.4f}",
            f"{s(gated_dn):>11.4f}", f"{s(gated_dn, 1.5):>11.4f}",
            f"{s(gated_ens):>11.4f}", f"{s(gated_ens, 1.5):>11.4f}",
        ]
        print(" | ".join(row))
        per_fold_metrics.append({
            "fold": fb.fold_idx,
            "n_bars": n_bars,
            "news_frac": float(news_t.mean()),
            "buy_hold_S": s(bh),
            "gated_ma_S": s(gated_ma),
            "gated_gao_S": s(gated_ga),
            "gated_donch_S": s(gated_dn),
            "gated_ens_S": s(gated_ens),
            "gated_ens_S15": s(gated_ens, 1.5),
        })

    print()
    print("===== WALK-FORWARD AGGREGATE (mean across 4 folds) =====")
    keys = ["buy_hold_S", "gated_ma_S", "gated_gao_S", "gated_donch_S", "gated_ens_S", "gated_ens_S15"]
    for k in keys:
        vals = [m[k] for m in per_fold_metrics]
        print(f"  {k:<18s}: mean={np.mean(vals):+.4f}  median={np.median(vals):+.4f}  min={np.min(vals):+.4f}  max={np.max(vals):+.4f}")

    print()
    print("Promotion-gate check (V1-SPEC §9.4 mini):")
    ens_means = [m["gated_ens_S"] for m in per_fold_metrics]
    ens_15 = [m["gated_ens_S15"] for m in per_fold_metrics]
    bh_means = [m["buy_hold_S"] for m in per_fold_metrics]
    print(f"  G1 (mean Sharpe >= 1.0 @ 1x)       : ens={np.mean(ens_means):+.4f}   {'PASS' if np.mean(ens_means) >= 1.0 else 'FAIL'}")
    print(f"  G2 (mean Sharpe >  0.5 @ 1.5x)     : ens={np.mean(ens_15):+.4f}      {'PASS' if np.mean(ens_15) > 0.5 else 'FAIL'}")
    print(f"  G3 (beats buy_hold by >= 0.2)      : delta={np.mean(ens_means) - np.mean(bh_means):+.4f} {'PASS' if np.mean(ens_means) - np.mean(bh_means) >= 0.2 else 'FAIL'}")
    print(f"  G_news (positive on every fold)    : count_pos={sum(1 for v in ens_means if v > 0)}/4")

    return 0


if __name__ == "__main__":
    sys.exit(main())
