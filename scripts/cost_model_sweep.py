"""Apply F2F cost model (0.7 bp) to our intraday GLD vol_target claim
for apples-to-apples comparison. Also test gross-of-cost (matches VLSTM's
framing) and our published 2bp baseline."""

from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from nanogld.data.walk_forward_splits import compute_fold_boundaries

BARS_PER_YEAR = 3276
GLD_IDX = 3


def _arr(x): return x.numpy() if hasattr(x, "numpy") else np.asarray(x)


def _sharpe(r):
    r = r[np.isfinite(r)]
    if len(r) < 2: return 0.0
    mu, sigma = r.mean(), r.std(ddof=1)
    if sigma < 1e-12: return 0.0
    return float(mu / sigma * np.sqrt(BARS_PER_YEAR))


def _rolling_std(x, w):
    n = len(x); out = np.zeros(n)
    cs = np.cumsum(x); cs2 = np.cumsum(x * x)
    for i in range(n):
        a = max(0, i - w + 1); c = i - a + 1
        if c < 2: continue
        m = (cs[i] - (cs[a - 1] if a > 0 else 0)) / c
        m2 = (cs2[i] - (cs2[a - 1] if a > 0 else 0)) / c
        v = max(0, m2 - m * m); out[i] = np.sqrt(v)
    return out


def _vol_target(rr, tv, w=64, cap=1.0):
    rv = _rolling_std(rr, w)
    raw = tv / (rv * np.sqrt(BARS_PER_YEAR) + 1e-8)
    p = np.clip(raw, 0, cap); p[~np.isfinite(p)] = 0.0
    return np.concatenate([[0.0], p[:-1]])


def pnl_at_cost(pos, nlr, cost_bps):
    p = np.nan_to_num(pos); r = np.nan_to_num(nlr)
    cf = cost_bps / 10000.0
    prev = np.concatenate([[0.0], p[:-1]])
    return p * r - cf * np.abs(p - prev)


def main():
    unified = torch.load(REPO_ROOT / "data" / "processed" / "training_v1_unified.pt", weights_only=False)
    features = _arr(unified["features"])
    bcn = _arr(unified["bar_close_utc_ns"]).astype(np.int64)
    close_full = features[:, GLD_IDX].astype(np.float64)
    folds = compute_fold_boundaries(bcn)

    print("Intraday GLD vol_target at different cost models")
    print(f"{'target_vol':<10s} {'cost (bp)':<10s} {'WF mean':<10s} per-fold")
    for tv in [0.03, 0.05, 0.07, 0.10, 0.15]:
        for cost_bps in [0.0, 0.7, 1.0, 2.0]:
            f_sharpes = []
            for fb in folds[:4]:
                side = torch.load(REPO_ROOT / "data" / "processed" / f"training_v1_sidecar_fold_{fb.fold_idx}.pt", weights_only=False)
                nlr = _arr(side["next_log_return"]).astype(np.float64)
                ts = slice(fb.test_start, fb.test_end)
                close_t = close_full[ts]
                rr = np.zeros_like(close_t)
                rr[1:] = np.log(np.maximum(close_t[1:], 1e-12) / np.maximum(close_t[:-1], 1e-12))
                rr = np.nan_to_num(rr)
                pos = _vol_target(rr, tv, 64)
                sh = _sharpe(pnl_at_cost(pos, nlr[ts], cost_bps))
                f_sharpes.append(sh)
            mean_sh = float(np.mean(f_sharpes))
            print(f"  {int(tv*100):>3d}%      {cost_bps:>5.2f}     {mean_sh:+.4f}    {[round(x,3) for x in f_sharpes]}")

    print()
    print("Benchmarks:")
    print(f"  F2F SOTA (daily gold futures, NET 0.7bp + sqrt-impact):  +2.88")
    print(f"  VLSTM SOTA (multi-asset portfolio, GROSS of cost):        +2.40")


if __name__ == "__main__":
    main()
