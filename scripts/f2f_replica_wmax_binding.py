"""Check if W_max=0.30 cap actually binds in the matched-vol F2F replica.

If cap never binds, scaling is purely linear and Sharpe identicality is
a tautology. If cap binds on some bars, the relationship is nonlinear
and the matched-vol experiment is informative."""

from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from f2f_replica_matched_vol import (
    fetch_gold_daily, simulate_f2f, _sharpe, _ann_vol, _ann_ret,
    DAILY_BPY,
)


def main():
    df = fetch_gold_daily(start="2005-01-01")
    n = len(df)

    train_len = 10 * DAILY_BPY
    test_len = 6 * (DAILY_BPY // 12)
    step = DAILY_BPY // 12
    bpm = step
    folds = []
    s = 0
    while s + train_len + test_len <= n:
        folds.append((s, s + train_len, s + train_len + test_len))
        s += step

    # Config 1: matched-vol (target_vol=2%, W_max=0.30)
    print("=== Config A: target_vol=2%, W_max=0.30 (matched-vol) ===")
    pnl_a = []
    pos_a = []
    for (a, b, c) in folds:
        pnl, pos = simulate_f2f(df, 0.99, 0.94, 0.02, 0.30, slice(a, b), slice(b, c))
        first_month = slice(b, min(c, b + bpm))
        pnl_a.append(pnl[first_month])
        pos_a.append(pos[first_month])
    chain_pnl_a = np.concatenate(pnl_a)
    chain_pos_a = np.concatenate(pos_a)
    nonzero = chain_pos_a[chain_pos_a > 0]
    print(f"  Sharpe: {_sharpe(chain_pnl_a):+.4f}")
    print(f"  ann_vol: {_ann_vol(chain_pnl_a)*100:.4f}%")
    print(f"  bars w/ position > 0: {len(nonzero)} / {len(chain_pos_a)}")
    if len(nonzero) > 0:
        bound_at_wmax = (np.abs(chain_pos_a - 0.30) < 1e-6).sum()
        print(f"  bars at W_max=0.30 cap: {bound_at_wmax}  ({100*bound_at_wmax/len(chain_pos_a):.2f}% of total, {100*bound_at_wmax/len(nonzero):.2f}% of active)")
        print(f"  active position stats: mean={nonzero.mean():.6f}, max={nonzero.max():.6f}, p99={np.percentile(nonzero, 99):.6f}, p95={np.percentile(nonzero, 95):.6f}")
    print()

    # Config 2: unscaled high-vol (target_vol=15%, W_max=2.0)
    print("=== Config B: target_vol=15%, W_max=2.0 (unscaled, original) ===")
    pnl_b = []
    pos_b = []
    for (a, b, c) in folds:
        pnl, pos = simulate_f2f(df, 0.99, 0.94, 0.15, 2.0, slice(a, b), slice(b, c))
        first_month = slice(b, min(c, b + bpm))
        pnl_b.append(pnl[first_month])
        pos_b.append(pos[first_month])
    chain_pnl_b = np.concatenate(pnl_b)
    chain_pos_b = np.concatenate(pos_b)
    nonzero_b = chain_pos_b[chain_pos_b > 0]
    print(f"  Sharpe: {_sharpe(chain_pnl_b):+.4f}")
    print(f"  ann_vol: {_ann_vol(chain_pnl_b)*100:.4f}%")
    print(f"  bars w/ position > 0: {len(nonzero_b)} / {len(chain_pos_b)}")
    if len(nonzero_b) > 0:
        bound_at_wmax_b = (np.abs(chain_pos_b - 2.0) < 1e-6).sum()
        print(f"  bars at W_max=2.0 cap: {bound_at_wmax_b}  ({100*bound_at_wmax_b/len(chain_pos_b):.2f}% of total, {100*bound_at_wmax_b/len(nonzero_b):.2f}% of active)")
        print(f"  active position stats: mean={nonzero_b.mean():.6f}, max={nonzero_b.max():.6f}")
    print()

    # Compare: position ratios
    print("=== Cross-check linearity ===")
    nz_a = chain_pos_a > 0
    nz_b = chain_pos_b > 0
    both_active = nz_a & nz_b
    if both_active.sum() > 0:
        ratios = chain_pos_b[both_active] / np.maximum(chain_pos_a[both_active], 1e-12)
        print(f"  bars where both A and B active: {both_active.sum()}")
        print(f"  position ratio B/A: mean={ratios.mean():.4f}, std={ratios.std():.4f}, min={ratios.min():.4f}, max={ratios.max():.4f}")
        print(f"  Linear scaling would give a constant ratio. If std > 0.1, scaling is nonlinear (cap binding).")


if __name__ == "__main__":
    main()
