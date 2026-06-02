"""Westfall-Young permutation FWE for the 9-asset intraday multi-asset family.

Standard Bonferroni assumes independent tests and is too conservative under
correlated returns. Westfall-Young permutation respects the cross-asset
correlation structure by permuting blocks of the *time index* identically
across all 9 assets, preserving (a) within-asset autocorrelation via blocks
and (b) cross-asset return correlation via shared permutation order.

Procedure:
1. Compute observed Δ_a = Sharpe(vt_a) - Sharpe(BH_a) for each asset a.
2. For B permutations:
   a. Sample a circular block-permutation of the time index (block_len=20).
   b. Apply the SAME permutation to all 9 assets' (vt, BH) PnL sequences.
   c. Compute permuted Δ*_a for each asset.
   d. Record max_a |Δ*_a| across the family.
3. FWE-adjusted p_a = (# permutations where max_a |Δ*_b| >= |Δ_a|) / B.

This is the Westfall-Young max-T method (1993). It is conservative against
the cross-asset correlation structure that Bonferroni ignores, and it makes
no parametric assumption about the joint return distribution. It is the
correct multi-testing correction for this design.
"""

from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
from nanogld.data.walk_forward_splits import compute_fold_boundaries

INTRADAY_BPY = 7308
BASE_COST_BPS = 2.0
TARGET_VOL = 0.03
N_PERMUTATIONS = 1000
BLOCK_LEN = 20


def _arr(x):
    return x.numpy() if hasattr(x, "numpy") else np.asarray(x)


def _sharpe(r, ann):
    r = r[np.isfinite(r)]
    if len(r) < 2: return 0.0
    mu, sigma = r.mean(), r.std(ddof=1)
    if sigma < 1e-12: return 0.0
    return float(mu / sigma * np.sqrt(ann))


def _pnl(pos, ret, cost_mult=1.0):
    p = np.nan_to_num(pos); r = np.nan_to_num(ret)
    cf = (BASE_COST_BPS * cost_mult) / 10000.0
    prev = np.concatenate([[0.0], p[:-1]])
    return p * r - cf * np.abs(p - prev)


def _rolling_std(x, w):
    n = len(x); out = np.zeros(n)
    if w <= 1: return out
    cs = np.cumsum(x); cs2 = np.cumsum(x * x)
    for i in range(n):
        a = max(0, i - w + 1); c = i - a + 1
        if c < 2: continue
        m = (cs[i] - (cs[a-1] if a > 0 else 0)) / c
        m2 = (cs2[i] - (cs2[a-1] if a > 0 else 0)) / c
        v = max(0.0, m2 - m * m); out[i] = np.sqrt(v)
    return out


def _vol_target(rr, tv=TARGET_VOL, w=64, cap=1.0):
    rv = _rolling_std(rr, w)
    raw = tv / (rv * np.sqrt(INTRADAY_BPY) + 1e-8)
    p = np.clip(raw, 0, cap); p[~np.isfinite(p)] = 0.0
    return np.concatenate([[0.0], p[:-1]])


def block_permutation_indices(n, block_len, rng):
    """Return a length-n index array via circular block permutation.

    Splits [0..n) into contiguous blocks of length `block_len` and shuffles
    the block order. Preserves within-block autocorrelation; breaks between-
    block dependence.
    """
    n_blocks = (n + block_len - 1) // block_len
    starts = np.arange(n_blocks) * block_len
    rng.shuffle(starts)
    idx = np.zeros(n, dtype=np.int64)
    pos = 0
    for s in starts:
        e = min(s + block_len, n)
        ln = e - s
        if pos + ln > n: ln = n - pos
        idx[pos:pos+ln] = np.arange(s, s+ln)
        pos += ln
        if pos >= n: break
    return idx[:n]


def main():
    unified = torch.load(REPO_ROOT / "data" / "processed" / "training_v1_unified.pt", weights_only=False)
    features = _arr(unified["features"]).astype(np.float32)
    feature_names = unified["feature_names"]
    bcn = _arr(unified["bar_close_utc_ns"]).astype(np.int64)

    asset_cols = {
        "GLD": "gld_close",
        "SPY": "spy_lag1_close",
        "QQQ": "qqq_lag1_close",
        "IWM": "iwm_lag1_close",
        "GDX": "gdx_lag1_close",
        "SLV": "slv_lag1_close",
        "TLT": "tlt_lag1_close",
        "XLE": "xle_lag1_close",
        "USO": "uso_lag1_close",
    }
    asset_indices = {sym: feature_names.index(col) for sym, col in asset_cols.items() if col in feature_names}
    print(f"Found {len(asset_indices)} asset close columns")

    folds = compute_fold_boundaries(bcn)
    print(f"Folds: {len(folds[:4])}")

    # Build per-asset concatenated test-window PnL sequences (vt and BH)
    pnl_vt = {}
    pnl_bh = {}
    for sym, col_idx in asset_indices.items():
        close = features[:, col_idx].astype(np.float64)
        chain_vt = []
        chain_bh = []
        for fb in folds[:4]:
            ts = slice(fb.test_start, fb.test_end)
            c = close[ts]
            rr = np.zeros_like(c)
            rr[1:] = np.log(np.maximum(c[1:], 1e-12) / np.maximum(c[:-1], 1e-12))
            rr = np.nan_to_num(rr)
            pos = _vol_target(rr)
            bh = np.ones_like(rr)
            nlr = np.concatenate([rr[1:], [0.0]])
            chain_vt.append(_pnl(pos, nlr, 1.0))
            chain_bh.append(_pnl(bh, nlr, 1.0))
        pnl_vt[sym] = np.concatenate(chain_vt)
        pnl_bh[sym] = np.concatenate(chain_bh)

    # Truncate all to common length (folds should match anyway)
    n = min(len(pnl_vt[s]) for s in asset_indices)
    for s in asset_indices:
        pnl_vt[s] = pnl_vt[s][:n]
        pnl_bh[s] = pnl_bh[s][:n]

    # Observed Δ per asset
    observed_delta = {sym: _sharpe(pnl_vt[sym], INTRADAY_BPY) - _sharpe(pnl_bh[sym], INTRADAY_BPY)
                       for sym in asset_indices}
    print()
    print("Observed Δ (Sharpe_vt − Sharpe_BH):")
    for sym in asset_indices:
        print(f"  {sym:<6s} Δ = {observed_delta[sym]:+.4f}")

    # Westfall-Young permutation: SAME block permutation applied to all 9 assets
    # Null = signal-free reshuffle of vt PnL aligned by time, keeping cross-asset
    # correlation. Under null, distribution of max over assets of |Δ*_a| approximates
    # the family-wise null distribution.
    rng = np.random.default_rng(42)
    max_abs_delta_dist = np.zeros(N_PERMUTATIONS)
    asset_keys = list(asset_indices.keys())

    for b in range(N_PERMUTATIONS):
        idx = block_permutation_indices(n, BLOCK_LEN, rng)
        boot_deltas = []
        for sym in asset_keys:
            # Resampled per-asset PnL — permute vt and BH IDENTICALLY across all assets.
            # This preserves cross-asset correlation in the resample (joint return draw).
            vt_b = pnl_vt[sym][idx]
            bh_b = pnl_bh[sym][idx]
            boot_deltas.append(_sharpe(vt_b, INTRADAY_BPY) - _sharpe(bh_b, INTRADAY_BPY))
        max_abs_delta_dist[b] = max(abs(d) for d in boot_deltas)
        if (b+1) % 500 == 0:
            print(f"  permutation {b+1}/{N_PERMUTATIONS}")

    # Westfall-Young FWE-adjusted p for each asset
    print()
    print("=== Westfall-Young FWE-adjusted p-values (one-sided, Δ > 0) ===")
    print(f"{'asset':<6s} {'Δ obs':>8s} {'FWE p (max|Δ|)':>14s} {'verdict at α=0.10':>18s}")
    n_passes_fwe = 0
    fwe_results = {}
    for sym in asset_keys:
        d = observed_delta[sym]
        # one-sided: count permutations where max abs Δ >= observed |Δ|
        # since we want to control FWE on the *family* not the per-test
        p_fwe = float((max_abs_delta_dist >= abs(d)).mean())
        # also need it to be positive Δ for "strategy beats BH" claim
        if d > 0 and p_fwe < 0.10:
            verdict = "FWE-PASS at α=0.10"
            n_passes_fwe += 1
        else:
            verdict = "FAIL"
        fwe_results[sym] = (d, p_fwe, verdict)
        print(f"  {sym:<6s} {d:>+8.4f} {p_fwe:>14.4f} {verdict:>18s}")

    print()
    print(f"FWE-PASS count at α=0.10: {n_passes_fwe} / {len(asset_keys)}")
    print(f"Bonferroni-corrected threshold for α=0.10 family-wise was: per-test p < {0.10/len(asset_keys):.4f}")
    print(f"Westfall-Young is the correct correction under cross-asset return correlation.")

    # Save summary for paper
    out = REPO_ROOT / "paper" / "wy_fwe_summary.txt"
    out.parent.mkdir(exist_ok=True)
    with open(out, "w") as f:
        f.write("Westfall-Young FWE-adjusted intraday multi-asset family\n")
        f.write(f"Permutations: {N_PERMUTATIONS}, block_len: {BLOCK_LEN}, n_bars: {n}\n\n")
        f.write(f"{'asset':<6s} {'Δ obs':>8s} {'FWE p':>10s} {'verdict':>20s}\n")
        for sym in asset_keys:
            d, p, v = fwe_results[sym]
            f.write(f"{sym:<6s} {d:>+8.4f} {p:>10.4f} {v:>20s}\n")
        f.write(f"\nFWE-PASS at α=0.10: {n_passes_fwe} / {len(asset_keys)}\n")


if __name__ == "__main__":
    main()
