"""Lean Westfall-Young permutation test using numpy only.

Uses the same 9-asset universe and vol_target_3pct recipe as the
paired-block-bootstrap test. Reduces N_PERMUTATIONS to 500 for speed.
"""

from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent

INTRADAY_BPY = 7308
BASE_COST_BPS = 2.0
TARGET_VOL = 0.03
N_PERMUTATIONS = 500
BLOCK_LEN = 20


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
    n_blocks = (n + block_len - 1) // block_len
    starts = (np.arange(n_blocks) * block_len).copy()
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
    print("Loading unified.pt via torch...", flush=True)
    import torch
    unified = torch.load(REPO_ROOT / "data" / "processed" / "training_v1_unified.pt", weights_only=False)
    features = unified["features"].numpy().astype(np.float32)
    feature_names = unified["feature_names"]
    bcn = unified["bar_close_utc_ns"].numpy().astype(np.int64)
    print(f"Loaded {features.shape} features, {len(feature_names)} names", flush=True)

    sys.path.insert(0, str(REPO_ROOT / "src"))
    from nanogld.data.walk_forward_splits import compute_fold_boundaries
    folds = compute_fold_boundaries(bcn)

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
    asset_keys = list(asset_indices.keys())
    print(f"Universe: {asset_keys}", flush=True)

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

    n = min(len(pnl_vt[s]) for s in asset_keys)
    for s in asset_keys:
        pnl_vt[s] = pnl_vt[s][:n]
        pnl_bh[s] = pnl_bh[s][:n]
    print(f"Test sequence length per asset: {n} bars", flush=True)

    pnl_vt_mat = np.stack([pnl_vt[s] for s in asset_keys])
    pnl_bh_mat = np.stack([pnl_bh[s] for s in asset_keys])

    observed_delta = np.zeros(len(asset_keys))
    for i, s in enumerate(asset_keys):
        observed_delta[i] = _sharpe(pnl_vt_mat[i], INTRADAY_BPY) - _sharpe(pnl_bh_mat[i], INTRADAY_BPY)
    print("\nObserved deltas:", flush=True)
    for i, s in enumerate(asset_keys):
        print(f"  {s:<6s}  {observed_delta[i]:+.4f}", flush=True)

    rng = np.random.default_rng(42)
    max_abs_delta_dist = np.zeros(N_PERMUTATIONS)

    print(f"\nRunning {N_PERMUTATIONS} permutations...", flush=True)
    for b in range(N_PERMUTATIONS):
        idx = block_permutation_indices(n, BLOCK_LEN, rng)
        vt_perm = pnl_vt_mat[:, idx]
        bh_perm = pnl_bh_mat[:, idx]
        deltas = np.zeros(len(asset_keys))
        for i in range(len(asset_keys)):
            deltas[i] = _sharpe(vt_perm[i], INTRADAY_BPY) - _sharpe(bh_perm[i], INTRADAY_BPY)
        max_abs_delta_dist[b] = np.max(np.abs(deltas))
        if (b + 1) % 100 == 0:
            print(f"  permutation {b+1}/{N_PERMUTATIONS}", flush=True)

    print("\n=== Westfall-Young FWE-adjusted p-values ===", flush=True)
    print(f"{'asset':<6s} {'Δ obs':>8s} {'WY p':>10s} {'verdict α=0.10':>18s}", flush=True)
    n_pass = 0
    results = []
    for i, s in enumerate(asset_keys):
        d = observed_delta[i]
        p_fwe = float((max_abs_delta_dist >= abs(d)).mean())
        if d > 0 and p_fwe < 0.10:
            verdict = "FWE-PASS"
            n_pass += 1
        else:
            verdict = "FAIL"
        results.append((s, d, p_fwe, verdict))
        print(f"  {s:<6s}  {d:+8.4f}  {p_fwe:>10.4f}  {verdict:>18s}", flush=True)

    print(f"\nWestfall-Young FWE-PASS at α=0.10: {n_pass} / 9", flush=True)

    out = REPO_ROOT / "paper" / "wy_fwe_summary.txt"
    out.parent.mkdir(exist_ok=True)
    with open(out, "w") as f:
        f.write(f"Westfall-Young permutation FWE for intraday 9-asset family\n")
        f.write(f"N_PERMUTATIONS={N_PERMUTATIONS}, BLOCK_LEN={BLOCK_LEN}, n_bars={n}\n\n")
        f.write(f"{'asset':<6s} {'D obs':>8s} {'WY p':>10s} {'verdict':>18s}\n")
        for s, d, p, v in results:
            f.write(f"{s:<6s}  {d:+8.4f}  {p:>10.4f}  {v:>18s}\n")
        f.write(f"\nWY FWE-PASS at alpha=0.10: {n_pass}/9\n")
    print(f"\nSaved summary to {out}", flush=True)


if __name__ == "__main__":
    main()
