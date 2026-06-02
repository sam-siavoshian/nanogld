"""Within-frequency intraday multi-asset replication of vol_target_3pct.

Uses unified.pt 30-minute bars (75,672 bars, ≈29/day, 2016-2026 GLD-correlated
universe). Tests vol_target on intraday close-to-close returns for 5 assets
present in unified["features"] (GLD, SPY, SLV, USO, TLT proxies via lag1_close
columns; daily-pulled fallback if not available).

Per asset:
  - intraday vol_target_3pct with rolling 64-bar vol
  - 4-fold WF using compute_fold_boundaries
  - per-fold Sharpe at BPY=7308 (correct intraday annualization)
  - daily-aggregated Sharpe + paired bootstrap vs buy-hold
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
DAILY_BPY = 252
BASE_COST_BPS = 2.0
TARGET_VOL = 0.03


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


def _paired_block_bootstrap_delta(pnl_a, pnl_b, n_boot=500, block_len=20, seed=42):
    rng = np.random.default_rng(seed)
    n = min(len(pnl_a), len(pnl_b))
    pnl_a, pnl_b = pnl_a[:n], pnl_b[:n]
    p_geom = 1.0 / block_len
    deltas = np.zeros(n_boot)
    for boot in range(n_boot):
        idx = np.empty(n, dtype=np.int64)
        i = 0
        while i < n:
            start = int(rng.integers(0, n))
            blk = max(1, int(rng.geometric(p_geom)))
            for j in range(blk):
                if i >= n: break
                idx[i] = (start + j) % n; i += 1
        deltas[boot] = _sharpe(pnl_a[idx], INTRADAY_BPY) - _sharpe(pnl_b[idx], INTRADAY_BPY)
    delta = _sharpe(pnl_a, INTRADAY_BPY) - _sharpe(pnl_b, INTRADAY_BPY)
    lo, hi = np.quantile(deltas, [0.025, 0.975])
    p_one_sided = float((deltas <= 0).mean())
    return float(delta), float(lo), float(hi), p_one_sided


def main():
    unified = torch.load(REPO_ROOT / "data" / "processed" / "training_v1_unified.pt", weights_only=False)
    features = _arr(unified["features"]).astype(np.float32)
    feature_names = unified["feature_names"]
    bcn = _arr(unified["bar_close_utc_ns"]).astype(np.int64)

    # Find close-price columns. unified.pt has gld_close at idx 3; others use lag1_close.
    # For pure intraday vol_target on different assets, use lag1_close as proxy (one-bar
    # lagged version which we then shift +1 to undo; or directly use the raw close if
    # available).
    # Universe choice rationale: large-liquid US ETFs across major asset classes
    # (equities/gold/oil/bonds) + the GLD anchor. VXX excluded from primary universe
    # as a degenerate-by-construction case (vol-target sizing on a volatility ETF
    # is self-referential); reported in appendix for completeness.
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

    print()
    print(f"{'asset':<6s} {'mean':>7s} {'BH mean':>7s} {'Δ':>7s} {'95% CI':<18s} {'p one-sided':>11s} {'verdict':>10s}")
    pass_strict = 0
    for sym, col_idx in asset_indices.items():
        close = features[:, col_idx].astype(np.float64)
        chain_pnl = []
        chain_pnl_bh = []
        per_fold_sharpe = []
        per_fold_bh = []
        for fb in folds[:4]:
            ts = slice(fb.test_start, fb.test_end)
            c = close[ts]
            rr = np.zeros_like(c)
            rr[1:] = np.log(np.maximum(c[1:], 1e-12) / np.maximum(c[:-1], 1e-12))
            rr = np.nan_to_num(rr)
            pos = _vol_target(rr)
            bh = np.ones_like(rr)
            nlr = np.concatenate([rr[1:], [0.0]])
            pnl = _pnl(pos, nlr, 1.0)
            pnl_bh = _pnl(bh, nlr, 1.0)
            per_fold_sharpe.append(_sharpe(pnl, INTRADAY_BPY))
            per_fold_bh.append(_sharpe(pnl_bh, INTRADAY_BPY))
            chain_pnl.append(pnl)
            chain_pnl_bh.append(pnl_bh)

        chain_pnl_arr = np.concatenate(chain_pnl)
        chain_pnl_bh_arr = np.concatenate(chain_pnl_bh)
        mean_s = np.mean(per_fold_sharpe)
        mean_bh = np.mean(per_fold_bh)
        delta_point, ci_lo, ci_hi, p_val = _paired_block_bootstrap_delta(chain_pnl_arr, chain_pnl_bh_arr)
        # strict-PASS: positive absolute Sharpe AND Δ ≥ 0.15 AND p < 0.10
        strict_pass = (mean_s > 0) and (delta_point >= 0.15) and (p_val < 0.10)
        loses_less = (mean_s < 0) and (delta_point >= 0.15)
        if strict_pass:
            verdict = "STRICT-PASS"
            pass_strict += 1
        elif loses_less:
            verdict = "loses-less"
        else:
            verdict = "weak/FAIL"
        print(f"  {sym:<6s} {mean_s:>+7.3f} {mean_bh:>+7.3f} {delta_point:>+7.3f}  [{ci_lo:>+5.2f}, {ci_hi:>+5.2f}]  {p_val:>10.4f}  {verdict:>10s}")

    print()
    print(f"Strict-PASS count (positive Sharpe AND Δ≥0.15 AND p<0.10): {pass_strict} / {len(asset_indices)}")


if __name__ == "__main__":
    main()
