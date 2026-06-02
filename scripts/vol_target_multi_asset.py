"""vol_target_3pct (PIT-clean) replication across 6 assets to test generalization.

Tests: GLD (gold ETF), SLV (silver ETF), SPY (US equities), TLT (long bonds),
USO (oil), BTC-USD (crypto). Same recipe, same 4-fold WF logic adapted to
each asset's available date range.

Per-asset:
  - daily Sharpe of vol_target_3pct (calibrated to each asset's vol level)
  - bootstrap 95% CI
  - paired vs buy_hold
  - per-fold breakdown

If vol_target replicates positive Sharpe on ≥4 of 6 assets, the method
generalizes; not a single-asset accident.
"""

from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
DAILY_BPY = 252
BASE_COST_BPS = 2.0
TARGET_VOL = 0.10  # 10% annualized (calibrated for daily; intraday-3pct equivalent)


def _arr(x):
    return x.numpy() if hasattr(x, "numpy") else np.asarray(x)


def _sharpe(r):
    r = r[np.isfinite(r)]
    if len(r) < 2: return 0.0
    mu, sigma = r.mean(), r.std(ddof=1)
    if sigma < 1e-12: return 0.0
    return float(mu / sigma * np.sqrt(DAILY_BPY))


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


def _vol_target(rr, tv, w=60, cap=1.0):
    rv = _rolling_std(rr, w)
    raw = tv / (rv * np.sqrt(DAILY_BPY) + 1e-8)
    p = np.clip(raw, 0, cap); p[~np.isfinite(p)] = 0.0
    return np.concatenate([[0.0], p[:-1]])


def _stat_block_bootstrap_sharpe(r, n_boot=500, block_len=20, seed=42):
    rng = np.random.default_rng(seed)
    r = r[np.isfinite(r)]
    n = len(r)
    if n < block_len + 2: return _sharpe(r), float("nan"), float("nan")
    p_geom = 1.0 / block_len
    sharpes = np.zeros(n_boot)
    for b in range(n_boot):
        sample = np.empty(n, dtype=r.dtype)
        i = 0
        while i < n:
            start = int(rng.integers(0, n))
            blk = max(1, int(rng.geometric(p_geom)))
            for j in range(blk):
                if i >= n: break
                sample[i] = r[(start + j) % n]; i += 1
        sharpes[b] = _sharpe(sample)
    lo, hi = np.quantile(sharpes, [0.025, 0.975])
    return _sharpe(r), float(lo), float(hi)


def _paired_bootstrap_delta(a, b, n_boot=500, block_len=20, seed=42):
    rng = np.random.default_rng(seed)
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
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
        deltas[boot] = _sharpe(a[idx]) - _sharpe(b[idx])
    delta = _sharpe(a) - _sharpe(b)
    lo, hi = np.quantile(deltas, [0.025, 0.975])
    p = float((deltas <= 0).mean())
    return float(delta), float(lo), float(hi), p


def fetch(sym, start="2010-01-01"):
    import yfinance as yf
    df = yf.download(sym, start=start, end="2026-05-25", progress=False, auto_adjust=True)
    if df is None or len(df) == 0:
        return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    df = df.reset_index()
    df["close"] = df["Close"].astype(float)
    df = df[["Date", "close"]].rename(columns={"Date": "ts"}).dropna()
    df["log_ret"] = np.log(df["close"] / df["close"].shift(1)).fillna(0.0)
    return df.reset_index(drop=True)


def evaluate_asset(sym, tv):
    df = fetch(sym)
    if df is None or len(df) < 1500:
        return None
    rr = df["log_ret"].values
    n = len(rr)
    pos = _vol_target(rr, tv, 60)
    bh = np.ones_like(rr)
    pnl = _pnl(pos, rr, 1.0)
    pnl_bh = _pnl(bh, rr, 1.0)

    # Per-fold split
    train_len = int(n * 0.45); val_len = int(n * 0.10); test_len = int(n * 0.12); step = int(n * 0.075)
    folds = []
    s = 0
    while s + train_len + val_len + test_len <= n:
        folds.append((s + train_len + val_len, s + train_len + val_len + test_len))
        s += step
        if len(folds) >= 4: break

    fold_sharpes = []
    fold_bh = []
    for (c, d) in folds:
        fold_sharpes.append(_sharpe(pnl[c:d]))
        fold_bh.append(_sharpe(pnl_bh[c:d]))

    chain = np.concatenate([pnl[c:d] for (c, d) in folds])
    chain_bh = np.concatenate([pnl_bh[c:d] for (c, d) in folds])
    s_vt, lo_vt, hi_vt = _stat_block_bootstrap_sharpe(chain)
    s_bh = _sharpe(chain_bh)
    delta, lo_d, hi_d, p_d = _paired_bootstrap_delta(chain, chain_bh)
    return {
        "symbol": sym,
        "bars": n,
        "fold_sharpes": fold_sharpes,
        "fold_bh": fold_bh,
        "chain_sharpe": s_vt,
        "chain_ci": (lo_vt, hi_vt),
        "chain_bh": s_bh,
        "delta": delta,
        "delta_ci": (lo_d, hi_d),
        "delta_p": p_d,
    }


def main():
    universe = [
        ("GLD", 0.10),    # gold ETF
        ("SLV", 0.20),    # silver
        ("SPY", 0.10),    # US equities
        ("TLT", 0.10),    # long bonds
        ("USO", 0.20),    # oil
        ("BTC-USD", 0.40),  # crypto (higher vol)
    ]
    print(f"=== vol_target multi-asset replication (PIT-clean WF) ===")
    print(f"{'asset':<10s} {'bars':>6s} {'per-fold sharpes':<30s} {'mean':>6s} | {'BH':>6s} | {'chain':>6s} {'[CI low, hi]':<18s} {'Δ vs BH':>7s} {'p':>6s}")
    pass_count = 0
    for sym, tv in universe:
        r = evaluate_asset(sym, tv)
        if r is None:
            print(f"  {sym:<8s}: SKIP")
            continue
        fold_str = " ".join(f"{x:+.2f}" for x in r["fold_sharpes"])
        bh_mean = np.mean(r["fold_bh"])
        passes = r["delta"] > 0.15 and r["delta_p"] < 0.15
        mark = "PASS" if passes else "weak"
        if passes: pass_count += 1
        print(f"  {sym:<10s} {r['bars']:>6d}  {fold_str:<30s} {np.mean(r['fold_sharpes']):>+6.2f} | {bh_mean:>+6.2f} | {r['chain_sharpe']:>+6.2f}  [{r['chain_ci'][0]:+.2f}, {r['chain_ci'][1]:+.2f}]  {r['delta']:>+6.2f}  {r['delta_p']:>5.3f}  {mark}")
    print()
    print(f"Assets where vol_target_3pct beats BH meaningfully: {pass_count}/6")
    print("Bonferroni-corrected significance threshold for 6 assets at α=0.05: p < 0.0083")


if __name__ == "__main__":
    main()
