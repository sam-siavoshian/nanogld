"""Multi-asset equal-weight vol-target portfolio — VLSTM-style SOTA attempt.

Hypothesis: VLSTM achieves Sharpe 2.40 on ~50-asset multi-asset futures
portfolio because of DIVERSIFICATION, not per-asset alpha. If each
asset Sharpe ~0.7 and we have ~25 weakly-correlated assets, portfolio
Sharpe ~= 0.7 × sqrt(N_effective) which can easily clear 2.0+.

Universe: 25 liquid ETFs covering:
  US equities:      SPY, IWM, QQQ, DIA, MDY
  International:    EFA, EEM, VWO
  Bonds:            AGG, TLT, IEF, BIL, HYG
  Commodities:      GLD, SLV, USO, UNG, DBC
  Real estate:      VNQ, IYR
  Sectors:          XLK, XLF, XLE, XLU, XLY

Per asset: vol_target_15pct with 60-day window, long-only, PIT-clean
(1-bar shift). Daily rebalance.

Portfolio: equal-weight average of per-asset positions × per-asset returns,
net of 2 bp turnover cost per asset.

Walk-forward: 4 folds, train_yrs=3 / val_yrs=1 / test_yrs=2, 1-yr step.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

DAILY_BARS_PER_YEAR = 252
BASE_COST_BPS = 2.0


def _arr(x): return x.numpy() if hasattr(x, "numpy") else np.asarray(x)


def _sharpe(r: np.ndarray) -> float:
    r = r[np.isfinite(r)]
    if len(r) < 2:
        return 0.0
    mu, sigma = r.mean(), r.std(ddof=1)
    if sigma < 1e-12:
        return 0.0
    return float(mu / sigma * np.sqrt(DAILY_BARS_PER_YEAR))


def _pnl(pos: np.ndarray, ret: np.ndarray, cost_mult: float = 1.0) -> np.ndarray:
    p = np.nan_to_num(pos, nan=0.0)
    r = np.nan_to_num(ret, nan=0.0)
    cost_frac = (BASE_COST_BPS * cost_mult) / 10_000.0
    prev = np.concatenate([[0.0], p[:-1]])
    return p * r - cost_frac * np.abs(p - prev)


def _rolling_std(x: np.ndarray, window: int) -> np.ndarray:
    n = len(x)
    out = np.zeros(n, dtype=np.float64)
    if window <= 1:
        return out
    cs = np.cumsum(x, dtype=np.float64)
    cs2 = np.cumsum(x * x, dtype=np.float64)
    for i in range(n):
        a = max(0, i - window + 1)
        c = i - a + 1
        if c < 2:
            continue
        m = (cs[i] - (cs[a - 1] if a > 0 else 0.0)) / c
        m2 = (cs2[i] - (cs2[a - 1] if a > 0 else 0.0)) / c
        v = max(0.0, m2 - m * m)
        out[i] = float(np.sqrt(v))
    return out


def _vol_target(rr: np.ndarray, target_vol: float, window: int = 60, cap: float = 1.0) -> np.ndarray:
    rv = _rolling_std(rr, window)
    raw = target_vol / (rv * np.sqrt(DAILY_BARS_PER_YEAR) + 1e-8)
    pos = np.clip(raw, 0.0, cap)
    pos[~np.isfinite(pos)] = 0.0
    return np.concatenate([[0.0], pos[:-1]])


def fetch_daily(sym: str, start: str = "2010-01-01") -> pd.DataFrame:
    import yfinance as yf  # noqa: PLC0415
    df = yf.download(sym, start=start, end="2026-05-25", progress=False, auto_adjust=True)
    if df is None or len(df) == 0:
        raise RuntimeError(f"empty {sym}")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    df = df.reset_index()
    df["close"] = df["Close"].astype(float)
    return df[["Date", "close"]].rename(columns={"Date": "ts"})


UNIVERSE = [
    "SPY", "IWM", "QQQ", "DIA", "MDY",      # US equities
    "EFA", "EEM", "VWO",                     # International
    "AGG", "TLT", "IEF", "BIL", "HYG",       # Bonds
    "GLD", "SLV", "USO", "UNG", "DBC",       # Commodities
    "VNQ", "IYR",                            # REITs
    "XLK", "XLF", "XLE", "XLU", "XLY",       # US sectors
]


def main() -> int:
    print(f"=== Multi-asset portfolio SOTA chase ===")
    print(f"universe size: {len(UNIVERSE)}")

    dfs: dict[str, pd.DataFrame] = {}
    for sym in UNIVERSE:
        try:
            df = fetch_daily(sym, start="2010-01-01")
            df = df.rename(columns={"close": f"close_{sym}"})
            dfs[sym] = df
            print(f"  {sym:<6s}: {len(df)} bars")
        except Exception as e:
            print(f"  {sym:<6s}: SKIP ({e})")
    print(f"loaded {len(dfs)}/{len(UNIVERSE)} assets")

    if len(dfs) < 5:
        print("not enough assets")
        return 1

    # Outer-join on ts then forward-fill
    merged = None
    for sym, df in dfs.items():
        if merged is None:
            merged = df
        else:
            merged = merged.merge(df, on="ts", how="outer")
    merged = merged.sort_values("ts").reset_index(drop=True)
    merged = merged.ffill().dropna().reset_index(drop=True)
    n = len(merged)
    print(f"merged + ffilled + dropna: {n} bars  {merged['ts'].iloc[0].date()} → {merged['ts'].iloc[-1].date()}")

    # Compute per-asset log returns
    rets = {}
    for sym in dfs:
        c = merged[f"close_{sym}"].values.astype(np.float64)
        r = np.zeros_like(c)
        r[1:] = np.log(np.maximum(c[1:], 1e-12) / np.maximum(c[:-1], 1e-12))
        r = np.nan_to_num(r, nan=0.0, posinf=0.0, neginf=0.0)
        rets[sym] = r

    syms = list(rets.keys())
    n_syms = len(syms)

    # Per-asset vol_target_15pct positions (long-only)
    positions = {sym: _vol_target(rets[sym], target_vol=0.15, window=60, cap=1.0) for sym in syms}

    # Per-asset PnL net 2 bp
    per_asset_pnl = {sym: _pnl(positions[sym], rets[sym], 1.0) for sym in syms}

    # Per-asset Sharpe (full series)
    per_asset_sharpe = {sym: _sharpe(per_asset_pnl[sym]) for sym in syms}
    print()
    print("Per-asset full-history Sharpe (vol_target_15pct, net 2bp, long-only):")
    for sym in sorted(syms, key=lambda s: -per_asset_sharpe[s]):
        print(f"  {sym:<6s}: {per_asset_sharpe[sym]:+.4f}")

    # Equal-weight portfolio PnL
    pnl_matrix = np.column_stack([per_asset_pnl[sym] for sym in syms])
    portfolio_pnl = pnl_matrix.mean(axis=1)  # equal weight
    portfolio_sharpe = _sharpe(portfolio_pnl)
    print()
    print(f"Equal-weight portfolio Sharpe (full): {portfolio_sharpe:+.4f}")

    # Walk-forward: 4 folds
    train_len = 3 * DAILY_BARS_PER_YEAR
    val_len = DAILY_BARS_PER_YEAR
    test_len = 2 * DAILY_BARS_PER_YEAR
    step = DAILY_BARS_PER_YEAR
    folds = []
    s = 0
    while s + train_len + val_len + test_len <= n:
        folds.append((s, s + train_len, s + train_len + val_len, s + train_len + val_len + test_len))
        s += step
    print(f"folds: {len(folds)}")
    for i, (a, b, c, d) in enumerate(folds):
        print(f"  fold {i}: train [{a},{b}) val [{b},{c}) test [{c},{d}) ({d-c} days)")

    fold_pnls = []
    fold_sharpes = []
    for fi, (a, b, c, d) in enumerate(folds):
        # No train-specific tuning; vol_target_15 is a fixed rule
        ts = slice(c, d)
        fold_pnl_matrix = pnl_matrix[ts]
        fold_port = fold_pnl_matrix.mean(axis=1)
        s = _sharpe(fold_port)
        fold_pnls.append(fold_port)
        fold_sharpes.append(s)
        print(f"  fold {fi}: portfolio Sharpe = {s:+.4f}")
    mean_fold = float(np.mean(fold_sharpes))
    median_fold = float(np.median(fold_sharpes))
    chain = np.concatenate(fold_pnls)
    agg = _sharpe(chain)
    print()
    print(f"Per-fold portfolio Sharpe: mean={mean_fold:+.4f}  median={median_fold:+.4f}  per_fold={[round(x,3) for x in fold_sharpes]}")
    print(f"Aggregate OOS chain Sharpe ({len(chain)} days): {agg:+.4f}")

    # Try multiple target_vols
    print()
    print("Target-vol sweep on equal-weight portfolio:")
    for tv in [0.05, 0.10, 0.15, 0.20, 0.30]:
        positions_tv = {sym: _vol_target(rets[sym], target_vol=tv, window=60, cap=1.0) for sym in syms}
        pnl_tv = np.column_stack([_pnl(positions_tv[sym], rets[sym], 1.0) for sym in syms]).mean(axis=1)
        # Per-fold aggregate
        f_sh = [_sharpe(pnl_tv[c:d]) for (a, b, c, d) in folds]
        print(f"  vol_target_{int(tv*100):>3d}%: per_fold={[round(x,3) for x in f_sh]}  mean={np.mean(f_sh):+.4f}  agg_chain={_sharpe(np.concatenate([pnl_tv[c:d] for (a,b,c,d) in folds])):+.4f}")

    print()
    print(f"VLSTM SOTA (multi-asset portfolio gross of cost): +2.40")
    print(f"F2F SOTA (daily gold single-asset):              +2.88")
    if mean_fold > 2.88:
        print(f">>> NEW SOTA: beats F2F by {mean_fold - 2.88:+.4f}")
    elif mean_fold > 2.40:
        print(f">>> beats VLSTM by {mean_fold - 2.40:+.4f}; F2F gap {mean_fold - 2.88:+.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
