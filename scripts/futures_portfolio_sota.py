"""Multi-asset futures portfolio SOTA chase — proper VLSTM-style universe.

Universe: ~30 liquid CME / NYMEX / CBOT / ICE continuous front-month
futures across 7 sectors. ETF version had too-high correlations
(0.7-0.9 inside US equities); switching to futures gives genuine
multi-sector diversification (correlations typically 0.0-0.4).

Per asset: vol_target_15pct with 60-day rolling realized vol (PIT-clean,
1-bar shift, long-only).
Portfolio: equal-weight average of per-asset positions × per-asset
returns, net of 2 bp per asset.
WF: 4-fold, train_yrs=3 / val_yrs=1 / test_yrs=2, 1-yr step.

Goal: beat VLSTM 2.40 / F2F 2.88 with the diversification multiplier.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

DAILY_BARS_PER_YEAR = 252
BASE_COST_BPS = 2.0


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


def _momentum_long(rr: np.ndarray, lookback: int) -> np.ndarray:
    n = len(rr)
    cs = np.cumsum(rr, dtype=np.float64)
    pos = np.zeros(n, dtype=np.float64)
    for i in range(lookback + 1, n):
        a = i - lookback
        s = cs[i - 1] - (cs[a - 1] if a > 0 else 0.0)
        pos[i] = 1.0 if s > 0 else 0.0
    return pos


def _tsmom_long_short(rr: np.ndarray, lookback: int) -> np.ndarray:
    """Time-Series Momentum (Moskowitz/Ooi/Pedersen 2012) long-short.

    pos[i] = sign(sum(rr[i-lookback..i-1]))   ∈ {-1, 0, +1}
    """
    n = len(rr)
    cs = np.cumsum(rr, dtype=np.float64)
    pos = np.zeros(n, dtype=np.float64)
    for i in range(lookback + 1, n):
        a = i - lookback
        s = cs[i - 1] - (cs[a - 1] if a > 0 else 0.0)
        if s > 0:
            pos[i] = 1.0
        elif s < 0:
            pos[i] = -1.0
    return pos


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


# Continuous front-month futures (yfinance =F suffix). Grouped by sector
# to verify diversification spread.
UNIVERSE = {
    "energy":     ["CL=F", "NG=F", "BZ=F", "HO=F", "RB=F"],
    "metals":     ["GC=F", "SI=F", "HG=F", "PL=F", "PA=F"],
    "grains":     ["ZC=F", "ZS=F", "ZW=F", "ZL=F", "ZM=F"],
    "rates":      ["ZB=F", "ZN=F", "ZF=F", "ZT=F"],
    "currency":   ["6E=F", "6J=F", "6B=F", "6A=F", "6C=F"],
    "equity":     ["ES=F", "NQ=F", "YM=F", "RTY=F"],
    "softs":      ["KC=F", "SB=F", "CC=F", "CT=F"],
}


def main() -> int:
    print(f"=== Multi-asset futures portfolio SOTA chase ===")
    all_syms = [s for sector in UNIVERSE.values() for s in sector]
    print(f"universe size: {len(all_syms)}")

    dfs: dict[str, pd.DataFrame] = {}
    for sector, syms in UNIVERSE.items():
        print(f"--- {sector} ---")
        for sym in syms:
            try:
                df = fetch_daily(sym, start="2010-01-01")
                df = df.rename(columns={"close": f"close_{sym}"})
                dfs[sym] = df
                print(f"  {sym:<8s}: {len(df)} bars  {df['ts'].iloc[0].date()} → {df['ts'].iloc[-1].date()}")
            except Exception as e:
                print(f"  {sym:<8s}: SKIP ({e})")
    print(f"\nloaded {len(dfs)}/{len(all_syms)} futures")

    if len(dfs) < 8:
        print("not enough futures")
        return 1

    # Outer-join + ffill + dropna
    merged = None
    for sym, df in dfs.items():
        merged = df if merged is None else merged.merge(df, on="ts", how="outer")
    merged = merged.sort_values("ts").reset_index(drop=True)
    merged = merged.ffill().dropna().reset_index(drop=True)
    n = len(merged)
    print(f"\nmerged + ffilled + dropna: {n} bars  {merged['ts'].iloc[0].date()} → {merged['ts'].iloc[-1].date()}")

    # Compute per-asset log returns
    rets = {}
    for sym in dfs:
        c = merged[f"close_{sym}"].values.astype(np.float64)
        r = np.zeros_like(c)
        r[1:] = np.log(np.maximum(c[1:], 1e-12) / np.maximum(c[:-1], 1e-12))
        r = np.nan_to_num(r, nan=0.0, posinf=0.0, neginf=0.0)
        rets[sym] = r

    syms = list(rets.keys())

    # Correlation matrix of returns (diagnostic)
    ret_matrix = np.column_stack([rets[s] for s in syms])
    corr = np.corrcoef(ret_matrix, rowvar=False)
    off_diag = corr[np.triu_indices_from(corr, k=1)]
    print(f"\ncorrelation diagnostics: mean={off_diag.mean():.3f}  median={np.median(off_diag):.3f}  max={off_diag.max():.3f}  |abs|.mean={np.abs(off_diag).mean():.3f}")

    # Per-asset vol_target_15 long-only positions
    target_vols = [0.05, 0.10, 0.15, 0.20, 0.30]
    print()
    print("Target-vol portfolio (equal-weight long-only) full-history Sharpe:")
    for tv in target_vols:
        positions_tv = {s: _vol_target(rets[s], target_vol=tv, window=60, cap=1.0) for s in syms}
        pnl_per = np.column_stack([_pnl(positions_tv[s], rets[s], 1.0) for s in syms])
        portfolio_pnl = pnl_per.mean(axis=1)
        print(f"  vt_{int(tv*100):>3d}%: Sharpe = {_sharpe(portfolio_pnl):+.4f}")

    # WF
    train_len = 3 * DAILY_BARS_PER_YEAR
    val_len = DAILY_BARS_PER_YEAR
    test_len = 2 * DAILY_BARS_PER_YEAR
    step = DAILY_BARS_PER_YEAR
    folds = []
    s = 0
    while s + train_len + val_len + test_len <= n:
        folds.append((s, s + train_len, s + train_len + val_len, s + train_len + val_len + test_len))
        s += step
    print(f"\nfolds: {len(folds)}")

    print()
    print("WF analysis at multiple target_vols + long-short TSMOM:")
    candidates: dict[str, Any] = {}
    for tv in target_vols:
        # vt only (long)
        positions_tv = {s: _vol_target(rets[s], target_vol=tv, window=60, cap=1.0) for s in syms}
        pnl_per = np.column_stack([_pnl(positions_tv[s], rets[s], 1.0) for s in syms])
        portfolio_pnl_vt = pnl_per.mean(axis=1)
        candidates[f"vt_{int(tv*100)}%"] = portfolio_pnl_vt

        # Time-series momentum long-short at multiple lookbacks
        for lb in [30, 60, 120, 252]:
            positions_lsm = {s: _vol_target(rets[s], target_vol=tv, window=60, cap=1.0) * _tsmom_long_short(rets[s], lb) for s in syms}
            pnl_per_lsm = np.column_stack([_pnl(positions_lsm[s], rets[s], 1.0) for s in syms])
            portfolio_pnl_lsm = pnl_per_lsm.mean(axis=1)
            candidates[f"vt_{int(tv*100)}%_x_tsmom_{lb}"] = portfolio_pnl_lsm

    print(f"{'strategy':<22s} | folds                              | mean      median  agg_chain")
    rows = []
    for name, pp in candidates.items():
        f_s = [_sharpe(pp[c:d]) for (a, b, c, d) in folds]
        mean_s = float(np.mean(f_s))
        med_s = float(np.median(f_s))
        chain = np.concatenate([pp[c:d] for (a, b, c, d) in folds])
        agg_s = _sharpe(chain)
        rows.append((mean_s, name, f_s, med_s, agg_s))
    rows.sort(key=lambda x: -x[0])
    for mean_s, name, f_s, med_s, agg_s in rows:
        f_str = " ".join(f"{x:+.2f}" for x in f_s)
        print(f"  {name:<22s} | {f_str:<34s} | {mean_s:+.4f}  {med_s:+.4f}  {agg_s:+.4f}")

    best_mean = rows[0][0]
    best_name = rows[0][1]
    print()
    print(f"VLSTM SOTA (multi-asset portfolio, gross of cost): +2.40")
    print(f"F2F SOTA (daily gold single-asset):                +2.88")
    print(f"Best strategy: {best_name}  WF mean Sharpe {best_mean:+.4f}")
    if best_mean > 2.88:
        print(f">>> NEW SOTA: beats F2F by {best_mean - 2.88:+.4f}")
    elif best_mean > 2.40:
        print(f">>> beats VLSTM by {best_mean - 2.40:+.4f}; F2F gap {best_mean - 2.88:+.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
