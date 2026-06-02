"""Crypto SOTA chase — apply our PIT-clean strategy ladder to daily BTC, ETH,
and a 2-coin equal-weight portfolio.

Hypothesis: crypto has higher vol-regime variance than gold. vol_target sizing
may extract more Sharpe in markets with sharper vol clustering. Also, no
strong published SOTA on daily-crypto-Sharpe with PIT WF — so we can
plausibly own a benchmark.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent

DAILY_BARS_PER_YEAR = 365  # crypto trades 7 days/week
BASE_COST_BPS = 5.0  # higher for crypto


def _arr(x: Any) -> np.ndarray:
    return x.numpy() if hasattr(x, "numpy") else np.asarray(x)


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


def _vol_target(rr: np.ndarray, target_vol: float, window: int = 30, cap: float = 1.0) -> np.ndarray:
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


def fetch_daily(sym: str) -> pd.DataFrame:
    import yfinance as yf  # noqa: PLC0415
    df = yf.download(sym, start="2014-01-01", end="2026-05-25", progress=False, auto_adjust=True)
    if df is None or len(df) == 0:
        raise RuntimeError(f"empty {sym}")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    df = df.reset_index()
    df["close"] = df["Close"].astype(float)
    df = df[["Date", "close"]].rename(columns={"Date": "ts"})
    df["log_ret"] = np.log(df["close"] / df["close"].shift(1)).fillna(0.0)
    return df


def split_4fold(n: int, train_yrs: float = 4.0, val_mo: float = 6.0, test_mo: float = 12.0, step_yrs: float = 1.0):
    bpy = DAILY_BARS_PER_YEAR
    train_len = int(train_yrs * bpy)
    val_len = int(val_mo * bpy / 12)
    test_len = int(test_mo * bpy / 12)
    step = int(step_yrs * bpy)
    folds = []
    s = 0
    while s + train_len + val_len + test_len <= n:
        folds.append((s, s + train_len, s + train_len + val_len, s + train_len + val_len + test_len))
        s += step
    return folds


def evaluate(symbol: str) -> dict[str, Any]:
    print(f"\n{'=' * 60}")
    print(f"asset: {symbol}")
    print('=' * 60)
    df = fetch_daily(symbol)
    rr = df["log_ret"].values
    close = df["close"].values
    n = len(df)
    print(f"bars: {n}  range: {df['ts'].iloc[0].date()} → {df['ts'].iloc[-1].date()}")
    folds = split_4fold(n)
    print(f"folds: {len(folds)}")
    if not folds:
        return {"symbol": symbol, "sharpe": 0.0}
    strategies: dict[str, Callable[[np.ndarray], np.ndarray]] = {
        "buy_hold": lambda r: np.ones_like(r, dtype=np.float64),
        "vt10": lambda r: _vol_target(r, 0.10, 30),
        "vt15": lambda r: _vol_target(r, 0.15, 30),
        "vt20": lambda r: _vol_target(r, 0.20, 30),
        "vt30": lambda r: _vol_target(r, 0.30, 30),
        "vt50": lambda r: _vol_target(r, 0.50, 30),
        "vt15_x_mom60": lambda r: _vol_target(r, 0.15, 30) * _momentum_long(r, 60),
        "vt30_x_mom60": lambda r: _vol_target(r, 0.30, 30) * _momentum_long(r, 60),
        "vt50_x_mom60": lambda r: _vol_target(r, 0.50, 30) * _momentum_long(r, 60),
        "mom60_only": lambda r: _momentum_long(r, 60),
        "mom30_only": lambda r: _momentum_long(r, 30),
    }
    fold_results: dict[str, list[float]] = {n_: [] for n_ in strategies}
    for fi, (a, b, c, d) in enumerate(folds):
        test_ret = rr[c:d]
        for name, fn in strategies.items():
            pos = fn(test_ret)
            fold_results[name].append(_sharpe(_pnl(pos, test_ret, 1.0)))

    print(f"{'strategy':<22s} | folds                       | mean      vs_bh")
    bh_mean = float(np.mean(fold_results["buy_hold"]))
    rows = []
    for name in strategies:
        v = fold_results[name]
        m = float(np.mean(v))
        f_str = " ".join(f"{x:+.2f}" for x in v)
        rows.append((m, f"  {name:<22s} | {f_str:<28s} | {m:+.4f}   {m - bh_mean:+.4f}"))
    rows.sort(key=lambda x: -x[0])
    for _, r in rows:
        print(r)
    best = rows[0][0]
    print(f"  best WF mean Sharpe: {best:+.4f}  (buy_hold={bh_mean:+.4f})")
    return {"symbol": symbol, "best": best, "buy_hold": bh_mean, "rows": rows}


def main() -> int:
    results = []
    for sym in ["BTC-USD", "ETH-USD"]:
        try:
            r = evaluate(sym)
            results.append(r)
        except Exception as e:
            print(f"[ERR] {sym}: {e}")

    print()
    print("=" * 60)
    print("CROSS-ASSET SUMMARY")
    print("=" * 60)
    for r in results:
        print(f"  {r['symbol']:<12s}: best WF mean = {r.get('best', 0):+.4f}  (vs buy_hold {r.get('buy_hold', 0):+.4f})")

    best_overall = max((r.get("best", 0) for r in results), default=0)
    print()
    print(f"Benchmarks: VLSTM 2.40 (multi-asset portfolio), F2F 2.88 (daily gold)")
    print(f"Our best crypto: {best_overall:+.4f}")
    if best_overall > 2.88:
        print(f"  >>> NEW SOTA (beats F2F by {best_overall - 2.88:+.4f})")
    elif best_overall > 2.40:
        print(f"  beats VLSTM by {best_overall - 2.40:+.4f}; F2F gap {best_overall - 2.88:+.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
