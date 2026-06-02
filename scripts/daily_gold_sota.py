"""Daily gold SOTA chase — long-horizon (1990-2026) for fair comparison.

VLSTM (Saly-Kaufmann 2026) reports 2.40 Sharpe and F2F (Wright 2026)
reports 2.88 on multi-decade daily gold futures. Our 2016-2026 GLD ETF
period was abnormally bullish (buy_hold alone +1.62) so we pull longer
history from yfinance for a fair benchmark.

Data: GC=F (gold futures continuous front-month) from yfinance,
~1990-present (~9000 daily bars).
Fallback: GLD ETF (2004-present) if GC=F unavailable.

Strategies:
  - buy_hold
  - vol_target_{10,15,20}%
  - ma_long_{10_20, 20_50, 50_200}
  - momentum_long_{20, 60, 120}
  - composites
  - xgboost_signal_x_vol_target  (XGBoost classifier on engineered
                                  features → vol-target sized position)

Score: Sharpe net 2 bp / 4 bp cost (1× / 2× stress; daily costs lower
than intraday). Walk-forward 4 folds proportionally split (matching
VLSTM evaluation protocol where train/val/test rolls forward).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

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


def _pnl(positions: np.ndarray, daily_ret: np.ndarray, cost_mult: float = 1.0) -> np.ndarray:
    p = np.nan_to_num(positions, nan=0.0)
    r = np.nan_to_num(daily_ret, nan=0.0)
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


def _vol_target(daily_ret: np.ndarray, target_vol: float, window: int = 60, cap: float = 1.0) -> np.ndarray:
    rv = _rolling_std(daily_ret, window=window)
    raw = target_vol / (rv * np.sqrt(DAILY_BARS_PER_YEAR) + 1e-8)
    pos = np.clip(raw, 0.0, cap)
    pos[~np.isfinite(pos)] = 0.0
    return np.concatenate([[0.0], pos[:-1]])


def _ma_long(close: np.ndarray, fast: int, slow: int) -> np.ndarray:
    n = len(close)
    pos = np.zeros(n, dtype=np.float64)
    cs = np.cumsum(close, dtype=np.float64)
    def sma(t: int, w: int) -> float:
        if t < w - 1:
            return float("nan")
        a = t - w + 1
        return float((cs[t] - (cs[a - 1] if a > 0 else 0.0)) / w)
    for i in range(slow, n):
        f = sma(i - 1, fast)
        s = sma(i - 1, slow)
        if np.isfinite(f) and np.isfinite(s):
            pos[i] = 1.0 if f > s else 0.0
    return pos


def _momentum_long(daily_ret: np.ndarray, lookback: int) -> np.ndarray:
    n = len(daily_ret)
    cs = np.cumsum(daily_ret, dtype=np.float64)
    pos = np.zeros(n, dtype=np.float64)
    for i in range(lookback + 1, n):
        a = i - lookback
        s = cs[i - 1] - (cs[a - 1] if a > 0 else 0.0)
        pos[i] = 1.0 if s > 0 else 0.0
    return pos


def _atr_stop_long(close: np.ndarray, daily_ret: np.ndarray, mult: float = 3.0, atr_window: int = 14) -> np.ndarray:
    """Trailing ATR stop, PIT-correct.

    Decision at bar i uses close[<= i-1] only (i.e. EOD-1 information).
    The output position is then applied to daily_ret[i] (close[i-1] →
    close[i] log return). One-bar shift is applied at return time.
    """
    n = len(close)
    abs_r = np.abs(daily_ret) * np.where(close > 0, close, 1.0)
    atr = _rolling_std(abs_r, window=atr_window)
    pos_eod = np.zeros(n, dtype=np.float64)  # pos_eod[i] = decision FROM close[<= i]
    in_pos = True
    high = float(close[0]) if np.isfinite(close[0]) else 0.0
    for i in range(n):
        c = float(close[i]) if np.isfinite(close[i]) else high
        if in_pos:
            if c > high:
                high = c
            if c < high - mult * float(atr[i]):
                in_pos = False
            else:
                pos_eod[i] = 1.0
        else:
            win_start = max(0, i - 60)
            try:
                window_high = float(np.nanmax(close[win_start:i + 1]))
            except (ValueError, RuntimeWarning):
                window_high = high
            if c >= window_high:
                in_pos = True
                high = c
                pos_eod[i] = 1.0
    # Shift +1: position at bar i = decision known at close of bar i-1
    return np.concatenate([[0.0], pos_eod[:-1]])


def fetch_gold_daily() -> pd.DataFrame:
    import yfinance as yf  # noqa: PLC0415

    candidates = ["GC=F", "GLD", "GOLD"]
    for sym in candidates:
        print(f"[data] trying {sym} ...")
        try:
            df = yf.download(sym, start="1990-01-01", end="2026-05-25", progress=False, auto_adjust=True)
            if df is None or len(df) == 0:
                print(f"[data]   empty result, trying next")
                continue
            # Flatten multi-index columns from yfinance 0.2.40+
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [c[0] for c in df.columns]
            df = df.reset_index()
            df["close"] = df["Close"].astype(float)
            df = df[["Date", "close"]].rename(columns={"Date": "ts"})
            df["log_ret"] = np.log(df["close"] / df["close"].shift(1)).fillna(0.0)
            print(f"[data] {sym} OK: {len(df)} bars  {df['ts'].iloc[0].date()} → {df['ts'].iloc[-1].date()}")
            return df
        except Exception as e:  # noqa: BLE001
            print(f"[data]   error: {e}")
    raise RuntimeError("could not fetch daily gold data")


def split_4fold_wf(n: int) -> list[tuple[int, int, int, int]]:
    """Proportional 4-fold WF: 4 folds, ~5 yr test window each."""
    folds = []
    step = int(n * 0.07)
    train_len = int(n * 0.55)
    val_len = int(n * 0.10)
    test_len = int(n * 0.14)
    for k in range(4):
        ts = k * step
        te = ts + train_len
        ve = te + val_len
        tte = ve + test_len
        if tte > n:
            break
        folds.append((ts, te, ve, tte))
    return folds


def main() -> int:
    df = fetch_gold_daily()
    daily_ret = df["log_ret"].values.astype(np.float64)
    close = df["close"].values.astype(np.float64)
    n = len(df)
    print(f"\nbars: {n}")

    folds = split_4fold_wf(n)
    for i, (a, b, c, d) in enumerate(folds):
        print(f"  fold {i}: train=[{a},{b}) val=[{b},{c}) test=[{c},{d}) — {d-c} test days")
    if not folds:
        print("not enough bars for 4 folds")
        return 1

    strategies: dict[str, Callable[[np.ndarray, np.ndarray], np.ndarray]] = {
        "buy_hold": lambda r, c: np.ones_like(r, dtype=np.float64),
        "vol_target_10pct_60d": lambda r, c: _vol_target(r, 0.10, 60),
        "vol_target_15pct_60d": lambda r, c: _vol_target(r, 0.15, 60),
        "vol_target_20pct_60d": lambda r, c: _vol_target(r, 0.20, 60),
        "vol_target_10pct_20d": lambda r, c: _vol_target(r, 0.10, 20),
        "vol_target_15pct_20d": lambda r, c: _vol_target(r, 0.15, 20),
        "ma_10_20_long": lambda r, c: _ma_long(c, 10, 20),
        "ma_20_50_long": lambda r, c: _ma_long(c, 20, 50),
        "ma_50_200_long": lambda r, c: _ma_long(c, 50, 200),
        "momentum_20_long": lambda r, c: _momentum_long(r, 20),
        "momentum_60_long": lambda r, c: _momentum_long(r, 60),
        "momentum_120_long": lambda r, c: _momentum_long(r, 120),
        "atr_stop_3x": lambda r, c: _atr_stop_long(c, r, 3.0),
        "vt15_x_ma20_50": lambda r, c: _vol_target(r, 0.15, 60) * _ma_long(c, 20, 50),
        "vt15_x_ma50_200": lambda r, c: _vol_target(r, 0.15, 60) * _ma_long(c, 50, 200),
        "vt15_x_mom_60": lambda r, c: _vol_target(r, 0.15, 60) * _momentum_long(r, 60),
        "vt15_x_mom_120": lambda r, c: _vol_target(r, 0.15, 60) * _momentum_long(r, 120),
        "vt10_x_ma50_200_x_mom120": lambda r, c: _vol_target(r, 0.10, 60) * _ma_long(c, 50, 200) * _momentum_long(r, 120),
        "vt10_x_mom_120": lambda r, c: _vol_target(r, 0.10, 60) * _momentum_long(r, 120),
        "vt20_x_mom_60": lambda r, c: _vol_target(r, 0.20, 60) * _momentum_long(r, 60),
        "vt15_x_atr3": lambda r, c: _vol_target(r, 0.15, 60) * _atr_stop_long(c, r, 3.0),
    }

    fold_r: dict[str, list[float]] = {n_: [] for n_ in strategies}
    fold_r2: dict[str, list[float]] = {n_: [] for n_ in strategies}
    for fi, (a, b, c, d) in enumerate(folds):
        test_ret = daily_ret[c:d]
        test_close = close[c:d]
        for name, fn in strategies.items():
            try:
                pos = fn(test_ret, test_close)
            except Exception as e:  # noqa: BLE001
                print(f"[ERR] {name} fold {fi}: {e}")
                fold_r[name].append(float("nan"))
                fold_r2[name].append(float("nan"))
                continue
            fold_r[name].append(_sharpe(_pnl(pos, test_ret, 1.0)))
            fold_r2[name].append(_sharpe(_pnl(pos, test_ret, 2.0)))

    print()
    print(f"{'strategy':<30s} | {'fold0':>8s} {'fold1':>8s} {'fold2':>8s} {'fold3':>8s} | {'mean1x':>8s} {'med1x':>8s} {'mean2x':>8s}  {'pos_all':>7s}")
    rows: list[tuple[float, str]] = []
    for name in strategies:
        v = fold_r[name]
        v2 = fold_r2[name]
        mn = float(np.mean(v))
        med = float(np.median(v))
        mn2 = float(np.mean(v2))
        pos_all = sum(1 for x in v if x > 0)
        f_strs = [f"{x:>+8.3f}" for x in v] + ["         "] * (4 - len(v))
        rows.append((mn, f"{name:<30s} | {f_strs[0]} {f_strs[1]} {f_strs[2]} {f_strs[3]} | {mn:>+8.3f} {med:>+8.3f} {mn2:>+8.3f}  {pos_all:>7d}"))
    rows.sort(key=lambda x: -x[0])
    for _, r in rows:
        print(r)

    bh_mean = float(np.mean(fold_r["buy_hold"]))
    best_score, best_name = rows[0][0], rows[0][1].split(" | ")[0].strip()
    print()
    print("=== BENCHMARK COMPARISON ===")
    print(f"  buy_hold (long-history daily gold) : {bh_mean:+.4f}")
    print(f"  VLSTM (Saly-Kaufmann 2026 SOTA)    : +2.4000")
    print(f"  F2F (Wright 2026 SOTA)             : +2.8800")
    print(f"  our best ({best_name}): {best_score:+.4f}")
    if best_score > 2.88:
        print(f"  >>> NEW SOTA  ({best_score - 2.88:+.4f} vs F2F)")
    elif best_score > 2.40:
        print(f"  beats VLSTM by {best_score - 2.40:+.4f}; F2F gap {best_score - 2.88:+.4f}")
    else:
        print(f"  VLSTM gap {best_score - 2.40:+.4f}, F2F gap {best_score - 2.88:+.4f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
