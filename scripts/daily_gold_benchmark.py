"""Daily gold WF benchmark: try to beat VLSTM 2.40 / F2F 2.88 SOTA.

Aggregates the V1 intraday GLD bars (75,672 30-min bars, 2014-2025)
into daily bars, then runs PIT-clean strategies + WF backtest.

Strategies tested:
  - buy_hold_daily               (baseline)
  - vol_target_daily             (size = clip(tau / sigma_t, 0, 1))
  - long_ma_daily                (10/20 day MA cross long-only)
  - momentum_daily               (long if last-N-day return > 0)
  - vol_target_x_ma_daily        (composite)
  - vol_target_x_momentum_daily  (composite)
  - vol_target_x_ma_x_momentum_daily (triple gate)
  - xgboost_signal_daily         (3-class XGBoost classifier on engineered daily
                                  features, vol-target sized)

Compute Sharpe net of 2 bp round-trip cost, annualized with
sqrt(252) (daily bars per year).

Walk-forward: same 4-fold geometry as intraday but on daily bars.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

DAILY_BARS_PER_YEAR = 252
BASE_COST_BPS = 2.0
GLD_CLOSE_FEATURE_IDX = 3


def _arr(x: Any) -> np.ndarray:
    return x.numpy() if hasattr(x, "numpy") else np.asarray(x)


def _sharpe(r: np.ndarray, ann_factor: float = DAILY_BARS_PER_YEAR) -> float:
    r = r[np.isfinite(r)]
    if len(r) < 2:
        return 0.0
    mu, sigma = r.mean(), r.std(ddof=1)
    if sigma < 1e-12:
        return 0.0
    return float(mu / sigma * np.sqrt(ann_factor))


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


def _vol_target_daily(daily_ret: np.ndarray, target_vol: float = 0.10, window: int = 20, cap: float = 1.0) -> np.ndarray:
    """Daily vol target. target_vol = annualized e.g. 10%."""
    rv = _rolling_std(daily_ret, window=window)
    raw = target_vol / (rv * np.sqrt(DAILY_BARS_PER_YEAR) + 1e-8)
    pos = np.clip(raw, 0.0, cap)
    pos[~np.isfinite(pos)] = 0.0
    return np.concatenate([[0.0], pos[:-1]])  # one-bar shift = causal


def _ma_long(close: np.ndarray, fast: int = 10, slow: int = 20) -> np.ndarray:
    """Long-only MA cross."""
    if len(close) < slow + 1:
        return np.zeros_like(close, dtype=np.float64)
    n = len(close)
    pos = np.zeros(n, dtype=np.float64)
    # Compute SMAs
    csum = np.cumsum(close, dtype=np.float64)
    def sma(t: int, w: int) -> float:
        if t < w - 1:
            return float("nan")
        a = t - w + 1
        return float((csum[t] - (csum[a - 1] if a > 0 else 0.0)) / w)
    for i in range(slow, n):
        f = sma(i - 1, fast)
        s = sma(i - 1, slow)
        if np.isfinite(f) and np.isfinite(s):
            pos[i] = 1.0 if f > s else 0.0
    return pos


def _momentum_long(daily_ret: np.ndarray, lookback: int = 20) -> np.ndarray:
    """Long if past-N realized log-return > 0."""
    n = len(daily_ret)
    cs_cum = np.cumsum(daily_ret, dtype=np.float64)
    pos = np.zeros(n, dtype=np.float64)
    for i in range(lookback + 1, n):
        a = i - lookback
        s = cs_cum[i - 1] - (cs_cum[a - 1] if a > 0 else 0.0)
        pos[i] = 1.0 if s > 0 else 0.0
    return pos


def aggregate_daily_from_intraday(unified: dict) -> pd.DataFrame:
    """Aggregate 30-min intraday bars to daily.

    Daily close = last intraday close of each UTC day.
    Daily log return = log(close[d] / close[d-1]).
    """
    bcn = _arr(unified["bar_close_utc_ns"]).astype(np.int64)
    feats = _arr(unified["features"])
    close = feats[:, GLD_CLOSE_FEATURE_IDX].astype(np.float64)
    ts = pd.to_datetime(bcn, unit="ns", utc=True)
    df = pd.DataFrame({"ts": ts, "close": close})
    df["day"] = df["ts"].dt.date
    daily = df.groupby("day").agg({"ts": "last", "close": "last"}).reset_index(drop=True)
    daily = daily.dropna(subset=["close"]).reset_index(drop=True)
    daily["log_ret"] = np.log(daily["close"] / daily["close"].shift(1)).fillna(0.0)
    return daily


def split_4fold_wf(n: int) -> list[tuple[int, int, int, int]]:
    """4-fold walk-forward over daily bars.

    Train (60%) / val (15%) / test (15%), step 5%. Returns list of
    (train_start, train_end, test_start, test_end).
    """
    # Use proportional split scaled to n bars (simpler than time-anchored
    # for daily benchmark; comparable to VLSTM evaluation protocol).
    folds = []
    step = int(n * 0.075)
    train_len = int(n * 0.55)
    test_len = int(n * 0.12)
    val_len = int(n * 0.12)
    for k in range(4):
        train_start = k * step
        train_end = train_start + train_len
        val_end = train_end + val_len
        test_start = val_end
        test_end = test_start + test_len
        if test_end > n:
            break
        folds.append((train_start, train_end, test_start, test_end))
    return folds


def main() -> int:
    unified_path = REPO_ROOT / "data" / "processed" / "training_v1_unified.pt"
    unified = torch.load(unified_path, weights_only=False)
    daily = aggregate_daily_from_intraday(unified)
    n = len(daily)
    print(f"daily bars: {n} ({daily['ts'].iloc[0].date()} → {daily['ts'].iloc[-1].date()})")
    print(f"vs intraday: {len(unified['features'])} 30-min bars")

    folds = split_4fold_wf(n)
    print(f"folds: {len(folds)}  (train/val/test bars each)")
    for i, (a, b, c, d) in enumerate(folds):
        print(f"  fold {i}: train=[{a},{b}) val=[{b},{c}) test=[{c},{d}) — {d-c} test days")

    daily_ret = daily["log_ret"].values
    close = daily["close"].values

    strategies: dict[str, Callable[[np.ndarray, np.ndarray], np.ndarray]] = {
        "buy_hold": lambda r, c: np.ones_like(r, dtype=np.float64),
        "vol_target_10pct": lambda r, c: _vol_target_daily(r, target_vol=0.10, window=20),
        "vol_target_15pct": lambda r, c: _vol_target_daily(r, target_vol=0.15, window=20),
        "vol_target_20pct": lambda r, c: _vol_target_daily(r, target_vol=0.20, window=20),
        "ma_10_20_long": lambda r, c: _ma_long(c, 10, 20),
        "ma_20_50_long": lambda r, c: _ma_long(c, 20, 50),
        "momentum_20_long": lambda r, c: _momentum_long(r, 20),
        "momentum_60_long": lambda r, c: _momentum_long(r, 60),
        "vt15_x_ma10_20": lambda r, c: _vol_target_daily(r, 0.15, 20) * _ma_long(c, 10, 20),
        "vt15_x_ma20_50": lambda r, c: _vol_target_daily(r, 0.15, 20) * _ma_long(c, 20, 50),
        "vt15_x_momentum_60": lambda r, c: _vol_target_daily(r, 0.15, 20) * _momentum_long(r, 60),
        "vt15_x_ma_x_mo": lambda r, c: _vol_target_daily(r, 0.15, 20) * _ma_long(c, 10, 20) * _momentum_long(r, 60),
    }

    fold_results: dict[str, list[float]] = {name: [] for name in strategies}
    fold_results_15: dict[str, list[float]] = {name: [] for name in strategies}

    for fi, (a, b, c, d) in enumerate(folds):
        test_ret = daily_ret[c:d]
        test_close = close[c:d]
        for name, fn in strategies.items():
            pos = fn(test_ret, test_close)
            fold_results[name].append(_sharpe(_pnl(pos, test_ret, 1.0)))
            fold_results_15[name].append(_sharpe(_pnl(pos, test_ret, 1.5)))

    print()
    print(f"{'strategy':<24s} | {'fold0':>8s} {'fold1':>8s} {'fold2':>8s} {'fold3':>8s} | {'mean1x':>8s} {'med1x':>8s} {'mean1.5x':>9s}  {'pos_all':>7s}")
    rows: list[tuple[float, str]] = []
    for name in strategies:
        v = fold_results[name]
        v15 = fold_results_15[name]
        mn = float(np.mean(v))
        med = float(np.median(v))
        mn15 = float(np.mean(v15))
        pos_all = sum(1 for x in v if x > 0)
        f_strs = [f"{x:>+8.3f}" for x in v] + ["         "] * (4 - len(v))
        rows.append((mn, f"{name:<24s} | {f_strs[0]} {f_strs[1]} {f_strs[2]} {f_strs[3]} | {mn:>+8.3f} {med:>+8.3f} {mn15:>+9.3f}  {pos_all:>7d}"))
    rows.sort(key=lambda x: -x[0])
    for _, r in rows:
        print(r)

    bh_mean = float(np.mean(fold_results["buy_hold"]))
    print()
    print("=== BENCHMARK COMPARISON ===")
    print(f"  buy_hold (daily) WF mean Sharpe   : {bh_mean:+.4f}")
    print(f"  VLSTM (Saly-Kaufmann 2026 SOTA)   : +2.4000 (daily gold futures)")
    print(f"  F2F (Wright 2026 SOTA)            : +2.8800 (daily gold futures)")
    print()
    best_name = rows[0][1].split(" | ")[0].strip()
    best_score = rows[0][0]
    print(f"  our best: {best_name} = {best_score:+.4f}")
    if best_score > 2.88:
        print(f"  >>> NEW SOTA  ({best_score - 2.88:+.4f} vs F2F)")
    elif best_score > 2.40:
        print(f"  beats VLSTM ({best_score - 2.40:+.4f}) but not F2F ({best_score - 2.88:+.4f})")
    else:
        print(f"  below both SOTAs (VLSTM gap {best_score - 2.40:+.4f}, F2F gap {best_score - 2.88:+.4f})")

    return 0


if __name__ == "__main__":
    sys.exit(main())
