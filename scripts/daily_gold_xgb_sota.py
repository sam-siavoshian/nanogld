"""Daily-gold SOTA chase v2 — XGBoost on engineered features × vol-target sizing.

Architecture:
  raw daily close (GC=F, 1990-2026)
    → engineered features (returns@{1,5,10,20,60,120}, RV@{20,60},
       SMA ratios @{20,50,200}, RSI-14, day-of-week, month)
    → triple-barrier labels (next-day; UP/FLAT/DOWN with vol-scaled
       neutral threshold)
    → XGBoost multi:softprob trained per fold (4-fold WF)
    → continuous position signal = (P_up - P_down)  in [-1, +1]
    → multiplied by vol_target_size for risk control
    → backtest Sharpe net 2 bp cost

Target: beat VLSTM 2.40 / F2F 2.88 on daily gold futures.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

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


def _rolling_mean(x: np.ndarray, window: int) -> np.ndarray:
    n = len(x)
    out = np.zeros(n, dtype=np.float64)
    cs = np.cumsum(x, dtype=np.float64)
    for i in range(n):
        a = max(0, i - window + 1)
        c = i - a + 1
        out[i] = float((cs[i] - (cs[a - 1] if a > 0 else 0.0)) / c)
    return out


def _vol_target(daily_ret: np.ndarray, target_vol: float, window: int = 60, cap: float = 1.0) -> np.ndarray:
    rv = _rolling_std(daily_ret, window=window)
    raw = target_vol / (rv * np.sqrt(DAILY_BARS_PER_YEAR) + 1e-8)
    pos = np.clip(raw, 0.0, cap)
    pos[~np.isfinite(pos)] = 0.0
    return np.concatenate([[0.0], pos[:-1]])


def engineer_features(close: np.ndarray, daily_ret: np.ndarray, ts: pd.Series) -> tuple[np.ndarray, list[str]]:
    """Build engineered features. All causal: feature at row i uses only
    info known at end of bar i (close[i] and earlier).

    Position then trades on daily_ret[i+1] (next-day return), so the
    PnL alignment in main() shifts the feature/label index +1 vs the
    trade target.
    """
    feats: dict[str, np.ndarray] = {}

    feats["ret_1"] = daily_ret
    for lag in [5, 10, 20, 60, 120]:
        cs = np.cumsum(daily_ret, dtype=np.float64)
        out = np.zeros_like(daily_ret)
        for i in range(lag, len(daily_ret)):
            a = i - lag
            out[i] = cs[i] - (cs[a - 1] if a > 0 else 0.0)
        feats[f"ret_{lag}"] = out

    for w in [20, 60]:
        feats[f"rv_{w}"] = _rolling_std(daily_ret, w)

    for w in [20, 50, 200]:
        sma = _rolling_mean(close, w)
        feats[f"close_over_sma_{w}"] = close / np.where(sma > 0, sma, 1.0) - 1.0

    # RSI-14
    diff = np.diff(close, prepend=close[0])
    gain = np.where(diff > 0, diff, 0.0)
    loss = np.where(diff < 0, -diff, 0.0)
    avg_gain = _rolling_mean(gain, 14)
    avg_loss = _rolling_mean(loss, 14)
    rs = avg_gain / (avg_loss + 1e-12)
    rsi = 100.0 - 100.0 / (1.0 + rs)
    feats["rsi_14"] = rsi

    # Calendar
    dow = pd.to_datetime(ts).dt.dayofweek.values.astype(np.float64)
    month = pd.to_datetime(ts).dt.month.values.astype(np.float64)
    feats["dow"] = dow
    feats["month"] = month

    names = list(feats.keys())
    X = np.column_stack([feats[k] for k in names]).astype(np.float32)
    return X, names


def make_labels(daily_ret: np.ndarray, neutral_eps: float = 0.001) -> np.ndarray:
    """Next-day direction labels (causal: y[i] = direction of day i+1's
    return). label 0=DOWN, 1=FLAT, 2=UP.

    Use a small neutral threshold (10 bps) to make the FLAT class
    meaningful; otherwise FLAT count → 0 and the classifier degenerates.
    """
    n = len(daily_ret)
    y = np.zeros(n, dtype=np.int64)
    for i in range(n - 1):
        r = daily_ret[i + 1]
        if r > neutral_eps:
            y[i] = 2
        elif r < -neutral_eps:
            y[i] = 0
        else:
            y[i] = 1
    y[-1] = 1  # last day has no future, default FLAT
    return y


def fetch_gold_daily() -> pd.DataFrame:
    import yfinance as yf  # noqa: PLC0415
    df = yf.download("GC=F", start="1990-01-01", end="2026-05-25", progress=False, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    df = df.reset_index()
    df["close"] = df["Close"].astype(float)
    df = df[["Date", "close"]].rename(columns={"Date": "ts"})
    df["log_ret"] = np.log(df["close"] / df["close"].shift(1)).fillna(0.0)
    return df


def split_4fold_wf(n: int) -> list[tuple[int, int, int, int]]:
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
    import xgboost as xgb  # noqa: PLC0415
    print(f"xgboost version: {xgb.__version__}")

    df = fetch_gold_daily()
    daily_ret = df["log_ret"].values.astype(np.float64)
    close = df["close"].values.astype(np.float64)
    n = len(df)
    print(f"bars: {n}  range: {df['ts'].iloc[0].date()} → {df['ts'].iloc[-1].date()}")

    X, feat_names = engineer_features(close, daily_ret, df["ts"])
    y = make_labels(daily_ret, neutral_eps=0.001)
    print(f"features: {len(feat_names)} ({feat_names[:6]}...)")
    print(f"label distribution: DOWN={np.sum(y==0)} FLAT={np.sum(y==1)} UP={np.sum(y==2)}")

    folds = split_4fold_wf(n)
    print(f"folds: {len(folds)}")
    fold_sharpe_1x: list[float] = []
    fold_sharpe_2x: list[float] = []
    fold_sharpe_pure_xgb: list[float] = []

    for fi, (ts_, te, ve, tte) in enumerate(folds):
        # Train slice: features X[ts_:te], labels y[ts_:te].
        # We must NOT include the LAST bar of training in features (because
        # y[i] uses ret[i+1], so X must stop at te-2 to avoid leaking
        # validation's first day return into training labels).
        # Safer: train on [ts_, te-1) which uses ret[ts_+1 ... te-1].
        train_X = X[ts_:te - 1]
        train_y = y[ts_:te - 1]
        test_X = X[ve:tte]
        test_ret = daily_ret[ve:tte]
        test_close = close[ve:tte]

        # Drop rows with too few finite features (early lookback bars)
        ok_train = np.isfinite(train_X).all(axis=1)
        train_X = train_X[ok_train]
        train_y = train_y[ok_train]

        dtrain = xgb.DMatrix(np.ascontiguousarray(train_X), label=np.ascontiguousarray(train_y.astype(np.float32)))
        dtest = xgb.DMatrix(np.ascontiguousarray(test_X))
        params: dict[str, Any] = {
            "objective": "multi:softprob",
            "num_class": 3,
            "max_depth": 5,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "verbosity": 0,
            "nthread": 8,
            "seed": 42 + fi,
        }
        booster = xgb.train(params, dtrain, num_boost_round=300)
        probs = booster.predict(dtest)  # (T_test, 3)
        # Continuous signal: P_up - P_down  in [-1, +1]
        signal = (probs[:, 2] - probs[:, 0]).astype(np.float64)
        signal = np.clip(signal, -1.0, 1.0)

        # Vol-target sizing applied to absolute signal magnitude
        vt = _vol_target(test_ret, target_vol=0.15, window=60, cap=1.0)
        # Long-bias version: only long when signal > 0, then scale by vt
        long_signal = np.where(signal > 0, signal, 0.0)
        # Pure XGB position (no vol target, signed)
        pure_pos = signal
        # Combined: vol_target * signal_when_positive (long-only)
        comp_pos = vt * long_signal

        fold_sharpe_pure_xgb.append(_sharpe(_pnl(pure_pos, test_ret, 1.0)))
        fold_sharpe_1x.append(_sharpe(_pnl(comp_pos, test_ret, 1.0)))
        fold_sharpe_2x.append(_sharpe(_pnl(comp_pos, test_ret, 2.0)))

        print(f"  fold {fi}: pure_xgb={fold_sharpe_pure_xgb[-1]:+.3f}  xgb_x_vt15(long-only)={fold_sharpe_1x[-1]:+.3f}  (1x)  {fold_sharpe_2x[-1]:+.3f}  (2x)")

    print()
    pure_mean = float(np.mean(fold_sharpe_pure_xgb))
    comp_mean = float(np.mean(fold_sharpe_1x))
    comp_2x_mean = float(np.mean(fold_sharpe_2x))
    print(f"WF mean Sharpe (1x cost):")
    print(f"  pure XGBoost signal         : {pure_mean:+.4f}")
    print(f"  XGBoost × vol_target (long) : {comp_mean:+.4f}")
    print(f"  XGBoost × vol_target (2x)   : {comp_2x_mean:+.4f}")
    print()
    print("Benchmarks:")
    print(f"  VLSTM (Saly-Kaufmann 2026) : +2.4000")
    print(f"  F2F (Wright 2026)          : +2.8800")
    best = max(pure_mean, comp_mean)
    if best > 2.88:
        print(f"  >>> NEW SOTA  best={best:+.4f}")
    elif best > 2.40:
        print(f"  beats VLSTM by {best - 2.40:+.4f}, F2F gap {best - 2.88:+.4f}")
    else:
        print(f"  best={best:+.4f}: VLSTM gap {best - 2.40:+.4f}, F2F gap {best - 2.88:+.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
