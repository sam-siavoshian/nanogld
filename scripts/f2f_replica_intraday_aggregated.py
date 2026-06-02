"""F2F replica v3 — aggregate our PIT-validated intraday GLD bars to daily
and run F2F's recipe on clean data (no front-month roll noise like yfinance GC=F).

Hypothesis: the ~2.2 Sharpe gap between our v2 F2F replica (+0.66) and the
paper's +2.88 may be partly explained by data quality. yfinance GC=F is a
continuous front-month series with roll noise. Our intraday GLD ETF bars
(2014-2026, validated PIT-clean by V1 data pipeline) aggregated to daily
should have cleaner OHLCV — GLD as an ETF rolls itself via NAV and has
no contract roll noise.

If our clean-data replica hits significantly higher Sharpe than yfinance
replica, this confirms the data-quality hypothesis and shows we can close
part of the F2F gap. Combine with our intraday vol_target as ensemble.

Architecture:
  Step 1: aggregate 30-min GLD bars to daily OHLC
  Step 2: run F2F-extracted-params recipe on this daily series
  Step 3: compare to yfinance-GC=F replica
  Step 4: ensemble with intraday vol_target_3pct as separate strategy
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

DAILY_BARS_PER_YEAR = 252
INTRADAY_BARS_PER_YEAR = 3276
COST_BPS = 0.7
GLD_CLOSE_IDX = 3
GLD_OPEN_IDX = 0
GLD_HIGH_IDX = 1
GLD_LOW_IDX = 2

OMEGA_F2F = 0.6
K_F2F = 50
THRESH_BULL = 0.55
THRESH_BEAR = 0.45
KELLY_FRACTION = 0.40
VOL_TARGET_ANN = 0.15
W_MAX = 2.0
ATR_WINDOW = 14
ATR_HARD_MULT = 2.0
ATR_TRAIL_MULT = 1.5
TIMEOUT_DAYS = 30


def _arr(x):
    return x.numpy() if hasattr(x, "numpy") else np.asarray(x)


def _sharpe(r, ann):
    r = r[np.isfinite(r)]
    if len(r) < 2:
        return 0.0
    mu, sigma = r.mean(), r.std(ddof=1)
    if sigma < 1e-12:
        return 0.0
    return float(mu / sigma * np.sqrt(ann))


def _ema(x, lam):
    out = np.zeros_like(x, dtype=np.float64)
    out[0] = x[0]
    for i in range(1, len(x)):
        out[i] = lam * out[i - 1] + (1 - lam) * x[i]
    return out


def _ewma_var(r, theta, init_var):
    n = len(r)
    sig2 = np.zeros(n, dtype=np.float64)
    sig2[0] = init_var
    for t in range(1, n):
        sig2[t] = theta * sig2[t - 1] + (1 - theta) * r[t - 1] ** 2
    return sig2


def _compute_atr(high, low, close, window=14):
    n = len(close)
    tr = np.zeros(n)
    for t in range(1, n):
        tr[t] = max(high[t] - low[t], abs(high[t] - close[t - 1]), abs(low[t] - close[t - 1]))
    atr = np.zeros(n)
    if n < window:
        return atr
    cs = np.cumsum(tr)
    for t in range(window - 1, n):
        a = t - window + 1
        atr[t] = (cs[t] - (cs[a - 1] if a > 0 else 0)) / window
    return atr


def _kelly_f_star(mu, sigma, k):
    if sigma <= 1e-12:
        return 0.0
    edge = mu - k
    if edge <= 0:
        return 0.0
    return float(edge / (sigma ** 2))


def aggregate_intraday_to_daily(unified):
    """Aggregate 30-min GLD bars to daily OHLC + log return.

    Uses bar_close_utc_ns to bucket into UTC days. open = first bar's open,
    high = max over day, low = min over day, close = last bar's close.
    """
    features = _arr(unified["features"])
    bcn = _arr(unified["bar_close_utc_ns"]).astype(np.int64)
    ts = pd.to_datetime(bcn, unit="ns", utc=True)

    o = features[:, GLD_OPEN_IDX].astype(np.float64)
    h = features[:, GLD_HIGH_IDX].astype(np.float64)
    low = features[:, GLD_LOW_IDX].astype(np.float64)
    c = features[:, GLD_CLOSE_IDX].astype(np.float64)
    # Drop bars where any OHLC is non-finite OR zero (RevIN-normalized features may
    # actually be z-scores, not raw prices — handle either case downstream).
    df = pd.DataFrame({"ts": ts, "open": o, "high": h, "low": low, "close": c})
    df["day"] = df["ts"].dt.date
    # daily OHLC: first/max/min/last
    daily = df.groupby("day").agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        n_bars=("close", "count"),
    ).reset_index()
    daily = daily.dropna(subset=["open", "high", "low", "close"]).reset_index(drop=True)
    daily["log_close"] = np.log(np.maximum(daily["close"].values.astype(np.float64), 1e-12))
    daily["log_ret"] = daily["log_close"].diff().fillna(0.0)
    return daily


def simulate_f2f(df, lam, theta, train_sl, test_sl):
    n = len(df)
    log_close = df["log_close"].values
    log_ret = df["log_ret"].values
    close = df["close"].values
    high = df["high"].values
    low = df["low"].values

    y_t = _ema(log_close, lam)
    d_yt = np.zeros_like(y_t)
    d_yt[1:] = y_t[1:] - y_t[:-1]
    mu_d = d_yt[train_sl].mean()
    sd_d = d_yt[train_sl].std(ddof=1) + 1e-12
    z = (d_yt - mu_d) / sd_d
    p_trend = (np.clip(z, -3, 3) + 3) / 6.0

    mom = np.zeros(n)
    for t in range(K_F2F, n):
        mom[t] = 1.0 if close[t] > close[t - K_F2F] else 0.0
    p_bull = OMEGA_F2F * p_trend + (1 - OMEGA_F2F) * mom

    init_var = max(1e-10, log_ret[train_sl].var(ddof=1))
    sig2 = _ewma_var(log_ret, theta, init_var)
    sigma_hat = np.sqrt(sig2)
    sigma_star = VOL_TARGET_ANN / np.sqrt(DAILY_BARS_PER_YEAR)
    w_vol = np.minimum(W_MAX, sigma_star / (sigma_hat + 1e-12))

    atr = _compute_atr(high, low, close, ATR_WINDOW)
    mu_r = log_ret[train_sl].mean()
    sig_r = log_ret[train_sl].std(ddof=1) + 1e-12
    k_lin = COST_BPS * 1e-4
    f_star = _kelly_f_star(mu_r, sig_r, k_lin)
    f_kelly = KELLY_FRACTION * f_star

    pos = np.zeros(n)
    in_pos = False
    entry_price = 0.0
    peak_price = 0.0
    age = 0
    for t in range(train_sl.stop, test_sl.stop):
        if t + 1 >= n:
            continue
        pb = p_bull[t]
        dyt = d_yt[t]
        if in_pos:
            age += 1
            if close[t] > peak_price:
                peak_price = close[t]
            stop_hard = entry_price - ATR_HARD_MULT * atr[t]
            stop_trail = peak_price - ATR_TRAIL_MULT * atr[t]
            if close[t] < stop_hard or close[t] < stop_trail or age >= TIMEOUT_DAYS or pb < THRESH_BEAR:
                in_pos = False
                pos[t + 1] = 0.0
                continue
            w_conf = w_vol[t] * max(0.0, (pb - 0.5) / 0.5)
            pos[t + 1] = max(0.0, min(W_MAX, f_kelly * w_conf))
        else:
            if pb >= THRESH_BULL and dyt > 0:
                in_pos = True
                entry_price = close[t]
                peak_price = close[t]
                age = 0
                w_conf = w_vol[t] * max(0.0, (pb - 0.5) / 0.5)
                pos[t + 1] = max(0.0, min(W_MAX, f_kelly * w_conf))

    cf = COST_BPS * 1e-4
    pos_prev = np.concatenate([[0.0], pos[:-1]])
    pnl = pos * log_ret - cf * np.abs(pos - pos_prev)
    return pnl, pos


def main():
    unified = torch.load(REPO_ROOT / "data" / "processed" / "training_v1_unified.pt", weights_only=False)
    daily = aggregate_intraday_to_daily(unified)
    n = len(daily)
    print(f"daily bars from intraday aggregation: {n}")
    print(f"date range: {daily['day'].iloc[0]} → {daily['day'].iloc[-1]}")
    print(f"close range: min={daily['close'].min():.4f}  max={daily['close'].max():.4f}  mean={daily['close'].mean():.4f}")
    print(f"first 5 closes: {daily['close'].head(5).tolist()}")
    print(f"last 5 closes:  {daily['close'].tail(5).tolist()}")
    print(f"log_ret stats: mean={daily['log_ret'].mean():.6f}  std={daily['log_ret'].std(ddof=1):.6f}")

    # WF: same as F2F (monthly step, 6-mo test). With ~2700 daily bars from
    # intraday, we get ~10 years coverage.
    # Use simpler: 3-yr train + 6-mo val + 6-mo test, 1-mo step.
    train_len = 3 * DAILY_BARS_PER_YEAR
    val_len = DAILY_BARS_PER_YEAR // 2
    test_len = DAILY_BARS_PER_YEAR // 2
    step = DAILY_BARS_PER_YEAR // 12
    bars_per_month = step
    folds = []
    s = 0
    while s + train_len + val_len + test_len <= n:
        folds.append((s, s + train_len, s + train_len + val_len, s + train_len + val_len + test_len))
        s += step
    print(f"folds: {len(folds)}")

    # F2F-extracted hyperparams + small grid
    print()
    print(f"{'lam':>6s} {'theta':>5s} | {'OOS Sharpe':>10s} {'ann_ret':>8s} {'ann_vol':>8s} {'active%':>8s}")
    for lam in [0.95, 0.966, 0.98, 0.99]:
        for theta in [0.94, 0.96]:
            test_pnl_chain = []
            test_pos_chain = []
            for (a, b, c, d) in folds:
                pnl, pos = simulate_f2f(daily, lam, theta, slice(a, b), slice(c, d))
                first_month = slice(c, min(d, c + bars_per_month))
                test_pnl_chain.append(pnl[first_month])
                test_pos_chain.append(pos[first_month])
            chain = np.concatenate(test_pnl_chain)
            pos_chain = np.concatenate(test_pos_chain)
            agg = _sharpe(chain, DAILY_BARS_PER_YEAR)
            ann_ret = chain.mean() * DAILY_BARS_PER_YEAR * 100
            ann_vol = chain.std(ddof=1) * np.sqrt(DAILY_BARS_PER_YEAR) * 100
            active = (np.abs(pos_chain) > 1e-9).mean() * 100
            print(f"  {lam:>5.3f} {theta:>5.2f}  | {agg:>+10.4f} {ann_ret:>+7.3f}% {ann_vol:>7.3f}% {active:>7.1f}%")

    print()
    print(f"F2F paper: Sharpe +2.88, ann_ret +2.62%, ann_vol 0.91%, active% ~40.5%")
    print(f"Our prior yfinance GC=F replica (clean params): +0.66")


if __name__ == "__main__":
    main()
