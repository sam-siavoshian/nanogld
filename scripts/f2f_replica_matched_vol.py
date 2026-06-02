"""F2F replica at MATCHED 0.91% realized vol (same as F2F paper).

Prior replica had 6.97% ann_vol vs F2F's 0.91% (7.6× higher), making
the artifacts non-comparable. Fix: scale positions down so realized vol
matches F2F's 0.91%. Re-measure Sharpe at matched vol.

Math: Sharpe is scale-invariant for purely linear strategies, BUT cost
drag scales differently. At smaller positions, the same per-bar cost
eats a larger fraction of return. So matched-vol Sharpe = original
Sharpe minus a cost adjustment. Expected: matched-vol Sharpe slightly
lower than +0.66 (our replica) at 0.91% vol due to cost amplification.
"""

from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
DAILY_BPY = 252

OMEGA_F2F = 0.6
K_F2F = 50
THRESH_BULL = 0.55
THRESH_BEAR = 0.45
KELLY_FRACTION = 0.40
COST_BPS = 0.7
W_MAX_DEFAULT = 2.0
ATR_WINDOW = 14
ATR_HARD_MULT = 2.0
ATR_TRAIL_MULT = 1.5
TIMEOUT_DAYS = 30


def _sharpe(r):
    r = r[np.isfinite(r)]
    if len(r) < 2:
        return 0.0
    mu, sigma = r.mean(), r.std(ddof=1)
    if sigma < 1e-12:
        return 0.0
    return float(mu / sigma * np.sqrt(DAILY_BPY))


def _ann_vol(r):
    return float(r.std(ddof=1) * np.sqrt(DAILY_BPY))


def _ann_ret(r):
    return float(r.mean() * DAILY_BPY)


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


def fetch_gold_daily(start="2000-01-01"):
    import yfinance as yf
    df = yf.download("GC=F", start=start, end="2026-05-25", progress=False, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    df = df.reset_index()
    for col in ("Open", "High", "Low", "Close"):
        df[col.lower()] = df[col].astype(float)
    df = df.rename(columns={"Date": "ts"})
    df = df[["ts", "open", "high", "low", "close"]].dropna().reset_index(drop=True)
    df["log_close"] = np.log(df["close"])
    df["log_ret"] = df["log_close"].diff().fillna(0.0)
    return df


def simulate_f2f(df, lam, theta, target_vol_ann, w_max, train_sl, test_sl):
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
    sigma_star = target_vol_ann / np.sqrt(DAILY_BPY)
    w_vol = np.minimum(w_max, sigma_star / (sigma_hat + 1e-12))

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
            pos[t + 1] = max(0.0, min(w_max, f_kelly * w_conf))
        else:
            if pb >= THRESH_BULL and dyt > 0:
                in_pos = True
                entry_price = close[t]
                peak_price = close[t]
                age = 0
                w_conf = w_vol[t] * max(0.0, (pb - 0.5) / 0.5)
                pos[t + 1] = max(0.0, min(w_max, f_kelly * w_conf))

    cf = COST_BPS * 1e-4
    pos_prev = np.concatenate([[0.0], pos[:-1]])
    pnl = pos * log_ret - cf * np.abs(pos - pos_prev)
    return pnl, pos


def main():
    df = fetch_gold_daily(start="2005-01-01")
    n = len(df)
    print(f"bars: {n}  {df['ts'].iloc[0].date()} → {df['ts'].iloc[-1].date()}")

    train_len = 10 * DAILY_BPY
    test_len = 6 * (DAILY_BPY // 12)
    step = DAILY_BPY // 12
    bpm = step
    folds = []
    s = 0
    while s + train_len + test_len <= n:
        folds.append((s, s + train_len, s + train_len + test_len))
        s += step
    print(f"folds: {len(folds)}")

    # Try different target_vol caps to find one that matches F2F's 0.91% realized vol
    print()
    print(f"{'tv_target':>10s} {'w_max':>6s} | {'chain_SR':>9s} {'ann_ret':>8s} {'ann_vol':>8s} {'active%':>8s} {'vs F2F vol':>11s}")
    for target_vol_ann in [0.01, 0.02, 0.03, 0.05, 0.10, 0.15]:
        for w_max in [0.10, 0.30, 0.50, 1.0, 2.0]:
            test_pnl_chain = []
            test_pos_chain = []
            for (a, b, c) in folds:
                pnl, pos = simulate_f2f(df, 0.99, 0.94, target_vol_ann, w_max, slice(a, b), slice(b, c))
                first_month = slice(b, min(c, b + bpm))
                test_pnl_chain.append(pnl[first_month])
                test_pos_chain.append(pos[first_month])
            chain = np.concatenate(test_pnl_chain)
            pos_chain = np.concatenate(test_pos_chain)
            sr = _sharpe(chain)
            ar = _ann_ret(chain) * 100
            av = _ann_vol(chain) * 100
            active = (np.abs(pos_chain) > 1e-9).mean() * 100
            ratio_vs_f2f = av / 0.91
            print(f"  {target_vol_ann:>8.2%} {w_max:>5.2f}  | {sr:>+9.4f} {ar:>+7.3f}% {av:>7.3f}% {active:>7.1f}% {ratio_vs_f2f:>10.2f}x")

    print()
    print(f"F2F paper: SR 2.88, ann_ret 2.62%, ann_vol 0.91%, active 40.5%")
    print("Looking for config with ann_vol ≈ 0.91% (vs F2F 1.00x)")


if __name__ == "__main__":
    main()
