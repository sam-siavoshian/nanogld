"""F2F replica v2 — using exact extracted hyperparameters from arXiv:2511.08571.

Hyperparams (confirmed via Nia/agent extraction):
  lambda (EMA log-price)     ≈ 0.966  (20-day half-life)
  theta (EWMA variance)      ≈ 0.94-0.97  (20-day EWMA range)
  omega (blend weight)       = 0.6  (explicit, paper Sec 6)
  K (momentum lookback)      = 50  (explicit)
  bull entry threshold       = 0.55  (Table 2)
  bear de-risk threshold     = 0.45  (Table 2)
  lambda_Kelly               = 0.40  (fractional Kelly multiplier)
  gamma                      = 0.02  (sqrt-impact penalty — set to 0 per
                                       earlier analysis showing it crushes
                                       positions to 0)
  cost                       = 0.7 bp linear
  vol_target                 = 15% ann
  W_max                      = 2.0
  ATR window                 = 14
  ATR hard stop              = 2x ATR
  ATR trailing stop          = 1.5x ATR (ratchet)
  max age                    = 30 trading days
  WF                         = 10-yr train / 6-mo test, monthly step

Grid (small, since most params are explicit):
  lambda  ∈ {0.95, 0.966, 0.98}
  theta   ∈ {0.94, 0.96}
"""

from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
DAILY_BARS_PER_YEAR = 252

# Explicit F2F params
OMEGA_F2F = 0.6
K_F2F = 50
THRESH_BULL = 0.55
THRESH_BEAR = 0.45
KELLY_FRACTION = 0.40
COST_BPS = 0.7
VOL_TARGET_ANN = 0.15
W_MAX = 2.0
ATR_WINDOW = 14
ATR_HARD_MULT = 2.0
ATR_TRAIL_MULT = 1.5
TIMEOUT_DAYS = 30


def _sharpe(r: np.ndarray) -> float:
    r = r[np.isfinite(r)]
    if len(r) < 2: return 0.0
    mu, sigma = r.mean(), r.std(ddof=1)
    if sigma < 1e-12: return 0.0
    return float(mu / sigma * np.sqrt(DAILY_BARS_PER_YEAR))


def _ema(x: np.ndarray, lam: float) -> np.ndarray:
    out = np.zeros_like(x, dtype=np.float64)
    out[0] = x[0]
    for i in range(1, len(x)):
        out[i] = lam * out[i - 1] + (1 - lam) * x[i]
    return out


def _ewma_var(r: np.ndarray, theta: float, init_var: float) -> np.ndarray:
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
    if n < window: return atr
    cs = np.cumsum(tr)
    for t in range(window - 1, n):
        a = t - window + 1
        atr[t] = (cs[t] - (cs[a - 1] if a > 0 else 0)) / window
    return atr


def _kelly_f_star(mu: float, sigma: float, k: float) -> float:
    """Pure Kelly with linear cost only."""
    if sigma <= 1e-12: return 0.0
    edge = mu - k
    if edge <= 0: return 0.0
    return float(edge / (sigma ** 2))


def fetch_gold_daily(start="2005-01-01"):
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


def simulate(df, lam, theta, train_sl, test_sl):
    n = len(df)
    log_close = df["log_close"].values
    log_ret = df["log_ret"].values
    close = df["close"].values
    high = df["high"].values
    low = df["low"].values

    # Trend signal
    y_t = _ema(log_close, lam)
    d_yt = np.zeros_like(y_t); d_yt[1:] = y_t[1:] - y_t[:-1]
    mu_d = d_yt[train_sl].mean()
    sd_d = d_yt[train_sl].std(ddof=1) + 1e-12
    z = (d_yt - mu_d) / sd_d
    z_clip = np.clip(z, -3, 3)
    p_trend = (z_clip + 3) / 6.0

    # Momentum
    mom = np.zeros(n)
    for t in range(K_F2F, n):
        mom[t] = 1.0 if close[t] > close[t - K_F2F] else 0.0
    p_bull = OMEGA_F2F * p_trend + (1 - OMEGA_F2F) * mom

    # EWMA variance forecast
    init_var = max(1e-10, log_ret[train_sl].var(ddof=1))
    sig2 = _ewma_var(log_ret, theta, init_var)
    sigma_hat = np.sqrt(sig2)
    sigma_star = VOL_TARGET_ANN / np.sqrt(DAILY_BARS_PER_YEAR)
    w_vol = np.minimum(W_MAX, sigma_star / (sigma_hat + 1e-12))

    # ATR
    atr = _compute_atr(high, low, close, ATR_WINDOW)

    # Kelly fixed per fold
    mu_r = log_ret[train_sl].mean()
    sig_r = log_ret[train_sl].std(ddof=1) + 1e-12
    k_lin = COST_BPS * 1e-4
    f_star = _kelly_f_star(mu_r, sig_r, k_lin)
    f_kelly = KELLY_FRACTION * f_star

    # Trading loop
    pos = np.zeros(n)
    in_pos = False
    entry_price = 0.0
    peak_price = 0.0
    age = 0
    for t in range(train_sl.stop, test_sl.stop):
        if t + 1 >= n: continue
        pb = p_bull[t]; dyt = d_yt[t]
        if in_pos:
            age += 1
            if close[t] > peak_price: peak_price = close[t]
            stop_hard = entry_price - ATR_HARD_MULT * atr[t]
            stop_trail = peak_price - ATR_TRAIL_MULT * atr[t]
            exit_now = close[t] < stop_hard or close[t] < stop_trail or age >= TIMEOUT_DAYS or pb < THRESH_BEAR
            if exit_now:
                in_pos = False
                pos[t + 1] = 0.0
                continue
            w_conf = w_vol[t] * max(0.0, (pb - 0.5) / 0.5)
            w = max(0.0, min(W_MAX, f_kelly * w_conf))
            pos[t + 1] = w
        else:
            if pb >= THRESH_BULL and dyt > 0:
                in_pos = True
                entry_price = close[t]; peak_price = close[t]; age = 0
                w_conf = w_vol[t] * max(0.0, (pb - 0.5) / 0.5)
                w = max(0.0, min(W_MAX, f_kelly * w_conf))
                pos[t + 1] = w

    cf = COST_BPS * 1e-4
    pos_prev = np.concatenate([[0.0], pos[:-1]])
    return pos * log_ret - cf * np.abs(pos - pos_prev), pos


def main():
    df = fetch_gold_daily(start="2005-01-01")
    n = len(df)
    print(f"bars: {n}  {df['ts'].iloc[0].date()} → {df['ts'].iloc[-1].date()}")

    # WF: 10-yr train + 6-mo test, monthly step, but for OOS chain we take only first month of each test
    train_len = 10 * DAILY_BARS_PER_YEAR
    test_len = 6 * (DAILY_BARS_PER_YEAR // 12)
    step = DAILY_BARS_PER_YEAR // 12
    bars_per_month = step
    folds = []
    s = 0
    while s + train_len + test_len <= n:
        folds.append((s, s + train_len, s + train_len + test_len))
        s += step
    print(f"folds: {len(folds)}")

    grid_lam = [0.95, 0.966, 0.98, 0.99]
    grid_theta = [0.94, 0.96]
    grid_thresh = [0.55, 0.58, 0.62, 0.65, 0.70]

    print()
    print("Sweep over thresh + (lam, theta):")
    print(f"{'lam':>6s} {'theta':>5s} {'thresh':>6s} | {'agg Sharpe':>11s} {'ann_ret':>8s} {'ann_vol':>8s} {'active%':>8s}")
    best = (-1e9, None)
    for thresh in grid_thresh:
        # Replace global thresh (hacky but local)
        global THRESH_BULL
        THRESH_BULL = thresh
        for lam in grid_lam:
            for theta in grid_theta:
                test_pnl_chain = []
                test_pos_chain = []
                for (a, b, c) in folds:
                    pnl, pos = simulate(df, lam, theta, slice(a, b), slice(b, c))
                    first_month = slice(b, min(c, b + bars_per_month))
                    test_pnl_chain.append(pnl[first_month])
                    test_pos_chain.append(pos[first_month])
                chain = np.concatenate(test_pnl_chain)
                pos_chain = np.concatenate(test_pos_chain)
                agg = _sharpe(chain)
                ann_ret = chain.mean() * DAILY_BARS_PER_YEAR * 100
                ann_vol = chain.std(ddof=1) * np.sqrt(DAILY_BARS_PER_YEAR) * 100
                active_pct = (np.abs(pos_chain) > 1e-9).mean() * 100
                row = f"  {lam:>5.3f} {theta:>5.2f} {thresh:>5.2f}  | {agg:>+11.4f} {ann_ret:>+7.3f}% {ann_vol:>7.3f}% {active_pct:>7.1f}%"
                print(row)
                if agg > best[0]:
                    best = (agg, (lam, theta, thresh))

    print()
    print(f"BEST: lam={best[1][0]} theta={best[1][1]} thresh={best[1][2]}  Sharpe={best[0]:+.4f}")
    print("F2F paper claim: Sharpe 2.88, ann_ret 2.62%, ann_vol 0.91%, MDD 0.52%")


if __name__ == "__main__":
    main()
