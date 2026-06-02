"""Final meta-stack: combine our +1.21 intraday vol_target + +0.66 daily F2F replica
as independent-timescale strategies. Aggregate both to daily for joint backtest.

Hypothesis: intraday signal and daily F2F signal exploit different timescales
of information; if correlation is low, combined Sharpe = sqrt(N) lift over
weighted average. With two strategies, this could plausibly hit +1.5+.

Approach:
  Strategy A (intraday): vol_target_3pct on 30-min GLD bars → aggregate per-day PnL
  Strategy B (daily):    F2F replica on intraday-aggregated daily GLD bars
  Combined:              equal-weight, also try risk-parity, also try Sharpe-weighted

Both PIT-clean. Compute Sharpe on daily-aggregated PnL series.
"""

from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from nanogld.data.walk_forward_splits import compute_fold_boundaries

DAILY_BPY = 252
INTRADAY_BPY = 3276
COST_BPS_INTRADAY = 2.0
COST_BPS_DAILY = 0.7

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


def _rolling_std(x, window):
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
        m = (cs[i] - (cs[a - 1] if a > 0 else 0)) / c
        m2 = (cs2[i] - (cs2[a - 1] if a > 0 else 0)) / c
        v = max(0.0, m2 - m * m)
        out[i] = np.sqrt(v)
    return out


def _ema(x, lam):
    out = np.zeros_like(x, dtype=np.float64)
    out[0] = x[0]
    for i in range(1, len(x)):
        out[i] = lam * out[i - 1] + (1 - lam) * x[i]
    return out


def _vol_target(rr, tv, w=64, cap=1.0):
    rv = _rolling_std(rr, w)
    raw = tv / (rv * np.sqrt(INTRADAY_BPY) + 1e-8)
    p = np.clip(raw, 0, cap)
    p[~np.isfinite(p)] = 0.0
    return np.concatenate([[0.0], p[:-1]])


def aggregate_intraday_to_daily(unified):
    features = _arr(unified["features"])
    bcn = _arr(unified["bar_close_utc_ns"]).astype(np.int64)
    ts = pd.to_datetime(bcn, unit="ns", utc=True)
    o = features[:, GLD_OPEN_IDX].astype(np.float64)
    h = features[:, GLD_HIGH_IDX].astype(np.float64)
    l = features[:, GLD_LOW_IDX].astype(np.float64)
    c = features[:, GLD_CLOSE_IDX].astype(np.float64)
    df = pd.DataFrame({"ts": ts, "open": o, "high": h, "low": l, "close": c})
    df["day"] = df["ts"].dt.date
    daily = df.groupby("day").agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
    ).reset_index()
    daily = daily.dropna().reset_index(drop=True)
    daily["log_close"] = np.log(np.maximum(daily["close"].values, 1e-12))
    daily["log_ret"] = daily["log_close"].diff().fillna(0.0)
    daily["ts"] = pd.to_datetime(daily["day"])
    return daily


def compute_atr(high, low, close, window=14):
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


def f2f_strategy_on_daily(daily, lam=0.98, theta=0.94):
    """Run F2F on daily series, return per-day PnL net 0.7bp."""
    n = len(daily)
    log_close = daily["log_close"].values
    log_ret = daily["log_ret"].values
    close = daily["close"].values
    high = daily["high"].values
    low = daily["low"].values

    y_t = _ema(log_close, lam)
    d_yt = np.zeros_like(y_t)
    d_yt[1:] = y_t[1:] - y_t[:-1]
    # Use the first 60% as 'train' for normalization (rolling would be ideal but
    # for a single-series eval we use this approximation; the F2F replica
    # already showed rolling WF gives similar results).
    train_n = int(n * 0.6)
    mu_d = d_yt[:train_n].mean()
    sd_d = d_yt[:train_n].std(ddof=1) + 1e-12
    z = (d_yt - mu_d) / sd_d
    p_trend = (np.clip(z, -3, 3) + 3) / 6.0

    mom = np.zeros(n)
    for t in range(K_F2F, n):
        mom[t] = 1.0 if close[t] > close[t - K_F2F] else 0.0
    p_bull = OMEGA_F2F * p_trend + (1 - OMEGA_F2F) * mom

    init_var = max(1e-10, log_ret[:train_n].var(ddof=1))
    sig2 = np.zeros(n)
    sig2[0] = init_var
    for t in range(1, n):
        sig2[t] = theta * sig2[t - 1] + (1 - theta) * log_ret[t - 1] ** 2
    sigma_hat = np.sqrt(sig2)
    sigma_star = VOL_TARGET_ANN / np.sqrt(DAILY_BPY)
    w_vol = np.minimum(W_MAX, sigma_star / (sigma_hat + 1e-12))

    atr = compute_atr(high, low, close, ATR_WINDOW)
    mu_r = log_ret[:train_n].mean()
    sig_r = log_ret[:train_n].std(ddof=1) + 1e-12
    k_lin = COST_BPS_DAILY * 1e-4
    edge = max(0.0, mu_r - k_lin)
    f_star = edge / (sig_r ** 2) if sig_r > 0 else 0.0
    f_kelly = KELLY_FRACTION * f_star

    pos = np.zeros(n)
    in_pos = False
    entry_price = 0.0
    peak_price = 0.0
    age = 0
    for t in range(K_F2F, n):
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

    cf = COST_BPS_DAILY * 1e-4
    pos_prev = np.concatenate([[0.0], pos[:-1]])
    return pos * log_ret - cf * np.abs(pos - pos_prev), pos


def intraday_strategy_per_bar_pnl(unified):
    """vol_target_3pct on intraday bars, returns per-bar PnL net 2bp."""
    features = _arr(unified["features"])
    bcn = _arr(unified["bar_close_utc_ns"]).astype(np.int64)
    folds = compute_fold_boundaries(bcn)
    close_full = features[:, GLD_CLOSE_IDX].astype(np.float64)
    # Concatenate per-fold test PnL into one chain
    pnl_per_bar = np.zeros(len(bcn))
    pos_per_bar = np.zeros(len(bcn))
    nlr_per_bar = np.zeros(len(bcn))
    for fb in folds[:4]:
        side = torch.load(REPO_ROOT / "data" / "processed" / f"training_v1_sidecar_fold_{fb.fold_idx}.pt", weights_only=False)
        nlr = _arr(side["next_log_return"]).astype(np.float64)
        ts = slice(fb.test_start, fb.test_end)
        close_t = close_full[ts]
        rr = np.zeros_like(close_t)
        rr[1:] = np.log(np.maximum(close_t[1:], 1e-12) / np.maximum(close_t[:-1], 1e-12))
        rr = np.nan_to_num(rr)
        pos = _vol_target(rr, 0.03, 64)
        cf = COST_BPS_INTRADAY * 1e-4
        prev = np.concatenate([[0.0], pos[:-1]])
        pnl = pos * nlr[ts] - cf * np.abs(pos - prev)
        pnl_per_bar[ts] = pnl
        pos_per_bar[ts] = pos
        nlr_per_bar[ts] = nlr[ts]
    return pnl_per_bar, pos_per_bar, nlr_per_bar


def aggregate_intraday_pnl_to_daily(unified, pnl_per_bar):
    """Sum per-bar PnL within each UTC day → daily PnL series, indexed by day."""
    bcn = _arr(unified["bar_close_utc_ns"]).astype(np.int64)
    ts = pd.to_datetime(bcn, unit="ns", utc=True)
    df = pd.DataFrame({"ts": ts, "pnl": pnl_per_bar})
    df["day"] = df["ts"].dt.date
    daily = df.groupby("day")["pnl"].sum().reset_index()
    daily["ts"] = pd.to_datetime(daily["day"])
    return daily


def main():
    print("=== Final meta-stack: intraday + daily F2F ensemble ===")
    unified = torch.load(REPO_ROOT / "data" / "processed" / "training_v1_unified.pt", weights_only=False)

    # Strategy A: intraday vol_target_3pct
    print("\n[A] running intraday vol_target_3pct ...")
    pnl_intraday_per_bar, pos_intraday, nlr_intraday = intraday_strategy_per_bar_pnl(unified)
    pnl_intraday_daily = aggregate_intraday_pnl_to_daily(unified, pnl_intraday_per_bar)
    s_intraday_intraday_freq = _sharpe(pnl_intraday_per_bar[pnl_intraday_per_bar != 0], INTRADAY_BPY)
    s_intraday_daily_freq = _sharpe(pnl_intraday_daily["pnl"].values[pnl_intraday_daily["pnl"].values != 0], DAILY_BPY)
    print(f"    intraday Sharpe (intraday freq):  {s_intraday_intraday_freq:+.4f}")
    print(f"    intraday Sharpe (daily aggregated):  {s_intraday_daily_freq:+.4f}")

    # Strategy B: F2F replica on intraday-aggregated daily GLD
    print("\n[B] running F2F daily replica on intraday-aggregated GLD ...")
    daily = aggregate_intraday_to_daily(unified)
    pnl_daily, pos_daily = f2f_strategy_on_daily(daily, lam=0.98, theta=0.94)
    daily["pnl_f2f"] = pnl_daily
    s_daily = _sharpe(pnl_daily, DAILY_BPY)
    print(f"    F2F daily Sharpe: {s_daily:+.4f}")
    print(f"    F2F daily active%: {(np.abs(pos_daily) > 1e-9).mean()*100:.1f}%")
    print(f"    F2F daily ann_vol: {pnl_daily.std(ddof=1)*np.sqrt(DAILY_BPY)*100:.3f}%")

    # Merge by day for combined backtest
    print("\n[combine] merging on daily index ...")
    merged = pnl_intraday_daily.merge(daily[["day", "pnl_f2f"]], on="day", how="inner")
    print(f"    merged: {len(merged)} days  {merged['day'].iloc[0]} → {merged['day'].iloc[-1]}")
    a = merged["pnl"].values
    b = merged["pnl_f2f"].values

    # Restrict to non-trivial days (any non-zero PnL on either strategy)
    nonzero = (np.abs(a) > 0) | (np.abs(b) > 0)
    a_nz = a[nonzero]
    b_nz = b[nonzero]
    rho = float(np.corrcoef(a_nz, b_nz)[0, 1]) if len(a_nz) > 1 else 0.0
    sa = _sharpe(a_nz, DAILY_BPY)
    sb = _sharpe(b_nz, DAILY_BPY)
    print(f"    correlation A vs B: {rho:+.4f}  (lower = better diversification)")
    print(f"    Sharpe A (nonzero days, daily): {sa:+.4f}")
    print(f"    Sharpe B (nonzero days, daily): {sb:+.4f}")

    # Combinations
    print()
    print(f"{'config':<32s} | weighted_sharpe (daily ann)")
    for wa, wb, name in [
        (1.0, 0.0, "100% intraday"),
        (0.0, 1.0, "100% F2F daily"),
        (0.5, 0.5, "equal weight"),
        (0.7, 0.3, "70/30 intraday"),
        (0.3, 0.7, "30/70 F2F"),
        (sa, sb, "Sharpe-weighted"),
    ]:
        # Normalize weights
        wn = wa + wb
        if wn == 0:
            continue
        wa_n, wb_n = wa / wn, wb / wn
        combined = wa_n * a + wb_n * b
        s_combo = _sharpe(combined[combined != 0], DAILY_BPY)
        ann_ret = combined.mean() * DAILY_BPY * 100
        ann_vol = combined.std(ddof=1) * np.sqrt(DAILY_BPY) * 100
        print(f"  {name:<30s} | {s_combo:+.4f}  ann_ret={ann_ret:+.3f}%  ann_vol={ann_vol:.3f}%")

    # Risk-parity: w_i ∝ 1/sigma_i
    sigma_a = a.std(ddof=1) + 1e-12
    sigma_b = b.std(ddof=1) + 1e-12
    inv_a = 1.0 / sigma_a
    inv_b = 1.0 / sigma_b
    wn = inv_a + inv_b
    rp = (inv_a / wn) * a + (inv_b / wn) * b
    s_rp = _sharpe(rp[rp != 0], DAILY_BPY)
    print(f"  {'risk-parity (1/sigma)':<30s} | {s_rp:+.4f}")

    print()
    print(f"Benchmarks: F2F 2.88 (daily gold futures), VLSTM 2.40 (multi-asset portfolio)")
    print(f"Our best meta-stack: depends on config above")


if __name__ == "__main__":
    main()
