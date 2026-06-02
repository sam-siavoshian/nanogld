"""Forecast-to-Fill (Singha, Aguilera-Toste, Lahiri 2025, arXiv:2511.08571)
replication on CME GC=F daily continuous front-month.

Implements the FAST configuration described in the paper:

  Trend signal:
    y_tilde_t = lambda * y_tilde_{t-1} + (1 - lambda) * log_close_t
    z_t = (y_tilde_t - y_tilde_{t-1} - mu_train) / sigma_train
    p_trend_t = (clip(z_t, -3, 3) + 3) / 6     in [0, 1]

  Momentum confirm:
    m_t = 1{ close_t / close_{t-K} > 1 }, K = 50

  Blend:
    p_bull_t = omega * p_trend_t + (1 - omega) * m_t
    p_bear_t = 1 - p_bull_t

  Activation (long-only):
    enter long if p_bull_t >= 0.52 AND delta_y_tilde_t > 0

  Vol target (EWMA RiskMetrics):
    sigma_hat_{t+1}^2 = theta * sigma_hat_t^2 + (1 - theta) * r_t^2
    sigma_star = 0.15 / sqrt(252)
    w_vol_t = min(W_max, sigma_star / sigma_hat_{t+1})

  Confidence shaping:
    w_conf_t = w_vol_t * (p_bull_t - 0.5) / 0.5

  Fractional Kelly:
    f_star = quadratic-root solution to growth objective with sqrt-impact
    f_tilde = 0.40 * f_star   (fractional Kelly = 0.40)
    w_t = f_tilde * w_conf_t   (capped at W_max)

  ATR exits:
    ATR_t = mean_{i=t-13..t}(TR_i),  TR_t = max(H-L, |H-C_{t-1}|, |L-C_{t-1}|)
    hard stop: entry_price - 2 * ATR_t
    trailing stop: peak_since_entry - 1.5 * ATR_t (ratchet)
    timeout: 30 trading days
    regime de-risk: close if p_bear_t > 0.50 (halve at 0.50)

  Walk-forward: train 10y rolling, test 6mo OOS, monthly step.

  Costs: per-day cost = 0.7 bp * |delta_w_t| + impact (already in Kelly via gamma).

Notes
-----
- We use close-only data (yfinance doesn't give clean intraday OHLC for GC=F
  back to 2005 reliably); ATR is built from rolling daily close-to-close
  absolute returns scaled by close as a proxy (paper uses true HLC; this
  is a documented approximation).
- We grid-search lambda, omega, theta on the train window per fold using
  train Sharpe net-of-cost as the criterion (paper says these are train-
  tuned; exact values not published).
- All decisions at bar t use only close[<= t]; positions applied to
  return at t+1 (paper's T+1 convention).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent

DAILY_BARS_PER_YEAR = 252
COST_BPS_PER_TURN = 0.7
KELLY_FRACTION = 0.40
VOL_TARGET_ANN = 0.15
W_MAX = 2.0
P_BULL_THRESH = 0.52
P_BEAR_DERISK = 0.50
MOMENTUM_LOOKBACK_K = 50
ATR_WINDOW = 14
ATR_HARD_MULT = 2.0
ATR_TRAIL_MULT = 1.5
TIMEOUT_DAYS = 30
IMPACT_GAMMA = 0.0  # Drop sqrt-impact penalty; see kelly_f_star() comment.

# Grid search (paper says train-tuned). Small grid to keep total ~quick.
LAMBDA_GRID = [0.7, 0.85, 0.95]   # EMA log-price smoother
OMEGA_GRID = [0.5, 0.6, 0.7]      # trend/momentum blend weight
THETA_GRID = [0.90, 0.94, 0.97]   # EWMA variance memory


def fetch_gold_daily(start: str = "2005-01-01") -> pd.DataFrame:
    import yfinance as yf  # noqa: PLC0415
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


def ema(x: np.ndarray, lam: float) -> np.ndarray:
    out = np.zeros_like(x, dtype=np.float64)
    out[0] = x[0]
    for i in range(1, len(x)):
        out[i] = lam * out[i - 1] + (1.0 - lam) * x[i]
    return out


def ewma_var(r: np.ndarray, theta: float, init_var: float) -> np.ndarray:
    """One-step-ahead forecast variance, RiskMetrics. sig2[t+1] uses r[<=t]."""
    n = len(r)
    sig2 = np.zeros(n, dtype=np.float64)
    sig2[0] = init_var
    for t in range(1, n):
        sig2[t] = theta * sig2[t - 1] + (1.0 - theta) * r[t - 1] * r[t - 1]
    return sig2


def compute_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, window: int = 14) -> np.ndarray:
    n = len(close)
    tr = np.zeros(n, dtype=np.float64)
    for t in range(1, n):
        a = high[t] - low[t]
        b = abs(high[t] - close[t - 1])
        c = abs(low[t] - close[t - 1])
        tr[t] = max(a, b, c)
    # simple mean (paper specifies simple, not Wilder)
    atr = np.zeros(n, dtype=np.float64)
    if n < window:
        return atr
    cs = np.cumsum(tr, dtype=np.float64)
    for t in range(window - 1, n):
        a = t - window + 1
        atr[t] = (cs[t] - (cs[a - 1] if a > 0 else 0.0)) / window
    return atr


def kelly_f_star(mu: float, sigma: float, n_round_trips: float, k: float, gamma: float = IMPACT_GAMMA) -> float:
    """Fractional Kelly with sqrt-impact penalty.

    Pure-Kelly path (gamma=0):  f_star = (mu - n*k) / sigma^2
    Sqrt-impact path (gamma>0): solve 2*sigma^2*x^2 + 3*gamma*n^(3/2)*x - 2*(mu - n*k) = 0,  f_star = x^2.

    Note on units: the paper's gamma=0.02 applied to (n*f)^(3/2) crushes
    f down to ~1e-6 if f is interpreted as a bankroll fraction in [0,1]
    and n=1, because gamma * 1^1.5 = 0.02 dominates the daily mu ~1e-4.
    For replication we fall back to pure Kelly (gamma=0) and let the
    linear cost + vol target + confidence shaping do the position sizing.
    """
    if sigma <= 1e-12:
        return 0.0
    edge = mu - n_round_trips * k
    if edge <= 0.0:
        return 0.0
    if gamma <= 0.0:
        f = edge / (sigma * sigma)
        return float(max(0.0, f))
    A = 2.0 * sigma * sigma
    B = 3.0 * gamma * n_round_trips ** 1.5
    C = -2.0 * edge
    disc = B * B - 4.0 * A * C
    if disc <= 0.0:
        return 0.0
    x = (-B + np.sqrt(disc)) / (2.0 * A)
    if x <= 0.0:
        return 0.0
    return float(x * x)


def build_signal(
    df: pd.DataFrame, lam: float, omega: float, train_slice: slice
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return p_bull, delta_y_tilde, p_trend over all bars.

    train_slice fits mu/sigma of standardized slope; subsequent bars
    use those frozen mu/sigma.
    """
    log_c = df["log_close"].values
    y_tilde = ema(log_c, lam)
    d_yt = np.zeros_like(y_tilde)
    d_yt[1:] = y_tilde[1:] - y_tilde[:-1]

    d_train = d_yt[train_slice]
    mu_train = float(d_train.mean())
    sigma_train = float(d_train.std(ddof=1) + 1e-12)
    z = (d_yt - mu_train) / sigma_train
    z_clip = np.clip(z, -3.0, 3.0)
    p_trend = (z_clip + 3.0) / 6.0  # [0,1]

    close = df["close"].values
    n = len(close)
    momentum = np.zeros(n, dtype=np.float64)
    for t in range(MOMENTUM_LOOKBACK_K, n):
        momentum[t] = 1.0 if close[t] > close[t - MOMENTUM_LOOKBACK_K] else 0.0

    p_bull = omega * p_trend + (1.0 - omega) * momentum
    return p_bull, d_yt, p_trend


def simulate_strategy(
    df: pd.DataFrame,
    lam: float,
    omega: float,
    theta: float,
    train_slice: slice,
    test_slice: slice,
    return_diag: bool = False,
) -> dict[str, Any]:
    """Run F2F FAST on test_slice (with train_slice used to fit moments).

    Returns a dict with daily PnL (net of linear cost) and the position
    vector (signed magnitude in [0, W_max]).
    """
    p_bull, d_yt, p_trend = build_signal(df, lam, omega, train_slice)
    n = len(df)
    log_ret = df["log_ret"].values

    high = df["high"].values
    low = df["low"].values
    close = df["close"].values
    atr = compute_atr(high, low, close, window=ATR_WINDOW)

    # EWMA variance from train_slice init then continuing causally.
    var_train = float(log_ret[train_slice].var(ddof=1)) if (train_slice.stop - train_slice.start) > 1 else 1e-6
    sig2 = ewma_var(log_ret, theta, init_var=var_train)
    sigma_hat = np.sqrt(sig2)
    sigma_star = VOL_TARGET_ANN / np.sqrt(DAILY_BARS_PER_YEAR)
    w_vol = np.minimum(W_MAX, sigma_star / (sigma_hat + 1e-12))

    # Train moments for Kelly mu, sigma estimators (unit-notional sleeve).
    mu_t = float(log_ret[train_slice].mean())
    sigma_kelly = float(log_ret[train_slice].std(ddof=1) + 1e-12)
    # n_round_trips capped at 1 per day per paper
    n_rt = 1.0

    pos = np.zeros(n, dtype=np.float64)
    in_pos = False
    entry_price = 0.0
    peak_price = 0.0
    age = 0
    for t in range(train_slice.stop, test_slice.stop):
        # Position decision uses info known at close of bar t,
        # applied to log_ret[t+1] (T+1 fill convention) — i.e. we trade
        # bar t+1 with the signal calculated at close of bar t.
        pb = p_bull[t]
        dyt = d_yt[t]
        if in_pos:
            age += 1
            if close[t] > peak_price:
                peak_price = close[t]
            stop_hard = entry_price - ATR_HARD_MULT * atr[t]
            stop_trail = peak_price - ATR_TRAIL_MULT * atr[t]
            exit_now = (
                close[t] < stop_hard
                or close[t] < stop_trail
                or age >= TIMEOUT_DAYS
                or pb < P_BEAR_DERISK
            )
            if exit_now:
                in_pos = False
                pos[t + 1 if t + 1 < n else t] = 0.0
                continue
            # Stay in position: recompute size based on current vol/conf.
            f_star = kelly_f_star(mu_t, sigma_kelly, n_rt, COST_BPS_PER_TURN * 1e-4, IMPACT_GAMMA)
            f_tilde = KELLY_FRACTION * f_star
            w_conf = w_vol[t] * (pb - 0.5) / 0.5
            w_t = max(0.0, min(W_MAX, f_tilde * w_conf))
            if t + 1 < n:
                pos[t + 1] = w_t
        else:
            if pb >= P_BULL_THRESH and dyt > 0:
                in_pos = True
                entry_price = close[t]
                peak_price = close[t]
                age = 0
                f_star = kelly_f_star(mu_t, sigma_kelly, n_rt, COST_BPS_PER_TURN * 1e-4, IMPACT_GAMMA)
                f_tilde = KELLY_FRACTION * f_star
                w_conf = w_vol[t] * (pb - 0.5) / 0.5
                w_t = max(0.0, min(W_MAX, f_tilde * w_conf))
                if t + 1 < n:
                    pos[t + 1] = w_t
            # else stay flat

    # Apply linear cost on absolute position change
    cost_frac = COST_BPS_PER_TURN / 10_000.0
    pos_prev = np.concatenate([[0.0], pos[:-1]])
    pnl = pos * log_ret - cost_frac * np.abs(pos - pos_prev)
    return {
        "pnl": pnl,
        "pos": pos,
        "p_bull": p_bull,
        "sigma_hat": sigma_hat,
        "test_slice": test_slice,
        "train_slice": train_slice,
    }


def sharpe(r: np.ndarray) -> float:
    r = r[np.isfinite(r)]
    if len(r) < 2:
        return 0.0
    mu, sigma = r.mean(), r.std(ddof=1)
    if sigma < 1e-12:
        return 0.0
    return float(mu / sigma * np.sqrt(DAILY_BARS_PER_YEAR))


def grid_search_train(df: pd.DataFrame, train_slice: slice, val_slice: slice) -> tuple[float, float, float, float]:
    """Pick lambda, omega, theta that maximize val Sharpe (PIT)."""
    best = (-1e9, 0.85, 0.6, 0.94)
    for lam in LAMBDA_GRID:
        for omega in OMEGA_GRID:
            for theta in THETA_GRID:
                out = simulate_strategy(df, lam, omega, theta, train_slice, val_slice)
                s = sharpe(out["pnl"][val_slice])
                if s > best[0]:
                    best = (s, lam, omega, theta)
    return best


def walk_forward_folds(df: pd.DataFrame, train_yrs: int = 10, test_months: int = 6, step_months: int = 1) -> list[dict]:
    """Build (train, val, test) folds. train = first 80% of train_yrs window,
    val = last 20% of train window, test = test_months OOS."""
    bars_per_year = 252
    train_len = train_yrs * bars_per_year
    test_len = int(test_months * (bars_per_year / 12))
    val_len = train_len // 5
    train_main_len = train_len - val_len
    step_len = int(step_months * (bars_per_year / 12))

    n = len(df)
    folds = []
    start = 0
    while True:
        train_main_end = start + train_main_len
        val_end = train_main_end + val_len
        test_end = val_end + test_len
        if test_end > n:
            break
        folds.append({
            "train": slice(start, train_main_end),
            "val": slice(train_main_end, val_end),
            "test": slice(val_end, test_end),
            "fold_idx": len(folds),
        })
        start += step_len
        if len(folds) >= 64:
            break
    return folds


def main() -> int:
    print(f"=== F2F replica (Singha et al. 2025, arXiv:2511.08571) ===")
    df = fetch_gold_daily(start="2005-01-01")
    n = len(df)
    print(f"bars: {n}  range: {df['ts'].iloc[0].date()} → {df['ts'].iloc[-1].date()}")

    folds = walk_forward_folds(df, train_yrs=10, test_months=6, step_months=1)
    print(f"folds: {len(folds)}")
    bars_per_month = int(252 / 12)
    test_pnl_concat: list[np.ndarray] = []
    test_pos_concat: list[np.ndarray] = []

    fold_sharpe: list[float] = []
    for fi, fold in enumerate(folds):
        s_best, lam, omega, theta = grid_search_train(df, fold["train"], fold["val"])
        out = simulate_strategy(df, lam, omega, theta, fold["train"], fold["test"])
        # Build NON-OVERLAPPING OOS chain: each fold contributes only the
        # FIRST `bars_per_month` bars of its test slice (= step size).
        # Otherwise 6mo test × monthly step => each bar counted 6x.
        first_month = slice(fold["test"].start, min(fold["test"].stop, fold["test"].start + bars_per_month))
        s_test = sharpe(out["pnl"][first_month])
        fold_sharpe.append(s_test)
        test_pnl_concat.append(out["pnl"][first_month])
        test_pos_concat.append(out["pos"][first_month])
        if fi % 10 == 0 or fi < 3:
            print(f"  fold {fi:3d}: train=[{fold['train'].start}:{fold['train'].stop}) val=[{fold['val'].start}:{fold['val'].stop}) test_first_mo=[{first_month.start}:{first_month.stop})")
            print(f"           best lam={lam} omega={omega} theta={theta}  val_sharpe={s_best:.3f}  test_first_mo_sharpe={s_test:+.3f}")

    print()
    print(f"folds tested: {len(fold_sharpe)}")
    print(f"per-fold test Sharpe: mean={np.mean(fold_sharpe):+.4f}  median={np.median(fold_sharpe):+.4f}  std={np.std(fold_sharpe):.4f}  min={np.min(fold_sharpe):+.4f}  max={np.max(fold_sharpe):+.4f}")

    # Aggregate: concatenate all OOS slices (overlap-free per design) →
    # one long OOS PnL series and report its Sharpe (this is the "all
    # OOS days" Sharpe the paper headlines).
    all_pnl = np.concatenate(test_pnl_concat)
    all_pos = np.concatenate(test_pos_concat)
    agg_sharpe = sharpe(all_pnl)
    print(f"aggregate OOS Sharpe ({len(all_pnl)} days, concat of test slices): {agg_sharpe:+.4f}")
    print(f"  ann_ret = {all_pnl.mean() * DAILY_BARS_PER_YEAR * 100:+.4f}%")
    print(f"  ann_vol = {all_pnl.std(ddof=1) * np.sqrt(DAILY_BARS_PER_YEAR) * 100:.4f}%")
    print(f"  active days (|pos|>0): {(np.abs(all_pos) > 1e-9).sum()} / {len(all_pos)} = {100*(np.abs(all_pos) > 1e-9).mean():.1f}%")
    print(f"  mean |pos|: {np.abs(all_pos).mean():.4f}")

    print()
    print(f"F2F paper claim: Sharpe 2.88 (CI [2.49, 3.27])")
    print(f"Replica result : Sharpe {agg_sharpe:+.4f}")
    if agg_sharpe > 2.88:
        print(f">>> BEATS F2F by {agg_sharpe - 2.88:+.4f}")
    elif agg_sharpe > 2.40:
        print(f"beats VLSTM portfolio number 2.40 by {agg_sharpe - 2.40:+.4f}, F2F gap {agg_sharpe - 2.88:+.4f}")
    else:
        print(f"VLSTM gap {agg_sharpe - 2.40:+.4f}, F2F gap {agg_sharpe - 2.88:+.4f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
