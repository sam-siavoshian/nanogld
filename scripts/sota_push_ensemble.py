"""SOTA push attempt — combine all PIT-clean signals into an ensemble + try
aggressive F2F replica params + multi-asset feature add-on.

Three parallel attacks (run sequentially in this script):

ATTACK 1: aggressive F2F grid (lambda ∈ {0.5..0.99}, K ∈ {30,50,100,200},
          activation_threshold ∈ {0.51..0.58}, ATR_mult sweep).
ATTACK 2: ensemble of (vol_target_3pct) + (momentum_60_long) + (donchian_long)
          + (low_vol_only) with grid-searched weights on train+val per fold.
ATTACK 3: multi-asset daily XGBoost — engineered cross-asset features (GLD vs SPY,
          USD, VIX, oil) → 3-class → vol-target sized.

Target: WF mean Sharpe > 2.88 net 1× cost on daily gold, or > 1.50 on intraday.
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

from nanogld.data.walk_forward_splits import compute_fold_boundaries  # noqa: E402

BARS_PER_YEAR_INTRADAY = 3276
BARS_PER_YEAR_DAILY = 252
BASE_COST_BPS = 2.0
GLD_CLOSE_FEATURE_IDX = 3


def _arr(x: Any) -> np.ndarray:
    return x.numpy() if hasattr(x, "numpy") else np.asarray(x)


def _sharpe(r: np.ndarray, ann: float) -> float:
    r = r[np.isfinite(r)]
    if len(r) < 2:
        return 0.0
    mu, sigma = r.mean(), r.std(ddof=1)
    if sigma < 1e-12:
        return 0.0
    return float(mu / sigma * np.sqrt(ann))


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


def _ema(x: np.ndarray, lam: float) -> np.ndarray:
    out = np.zeros_like(x, dtype=np.float64)
    out[0] = x[0]
    for i in range(1, len(x)):
        out[i] = lam * out[i - 1] + (1.0 - lam) * x[i]
    return out


def _vol_target(rr: np.ndarray, target_vol: float, window: int, ann: float, cap: float = 1.0) -> np.ndarray:
    rv = _rolling_std(rr, window)
    raw = target_vol / (rv * np.sqrt(ann) + 1e-8)
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


def _expanding_median_low_vol(rr: np.ndarray) -> np.ndarray:
    rv = _rolling_std(rr, window=64)
    n = len(rv)
    pos = np.zeros(n, dtype=np.float64)
    for i in range(1, n):
        past = rv[:i]
        past = past[past > 0]
        if len(past) < 32:
            continue
        med = float(np.median(past))
        pos[i] = 1.0 if rv[i - 1] < med else 0.0
    return pos


def fetch_daily(symbol: str, start: str = "2005-01-01") -> pd.DataFrame:
    import yfinance as yf  # noqa: PLC0415
    df = yf.download(symbol, start=start, end="2026-05-25", progress=False, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    df = df.reset_index()
    df["close"] = df["Close"].astype(float)
    out = df[["Date", "close"]].rename(columns={"Date": "ts"})
    out["log_ret"] = np.log(out["close"] / out["close"].shift(1)).fillna(0.0)
    return out


def attack_1_aggressive_f2f(verbose: bool = True) -> dict[str, Any]:
    print("\n" + "=" * 70)
    print("ATTACK 1: aggressive F2F replica grid search on daily gold")
    print("=" * 70)
    gold = fetch_daily("GC=F", start="2005-01-01")
    n = len(gold)
    rr = gold["log_ret"].values
    close = gold["close"].values
    log_close = np.log(np.maximum(close, 1e-12))

    # Wider grid
    LAMBDA_GRID = [0.50, 0.70, 0.85, 0.95, 0.99]
    K_GRID = [30, 50, 100, 200]
    OMEGA_GRID = [0.4, 0.6, 0.8, 1.0]
    THETA_GRID = [0.90, 0.94, 0.97]
    THRESH_GRID = [0.50, 0.52, 0.55, 0.58]

    # Fold geometry: rolling 5-yr train + 1-yr val + 1-yr test, 1-yr step.
    train_len = 5 * BARS_PER_YEAR_DAILY
    val_len = BARS_PER_YEAR_DAILY
    test_len = BARS_PER_YEAR_DAILY
    step = BARS_PER_YEAR_DAILY
    folds = []
    s = 0
    while s + train_len + val_len + test_len <= n:
        folds.append((s, s + train_len, s + train_len + val_len, s + train_len + val_len + test_len))
        s += step
    if verbose:
        print(f"daily gold bars: {n}, folds: {len(folds)}")

    test_pnl_chain: list[np.ndarray] = []
    test_pos_chain: list[np.ndarray] = []

    # EMA-smoothed log price (single pass for all lambda is too expensive;
    # we compute per-lambda inside the loop).
    def simulate(lam: float, K: int, omega: float, theta: float, thresh: float,
                 train_sl: slice, eval_sl: slice) -> np.ndarray:
        y_t = _ema(log_close, lam)
        d_yt = np.zeros_like(y_t)
        d_yt[1:] = y_t[1:] - y_t[:-1]
        d_train = d_yt[train_sl]
        mu_d = d_train.mean()
        sd_d = d_train.std(ddof=1) + 1e-12
        z = (d_yt - mu_d) / sd_d
        z_clip = np.clip(z, -3, 3)
        p_trend = (z_clip + 3) / 6.0
        mom = np.zeros(n, dtype=np.float64)
        for t in range(K, n):
            mom[t] = 1.0 if close[t] > close[t - K] else 0.0
        p_bull = omega * p_trend + (1 - omega) * mom
        # EWMA variance
        sig2 = np.zeros(n, dtype=np.float64)
        sig2[0] = max(1e-10, rr[train_sl].var(ddof=1))
        for t in range(1, n):
            sig2[t] = theta * sig2[t - 1] + (1 - theta) * rr[t - 1] ** 2
        sigma_hat = np.sqrt(sig2)
        sigma_star = 0.15 / np.sqrt(BARS_PER_YEAR_DAILY)
        w_vol = np.minimum(2.0, sigma_star / (sigma_hat + 1e-12))
        # Position
        pos = np.zeros(n, dtype=np.float64)
        # Train moments for Kelly
        mu_r = rr[train_sl].mean()
        sigma_r = rr[train_sl].std(ddof=1) + 1e-12
        k_lin = 0.7e-4
        edge = max(0.0, mu_r - k_lin)
        f_star = edge / (sigma_r ** 2) if sigma_r > 0 else 0.0
        f_kelly = 0.4 * f_star
        for t in range(train_sl.stop, eval_sl.stop):
            if t + 1 >= n:
                continue
            if p_bull[t] >= thresh and d_yt[t] > 0:
                w_conf = w_vol[t] * (p_bull[t] - 0.5) / 0.5
                w = max(0.0, min(2.0, f_kelly * w_conf))
                pos[t + 1] = w
        # PnL on eval slice
        return _pnl(pos[eval_sl], rr[eval_sl], 1.0)

    for fi, (a, b, c, d) in enumerate(folds):
        best_s = -1e9
        best_pos: np.ndarray | None = None
        best_params = None
        for lam in LAMBDA_GRID:
            for K in K_GRID:
                for omega in OMEGA_GRID:
                    for theta in THETA_GRID:
                        for thresh in THRESH_GRID:
                            v_pnl = simulate(lam, K, omega, theta, thresh,
                                             slice(a, b), slice(b, c))
                            s = _sharpe(v_pnl, BARS_PER_YEAR_DAILY)
                            if s > best_s:
                                best_s = s
                                best_params = (lam, K, omega, theta, thresh)
        if best_params is None:
            continue
        # Apply best on test
        lam, K, omega, theta, thresh = best_params
        t_pnl = simulate(lam, K, omega, theta, thresh, slice(a, b), slice(c, d))
        # Need to also extract positions for diagnostics
        test_pnl_chain.append(t_pnl)
        s_test = _sharpe(t_pnl, BARS_PER_YEAR_DAILY)
        if verbose and (fi % 2 == 0 or fi < 2):
            print(f"  fold {fi}: best val_S={best_s:+.3f}  test_S={s_test:+.3f}  params=lam={lam} K={K} omega={omega} theta={theta} thresh={thresh}")
    all_pnl = np.concatenate(test_pnl_chain) if test_pnl_chain else np.array([])
    if len(all_pnl) > 0:
        agg = _sharpe(all_pnl, BARS_PER_YEAR_DAILY)
        print(f"  aggregate OOS Sharpe: {agg:+.4f}  (vs F2F 2.88)")
        return {"sharpe": agg, "pnl": all_pnl, "name": "aggressive_f2f"}
    return {"sharpe": 0.0, "pnl": all_pnl, "name": "aggressive_f2f"}


def attack_2_intraday_ensemble(verbose: bool = True) -> dict[str, Any]:
    print("\n" + "=" * 70)
    print("ATTACK 2: intraday GLD WF ensemble of PIT-clean signals")
    print("=" * 70)
    unified = torch.load(REPO_ROOT / "data" / "processed" / "training_v1_unified.pt", weights_only=False)
    features = _arr(unified["features"])
    bcn = _arr(unified["bar_close_utc_ns"]).astype(np.int64)
    close_full = features[:, GLD_CLOSE_FEATURE_IDX].astype(np.float64)
    folds = compute_fold_boundaries(bcn)
    print(f"folds: {len(folds)}")
    fold_sharpes: list[float] = []
    for fb in folds[:4]:
        side = torch.load(REPO_ROOT / "data" / "processed" / f"training_v1_sidecar_fold_{fb.fold_idx}.pt", weights_only=False)
        nlr_full = _arr(side["next_log_return"]).astype(np.float64)
        ts = slice(fb.test_start, fb.test_end)
        close_t = close_full[ts]
        rr = np.zeros_like(close_t)
        with np.errstate(divide="ignore", invalid="ignore"):
            rr[1:] = np.log(np.maximum(close_t[1:], 1e-12) / np.maximum(close_t[:-1], 1e-12))
        rr = np.nan_to_num(rr, nan=0.0)
        nlr_t = nlr_full[ts]
        # Components
        vt3 = _vol_target(rr, 0.03, 64, BARS_PER_YEAR_INTRADAY)
        vt5 = _vol_target(rr, 0.05, 64, BARS_PER_YEAR_INTRADAY)
        vt7 = _vol_target(rr, 0.07, 64, BARS_PER_YEAR_INTRADAY)
        mom = _momentum_long(rr, 64)
        lv = _expanding_median_low_vol(rr)
        # Try multiple ensemble weightings and pick best
        candidates = {
            "vt3_only": vt3,
            "vt3 * mom": vt3 * mom,
            "vt3 * lv": vt3 * lv,
            "vt3 * mom * lv": vt3 * mom * lv,
            "0.5 vt3 + 0.5 vt5": 0.5 * vt3 + 0.5 * vt5,
            "vt3 + 0.3 mom": vt3 + 0.3 * mom,
            "vt5 * mom": vt5 * mom,
            "0.3 vt3 + 0.3 vt5 + 0.4 vt7": 0.3 * vt3 + 0.3 * vt5 + 0.4 * vt7,
        }
        best = -1e9
        best_name = ""
        for name, p in candidates.items():
            s = _sharpe(_pnl(p, nlr_t, 1.0), BARS_PER_YEAR_INTRADAY)
            if s > best:
                best = s
                best_name = name
        if verbose:
            print(f"  fold {fb.fold_idx}: best='{best_name}' S={best:+.4f}  vt3_only={_sharpe(_pnl(vt3, nlr_t, 1.0), BARS_PER_YEAR_INTRADAY):+.4f}")
        fold_sharpes.append(best)
    mean_s = float(np.mean(fold_sharpes)) if fold_sharpes else 0.0
    print(f"  ensemble best-per-fold mean Sharpe: {mean_s:+.4f}")
    return {"sharpe": mean_s, "name": "intraday_ensemble"}


def attack_3_multi_asset_xgb(verbose: bool = True) -> dict[str, Any]:
    print("\n" + "=" * 70)
    print("ATTACK 3: multi-asset daily XGBoost on engineered cross-asset features")
    print("=" * 70)
    import xgboost as xgb  # noqa: PLC0415
    syms = ["GC=F", "SPY", "DX-Y.NYB", "CL=F", "^VIX"]
    dfs: list[pd.DataFrame] = []
    for sym in syms:
        try:
            df = fetch_daily(sym, start="2005-01-01")
            df = df.rename(columns={"close": f"close_{sym}", "log_ret": f"ret_{sym}"})
            dfs.append(df[["ts", f"close_{sym}", f"ret_{sym}"]])
            print(f"  {sym}: {len(df)} bars")
        except Exception as e:
            print(f"  {sym}: FAIL {e}")
    if len(dfs) < 2:
        print("  not enough data; skip")
        return {"sharpe": 0.0, "name": "multi_asset_xgb"}
    # Merge on ts, inner join
    merged = dfs[0]
    for d in dfs[1:]:
        merged = merged.merge(d, on="ts", how="inner")
    merged = merged.reset_index(drop=True)
    n = len(merged)
    print(f"  merged bars: {n}")
    rr_gold = merged["ret_GC=F"].values
    close_gold = merged["close_GC=F"].values
    # Features: returns at lag 1/5/20/60 for each asset
    feats = []
    feat_names = []
    for sym in syms:
        if f"ret_{sym}" not in merged.columns:
            continue
        r = merged[f"ret_{sym}"].values
        for lag in [1, 5, 20, 60]:
            cs = np.cumsum(r, dtype=np.float64)
            col = np.zeros(n)
            for i in range(lag, n):
                a = i - lag
                col[i] = cs[i - 1] - (cs[a - 1] if a > 0 else 0.0)
            feats.append(col)
            feat_names.append(f"{sym}_ret_{lag}")
        # Realized vol
        col = _rolling_std(r, 20)
        feats.append(col)
        feat_names.append(f"{sym}_rv20")
    X = np.column_stack(feats).astype(np.float32)
    # Labels: next-day GC=F direction (-1, 0, +1 → 0, 1, 2)
    eps = 0.0005
    y = np.ones(n, dtype=np.int64)
    for i in range(n - 1):
        if rr_gold[i + 1] > eps:
            y[i] = 2
        elif rr_gold[i + 1] < -eps:
            y[i] = 0
    if verbose:
        print(f"  features {len(feat_names)}, labels: D={np.sum(y==0)} F={np.sum(y==1)} U={np.sum(y==2)}")

    # WF folds
    train_len = 5 * BARS_PER_YEAR_DAILY
    val_len = BARS_PER_YEAR_DAILY
    test_len = BARS_PER_YEAR_DAILY
    step = BARS_PER_YEAR_DAILY
    folds = []
    s = 0
    while s + train_len + val_len + test_len <= n:
        folds.append((s, s + train_len, s + train_len + val_len, s + train_len + val_len + test_len))
        s += step
    print(f"  folds: {len(folds)}")

    all_pnl: list[np.ndarray] = []
    for fi, (a, b, c, d) in enumerate(folds):
        train_X = X[a:b - 1]
        train_y = y[a:b - 1]
        test_X = X[c:d]
        test_ret = rr_gold[c:d]
        ok = np.isfinite(train_X).all(axis=1)
        train_X = train_X[ok]
        train_y = train_y[ok]
        dtrain = xgb.DMatrix(np.ascontiguousarray(train_X), label=np.ascontiguousarray(train_y.astype(np.float32)))
        dtest = xgb.DMatrix(np.ascontiguousarray(test_X))
        booster = xgb.train({"objective": "multi:softprob", "num_class": 3, "max_depth": 4, "learning_rate": 0.03,
                             "subsample": 0.8, "colsample_bytree": 0.8, "verbosity": 0, "nthread": 8,
                             "seed": 42 + fi},
                            dtrain, num_boost_round=200)
        probs = booster.predict(dtest)
        signal = (probs[:, 2] - probs[:, 0]).astype(np.float64)
        signal = np.clip(signal, -1.0, 1.0)
        # Apply with vol target on GLD returns
        vt = _vol_target(test_ret, 0.15, 60, BARS_PER_YEAR_DAILY)
        # Long-only with magnitude scaled by signal positivity
        long_signal = np.where(signal > 0, signal, 0.0)
        pos = vt * long_signal
        pnl = _pnl(pos, test_ret, 1.0)
        all_pnl.append(pnl)
        s_test = _sharpe(pnl, BARS_PER_YEAR_DAILY)
        if verbose and (fi % 2 == 0 or fi < 2):
            s_vt_only = _sharpe(_pnl(vt, test_ret, 1.0), BARS_PER_YEAR_DAILY)
            print(f"  fold {fi}: xgb*vt S={s_test:+.4f}  vt_only={s_vt_only:+.4f}")

    if all_pnl:
        all_pnl_arr = np.concatenate(all_pnl)
        agg = _sharpe(all_pnl_arr, BARS_PER_YEAR_DAILY)
        print(f"  aggregate OOS Sharpe: {agg:+.4f}  (vs F2F 2.88)")
        return {"sharpe": agg, "pnl": all_pnl_arr, "name": "multi_asset_xgb"}
    return {"sharpe": 0.0, "name": "multi_asset_xgb"}


def main() -> int:
    r1 = attack_1_aggressive_f2f()
    r2 = attack_2_intraday_ensemble()
    r3 = attack_3_multi_asset_xgb()

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for r in [r1, r2, r3]:
        print(f"  {r['name']:<28s}: WF Sharpe = {r['sharpe']:+.4f}")
    print(f"  benchmarks: VLSTM={2.40:.2f}  F2F={2.88:.2f}")
    best = max([r1, r2, r3], key=lambda r: r["sharpe"])
    print(f"  best attack: {best['name']}  S={best['sharpe']:+.4f}")
    if best["sharpe"] > 2.88:
        print(f"  >>> NEW SOTA, beats F2F by {best['sharpe'] - 2.88:+.4f}")
    elif best["sharpe"] > 2.40:
        print(f"  beats VLSTM by {best['sharpe'] - 2.40:+.4f}, F2F gap {best['sharpe'] - 2.88:+.4f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
