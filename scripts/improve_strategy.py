"""Improved-strategy search: composite signals built on top of
vol_target_buy_hold (the §6.8 ship strategy) — can we push beyond +1.195
WF mean Sharpe?

All strategies are PIT-clean (audited) and trade GLD with cost 2 bp
round-trip. Backtested on the same 4-fold WF geometry as V4/V4d.

Composites tested:

  V1: vol_target_buy_hold  (baseline; §6.8)
  V2: vol_target × ma_20_50_long  (only long when also trending up)
  V3: vol_target × atr_stop_3x   (only long when also above ATR stop)
  V4: vol_target × ma × atr      (triple-gate, most restrictive)
  V5: vol_target × momentum_64_long
  V6: vol_target × low_vol_only  (double vol filter)
  V7: ensemble_weighted = 0.5*vol_target + 0.3*ma_20_50_long + 0.2*atr_stop_3x
  V8: leveraged_vol_target (target_vol 7% instead of 5%, cap 1.5x)
  V9: lower-target_vol (3%) for higher Sharpe at lower returns
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from nanogld.backtest.baselines import (  # noqa: E402
    donchian_positions,
    ma_cross_positions,
)
from nanogld.data.walk_forward_splits import compute_fold_boundaries  # noqa: E402

BARS_PER_YEAR = 3276
BASE_COST_BPS = 2.0
GLD_CLOSE_FEATURE_IDX = 3


def _arr(x: Any) -> np.ndarray:
    return x.numpy() if hasattr(x, "numpy") else np.asarray(x)


def _sharpe(r: np.ndarray) -> float:
    r = r[np.isfinite(r)]
    if len(r) < 2:
        return 0.0
    mu, sigma = r.mean(), r.std(ddof=1)
    if sigma < 1e-12:
        return 0.0
    return float(mu / sigma * np.sqrt(BARS_PER_YEAR))


def _pnl(positions: np.ndarray, nlr: np.ndarray, cost_mult: float = 1.0) -> np.ndarray:
    p = np.nan_to_num(positions, nan=0.0)
    r = np.nan_to_num(nlr, nan=0.0)
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


def _vol_target(realized_ret: np.ndarray, target_vol: float, cap: float = 1.0) -> np.ndarray:
    rv = _rolling_std(realized_ret, window=64)
    raw = target_vol / (rv * np.sqrt(BARS_PER_YEAR) + 1e-8)
    pos = np.clip(raw, 0.0, cap)
    pos[~np.isfinite(pos)] = 0.0
    return np.concatenate([[0.0], pos[:-1]])


def _long_only_ma(close: np.ndarray, fast: int, slow: int) -> np.ndarray:
    p = ma_cross_positions(close, fast_span=fast, slow_span=slow).astype(np.float64)
    p[p < 0] = 0.0
    return p


def _atr_stop_long(close: np.ndarray, atr: np.ndarray, mult: float = 3.0) -> np.ndarray:
    n = len(close)
    pos = np.zeros(n, dtype=np.float64)
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
                pos[i] = 1.0
        else:
            win_start = max(0, i - 20)
            window_high = float(np.nanmax(close[win_start:i + 1]))
            if c >= window_high:
                in_pos = True
                high = c
                pos[i] = 1.0
    return pos


def _momentum_long(realized_ret: np.ndarray, lookback: int = 64) -> np.ndarray:
    csum = np.zeros_like(realized_ret, dtype=np.float64)
    csum_cum = np.cumsum(realized_ret, dtype=np.float64)
    for i in range(len(csum)):
        a = max(0, i - lookback + 1)
        csum[i] = csum_cum[i] - (csum_cum[a - 1] if a > 0 else 0.0)
    pos = (csum > 0.0).astype(np.float64)
    return np.concatenate([[0.0], pos[:-1]])


def _expanding_median_low_vol(realized_ret: np.ndarray) -> np.ndarray:
    rv = _rolling_std(realized_ret, window=64)
    pos = np.zeros_like(rv)
    for i in range(1, len(rv)):
        past = rv[:i]
        past = past[past > 0]
        if len(past) < 32:
            continue
        med = float(np.median(past))
        pos[i] = 1.0 if rv[i - 1] < med else 0.0
    return pos


# === STRATEGY DEFINITIONS ==================================================
def strat_vol_target_5(ctx):
    return _vol_target(ctx["rr"], target_vol=0.05)


def strat_vol_target_x_ma(ctx):
    vt = _vol_target(ctx["rr"], target_vol=0.05)
    ma = _long_only_ma(ctx["close"], 20, 50)
    return vt * ma


def strat_vol_target_x_atr(ctx):
    vt = _vol_target(ctx["rr"], target_vol=0.05)
    at = _atr_stop_long(ctx["close"], ctx["atr"], mult=3.0)
    return vt * at


def strat_vol_target_x_ma_atr(ctx):
    vt = _vol_target(ctx["rr"], target_vol=0.05)
    ma = _long_only_ma(ctx["close"], 20, 50)
    at = _atr_stop_long(ctx["close"], ctx["atr"], mult=3.0)
    return vt * ma * at


def strat_vol_target_x_momentum(ctx):
    vt = _vol_target(ctx["rr"], target_vol=0.05)
    mo = _momentum_long(ctx["rr"], lookback=64)
    return vt * mo


def strat_vol_target_x_lowvol(ctx):
    vt = _vol_target(ctx["rr"], target_vol=0.05)
    lv = _expanding_median_low_vol(ctx["rr"])
    return vt * lv


def strat_ensemble_weighted(ctx):
    vt = _vol_target(ctx["rr"], target_vol=0.05)
    ma = _long_only_ma(ctx["close"], 20, 50)
    at = _atr_stop_long(ctx["close"], ctx["atr"], mult=3.0)
    return 0.5 * vt + 0.3 * ma + 0.2 * at


def strat_vol_target_7_cap15(ctx):
    return _vol_target(ctx["rr"], target_vol=0.07, cap=1.5)


def strat_vol_target_3(ctx):
    return _vol_target(ctx["rr"], target_vol=0.03)


def strat_vol_target_x_ma1020(ctx):
    vt = _vol_target(ctx["rr"], target_vol=0.05)
    ma = _long_only_ma(ctx["close"], 10, 20)
    return vt * ma


def strat_buy_hold(ctx):
    return np.ones(len(ctx["rr"]), dtype=np.float64)


STRATEGIES: dict[str, Callable[[dict], np.ndarray]] = {
    "buy_hold": strat_buy_hold,
    "vol_target_5pct": strat_vol_target_5,
    "vol_target_3pct": strat_vol_target_3,
    "vol_target_7pct_cap15": strat_vol_target_7_cap15,
    "vt_x_ma_20_50": strat_vol_target_x_ma,
    "vt_x_atr_3x": strat_vol_target_x_atr,
    "vt_x_ma_x_atr": strat_vol_target_x_ma_atr,
    "vt_x_momentum_64": strat_vol_target_x_momentum,
    "vt_x_lowvol": strat_vol_target_x_lowvol,
    "vt_x_ma_10_20": strat_vol_target_x_ma1020,
    "ensemble_weighted": strat_ensemble_weighted,
}


def main() -> int:
    unified = torch.load(REPO_ROOT / "data" / "processed" / "training_v1_unified.pt", weights_only=False)
    features = _arr(unified["features"])
    bcn = _arr(unified["bar_close_utc_ns"]).astype(np.int64)
    close_full = features[:, GLD_CLOSE_FEATURE_IDX].astype(np.float64)
    fold_boundaries = compute_fold_boundaries(bcn)
    print(f"folds: {len(fold_boundaries)}")

    fold_results: dict[str, list[float]] = {name: [] for name in STRATEGIES}
    fold_results_15: dict[str, list[float]] = {name: [] for name in STRATEGIES}

    for fb in fold_boundaries[:4]:
        side_path = REPO_ROOT / "data" / "processed" / f"training_v1_sidecar_fold_{fb.fold_idx}.pt"
        side = torch.load(side_path, weights_only=False)
        atr_full = _arr(side["gld_atr_14"]).astype(np.float64)
        nlr_full = _arr(side["next_log_return"]).astype(np.float64)

        ts = slice(fb.test_start, fb.test_end)
        close_t = close_full[ts]
        atr_t = atr_full[ts]
        # PIT-correct realized return
        rr = np.zeros_like(close_t, dtype=np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            rr[1:] = np.log(np.maximum(close_t[1:], 1e-12) / np.maximum(close_t[:-1], 1e-12))
        rr = np.nan_to_num(rr, nan=0.0, posinf=0.0, neginf=0.0)
        nlr_t = nlr_full[ts]

        ctx = {"close": close_t, "atr": atr_t, "rr": rr, "nlr": nlr_t}

        for name, fn in STRATEGIES.items():
            try:
                pos = fn(ctx)
            except Exception as e:  # noqa: BLE001
                print(f"    [ERR] {name}: {e}")
                fold_results[name].append(float("nan"))
                fold_results_15[name].append(float("nan"))
                continue
            fold_results[name].append(_sharpe(_pnl(pos, nlr_t, 1.0)))
            fold_results_15[name].append(_sharpe(_pnl(pos, nlr_t, 1.5)))

    print()
    print(f"{'strategy':<28s} | {'fold0':>8s} {'fold1':>8s} {'fold2':>8s} {'fold3':>8s} | {'mean1x':>8s} {'med1x':>8s} {'mean1.5x':>9s} {'pos_all':>7s} {'vs_bh':>7s}")
    bh_mean = float(np.mean(fold_results["buy_hold"]))
    rows: list[tuple[float, str]] = []
    for name in STRATEGIES:
        v = fold_results[name]
        v15 = fold_results_15[name]
        mn = float(np.mean(v))
        med = float(np.median(v))
        mn15 = float(np.mean(v15))
        pos_all = sum(1 for x in v if x > 0)
        rows.append((mn, f"{name:<28s} | {v[0]:>+8.3f} {v[1]:>+8.3f} {v[2]:>+8.3f} {v[3]:>+8.3f} | {mn:>+8.3f} {med:>+8.3f} {mn15:>+9.3f} {pos_all:>7d} {mn - bh_mean:>+7.3f}"))
    rows.sort(key=lambda x: -x[0])
    for _, r in rows:
        print(r)
    print()
    print(f"buy_hold WF mean: {bh_mean:+.4f}")
    print("vs_bh = delta vs buy_hold (positive = improvement)")
    print("pos_all = how many of 4 folds are positive (4 = ideal)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
