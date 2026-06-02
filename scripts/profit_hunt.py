"""Profit-hunt: 4-fold walk-forward Sharpe over ~20 candidate strategies.

Goal: find an active strategy that beats buy_hold's +0.85 WF mean Sharpe
on the same 4 fold geometry as V4. CPU-only. No model training.

Strategies fall into 4 families:

  1. Long-only filters of trend-following baselines (ma_cross / donchian)
  2. ATR-stop overlays on buy_hold (capture trend, exit on volatility spikes)
  3. Regime-gated buy_hold (only long during low-vol / news-present / FOMC)
  4. Momentum-pulse (long when N-bar return > threshold, flat otherwise)

Each strategy emits a (T_test,) position vector in [-1, +1]. We score
Sharpe net of cost at 1.0× and 1.5× the 2 bp round-trip rate.
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
    """Causal rolling std of x with the given window."""
    out = np.zeros_like(x, dtype=np.float64)
    if window <= 1:
        return out
    cs = np.cumsum(x, dtype=np.float64)
    cs2 = np.cumsum(x * x, dtype=np.float64)
    for i in range(len(x)):
        a = max(0, i - window + 1)
        n = i - a + 1
        if n < 2:
            out[i] = 0.0
            continue
        m = (cs[i] - (cs[a - 1] if a > 0 else 0.0)) / n
        m2 = (cs2[i] - (cs2[a - 1] if a > 0 else 0.0)) / n
        v = max(0.0, m2 - m * m)
        out[i] = float(np.sqrt(v))
    return out


def _rolling_mean(x: np.ndarray, window: int) -> np.ndarray:
    out = np.zeros_like(x, dtype=np.float64)
    cs = np.cumsum(x, dtype=np.float64)
    for i in range(len(x)):
        a = max(0, i - window + 1)
        n = i - a + 1
        out[i] = float((cs[i] - (cs[a - 1] if a > 0 else 0.0)) / n)
    return out


def _ffill(x: np.ndarray) -> np.ndarray:
    """Forward-fill NaNs globally."""
    out = x.astype(np.float64).copy()
    last = float("nan")
    for i in range(len(out)):
        if np.isfinite(out[i]):
            last = float(out[i])
        else:
            out[i] = last
    return out


# === STRATEGY DEFINITIONS ==================================================
def buy_hold(ctx: dict) -> np.ndarray:
    return np.ones(len(ctx["nlr"]), dtype=np.float64)


def long_only_ma_cross(ctx: dict) -> np.ndarray:
    p = ma_cross_positions(ctx["close"], fast_span=10, slow_span=20).astype(np.float64)
    p[p < 0] = 0.0
    return p


def long_only_donchian(ctx: dict) -> np.ndarray:
    p = donchian_positions(ctx["close"], window=20).astype(np.float64)
    p[p < 0] = 0.0
    return p


def long_only_ma_long(ctx: dict) -> np.ndarray:
    p = ma_cross_positions(ctx["close"], fast_span=20, slow_span=50).astype(np.float64)
    p[p < 0] = 0.0
    return p


def long_only_ma_short(ctx: dict) -> np.ndarray:
    p = ma_cross_positions(ctx["close"], fast_span=5, slow_span=10).astype(np.float64)
    p[p < 0] = 0.0
    return p


def buy_hold_atr_stop(ctx: dict, mult: float = 2.0) -> np.ndarray:
    """Long 1.0 until close drops mult × ATR-14 below the trailing high.
    Re-enter long on a new local high."""
    close = ctx["close"]
    atr = ctx["atr"]
    n = len(close)
    pos = np.zeros(n, dtype=np.float64)
    in_pos = True
    high = close[0]
    for i in range(n):
        if in_pos:
            if close[i] > high:
                high = close[i]
            if close[i] < high - mult * atr[i]:
                in_pos = False
            else:
                pos[i] = 1.0
        else:
            # Re-enter when close exceeds the prior 20-bar high (ratchet)
            window_high = close[max(0, i - 20):i + 1].max()
            if close[i] >= window_high:
                in_pos = True
                high = close[i]
                pos[i] = 1.0
    return pos


def buy_hold_atr_stop_1_5(ctx: dict) -> np.ndarray:
    return buy_hold_atr_stop(ctx, mult=1.5)


def buy_hold_atr_stop_3(ctx: dict) -> np.ndarray:
    return buy_hold_atr_stop(ctx, mult=3.0)


def vol_target_buy_hold(ctx: dict, target_vol: float = 0.05) -> np.ndarray:
    """Position size = target_vol / realized_vol, clamped to [0, 1] (long-only).

    PIT-correct: uses `realized_ret` (return between bar i-1 and bar i,
    known at bar i open) not next_log_return (which would peek at bar
    i+1's close).
    """
    rr = ctx["realized_ret"]
    rv = _rolling_std(rr, window=64)
    rv = np.where(rv > 1e-8, rv, np.nan)
    rv = _ffill(rv)
    raw = target_vol / (rv * np.sqrt(BARS_PER_YEAR) + 1e-8)
    pos = np.clip(raw, 0.0, 1.0)
    # Shift forward by 1 bar so position at bar i uses information known
    # at bar i-1's close (no peek at bar i's own next_log_return).
    return np.concatenate([[0.0], pos[:-1]])


def low_vol_only(ctx: dict) -> np.ndarray:
    """Long only when rolling vol below its expanding median. PIT-correct.

    Uses an expanding median over rv[<= i-1] so the threshold at bar i
    does not depend on future bars. Position then shifted +1 to apply
    after the bar's information set is known.
    """
    rr = ctx["realized_ret"]
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


def momentum_pulse(ctx: dict, lookback: int = 26, thresh: float = 0.0) -> np.ndarray:
    """Long if cumulative realized log-return over last `lookback` bars > thresh.

    PIT-correct: uses `realized_ret` (return between bar i-1 and bar i,
    known at bar i open) not next_log_return.
    """
    rr = ctx["realized_ret"]
    cs = _rolling_mean(rr, window=lookback)
    pos = (cs > thresh).astype(np.float64)
    # Shift +1: decision uses realized_ret up to bar i, applied at bar i+1
    return np.concatenate([[0.0], pos[:-1]])


def momentum_pulse_short(ctx: dict) -> np.ndarray:
    return momentum_pulse(ctx, lookback=13, thresh=0.0)


def momentum_pulse_long(ctx: dict) -> np.ndarray:
    return momentum_pulse(ctx, lookback=64, thresh=0.0)


def news_present_long(ctx: dict) -> np.ndarray:
    return ctx["news"].astype(np.float64)


def news_absent_long(ctx: dict) -> np.ndarray:
    return (1.0 - ctx["news"]).astype(np.float64)


def high_vol_long(ctx: dict) -> np.ndarray:
    return ctx["is_high_vol"].astype(np.float64)


def low_vol_long(ctx: dict) -> np.ndarray:
    return (1.0 - ctx["is_high_vol"]).astype(np.float64)


def ma_long_atr_stop(ctx: dict) -> np.ndarray:
    """Long when ma_cross long AND ATR stop says in_pos."""
    ma = long_only_ma_cross(ctx)
    stop = buy_hold_atr_stop(ctx, mult=2.0)
    return ma * stop


def ensemble_long(ctx: dict) -> np.ndarray:
    """Average of long-only signals."""
    sigs = [
        long_only_ma_cross(ctx),
        long_only_donchian(ctx),
        long_only_ma_long(ctx),
        momentum_pulse_long(ctx),
        buy_hold_atr_stop(ctx, mult=2.0),
    ]
    return np.mean(sigs, axis=0)


def regime_smart(ctx: dict) -> np.ndarray:
    """Long during low-vol regime AND when ma cross trending up; flat otherwise.

    PIT-correct: vol from realized_ret + expanding median (no future peek).
    """
    rr = ctx["realized_ret"]
    rv = _rolling_std(rr, window=64)
    n = len(rv)
    low_vol = np.zeros(n, dtype=np.float64)
    for i in range(1, n):
        past = rv[:i]
        past = past[past > 0]
        if len(past) < 32:
            continue
        med = float(np.median(past))
        low_vol[i] = 1.0 if rv[i - 1] < med else 0.0
    ma = long_only_ma_cross(ctx)
    return ma * low_vol


STRATEGIES: dict[str, Callable[[dict], np.ndarray]] = {
    "buy_hold": buy_hold,
    "long_ma_10_20": long_only_ma_cross,
    "long_donchian_20": long_only_donchian,
    "long_ma_20_50": long_only_ma_long,
    "long_ma_5_10": long_only_ma_short,
    "buy_hold_atr_stop_2x": buy_hold_atr_stop,
    "buy_hold_atr_stop_1.5x": buy_hold_atr_stop_1_5,
    "buy_hold_atr_stop_3x": buy_hold_atr_stop_3,
    "vol_target_buy_hold": vol_target_buy_hold,
    "low_vol_only": low_vol_only,
    "momentum_pulse_13": momentum_pulse_short,
    "momentum_pulse_64": momentum_pulse_long,
    "news_present_long": news_present_long,
    "news_absent_long": news_absent_long,
    "high_vol_long": high_vol_long,
    "low_vol_long": low_vol_long,
    "ma_long_atr_stop": ma_long_atr_stop,
    "ensemble_long": ensemble_long,
    "regime_smart": regime_smart,
}


def main() -> int:
    unified = torch.load(REPO_ROOT / "data" / "processed" / "training_v1_unified.pt", weights_only=False)
    features = _arr(unified["features"])
    bcn = _arr(unified["bar_close_utc_ns"]).astype(np.int64)
    close_full = features[:, GLD_CLOSE_FEATURE_IDX].astype(np.float64)
    bn_off = _arr(unified["bar_news_offsets"]).astype(np.int64)
    news_full = (np.diff(bn_off) > 0).astype(np.float64)
    fold_boundaries = compute_fold_boundaries(bcn)
    print(f"folds: {len(fold_boundaries)}")

    fold_results: dict[str, list[float]] = {name: [] for name in STRATEGIES}
    fold_results_15: dict[str, list[float]] = {name: [] for name in STRATEGIES}

    for fb in fold_boundaries[:4]:
        side_path = REPO_ROOT / "data" / "processed" / f"training_v1_sidecar_fold_{fb.fold_idx}.pt"
        side = torch.load(side_path, weights_only=False)
        atr_full = _arr(side["gld_atr_14"]).astype(np.float64)
        h5x = _arr(side["gld_h5_x_vol_high"])
        nlr_full = _arr(side["next_log_return"]).astype(np.float64)
        # is_high_vol per-day propagation
        day_key = bcn // 86_400_000_000_000
        hv_days = set(day_key[np.abs(h5x) > 1e-10].tolist())
        is_hv_full = np.array([int(d) in hv_days for d in day_key], dtype=bool)

        ts = slice(fb.test_start, fb.test_end)
        close_t = close_full[ts]
        # PIT-correct realized return: log(close[i]/close[i-1]). Known at
        # bar i's open. Position at bar i uses realized_ret[<=i-1] only.
        realized_ret = np.zeros_like(close_t, dtype=np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            realized_ret[1:] = np.log(np.maximum(close_t[1:], 1e-12) / np.maximum(close_t[:-1], 1e-12))
        realized_ret = np.nan_to_num(realized_ret, nan=0.0, posinf=0.0, neginf=0.0)
        ctx = {
            "close": close_t,
            "atr": atr_full[ts],
            "nlr": nlr_full[ts],
            "realized_ret": realized_ret,
            "news": news_full[ts],
            "is_high_vol": is_hv_full[ts].astype(np.float64),
        }
        ts_len = fb.test_end - fb.test_start
        print(f"  fold {fb.fold_idx}: test=[{fb.test_start},{fb.test_end}) bars={ts_len} news_frac={ctx['news'].mean():.3f}")

        for name, fn in STRATEGIES.items():
            try:
                pos = fn(ctx)
            except Exception as e:  # noqa: BLE001
                print(f"    [ERR] {name}: {e}")
                fold_results[name].append(float("nan"))
                fold_results_15[name].append(float("nan"))
                continue
            fold_results[name].append(_sharpe(_pnl(pos, ctx["nlr"], 1.0)))
            fold_results_15[name].append(_sharpe(_pnl(pos, ctx["nlr"], 1.5)))

    print()
    print(f"{'strategy':<28s} | {'fold0':>8s} {'fold1':>8s} {'fold2':>8s} {'fold3':>8s} | {'mean1x':>8s} {'med1x':>8s} {'mean1.5x':>9s} {'PASS_G3?':>9s}")
    bh_mean = float(np.mean(fold_results["buy_hold"]))
    rows: list[tuple[float, str]] = []
    for name in STRATEGIES:
        v = fold_results[name]
        v15 = fold_results_15[name]
        mn = float(np.mean(v))
        med = float(np.median(v))
        mn15 = float(np.mean(v15))
        pass_g3 = (mn - bh_mean) >= 0.2
        rows.append((mn, f"{name:<28s} | {v[0]:>+8.3f} {v[1]:>+8.3f} {v[2]:>+8.3f} {v[3]:>+8.3f} | {mn:>+8.3f} {med:>+8.3f} {mn15:>+9.3f} {'YES' if pass_g3 else 'no'}"))
    rows.sort(key=lambda x: -x[0])
    for _, r in rows:
        print(r)

    print()
    print(f"buy_hold mean: {bh_mean:+.4f}")
    print(f"PASS_G3 = beats buy_hold by >= 0.2 Sharpe")
    return 0


if __name__ == "__main__":
    sys.exit(main())
