"""Recompute Deflated Sharpe Ratio with realistic n_trials count.

Reviewer flagged n_trials=11 as implausibly low. Realistic count of strategies
tested in this paper:

  §6.4 baseline ladder (real evaluations only):  ~7
  §6.8 vol-target sweep + composites:            ~9 (vt_{3,5,7,10,15}% + 4 composites)
  §6.9 daily-gold strategies (sotaa_sweep):      ~10 (vt × multiple windows + atr + ma + momentum)
  §6.10 multi-asset futures sweep (TSMOM grid):  ~32 (4 lookbacks × 4 vol-targets × 2 weightings)
  §6.11 14-strategy meta + per-strat combo:      ~14 + 3 meta-combos
  §6.12 multimodal LSTM × 4 sizing schemes:      ~4
  Frequency selection (intraday vs daily):       ×2
  Cost convention sweep:                         ×4

Total candidate strategy/parameter combos backtested under the same WF
geometry ≈ 90-120. Use n_trials = 100 as a defensible upper bound;
also report at n_trials = 50 (lower bound, ignoring some grid-search
redundancy) and n_trials = 11 (the paper's original claim) for comparison.
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

INTRADAY_BPY = 3276
DAILY_BPY = 252
BASE_COST_BPS = 2.0
GLD_CLOSE_IDX = 3
TARGET_VOL = 0.03
VT_WINDOW = 64


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


def _rolling_std(x, w):
    n = len(x)
    out = np.zeros(n)
    cs = np.cumsum(x); cs2 = np.cumsum(x * x)
    for i in range(n):
        a = max(0, i - w + 1); c = i - a + 1
        if c < 2: continue
        m = (cs[i] - (cs[a - 1] if a > 0 else 0)) / c
        m2 = (cs2[i] - (cs2[a - 1] if a > 0 else 0)) / c
        v = max(0, m2 - m * m); out[i] = np.sqrt(v)
    return out


def _vol_target(rr, tv=TARGET_VOL, w=VT_WINDOW, cap=1.0):
    rv = _rolling_std(rr, w)
    raw = tv / (rv * np.sqrt(INTRADAY_BPY) + 1e-8)
    p = np.clip(raw, 0, cap); p[~np.isfinite(p)] = 0.0
    return np.concatenate([[0.0], p[:-1]])


def deflated_sharpe(sr_obs: float, n_obs: int, returns: np.ndarray, n_trials: int) -> tuple[float, float]:
    """Bailey-Lopez de Prado DSR.

    Returns (DSR_probability, E[max SR | null with n_trials]) where DSR is the
    probability that the observed SR exceeds the expected maximum under null.
    """
    from scipy import stats
    gamma_const = 0.5772156649
    skew = float(stats.skew(returns)) if len(returns) > 3 else 0.0
    kurt = float(stats.kurtosis(returns, fisher=True)) if len(returns) > 3 else 0.0
    # E[max SR | null with N trials]
    e_max = (1 - gamma_const) * stats.norm.ppf(1 - 1.0 / n_trials) + gamma_const * stats.norm.ppf(1 - 1.0 / (n_trials * np.e))
    # Lo (2002) variance of SR estimator
    var_sr = (1.0 - skew * sr_obs + (kurt - 1.0) / 4.0 * sr_obs ** 2) / (n_obs - 1)
    sigma_sr = max(np.sqrt(max(var_sr, 1e-12)), 1e-12)
    dsr_stat = (sr_obs - e_max) / sigma_sr
    dsr_prob = float(stats.norm.cdf(dsr_stat))
    return float(dsr_prob), float(e_max)


def main():
    unified = torch.load(REPO_ROOT / "data" / "processed" / "training_v1_unified.pt", weights_only=False)
    features = _arr(unified["features"]).astype(np.float32)
    bcn = _arr(unified["bar_close_utc_ns"]).astype(np.int64)
    close_full = features[:, GLD_CLOSE_IDX].astype(np.float64)
    folds = compute_fold_boundaries(bcn)

    all_intraday = []
    all_daily = []
    for fb in folds[:4]:
        side = torch.load(REPO_ROOT / "data" / "processed" / f"training_v1_sidecar_fold_{fb.fold_idx}.pt", weights_only=False)
        nlr = _arr(side["next_log_return"]).astype(np.float64)
        ts = slice(fb.test_start, fb.test_end)
        close_t = close_full[ts]
        rr = np.zeros_like(close_t)
        rr[1:] = np.log(np.maximum(close_t[1:], 1e-12) / np.maximum(close_t[:-1], 1e-12))
        rr = np.nan_to_num(rr)
        pos = _vol_target(rr)
        cf = BASE_COST_BPS * 1e-4
        prev = np.concatenate([[0.0], pos[:-1]])
        pnl = pos * nlr[ts] - cf * np.abs(pos - prev)
        all_intraday.append(pnl)
        # Daily aggregate
        bcn_t = bcn[ts]
        ts_pd = pd.to_datetime(bcn_t, unit="ns", utc=True)
        df = pd.DataFrame({"ts": ts_pd, "pnl": pnl})
        df["day"] = df["ts"].dt.date
        daily = df.groupby("day")["pnl"].sum().values
        all_daily.append(daily)

    chain_i = np.concatenate(all_intraday)
    chain_d = np.concatenate(all_daily)
    sr_i = _sharpe(chain_i, INTRADAY_BPY)
    sr_d = _sharpe(chain_d, DAILY_BPY)
    print(f"chain Sharpe: intraday={sr_i:+.4f} (n_obs={len(chain_i)}), daily={sr_d:+.4f} (n_obs={len(chain_d)})")
    print()
    print(f"{'n_trials':>10s} {'E[max|null,intra]':>20s} {'DSR_intraday':>14s} {'E[max|null,daily]':>20s} {'DSR_daily':>11s}")
    for n_trials in [11, 25, 50, 100, 200]:
        dsr_i, emax_i = deflated_sharpe(sr_i, len(chain_i), chain_i, n_trials)
        dsr_d, emax_d = deflated_sharpe(sr_d, len(chain_d), chain_d, n_trials)
        print(f"  {n_trials:>8d} {emax_i:>18.4f} {dsr_i:>14.4f} {emax_d:>18.4f} {dsr_d:>11.4f}")
    print()
    print(">0.95 DSR = passes multiple-testing gate at 5% sig")


if __name__ == "__main__":
    main()
