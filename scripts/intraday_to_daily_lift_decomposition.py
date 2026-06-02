"""Decompose the +51% intraday→daily Sharpe lift.

Reviewer flagged: lag-1 ρ ≈ -0.05 mathematically gives only ~5% Sharpe lift
under iid-bar null + ρ correction. The observed +51% lift must come from
elsewhere. Hypothesis: multi-lag autocorr (within-day mean reversion at
multiple bar-spacings) + position-vs-return autocorrelation interaction.

Compute:
  1. Lag k=1..30 autocorrelation of per-bar PnL by fold
  2. Cumulative variance ratio:
     Var(sum_{N=13} pnl_t) / N · Var(pnl_t) = 1 + 2·sum_{k=1..N-1} (1-k/N) ρ_k
  3. Implied Sharpe lift factor = sqrt(N) / sqrt(N · variance_ratio)
                                = 1 / sqrt(variance_ratio)
"""

from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from nanogld.data.walk_forward_splits import compute_fold_boundaries  # noqa: E402

INTRADAY_BPY = 3276
DAILY_BPY = 252
BASE_COST_BPS = 2.0
GLD_CLOSE_IDX = 3
TARGET_VOL = 0.03


def _arr(x):
    return x.numpy() if hasattr(x, "numpy") else np.asarray(x)


def _rolling_std(x, w):
    n = len(x); out = np.zeros(n)
    cs = np.cumsum(x); cs2 = np.cumsum(x * x)
    for i in range(n):
        a = max(0, i - w + 1); c = i - a + 1
        if c < 2: continue
        m = (cs[i] - (cs[a - 1] if a > 0 else 0)) / c
        m2 = (cs2[i] - (cs2[a - 1] if a > 0 else 0)) / c
        v = max(0, m2 - m * m); out[i] = np.sqrt(v)
    return out


def _vol_target(rr, tv=TARGET_VOL, w=64, cap=1.0):
    rv = _rolling_std(rr, w)
    raw = tv / (rv * np.sqrt(INTRADAY_BPY) + 1e-8)
    p = np.clip(raw, 0, cap); p[~np.isfinite(p)] = 0.0
    return np.concatenate([[0.0], p[:-1]])


def autocorr(x, lag):
    x = x[np.isfinite(x)]
    if len(x) < lag + 2:
        return float("nan")
    return float(np.corrcoef(x[lag:], x[:-lag])[0, 1])


def variance_ratio(pnl, N):
    """Var(sum_N(pnl)) / (N · Var(pnl)) via lagged autocorr."""
    rhos = [autocorr(pnl, k) for k in range(1, N)]
    rhos = [r if np.isfinite(r) else 0.0 for r in rhos]
    factor = 1.0 + 2.0 * sum((1.0 - k / N) * rhos[k - 1] for k in range(1, N))
    return factor, rhos


def main():
    unified = torch.load(REPO_ROOT / "data" / "processed" / "training_v1_unified.pt", weights_only=False)
    features = _arr(unified["features"]).astype(np.float32)
    bcn = _arr(unified["bar_close_utc_ns"]).astype(np.int64)
    close_full = features[:, GLD_CLOSE_IDX].astype(np.float64)
    folds = compute_fold_boundaries(bcn)

    print("Per-fold autocorrelation analysis of vol_target_3pct PnL")
    print()
    BARS_PER_DAY_APPROX = 13  # 6.5h RTH / 30-min = 13 bars

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
        nz = pnl[pnl != 0]

        # Multi-lag autocorr
        rhos = [autocorr(nz, k) for k in range(1, 14)]
        factor_13, _ = variance_ratio(nz, BARS_PER_DAY_APPROX)
        implied_lift = 1.0 / np.sqrt(max(factor_13, 1e-6)) - 1.0  # as fraction

        # Position autocorr (sticky?)
        nz_pos_mask = pos > 0
        pos_nz = pos[nz_pos_mask]
        pos_rhos = [autocorr(pos_nz, k) for k in range(1, 14)]

        # Aggregate to daily
        bcn_t = bcn[ts]
        ts_pd = pd.to_datetime(bcn_t, unit="ns", utc=True)
        df = pd.DataFrame({"ts": ts_pd, "pnl": pnl, "pos": pos})
        df["day"] = df["ts"].dt.date
        daily_pnl = df.groupby("day")["pnl"].sum().values
        bars_per_day_actual = df.groupby("day").size().mean()

        # Actual Sharpe lift
        sr_intra = pnl.mean() / pnl.std(ddof=1) * np.sqrt(INTRADAY_BPY)
        sr_daily = daily_pnl.mean() / daily_pnl.std(ddof=1) * np.sqrt(DAILY_BPY)
        observed_lift = sr_daily / sr_intra - 1.0

        print(f"=== Fold {fb.fold_idx} ===")
        print(f"  bars per day (actual): {bars_per_day_actual:.2f}")
        print(f"  PnL ρ(k=1..13): {[round(r, 3) for r in rhos]}")
        print(f"  Position ρ(k=1..13): {[round(r, 3) for r in pos_rhos]}")
        print(f"  Variance-ratio factor @ N=13: {factor_13:.4f}")
        print(f"  Implied Sharpe lift from autocorr only: {implied_lift*100:+.2f}%")
        print(f"  Observed Sharpe: intraday={sr_intra:+.4f}, daily={sr_daily:+.4f}")
        print(f"  Observed Sharpe lift: {observed_lift*100:+.2f}%")
        print(f"  Gap between implied and observed: {(observed_lift - implied_lift)*100:+.2f}%")
        print()


if __name__ == "__main__":
    main()
