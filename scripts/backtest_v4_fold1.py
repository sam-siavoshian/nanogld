"""Backtest V4 fold 1 LLRD checkpoint (pulled from Spark).

Computes:
- Position weight distribution (mean, abs_mean, saturation %)
- 3-class prediction breakdown
- Intraday Sharpe (BPY=7308) + daily-aggregated Sharpe (BPY=252)
- Per-bucket {news-present, news-absent, both} Sharpes
- Cost-stress at {0.5×, 1.0×, 1.5×} of 2bp base
- Position-saturation diagnostic vs fold 0 collapse pattern

Output: prints summary table; saves results.json for paper update.
"""

from __future__ import annotations
import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from nanogld.data.dataset import NanoGLDDataset
from nanogld.model.model import nanoGLDV1

INTRADAY_BPY = 7308
DAILY_BPY = 252
BASE_COST_BPS = 2.0


def _build_model(cfg):
    m = cfg["model"]
    return nanoGLDV1(
        numeric_dim=int(m["numeric_dim"]),
        d_model=int(m["d_model"]),
        num_heads=int(m["num_heads"]),
        num_transformer_layers=int(m["num_transformer_layers"]),
        num_slstm_layers=int(m["num_slstm_layers"]),
        t_bars=int(m["t_bars"]),
        patch_len=int(m["patch_len"]),
        patch_stride=int(m["patch_stride"]),
        n_classes=int(m["n_classes"]),
        regime_dim=int(m["regime_dim"]),
        d_text=int(m["d_text"]),
        n_news_slots=int(m["n_news_slots"]),
        dropout=float(m["dropout"]),
        drop_path_max=float(m["drop_path_max"]),
        decomposition_kernel=int(m["decomposition_kernel"]),
    )


def _sharpe(r, ann):
    r = r[np.isfinite(r)]
    if len(r) < 2:
        return 0.0
    mu, sigma = r.mean(), r.std(ddof=1)
    if sigma < 1e-12:
        return 0.0
    return float(mu / sigma * np.sqrt(ann))


def _pnl(pos, ret, cost_mult=1.0):
    p = np.nan_to_num(pos)
    r = np.nan_to_num(ret)
    cf = (BASE_COST_BPS * cost_mult) / 10000.0
    prev = np.concatenate([[0.0], p[:-1]])
    return p * r - cf * np.abs(p - prev)


def main():
    cfg_path = REPO_ROOT / "configs" / "v4_wf.yaml"
    ckpt_path = REPO_ROOT / "checkpoints" / "v4_spark" / "fold_1" / "llrd" / "llrd_final.pt"
    unified = REPO_ROOT / "data" / "processed" / "training_v1_unified.pt"
    sidecar = REPO_ROOT / "data" / "processed" / "training_v1_sidecar_fold_1.pt"

    if not ckpt_path.exists():
        print(f"ckpt not found: {ckpt_path}")
        return 1

    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    print("starting backtest", flush=True)
    device = torch.device("cpu")  # force CPU — MPS hangs in background
    print(f"device: {device}", flush=True)

    model = _build_model(cfg).to(device)
    state = torch.load(ckpt_path, map_location=device, weights_only=False)["model_state"]
    model.load_state_dict(state)
    model.train(mode=False)
    print(f"loaded ckpt: {ckpt_path}")

    dl_cfg = cfg["dataloader"]
    ds = NanoGLDDataset(
        unified_path=unified,
        sidecar_path=sidecar,
        split="test",
        lookback_T=int(dl_cfg["lookback_T"]),
        n_news_slots=int(dl_cfg["n_news_slots"]),
        label_mode=dl_cfg["label_mode"],
        fold_idx=1,
        n_folds=int(cfg.get("walk_forward", {}).get("n_folds", 4)),
    )
    print(f"fold 1 test bars: {len(ds)}")

    positions = []
    logits = []
    labels = []
    nlrs = []
    news_pres = []
    bcn_list = []

    loader = torch.utils.data.DataLoader(ds, batch_size=16, shuffle=False, num_workers=0)
    with torch.no_grad():
        for i, batch in enumerate(loader):
            ch = batch["channel_inputs"].float().nan_to_num_(0.0).to(device)
            ne = batch["news_embeddings"].float().to(device)
            nm = batch["news_mask"].float().to(device)
            inp = batch["is_news_present"].long().to(device)
            rv = batch["regime_vec"].float().to(device)
            out = model(
                channel_inputs=ch,
                news_embeddings=ne,
                news_mask=nm,
                is_news_present=inp,
                regime_vec=rv,
            )
            positions.append(out["position_weight"].cpu().numpy())
            logits.append(out["logits_3class"].cpu().numpy())
            labels.append(batch["label_3class"].long().cpu().numpy())
            nlrs.append(batch["next_log_return"].float().cpu().numpy())
            news_pres.append(batch["is_news_present"].long().cpu().numpy())
            if "bar_close_utc_ns" in batch:
                bcn_list.append(batch["bar_close_utc_ns"].long().cpu().numpy())
            if i % 25 == 0:
                print(f"  batch {i}/{len(loader)}")

    positions = np.concatenate(positions, axis=0)
    if positions.ndim > 1:
        positions = positions.squeeze(-1)
    logits = np.concatenate(logits, axis=0)
    labels = np.concatenate(labels, axis=0)
    nlrs = np.concatenate(nlrs, axis=0)
    news_pres = np.concatenate(news_pres, axis=0).astype(bool)
    preds = logits.argmax(axis=1)

    # PnL net of cost
    pnl_1x = _pnl(positions, nlrs, 1.0)
    pnl_05x = _pnl(positions, nlrs, 0.5)
    pnl_15x = _pnl(positions, nlrs, 1.5)

    s_intra_1x = _sharpe(pnl_1x, INTRADAY_BPY)
    s_intra_05x = _sharpe(pnl_05x, INTRADAY_BPY)
    s_intra_15x = _sharpe(pnl_15x, INTRADAY_BPY)

    # Daily aggregation (29 bars/day approx → group every 29 bars)
    n = len(pnl_1x)
    bars_per_day = 29
    n_days = n // bars_per_day
    daily_pnl_1x = pnl_1x[: n_days * bars_per_day].reshape(n_days, bars_per_day).sum(axis=1)
    daily_pnl_05x = pnl_05x[: n_days * bars_per_day].reshape(n_days, bars_per_day).sum(axis=1)
    daily_pnl_15x = pnl_15x[: n_days * bars_per_day].reshape(n_days, bars_per_day).sum(axis=1)

    s_daily_1x = _sharpe(daily_pnl_1x, DAILY_BPY)
    s_daily_05x = _sharpe(daily_pnl_05x, DAILY_BPY)
    s_daily_15x = _sharpe(daily_pnl_15x, DAILY_BPY)

    # Per-bucket Sharpe
    pnl_news = pnl_1x.copy()
    pnl_news[~news_pres] = 0
    pnl_nonews = pnl_1x.copy()
    pnl_nonews[news_pres] = 0
    s_news_intra = _sharpe(pnl_news[news_pres], INTRADAY_BPY) if news_pres.sum() > 1 else 0
    s_nonews_intra = _sharpe(pnl_nonews[~news_pres], INTRADAY_BPY) if (~news_pres).sum() > 1 else 0

    # Position saturation diagnostic
    abs_pos_mean = float(np.abs(positions).mean())
    pct_sat_99 = float((np.abs(positions) > 0.99).mean() * 100)
    pct_sat_95 = float((np.abs(positions) > 0.95).mean() * 100)

    # Buy-hold comparison (positions = +1)
    pnl_bh = nlrs.copy()
    s_bh_intra = _sharpe(pnl_bh, INTRADAY_BPY)
    daily_bh = pnl_bh[: n_days * bars_per_day].reshape(n_days, bars_per_day).sum(axis=1)
    s_bh_daily = _sharpe(daily_bh, DAILY_BPY)

    print()
    print("===== V4 FOLD 1 BACKTEST RESULTS =====")
    print(f"test bars: {n}, days: {n_days}")
    print()
    print("Position diagnostics:")
    print(f"  abs_pos_mean:      {abs_pos_mean:.4f}")
    print(f"  pct |pos| > 0.99:  {pct_sat_99:.1f}%  (fold 0 collapsed at 100% saturation)")
    print(f"  pct |pos| > 0.95:  {pct_sat_95:.1f}%")
    print(f"  mean position:     {positions.mean():+.4f}")
    print(f"  std position:      {positions.std():.4f}")
    print(f"  long (>0):  {(positions > 0).mean()*100:.1f}%")
    print(f"  short (<0): {(positions < 0).mean()*100:.1f}%")
    print()
    print("Prediction breakdown:")
    print(f"  acc:        {(preds == labels).mean():.4f} (random=0.333)")
    for c in [0, 1, 2]:
        print(f"  pred  class {c}: {(preds == c).mean()*100:.1f}%")
        print(f"  label class {c}: {(labels == c).mean()*100:.1f}%")
    print()
    print("Sharpe (V4 model, intraday + daily-agg) vs Buy-Hold:")
    print(f"  V4    intraday  Sharpe 1.0x cost: {s_intra_1x:+.4f}  | BH: {s_bh_intra:+.4f}")
    print(f"  V4    intraday  Sharpe 0.5x cost: {s_intra_05x:+.4f}")
    print(f"  V4    intraday  Sharpe 1.5x cost: {s_intra_15x:+.4f}")
    print(f"  V4    daily-agg Sharpe 1.0x cost: {s_daily_1x:+.4f}  | BH: {s_bh_daily:+.4f}")
    print(f"  V4    daily-agg Sharpe 0.5x cost: {s_daily_05x:+.4f}")
    print(f"  V4    daily-agg Sharpe 1.5x cost: {s_daily_15x:+.4f}")
    print()
    print("Per-bucket (1.0x cost, intraday-frequency):")
    print(f"  news-present (n={news_pres.sum()}):  Sharpe = {s_news_intra:+.4f}")
    print(f"  news-absent  (n={(~news_pres).sum()}): Sharpe = {s_nonews_intra:+.4f}")
    print()
    print("Compare to fold 0:")
    print(f"  fold 0 Sharpe (collapsed): -0.863  vs BH +0.859")
    print(f"  fold 1 Sharpe:             {s_intra_1x:+.4f}  vs BH {s_bh_intra:+.4f}")
    print(f"  delta vs BH (fold 1):      {s_intra_1x - s_bh_intra:+.4f}")
    print()

    # Verdict
    collapsed = pct_sat_99 > 90 and s_intra_1x < s_bh_intra - 0.5
    print("===== VERDICT =====")
    if collapsed:
        print("FOLD 1 ALSO COLLAPSED (>90% saturation + Sharpe << BH)")
    elif s_intra_1x < s_bh_intra - 0.2:
        print("FOLD 1: model produces negative alpha vs BH (no collapse pattern but bad Sharpe)")
    elif abs(s_intra_1x - s_bh_intra) < 0.2:
        print("FOLD 1: model matches BH (no incremental edge)")
    elif s_intra_1x > s_bh_intra + 0.2:
        print("FOLD 1: model BEATS BH — first positive V4 result on any fold")
    else:
        print("FOLD 1: ambiguous, see detailed numbers")

    out = {
        "fold": 1,
        "test_bars": n,
        "test_days": n_days,
        "position": {
            "abs_pos_mean": abs_pos_mean,
            "pct_sat_99": pct_sat_99,
            "pct_sat_95": pct_sat_95,
            "mean": float(positions.mean()),
            "std": float(positions.std()),
            "pct_long": float((positions > 0).mean() * 100),
            "pct_short": float((positions < 0).mean() * 100),
        },
        "accuracy": float((preds == labels).mean()),
        "sharpe": {
            "v4_intraday_1x": s_intra_1x,
            "v4_intraday_05x": s_intra_05x,
            "v4_intraday_15x": s_intra_15x,
            "v4_daily_1x": s_daily_1x,
            "v4_daily_05x": s_daily_05x,
            "v4_daily_15x": s_daily_15x,
            "bh_intraday_1x": s_bh_intra,
            "bh_daily_1x": s_bh_daily,
            "news_present": s_news_intra,
            "news_absent": s_nonews_intra,
        },
        "delta_vs_bh_intraday": s_intra_1x - s_bh_intra,
        "verdict": "collapsed" if collapsed else ("matches_bh" if abs(s_intra_1x - s_bh_intra) < 0.2 else ("negative_alpha" if s_intra_1x < s_bh_intra else "positive_alpha")),
    }
    out_path = REPO_ROOT / "paper" / "v4_fold1_backtest.json"
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {out_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
