"""Quick diagnostic for V4 fold 0 inference.

Runs the model on fold 0's test split (3721 bars) and prints:
- Position distribution (long / short / zero counts)
- Predicted class distribution (UP / FLAT / DOWN counts vs labels)
- Position weight summary (mean, abs-mean, std)
- Correlation between predicted position and next_log_return (should be > 0)

Goal: figure out if V4 negative Sharpe is from
(a) shorting an uptrend (sign bug),
(b) constant zero positions (conformal floor too strict), or
(c) random noise (model did not learn).

Run on Spark CPU. No GPU. Self-contained, no calibration step.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from nanogld.data.dataset import NanoGLDDataset  # noqa: E402
from nanogld.model.model import nanoGLDV1  # noqa: E402


def _build_model(cfg: dict[str, Any]) -> torch.nn.Module:
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


def main() -> int:
    cfg_path = REPO_ROOT / "configs" / "v4_wf.yaml"
    ckpt_path = REPO_ROOT / "checkpoints" / "v4" / "fold_0" / "llrd" / "llrd_final.pt"
    unified = REPO_ROOT / "data" / "processed" / "training_v1_unified.pt"
    sidecar = REPO_ROOT / "data" / "processed" / "training_v1_sidecar_fold_0.pt"

    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cpu")
    model = _build_model(cfg).to(device)
    state = torch.load(ckpt_path, map_location=device, weights_only=False)["model_state"]
    model.load_state_dict(state)
    model.train(mode=False)

    dl_cfg = cfg["dataloader"]
    ds = NanoGLDDataset(
        unified_path=unified,
        sidecar_path=sidecar,
        split="test",
        lookback_T=int(dl_cfg["lookback_T"]),
        n_news_slots=int(dl_cfg["n_news_slots"]),
        label_mode=dl_cfg["label_mode"],
        fold_idx=0,
        n_folds=int(cfg.get("walk_forward", {}).get("n_folds", 4)),
    )
    print(f"dataset: {len(ds)} bars")

    logits_list: list[np.ndarray] = []
    pos_list: list[np.ndarray] = []
    label_list: list[np.ndarray] = []
    nlr_list: list[np.ndarray] = []
    news_pres_list: list[np.ndarray] = []

    loader = torch.utils.data.DataLoader(ds, batch_size=16, shuffle=False, num_workers=0)
    with torch.no_grad():
        for i, batch in enumerate(loader):
            ch = batch["channel_inputs"].float().nan_to_num_(0.0)
            ne = batch["news_embeddings"].float()
            nm = batch["news_mask"].float()
            inp = batch["is_news_present"].long()
            rv = batch["regime_vec"].float()
            out = model(
                channel_inputs=ch,
                news_embeddings=ne,
                news_mask=nm,
                is_news_present=inp,
                regime_vec=rv,
            )
            logits_list.append(out["logits_3class"].cpu().numpy())
            pos_list.append(out["position_weight"].cpu().numpy())
            label_list.append(batch["label_3class"].long().cpu().numpy())
            nlr_list.append(batch["next_log_return"].float().cpu().numpy())
            news_pres_list.append(batch["is_news_present"].long().cpu().numpy())
            if i % 50 == 0:
                print(f"  batch {i}/{len(loader)}")

    logits = np.concatenate(logits_list, axis=0)
    positions = np.concatenate(pos_list, axis=0).squeeze(-1) if pos_list[0].ndim > 1 else np.concatenate(pos_list, axis=0)
    labels = np.concatenate(label_list, axis=0)
    nlrs = np.concatenate(nlr_list, axis=0)
    news_pres = np.concatenate(news_pres_list, axis=0)
    preds = logits.argmax(axis=1)

    print("\n===== POSITION WEIGHT DISTRIBUTION =====")
    print(f"  n_bars:     {len(positions)}")
    print(f"  mean:       {positions.mean():+.6f}")
    print(f"  abs_mean:   {np.abs(positions).mean():.6f}")
    print(f"  std:        {positions.std():.6f}")
    print(f"  min,max:    {positions.min():+.6f}, {positions.max():+.6f}")
    print(f"  long  (>0): {(positions > 0).sum()} ({(positions > 0).mean()*100:.1f}%)")
    print(f"  short (<0): {(positions < 0).sum()} ({(positions < 0).mean()*100:.1f}%)")
    print(f"  zero  (==0):{(positions == 0).sum()} ({(positions == 0).mean()*100:.1f}%)")

    print("\n===== PRED vs LABEL =====")
    print("class 0=DOWN, 1=FLAT, 2=UP (V1-SPEC §4)")
    for c in [0, 1, 2]:
        print(f"  label  class {c}: count={(labels == c).sum()} ({(labels == c).mean()*100:.1f}%)")
    for c in [0, 1, 2]:
        print(f"  pred   class {c}: count={(preds == c).sum()} ({(preds == c).mean()*100:.1f}%)")
    acc = (preds == labels).mean()
    print(f"  raw accuracy: {acc:.4f} (random=0.333)")

    print("\n===== POSITION X NEXT_LOG_RETURN =====")
    print(f"  corr(position, next_log_return) = {np.corrcoef(positions, nlrs)[0,1]:+.4f}  (>0 means positions align with realized returns)")
    print(f"  realized PnL ignore-cost = sum(position * next_log_return) = {(positions * nlrs).sum():+.6f}")
    print(f"  mean next_log_return     = {nlrs.mean():+.8f}  (>0 = uptrend, market bias)")
    print(f"  mean position            = {positions.mean():+.6f}  (compare sign vs market)")

    print("\n===== NEWS PRESENT BREAKDOWN =====")
    is_pres = news_pres.astype(bool)
    print(f"  present bars: {is_pres.sum()} ({is_pres.mean()*100:.1f}%)")
    print(f"  pos mean (news present) = {positions[is_pres].mean():+.6f}")
    print(f"  pos mean (news absent)  = {positions[~is_pres].mean():+.6f}")

    print("\n===== INTERPRETATION HINTS =====")
    if (positions == 0).mean() > 0.90:
        print("  >90% zero positions: conformal floor / sizer is gating everything. cal step or floor too strict.")
    elif positions.mean() < -0.05 and nlrs.mean() > 0:
        print("  Model is SHORT during an uptrend. Sign bug suspected (label / position-weight tanh inversion).")
    elif acc < 0.34:
        print("  Predictions ~random. Model did not learn during LLRD. SSL warm-start may be wrong.")
    elif np.abs(np.corrcoef(positions, nlrs)[0,1]) < 0.02:
        print("  Position weights uncorrelated with realized returns. Calibration may fix or model truly random.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
