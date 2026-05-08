"""Top-level calibration orchestrator.

Run sequence (per fold, post-training):
  1. T-scaling on val_b → fitted T in [0.7, 3.0]
  2. RAPS Mondrian quantile fit on val_c → q_hat per class
  3. Laplace last-layer fit on train (or train_subset) → posterior
  4. AgACI online wrapper initialized with target alpha=0.10

Saves artifacts to `output_dir / "calibration_<fold>"`:
  t_scaler.pt
  raps_quantiles.json
  laplace.joblib
  agaci_state.json
  meta.json

Spec: plan/V1-SPEC.md §5.6.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import Tensor

from nanogld.calibration.agaci import AgACI
from nanogld.calibration.ece import classwise_ada_ece, per_bucket_ece
from nanogld.calibration.raps import fit_raps_quantile
from nanogld.calibration.temperature_scaling import TemperatureScaler


@dataclass(frozen=True)
class CalibrationConfig:
    """Calibration run config."""

    fold_idx: int
    output_dir: Path
    alpha_target: float = 0.10
    raps_lambda: float = 0.01
    raps_kreg: int = 1
    t_scaling_max_iter: int = 50


@dataclass(frozen=True)
class CalibrationArtifacts:
    """Paths to the four saved calibration artifacts."""

    t_scaler_path: Path
    raps_quantiles_path: Path
    agaci_state_path: Path
    meta_path: Path


def _atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp, path)


def _atomic_save_torch(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def calibrate(
    cfg: CalibrationConfig,
    val_b_logits: Tensor,
    val_b_labels: Tensor,
    val_c_logits: Tensor,
    val_c_labels: Tensor,
    val_c_news_present_mask: Tensor | None = None,
) -> CalibrationArtifacts:
    """Fit T-scaling, RAPS, and AgACI; save artifacts.

    Args:
        cfg: calibration config.
        val_b_logits, val_b_labels: temperature-scaling set.
        val_c_logits, val_c_labels: conformal set.
        val_c_news_present_mask: optional bool mask for per-bucket ECE
            diagnostic.

    Returns:
        CalibrationArtifacts with paths to the four saved files.
    """
    out_dir = cfg.output_dir / f"calibration_{cfg.fold_idx}"
    out_dir.mkdir(parents=True, exist_ok=True)

    t_scaler = TemperatureScaler(init_T=1.0)
    fitted_T = t_scaler.fit(val_b_logits, val_b_labels, max_iter=cfg.t_scaling_max_iter)
    t_path = out_dir / "t_scaler.pt"
    _atomic_save_torch(t_path, {"log_T": t_scaler.log_T.detach(), "T": fitted_T})

    val_c_probs = t_scaler.calibrated_probs(val_c_logits)
    q_hats = fit_raps_quantile(
        val_c_probs,
        val_c_labels,
        alpha=cfg.alpha_target,
        mondrian=True,
        lambda_reg=cfg.raps_lambda,
        k_reg=cfg.raps_kreg,
    )
    raps_path = out_dir / "raps_quantiles.json"
    _atomic_write_json(raps_path, {"q_hats": {str(k): v for k, v in q_hats.items()}})

    agaci = AgACI(alpha_target=cfg.alpha_target)
    agaci_path = out_dir / "agaci_state.json"
    _atomic_write_json(agaci_path, agaci.state_dict())

    macro_ece, worst_ece = classwise_ada_ece(val_c_probs, val_c_labels)
    meta = {
        "fold_idx": cfg.fold_idx,
        "fitted_T": fitted_T,
        "macro_ada_ece": macro_ece,
        "worst_ada_ece": worst_ece,
    }
    if val_c_news_present_mask is not None:
        meta["per_bucket_ece"] = per_bucket_ece(val_c_probs, val_c_labels, val_c_news_present_mask)
    meta["config"] = {**asdict(cfg), "output_dir": str(cfg.output_dir)}

    meta_path = out_dir / "meta.json"
    _atomic_write_json(meta_path, meta)

    return CalibrationArtifacts(
        t_scaler_path=t_path,
        raps_quantiles_path=raps_path,
        agaci_state_path=agaci_path,
        meta_path=meta_path,
    )


def load_calibration(artifact_dir: Path) -> dict:
    """Load saved calibration artifacts back into a usable dict."""
    artifact_dir = Path(artifact_dir)
    if not artifact_dir.exists():
        raise FileNotFoundError(f"calibration dir not found: {artifact_dir}")

    t_scaler = torch.load(artifact_dir / "t_scaler.pt", weights_only=False)
    with open(artifact_dir / "raps_quantiles.json") as f:
        raps = json.load(f)
    with open(artifact_dir / "agaci_state.json") as f:
        agaci_state = json.load(f)
    with open(artifact_dir / "meta.json") as f:
        meta = json.load(f)

    agaci = AgACI(alpha_target=meta["config"]["alpha_target"])
    agaci.load_state_dict(agaci_state)
    q_hats = {int(k) if k != "-1" else -1: v for k, v in raps["q_hats"].items()}

    return {
        "t_scaler_T": t_scaler["T"],
        "t_scaler_log_T": t_scaler["log_T"],
        "raps_quantiles": q_hats,
        "agaci": agaci,
        "meta": meta,
    }
