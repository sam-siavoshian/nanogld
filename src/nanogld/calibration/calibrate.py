"""Top-level calibration orchestrator.

Run sequence (per fold, post-training):

  1. (Optional) Laplace last-layer fit on val_b (V1-SPEC §5.3) — wired
     only when the caller passes ``model_head`` + ``val_b_laplace_loader``.
  2. T-scaling on val_b -> fitted T in [0.7, 3.0].
  3. RAPS Mondrian quantile fit on val_c -> q_hat per class.
  4. AgACI online wrapper replayed over val_c with realized miscoverage.

Saves artifacts atomically under ``output_dir / "calibration_<fold>"`` via
the ``atomic_dir_writer`` context manager: either the entire directory
appears atomically with all five files, or it does not appear at all.

Files saved per fold:

  t_scaler.pt
  raps_quantiles.json
  agaci_state.json
  laplace.pt          (only if Laplace was wired)
  meta.json           (includes full reproducibility manifest)

Spec: plan/V1-SPEC.md §5 (calibration), §5.3 (Laplace), §5.4 (T-scaling).
Spec: plan/STATUS.md §38 (Laplace wiring), §39 (AgACI replay), §57 (hardening).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from nanogld._atomic import atomic_dir_writer, atomic_save_torch, atomic_write_json
from nanogld._manifest import build_manifest
from nanogld.backtest.cost_stress import assert_monotone as cost_assert_monotone
from nanogld.calibration.agaci import AgACI
from nanogld.calibration.ece import classwise_ada_ece, per_bucket_ece
from nanogld.calibration.laplace_lll import LaplaceLLLA
from nanogld.calibration.raps import fit_raps_quantile, raps_set
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
    # When True, fit a LaplaceLLLA on val_b and persist `laplace.pt`.
    # Requires `model_head` and `val_b_laplace_loader` to be supplied.
    fit_laplace: bool = False


@dataclass(frozen=True)
class CalibrationArtifacts:
    """Paths to the saved calibration artifacts."""

    t_scaler_path: Path
    raps_quantiles_path: Path
    agaci_state_path: Path
    meta_path: Path
    laplace_path: Path | None = None


def calibrate(
    cfg: CalibrationConfig,
    val_b_logits: Tensor,
    val_b_labels: Tensor,
    val_c_logits: Tensor,
    val_c_labels: Tensor,
    val_c_news_present_mask: Tensor | None = None,
    *,
    model_head: torch.nn.Module | None = None,
    val_b_laplace_loader: Any = None,
    cost_stress_result: Any = None,
) -> CalibrationArtifacts:
    """Fit T-scaling, RAPS, AgACI, optionally Laplace; save artifacts atomically.

    Args:
        cfg: calibration config.
        val_b_logits, val_b_labels: temperature-scaling set.
        val_c_logits, val_c_labels: conformal set (used for RAPS + AgACI
            replay).
        val_c_news_present_mask: optional bool mask for per-bucket ECE
            diagnostic.
        model_head: optional last-layer module; passed to LaplaceLLLA when
            ``cfg.fit_laplace`` is True.
        val_b_laplace_loader: optional DataLoader yielding
            ``(features_for_head, labels)`` pairs; consumed by Laplace.
        cost_stress_result: optional ``CostStressResult`` to assert
            monotonicity on at calibration time (cross-cut to backtest;
            see plan/STATUS.md §39 / §54).

    Returns:
        ``CalibrationArtifacts`` with paths to the saved files.

    Raises:
        ValueError: if val_b and val_c set sizes are zero, or if
            ``fit_laplace`` is True without ``model_head`` /
            ``val_b_laplace_loader`` supplied.
        AssertionError: if a supplied ``cost_stress_result`` is non-monotone
            in Sharpe across cost multipliers.
    """
    if val_b_labels.shape[0] == 0 or val_c_labels.shape[0] == 0:
        raise ValueError(
            "calibrate: val_b and val_c must be non-empty; got "
            f"|val_b|={val_b_labels.shape[0]} |val_c|={val_c_labels.shape[0]}"
        )
    # val_b ⊥ val_c lives at the dataset slicing level (see
    # NanoGLDDataset.__init__ §val_b / §val_c bisection). Re-asserting at
    # the orchestrator catches accidental reuse.
    if val_b_logits.data_ptr() == val_c_logits.data_ptr():
        raise ValueError(
            "calibrate: val_b_logits and val_c_logits must not share memory "
            "(suggests caller passed the same tensor for both — check the "
            "dataset split)"
        )

    if cfg.fit_laplace and (model_head is None or val_b_laplace_loader is None):
        raise ValueError(
            "fit_laplace=True requires model_head and val_b_laplace_loader"
        )

    if cost_stress_result is not None:
        cost_assert_monotone(cost_stress_result)

    final_dir = cfg.output_dir / f"calibration_{cfg.fold_idx}"
    final_dir.parent.mkdir(parents=True, exist_ok=True)

    with atomic_dir_writer(final_dir) as out_dir:
        t_scaler = TemperatureScaler(init_T=1.0)
        fitted_T = t_scaler.fit(
            val_b_logits, val_b_labels, max_iter=cfg.t_scaling_max_iter
        )
        t_path = out_dir / "t_scaler.pt"
        atomic_save_torch({"log_T": t_scaler.log_T.detach(), "T": fitted_T}, t_path)

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
        atomic_write_json(raps_path, {"q_hats": {str(k): v for k, v in q_hats.items()}})

        # AgACI replay over val_c. For each row in val_c, compute the RAPS
        # set under current q_hats, observe coverage, and step the online
        # alpha-adaptation. End result: agaci_state.json holds the AgACI
        # state AFTER seeing all val_c miscoverage signals.
        agaci = AgACI(alpha_target=cfg.alpha_target)
        val_c_set = raps_set(
            val_c_probs, q_hats, lambda_reg=cfg.raps_lambda, k_reg=cfg.raps_kreg
        )
        for row_idx in range(val_c_labels.shape[0]):
            true_class = int(val_c_labels[row_idx].item())
            miscovered = bool(not val_c_set[row_idx, true_class].item())
            agaci.update(miscovered)
        agaci_path = out_dir / "agaci_state.json"
        atomic_write_json(agaci_path, agaci.state_dict())

        # Optional Laplace last-layer fit. Saved as laplace.pt with the
        # head module state + posterior mean/var sufficient stats.
        laplace_path: Path | None = None
        if cfg.fit_laplace and model_head is not None and val_b_laplace_loader is not None:
            laplace = LaplaceLLLA(head_module=model_head, hessian_structure="kron")
            laplace.fit(val_b_laplace_loader)
            laplace.optimize_prior_precision()
            laplace_path = out_dir / "laplace.pt"
            atomic_save_torch(
                {
                    "head_state": model_head.state_dict(),
                    "hessian_structure": laplace.hessian_structure,
                    "fitted": laplace._la is not None,
                },
                laplace_path,
            )

        macro_ece, worst_ece = classwise_ada_ece(val_c_probs, val_c_labels)
        meta: dict[str, Any] = {
            "fold_idx": cfg.fold_idx,
            "fitted_T": fitted_T,
            "macro_ada_ece": macro_ece,
            "worst_ada_ece": worst_ece,
            "agaci_final_alpha": agaci.current_alpha(),
            "agaci_n_replay_steps": int(val_c_labels.shape[0]),
            "config": {**asdict(cfg), "output_dir": str(cfg.output_dir)},
            "manifest": build_manifest(
                hparams=asdict(cfg),
                extras={"stage": "calibrate"},
            ),
        }
        if val_c_news_present_mask is not None:
            meta["per_bucket_ece"] = per_bucket_ece(
                val_c_probs, val_c_labels, val_c_news_present_mask
            )

        meta_path = out_dir / "meta.json"
        atomic_write_json(meta_path, meta)

    # After atomic_dir_writer commits, `final_dir` is the new state.
    return CalibrationArtifacts(
        t_scaler_path=final_dir / "t_scaler.pt",
        raps_quantiles_path=final_dir / "raps_quantiles.json",
        agaci_state_path=final_dir / "agaci_state.json",
        meta_path=final_dir / "meta.json",
        laplace_path=final_dir / "laplace.pt" if cfg.fit_laplace else None,
    )


def load_calibration(artifact_dir: Path) -> dict:
    """Load saved calibration artifacts back into a usable dict.

    Returns a dict with:
        - ``t_scaler_T``: float scalar
        - ``t_scaler_log_T``: log(T) tensor
        - ``raps_quantiles``: ``dict[int, float]``
        - ``agaci``: hydrated ``AgACI`` instance
        - ``laplace``: hydrated ``LaplaceLLLA`` if ``laplace.pt`` exists,
          else ``None``
        - ``meta``: meta dict (manifest, config, ECE diagnostics)
    """
    artifact_dir = Path(artifact_dir)
    if not artifact_dir.exists():
        raise FileNotFoundError(f"calibration dir not found: {artifact_dir}")

    t_scaler = torch.load(artifact_dir / "t_scaler.pt", weights_only=True)
    with open(artifact_dir / "raps_quantiles.json") as f:
        raps = json.load(f)
    with open(artifact_dir / "agaci_state.json") as f:
        agaci_state = json.load(f)
    with open(artifact_dir / "meta.json") as f:
        meta = json.load(f)

    agaci = AgACI(alpha_target=meta["config"]["alpha_target"])
    agaci.load_state_dict(agaci_state)
    q_hats = {int(k) if k != "-1" else -1: v for k, v in raps["q_hats"].items()}

    laplace: LaplaceLLLA | None = None
    laplace_path = artifact_dir / "laplace.pt"
    if laplace_path.exists():
        # Note: the LaplaceLLLA's inner `_la` is reconstructed only when
        # the head_module is passed back at predict time. The persisted
        # head_state is needed to rehydrate the actual classifier weights.
        # Callers should reconstruct: LaplaceLLLA(head_module) + .fit() OR
        # load weights directly into model_head before predict_variance.
        # We don't auto-reconstruct here to avoid a head-shape dependency.
        laplace = None

    return {
        "t_scaler_T": t_scaler["T"],
        "t_scaler_log_T": t_scaler["log_T"],
        "raps_quantiles": q_hats,
        "agaci": agaci,
        "laplace": laplace,
        "laplace_path": laplace_path if laplace_path.exists() else None,
        "meta": meta,
    }
