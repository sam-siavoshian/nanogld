"""Inference-time calibration orchestrator.

Given raw model logits and a loaded calibration artifact dict (output of
``load_calibration``), produce the four signals the sizer needs:

- ``probs``: T-scaled softmax probabilities, shape ``(B, n_classes)``.
- ``prediction_set``: RAPS conformal prediction set, shape ``(B, n_classes)``
  bool. ``True`` iff the class is in the set at the configured target alpha.
- ``aps_lower_bound``: per-row lower bound on the top-class probability
  inside the prediction set. Defined as the *minimum* probability of any
  class included in the set (Angelopoulos 2020 APS lower-bound semantics).
  This is what the sizer's conformal floor compares against the 0.40 cutoff
  (V1-SPEC §10.1 line 435).
- ``laplace_var``: per-row epistemic variance estimate from the Laplace
  last-layer module if a fitted ``LaplaceLLLA`` is supplied; ``None``
  otherwise.

This module is intentionally side-effect-free: it does not mutate the
calibration artifacts (the AgACI online state is updated by the backtest
loop with realized miscoverage signals, not at inference time).

Spec: plan/V1-SPEC.md §5 (calibration stack) + §10.1 (sizing conformal floor).
"""

from __future__ import annotations

from typing import Any, NamedTuple

import torch
import torch.nn.functional as F
from torch import Tensor

from nanogld.calibration.laplace_lll import LaplaceLLLA
from nanogld.calibration.raps import raps_set


class CalibratedPrediction(NamedTuple):
    """Output of :func:`predict_calibrated`."""

    probs: Tensor
    prediction_set: Tensor
    aps_lower_bound: Tensor
    laplace_var: Tensor | None


def predict_calibrated(
    model_logits: Tensor,
    calib: dict[str, Any],
    *,
    laplace: LaplaceLLLA | None = None,
    lambda_reg: float = 0.01,
    k_reg: int = 1,
) -> CalibratedPrediction:
    """Run the full T-scale -> RAPS -> APS-lower-bound pipeline.

    Args:
        model_logits: ``(B, n_classes)`` raw logits from the model's
            classification head.
        calib: dict returned by :func:`load_calibration`. Must contain
            ``t_scaler_T`` and ``raps_quantiles``. ``agaci`` is read for
            reporting but not mutated.
        laplace: optional fitted :class:`LaplaceLLLA` module. If provided,
            ``laplace_var`` is filled with per-row posterior variance;
            otherwise ``None``.
        lambda_reg: RAPS size penalty (must match calibration-time value).
        k_reg: RAPS rank threshold (must match calibration-time value).

    Returns:
        :class:`CalibratedPrediction` named tuple.

    Raises:
        ValueError: if model_logits is empty or the calibration dict is
            missing required keys.
    """
    if model_logits.ndim != 2:
        raise ValueError(
            f"model_logits must be (B, C); got shape {tuple(model_logits.shape)}"
        )
    if model_logits.shape[0] == 0:
        raise ValueError("model_logits has zero batch dimension")
    if "t_scaler_T" not in calib:
        raise ValueError("calib dict missing required key 't_scaler_T'")
    if "raps_quantiles" not in calib:
        raise ValueError("calib dict missing required key 'raps_quantiles'")

    t = float(calib["t_scaler_T"])
    if t <= 0:
        raise ValueError(f"calib t_scaler_T must be positive; got {t}")

    probs = F.softmax(model_logits / t, dim=-1)

    q_hats = calib["raps_quantiles"]
    prediction_set = raps_set(
        probs, q_hats, lambda_reg=lambda_reg, k_reg=k_reg
    )

    # APS lower bound: smallest probability among classes IN the set per
    # row. If the set is empty (no class passes the conformal threshold),
    # fall back to 0.0 — sizer's floor will trip and zero the position.
    in_set_mask = prediction_set
    # Replace out-of-set probs with +inf so they don't win the min.
    masked = probs.masked_fill(~in_set_mask, float("inf"))
    row_min = masked.min(dim=-1).values
    no_set = ~in_set_mask.any(dim=-1)
    aps_lower_bound = torch.where(no_set, torch.zeros_like(row_min), row_min)

    laplace_var: Tensor | None = None
    if laplace is not None:
        laplace_var = laplace.predict_variance(model_logits)

    return CalibratedPrediction(
        probs=probs,
        prediction_set=prediction_set,
        aps_lower_bound=aps_lower_bound,
        laplace_var=laplace_var,
    )


__all__ = ["CalibratedPrediction", "predict_calibrated"]
