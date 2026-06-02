"""Regression locks for ``calibration.inference.predict_calibrated``.

The sizer consumes ``aps_lower_bound`` as its conformal-floor signal
(V1-SPEC §10.1 line 435: ``aps_lower_bound < 0.40 -> force position 0``).
Pre-fix, no producer existed for this signal. These tests pin the new
producer's contract.

Locks:

- Shape contract: ``(B, C)`` logits -> ``(B, C)`` probs + set, ``(B,)``
  lower bound.
- ``probs.sum(dim=-1) == 1`` at all rows.
- ``aps_lower_bound`` in ``[0, 1]``.
- Empty prediction set (low confidence + tight q_hats) -> lower bound 0.
- Laplace path supplies variance when ``laplace`` arg is non-None.
"""

from __future__ import annotations

import torch

from nanogld.calibration import predict_calibrated
from nanogld.calibration.laplace_lll import LaplaceLLLA


def _toy_calib(t: float = 1.0) -> dict:
    return {
        "t_scaler_T": t,
        "raps_quantiles": {0: 0.9, 1: 0.9, 2: 0.9},
    }


def test_predict_calibrated_shapes() -> None:
    logits = torch.randn(8, 3)
    out = predict_calibrated(logits, _toy_calib(), lambda_reg=0.01, k_reg=1)
    assert out.probs.shape == (8, 3)
    assert out.prediction_set.shape == (8, 3)
    assert out.aps_lower_bound.shape == (8,)
    assert out.laplace_var is None


def test_probs_sum_to_one() -> None:
    logits = torch.randn(16, 3)
    out = predict_calibrated(logits, _toy_calib(), lambda_reg=0.01, k_reg=1)
    sums = out.probs.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)


def test_aps_lower_bound_in_unit_interval() -> None:
    logits = torch.randn(32, 3) * 3.0
    out = predict_calibrated(logits, _toy_calib(), lambda_reg=0.01, k_reg=1)
    assert (out.aps_lower_bound >= 0).all()
    assert (out.aps_lower_bound <= 1).all()


def test_aps_lower_bound_zero_when_set_empty() -> None:
    """Tight q_hats + very flat probs => no class in set => lower bound = 0."""
    # Uniform logits => uniform probs (each ~ 0.333). RAPS score = 1 - 0.333 +
    # penalty. With q_hat = 0.01, score > q_hat => set is empty.
    logits = torch.zeros(4, 3)
    calib = {
        "t_scaler_T": 1.0,
        "raps_quantiles": {0: 0.01, 1: 0.01, 2: 0.01},
    }
    out = predict_calibrated(logits, calib, lambda_reg=0.01, k_reg=1)
    assert (~out.prediction_set.any(dim=-1)).all()
    assert torch.all(out.aps_lower_bound == 0.0)


def test_aps_lower_bound_top_class_when_set_singleton() -> None:
    """When the set is a singleton, lower bound == that class's probability."""
    # Very peaked logits at class 1: probs ~ (0.01, 0.98, 0.01).
    logits = torch.tensor([[0.0, 6.0, 0.0]])
    calib = {
        "t_scaler_T": 1.0,
        # q_hat low so only the top class survives.
        "raps_quantiles": {0: 0.05, 1: 0.5, 2: 0.05},
    }
    out = predict_calibrated(logits, calib, lambda_reg=0.01, k_reg=1)
    in_set = out.prediction_set[0]
    assert in_set[1].item() is True
    expected_lower = out.probs[0, 1].item()
    assert abs(out.aps_lower_bound[0].item() - expected_lower) < 1e-6


def test_t_scaling_softens_probs() -> None:
    """Higher T flattens the softmax distribution."""
    logits = torch.tensor([[5.0, -3.0, -2.0]])
    out_t1 = predict_calibrated(logits, _toy_calib(t=1.0), lambda_reg=0.01, k_reg=1)
    out_t3 = predict_calibrated(logits, _toy_calib(t=3.0), lambda_reg=0.01, k_reg=1)
    # Hotter T => less peaked => top-class prob smaller, others larger.
    assert out_t1.probs[0, 0] > out_t3.probs[0, 0]


def test_invalid_t_raises() -> None:
    import pytest

    logits = torch.zeros(4, 3)
    with pytest.raises(ValueError, match="t_scaler_T"):
        predict_calibrated(logits, {"raps_quantiles": {0: 0.5}})
    with pytest.raises(ValueError, match="must be positive"):
        predict_calibrated(logits, {"t_scaler_T": -1.0, "raps_quantiles": {0: 0.5}})


def test_missing_raps_quantiles_raises() -> None:
    import pytest

    with pytest.raises(ValueError, match="raps_quantiles"):
        predict_calibrated(torch.zeros(2, 3), {"t_scaler_T": 1.0})


def test_empty_batch_raises() -> None:
    import pytest

    with pytest.raises(ValueError, match="zero batch"):
        predict_calibrated(torch.zeros(0, 3), _toy_calib())


def test_laplace_var_returned_when_supplied() -> None:
    """When a fitted Laplace module is passed in, ``laplace_var`` is filled."""
    # Build a minimal Laplace shim that returns a known-shape variance.
    class FakeLaplace:
        def predict_variance(self, logits: torch.Tensor) -> torch.Tensor:
            return torch.ones(logits.shape[0])

    out = predict_calibrated(
        torch.randn(8, 3), _toy_calib(), laplace=FakeLaplace()  # type: ignore[arg-type]
    )
    assert out.laplace_var is not None
    assert out.laplace_var.shape == (8,)
    assert torch.all(out.laplace_var == 1.0)
