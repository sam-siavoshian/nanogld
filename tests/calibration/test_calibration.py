"""Unit tests for V1 calibration stack."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from nanogld.calibration.agaci import AgACI
from nanogld.calibration.calibrate import (
    CalibrationConfig,
    calibrate,
    load_calibration,
)
from nanogld.calibration.ece import (
    adaptive_ece,
    classwise_ada_ece,
    macro_brier,
    per_bucket_ece,
)
from nanogld.calibration.laplace_lll import kelly_multiplier
from nanogld.calibration.raps import (
    fit_raps_quantile,
    raps_score,
    raps_set,
)
from nanogld.calibration.temperature_scaling import T_MAX, T_MIN, TemperatureScaler


@pytest.mark.smoke
def test_temperature_scaler_fits_in_bounds() -> None:
    torch.manual_seed(0)
    logits = torch.randn(200, 3) * 5.0
    labels = torch.randint(0, 3, (200,))
    ts = TemperatureScaler(init_T=1.0)
    fitted = ts.fit(logits, labels, max_iter=50)
    assert T_MIN <= fitted <= T_MAX


@pytest.mark.smoke
def test_temperature_scaler_preserves_argmax() -> None:
    logits = torch.randn(50, 3)
    labels = torch.randint(0, 3, (50,))
    ts = TemperatureScaler(init_T=1.0)
    ts.fit(logits, labels, max_iter=10)
    assert torch.equal(logits.argmax(dim=-1), ts(logits).argmax(dim=-1))


@pytest.mark.smoke
def test_raps_score_lower_when_label_top() -> None:
    probs = torch.tensor([[0.9, 0.05, 0.05], [0.1, 0.8, 0.1]])
    labels = torch.tensor([0, 1])
    score = raps_score(probs, labels, lambda_reg=0.0, k_reg=1)
    assert score[0] < 0.2 and score[1] < 0.3


@pytest.mark.smoke
def test_raps_quantile_mondrian_returns_per_class() -> None:
    torch.manual_seed(0)
    probs = F.softmax(torch.randn(60, 3), dim=-1)
    labels = torch.randint(0, 3, (60,))
    q = fit_raps_quantile(probs, labels, alpha=0.10, mondrian=True)
    assert set(q.keys()) == {0, 1, 2}


@pytest.mark.smoke
def test_raps_set_shape_and_dtype() -> None:
    probs = torch.tensor([[0.7, 0.2, 0.1], [0.4, 0.4, 0.2]])
    q = {0: 0.5, 1: 0.5, 2: 0.5}
    s = raps_set(probs, q)
    assert s.shape == probs.shape
    assert s.dtype == torch.bool


@pytest.mark.smoke
def test_agaci_alpha_in_range() -> None:
    a = AgACI(alpha_target=0.10)
    for _ in range(50):
        a.update(miscovered=False)
    assert 0.001 <= a.current_alpha() <= 0.5


@pytest.mark.smoke
def test_agaci_state_round_trip() -> None:
    a = AgACI(alpha_target=0.10)
    for i in range(20):
        a.update(miscovered=bool(i % 3 == 0))
    state = a.state_dict()
    b = AgACI()
    b.load_state_dict(state)
    assert b.current_alpha() == pytest.approx(a.current_alpha())


@pytest.mark.smoke
def test_adaptive_ece_calibrated_synthetic() -> None:
    torch.manual_seed(0)
    n = 400
    probs = torch.full((n, 3), 1.0 / 3.0)
    labels = torch.randint(0, 3, (n,))
    ece = adaptive_ece(probs, labels)
    assert ece < 0.20


@pytest.mark.smoke
def test_classwise_ada_ece_returns_pair() -> None:
    probs = F.softmax(torch.randn(50, 3), dim=-1)
    labels = torch.randint(0, 3, (50,))
    macro, worst = classwise_ada_ece(probs, labels)
    assert macro <= worst


@pytest.mark.smoke
def test_per_bucket_ece_keys() -> None:
    probs = F.softmax(torch.randn(50, 3), dim=-1)
    labels = torch.randint(0, 3, (50,))
    mask = torch.randint(0, 2, (50,)).bool()
    out = per_bucket_ece(probs, labels, mask)
    assert set(out.keys()) == {"present", "absent", "both"}


@pytest.mark.smoke
def test_macro_brier_zero_for_perfect_probs() -> None:
    probs = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    labels = torch.tensor([0, 1])
    b = macro_brier(probs, labels)
    assert b == pytest.approx(0.0)


@pytest.mark.smoke
def test_kelly_multiplier_in_unit_interval() -> None:
    var = torch.tensor([1e-6, 1e-2, 1e+2])
    m = kelly_multiplier(var, sigma_target=0.05, floor=0.0, ceil=1.0)
    assert (m >= 0.0).all() and (m <= 1.0).all()


@pytest.mark.smoke
def test_calibrate_round_trip(tmp_path: Path) -> None:
    torch.manual_seed(0)
    val_b_logits = torch.randn(100, 3)
    val_b_labels = torch.randint(0, 3, (100,))
    val_c_logits = torch.randn(80, 3)
    val_c_labels = torch.randint(0, 3, (80,))
    cfg = CalibrationConfig(fold_idx=0, output_dir=tmp_path)
    artifacts = calibrate(cfg, val_b_logits, val_b_labels, val_c_logits, val_c_labels)
    assert artifacts.t_scaler_path.exists()
    loaded = load_calibration(tmp_path / "calibration_0")
    assert "agaci" in loaded
    assert "raps_quantiles" in loaded
    assert "t_scaler_T" in loaded
