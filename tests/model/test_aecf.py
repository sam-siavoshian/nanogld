"""Unit tests for AECF."""

from __future__ import annotations

import pytest
import torch

from nanogld.model.aecf import AECFMask, aecf_entropy_reg


@pytest.mark.smoke
def test_sample_p_in_range() -> None:
    m = AECFMask(p_min=0.1, p_max=0.9, curriculum_steps=100)
    for step in (0, 50, 100, 1000):
        p = m.sample_p(training_step=step)
        assert m.p_min <= p <= m.p_max


@pytest.mark.smoke
def test_curriculum_starts_at_p_min() -> None:
    m = AECFMask(p_min=0.0, p_max=0.9, curriculum_steps=100)
    p = m.sample_p(training_step=0)
    assert p == 0.0


@pytest.mark.smoke
def test_sample_mask_shape_and_dtype() -> None:
    m = AECFMask(p_min=0.0, p_max=0.5)
    mask = m.sample_mask(batch_size=16, training_step=10_000)
    assert mask.shape == (16,)
    assert mask.dtype == torch.float32
    assert ((mask == 0) | (mask == 1)).all()


@pytest.mark.smoke
def test_invalid_bounds_raise() -> None:
    with pytest.raises(ValueError):
        AECFMask(p_min=0.5, p_max=0.4)


@pytest.mark.smoke
def test_entropy_reg_non_negative() -> None:
    g = torch.softmax(torch.randn(8, 4), dim=-1)
    reg = aecf_entropy_reg(g, lambda_x=0.01)
    assert reg.item() >= 0.0


@pytest.mark.smoke
def test_entropy_reg_max_at_uniform() -> None:
    """Uniform distribution gives max entropy."""
    uniform = torch.full((4, 3), 1.0 / 3.0)
    spike = torch.zeros(4, 3)
    spike[:, 0] = 1.0
    r_uniform = aecf_entropy_reg(uniform, lambda_x=1.0).item()
    r_spike = aecf_entropy_reg(spike, lambda_x=1.0).item()
    assert r_uniform > r_spike
