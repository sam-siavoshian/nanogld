"""Unit tests for RMSNorm."""

from __future__ import annotations

import pytest
import torch

from nanogld.model.rms_norm import RMSNorm


@pytest.mark.smoke
def test_shape_preservation() -> None:
    n = RMSNorm(dim=384)
    x = torch.randn(2, 16, 384)
    y = n(x)
    assert y.shape == x.shape


@pytest.mark.smoke
def test_no_bias_param_count() -> None:
    n = RMSNorm(dim=384)
    n_params = sum(p.numel() for p in n.parameters())
    assert n_params == 384


@pytest.mark.smoke
def test_init_weights_are_one() -> None:
    n = RMSNorm(dim=384)
    assert torch.equal(n.weight, torch.ones(384))


@pytest.mark.smoke
def test_numerical_match_reference() -> None:
    """Hand-rolled RMSNorm reference."""
    torch.manual_seed(0)
    n = RMSNorm(dim=64, eps=1e-6)
    x = torch.randn(4, 8, 64)
    y = n(x)

    rms = x.pow(2).mean(dim=-1, keepdim=True).add(1e-6).rsqrt()
    expected = x * rms * n.weight
    torch.testing.assert_close(y, expected, atol=1e-6, rtol=1e-6)
