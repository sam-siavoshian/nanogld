"""Unit tests for FiLM conditioner."""

from __future__ import annotations

import pytest
import torch

from nanogld.model.film import FiLMConditioner


@pytest.mark.smoke
def test_shape_preservation() -> None:
    f = FiLMConditioner(d_model=384, regime_dim=12)
    x = torch.randn(2, 16, 384)
    regime = torch.randn(2, 12)
    y = f(x, regime)
    assert y.shape == x.shape


@pytest.mark.smoke
def test_init_is_identity() -> None:
    """FiLM at init must be identity (gamma=0 + beta=0 + (1+0)*x + 0 = x)."""
    f = FiLMConditioner(d_model=64, regime_dim=12)
    x = torch.randn(2, 4, 64)
    regime = torch.randn(2, 12)
    y = f(x, regime)
    torch.testing.assert_close(y, x)


@pytest.mark.smoke
def test_param_count() -> None:
    """One Linear(regime_dim, 2*d_model) with bias."""
    d_model, regime_dim = 384, 12
    f = FiLMConditioner(d_model=d_model, regime_dim=regime_dim)
    n_params = sum(p.numel() for p in f.parameters())
    expected = regime_dim * 2 * d_model + 2 * d_model
    assert n_params == expected
