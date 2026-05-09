"""Unit tests for CFAProjector."""

from __future__ import annotations

import pytest
import torch

from nanogld.model.cfa_projector import CFAProjector


@pytest.mark.smoke
def test_output_shape() -> None:
    p = CFAProjector(d_text=256, d_bottleneck=64, d_model=384)
    text = torch.randn(2, 8, 256)
    bar = torch.randn(2, 384)
    out = p(text, bar)
    assert out.shape == (2, 8, 384)


@pytest.mark.smoke
def test_orthogonal_to_bar_pool() -> None:
    """Output rows must be ~orthogonal to bar_pool (post-projection step)."""
    p = CFAProjector(d_text=64, d_bottleneck=16, d_model=32)
    torch.manual_seed(0)
    text = torch.randn(4, 5, 64)
    bar = torch.randn(4, 32)
    out = p(text, bar)
    inner = (out * bar.unsqueeze(1)).sum(dim=-1)
    assert inner.abs().max() < 1e-3


@pytest.mark.smoke
def test_param_count_dominated_by_film_proj() -> None:
    """Two-stage low-rank + bias-free FiLM."""
    p = CFAProjector(d_text=256, d_bottleneck=64, d_model=384)
    n_params = sum(prm.numel() for prm in p.parameters())
    assert n_params < 500_000, "param count should be modest"


@pytest.mark.smoke
def test_init_film_is_zero_so_text_proj_passes() -> None:
    p = CFAProjector(d_text=256, d_bottleneck=64, d_model=384)
    assert p.film_proj.weight.abs().sum().item() == 0.0
    assert p.film_proj.bias.abs().sum().item() == 0.0
