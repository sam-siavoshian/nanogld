"""Unit tests for sLSTMBlock."""

from __future__ import annotations

import pytest
import torch

from nanogld.model.slstm_block import sLSTMBlock


@pytest.mark.smoke
def test_shape_preservation() -> None:
    b = sLSTMBlock(d_model=64)
    x = torch.randn(2, 16, 64)
    y = b(x)
    assert y.shape == x.shape


@pytest.mark.smoke
def test_residual_connection() -> None:
    b = sLSTMBlock(d_model=32).train(False)
    x = torch.randn(1, 4, 32)
    y = b(x)
    diff = (y - x).abs().mean()
    assert diff.item() > 0.0


@pytest.mark.smoke
def test_backward_no_nan() -> None:
    b = sLSTMBlock(d_model=16)
    x = torch.randn(2, 8, 16, requires_grad=True)
    y = b(x)
    y.sum().backward()
    for p in b.parameters():
        if p.grad is not None:
            assert not torch.isnan(p.grad).any()
