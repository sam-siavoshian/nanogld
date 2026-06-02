"""Unit tests for SwiGLU FFN."""

from __future__ import annotations

import pytest
import torch

from nanogld.model.swiglu import SwiGLU, swiglu_hidden_dim


@pytest.mark.smoke
def test_hidden_dim_at_d384() -> None:
    assert swiglu_hidden_dim(384) == 1024


@pytest.mark.smoke
def test_shape_preservation() -> None:
    m = SwiGLU(d_model=384)
    x = torch.randn(2, 16, 384)
    y = m(x)
    assert y.shape == x.shape


@pytest.mark.smoke
def test_no_bias() -> None:
    m = SwiGLU(d_model=384)
    for name, _p in m.named_parameters():
        assert "bias" not in name, f"unexpected bias param: {name}"


@pytest.mark.smoke
def test_inference_mode_deterministic() -> None:
    m = SwiGLU(d_model=64, hidden_dim=128).train(False)
    x = torch.randn(2, 4, 64)
    y1 = m(x)
    y2 = m(x)
    torch.testing.assert_close(y1, y2)
