"""Unit tests for MultiHeadAttention."""

from __future__ import annotations

import pytest
import torch

from nanogld.model.attention import MultiHeadAttention


@pytest.mark.smoke
def test_shape_preservation() -> None:
    m = MultiHeadAttention(d_model=384, num_heads=6, max_seq=64)
    x = torch.randn(2, 16, 384)
    out, v = m(x)
    assert out.shape == (2, 16, 384)
    assert v.shape == (2, 16, 384)


@pytest.mark.smoke
def test_no_bias_anywhere() -> None:
    m = MultiHeadAttention(d_model=64, num_heads=4, max_seq=16)
    for name, _p in m.named_parameters():
        if "norm" in name:
            continue
        if "head_gate" in name:
            continue
        assert "bias" not in name, f"unexpected bias param: {name}"


@pytest.mark.smoke
def test_per_head_gate_count() -> None:
    m = MultiHeadAttention(d_model=384, num_heads=6, max_seq=16)
    assert m.head_gate.numel() == 6


@pytest.mark.smoke
def test_value_residual_chain_no_crash() -> None:
    m1 = MultiHeadAttention(d_model=64, num_heads=4, max_seq=16)
    m2 = MultiHeadAttention(d_model=64, num_heads=4, max_seq=16)
    x = torch.randn(1, 8, 64)
    out1, v1 = m1(x, prev_v=None)
    out2, v2 = m2(out1, prev_v=v1)
    assert out2.shape == x.shape


@pytest.mark.smoke
def test_d_model_must_divide_num_heads() -> None:
    with pytest.raises(ValueError):
        MultiHeadAttention(d_model=100, num_heads=6, max_seq=16)


@pytest.mark.smoke
def test_backward_no_nan() -> None:
    torch.manual_seed(0)
    m = MultiHeadAttention(d_model=64, num_heads=4, max_seq=16)
    x = torch.randn(2, 8, 64, requires_grad=True)
    out, _ = m(x)
    out.sum().backward()
    for p in m.parameters():
        if p.grad is not None:
            assert not torch.isnan(p.grad).any(), "NaN in gradient"
