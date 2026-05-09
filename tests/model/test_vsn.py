"""Unit tests for Variable Selection Network."""

from __future__ import annotations

import pytest
import torch

from nanogld.model.vsn import GRN, VSN


@pytest.mark.smoke
def test_grn_shape_preservation() -> None:
    grn = GRN(input_dim=64, hidden_dim=32, output_dim=64)
    x = torch.randn(2, 16, 64)
    y = grn(x)
    assert y.shape == x.shape


@pytest.mark.smoke
def test_grn_input_output_dim_mismatch_uses_skip() -> None:
    grn = GRN(input_dim=64, hidden_dim=32, output_dim=128)
    x = torch.randn(2, 16, 64)
    y = grn(x)
    assert y.shape == (2, 16, 128)


@pytest.mark.smoke
def test_vsn_shape_preservation() -> None:
    v = VSN(num_features=128, hidden_dim=32)
    x = torch.randn(2, 64, 128)
    out, gate = v(x)
    assert out.shape == x.shape
    assert gate.shape == x.shape


@pytest.mark.smoke
def test_vsn_gate_softmax_sums_to_one() -> None:
    v = VSN(num_features=64, hidden_dim=32)
    x = torch.randn(2, 8, 64)
    _out, gate = v(x)
    sums = gate.sum(dim=-1)
    torch.testing.assert_close(sums, torch.ones_like(sums), atol=1e-4, rtol=1e-4)


@pytest.mark.smoke
def test_vsn_backward_no_nan() -> None:
    v = VSN(num_features=32, hidden_dim=16)
    x = torch.randn(2, 8, 32, requires_grad=True)
    out, _ = v(x)
    out.sum().backward()
    for p in v.parameters():
        if p.grad is not None:
            assert not torch.isnan(p.grad).any()
