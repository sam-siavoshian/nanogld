"""Unit tests for sLSTM cell + sequence wrapper."""

from __future__ import annotations

import pytest
import torch

from nanogld.model.slstm import sLSTM, sLSTMCell


@pytest.mark.smoke
def test_cell_shape() -> None:
    cell = sLSTMCell(d_model=64)
    x = torch.randn(2, 64)
    h = torch.zeros(2, 64)
    c = torch.zeros(2, 64)
    h_new, (h_n2, c_n2) = cell(x, (h, c))
    assert h_new.shape == (2, 64)
    assert torch.equal(h_new, h_n2)
    assert c_n2.shape == (2, 64)


@pytest.mark.smoke
def test_seq_shape_preservation() -> None:
    m = sLSTM(d_model=64)
    x = torch.randn(2, 16, 64)
    y = m(x)
    assert y.shape == x.shape


@pytest.mark.smoke
def test_backward_no_nan() -> None:
    torch.manual_seed(0)
    m = sLSTM(d_model=32)
    x = torch.randn(2, 8, 32, requires_grad=True)
    y = m(x)
    y.sum().backward()
    for p in m.parameters():
        if p.grad is not None:
            assert not torch.isnan(p.grad).any()


@pytest.mark.smoke
def test_inference_deterministic() -> None:
    m = sLSTM(d_model=16).train(False)
    x = torch.randn(1, 4, 16)
    y1 = m(x)
    y2 = m(x)
    torch.testing.assert_close(y1, y2)
