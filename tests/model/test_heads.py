"""Unit tests for MultiTaskHead."""

from __future__ import annotations

import pytest
import torch

from nanogld.model.heads import MultiTaskHead


@pytest.mark.smoke
def test_output_shapes() -> None:
    h = MultiTaskHead(d_model=384, n_classes=3)
    pooled = torch.randn(4, 384)
    logits, pos = h(pooled)
    assert logits.shape == (4, 3)
    assert pos.shape == (4,)


@pytest.mark.smoke
def test_position_in_unit_interval() -> None:
    h = MultiTaskHead(d_model=64)
    pooled = torch.randn(8, 64) * 100.0
    _, pos = h(pooled)
    assert (pos >= -1.0).all()
    assert (pos <= 1.0).all()


@pytest.mark.smoke
def test_no_bias() -> None:
    h = MultiTaskHead(d_model=64)
    for name, _p in h.named_parameters():
        assert "bias" not in name, f"unexpected bias param: {name}"
