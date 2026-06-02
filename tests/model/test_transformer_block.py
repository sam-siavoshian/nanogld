"""Unit tests for TransformerBlock."""

from __future__ import annotations

import pytest
import torch

from nanogld.model.transformer_block import TransformerBlock


@pytest.mark.smoke
def test_basic_block_shape() -> None:
    b = TransformerBlock(d_model=64, num_heads=4, max_seq=16)
    x = torch.randn(2, 16, 64)
    out, v = b(x)
    assert out.shape == x.shape
    assert v.shape == x.shape


@pytest.mark.smoke
def test_block_with_film_shape() -> None:
    b = TransformerBlock(d_model=64, num_heads=4, max_seq=16, has_film=True, regime_dim=12)
    x = torch.randn(2, 16, 64)
    regime = torch.randn(2, 12)
    out, _ = b(x, regime=regime)
    assert out.shape == x.shape


@pytest.mark.smoke
def test_block_with_cross_attn_shape() -> None:
    b = TransformerBlock(
        d_model=64,
        num_heads=4,
        max_seq=16,
        has_cross_attn=True,
        d_text=128,
        n_news_slots=4,
    )
    x = torch.randn(2, 16, 64)
    bar_pool = torch.randn(2, 64)
    news = torch.randn(2, 4, 128)
    news_mask = torch.ones(2, 4)
    is_news_present = torch.ones(2)
    out, _ = b(
        x,
        bar_pool=bar_pool,
        news=news,
        news_mask=news_mask,
        is_news_present=is_news_present,
    )
    assert out.shape == x.shape


@pytest.mark.smoke
def test_drop_path_suppressed_in_inference() -> None:
    """In inference mode, drop_path is suppressed regardless of value."""
    b = TransformerBlock(d_model=32, num_heads=4, max_seq=16, drop_path=0.5).train(False)
    x = torch.randn(2, 8, 32)
    y1, _ = b(x)
    y2, _ = b(x)
    torch.testing.assert_close(y1, y2)


@pytest.mark.smoke
def test_value_residual_chain() -> None:
    b1 = TransformerBlock(d_model=32, num_heads=4, max_seq=16)
    b2 = TransformerBlock(d_model=32, num_heads=4, max_seq=16)
    x = torch.randn(1, 8, 32)
    h, v = b1(x)
    out, _ = b2(h, prev_v=v)
    assert out.shape == x.shape
