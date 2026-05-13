"""Unit tests for HybridEncoder."""

from __future__ import annotations

import pytest
import torch

from nanogld.model.encoder import (
    DEFAULT_CROSS_ATTN_LAYERS,
    DEFAULT_FILM_LAYERS,
    HybridEncoder,
    TOTAL_LAYERS,
)


@pytest.mark.smoke
def test_total_layers_is_12() -> None:
    assert TOTAL_LAYERS == 12


@pytest.mark.smoke
def test_film_layers_default() -> None:
    assert DEFAULT_FILM_LAYERS == (2, 4, 6, 8, 10)


@pytest.mark.smoke
def test_cross_attn_layers_default() -> None:
    assert DEFAULT_CROSS_ATTN_LAYERS == (3, 7, 11)


@pytest.mark.smoke
def test_invalid_layer_split_raises() -> None:
    with pytest.raises(ValueError):
        HybridEncoder(num_transformer_layers=8, num_slstm_layers=2)


@pytest.mark.smoke
def test_forward_shape() -> None:
    enc = HybridEncoder(
        d_model=32,
        num_heads=4,
        max_seq=16,
        num_transformer_layers=10,
        num_slstm_layers=2,
        d_text=64,
        n_news_slots=4,
        regime_dim=12,
    )
    b_prime = 4
    x = torch.randn(b_prime, 16, 32)
    regime = torch.randn(b_prime, 12)
    news = torch.randn(b_prime, 4, 64)
    news_mask = torch.ones(b_prime, 4)
    is_news_present = torch.ones(b_prime)
    out = enc(x, regime=regime, news=news, news_mask=news_mask, is_news_present=is_news_present)
    assert out.shape == x.shape


@pytest.mark.smoke
def test_film_and_cross_attn_blocks_present() -> None:
    enc = HybridEncoder(
        d_model=32,
        num_heads=4,
        max_seq=16,
        d_text=64,
        n_news_slots=4,
    )
    has_film_count = sum(1 for b in enc.transformer_blocks if b.has_film)
    has_xattn_count = sum(1 for b in enc.transformer_blocks if b.has_cross_attn)
    assert has_film_count == 5
    assert has_xattn_count == 2
