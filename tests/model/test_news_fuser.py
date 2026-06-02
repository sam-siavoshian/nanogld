"""Unit tests for NewsFuser."""

from __future__ import annotations

import pytest
import torch

from nanogld.model.news_fuser import NewsFuser


@pytest.mark.smoke
def test_output_shape() -> None:
    f = NewsFuser(d_model=64, d_text=256, n_heads=4, n_news_slots=8)
    bar_tokens = torch.randn(2, 16, 64)
    bar_pool = torch.randn(2, 64)
    news = torch.randn(2, 8, 256)
    news_mask = torch.ones(2, 8)
    is_news_present = torch.ones(2)
    out = f(bar_tokens, bar_pool, news, news_mask, is_news_present)
    assert out.shape == (2, 16, 64)


@pytest.mark.smoke
def test_alpha_init_zero_makes_pass_through() -> None:
    """At init, tanh(alpha=0) = 0 → output equals bar_tokens."""
    f = NewsFuser(d_model=64, d_text=128, n_heads=4, n_news_slots=4).train(False)
    bar_tokens = torch.randn(2, 8, 64)
    bar_pool = torch.randn(2, 64)
    news = torch.randn(2, 4, 128)
    news_mask = torch.ones(2, 4)
    is_news_present = torch.ones(2)
    out = f(bar_tokens, bar_pool, news, news_mask, is_news_present)
    torch.testing.assert_close(out, bar_tokens, atol=1e-5, rtol=1e-5)


@pytest.mark.smoke
def test_d_model_must_divide_n_heads() -> None:
    with pytest.raises(ValueError):
        NewsFuser(d_model=100, d_text=256, n_heads=6)


@pytest.mark.smoke
def test_no_news_path_zero_mask() -> None:
    """All news_mask = 0 → fuser still returns shape and no NaN."""
    f = NewsFuser(d_model=32, d_text=64, n_heads=4, n_news_slots=4)
    bar_tokens = torch.randn(1, 4, 32)
    bar_pool = torch.randn(1, 32)
    news = torch.randn(1, 4, 64)
    news_mask = torch.zeros(1, 4)
    is_news_present = torch.zeros(1)
    out = f(bar_tokens, bar_pool, news, news_mask, is_news_present)
    assert out.shape == (1, 4, 32)
    assert not torch.isnan(out).any()
