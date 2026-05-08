"""Unit tests for nanoGLDV1 top-level model."""

from __future__ import annotations

import inspect

import pytest
import torch

from nanogld.model import nanoGLDV1


def _build_tiny_model() -> nanoGLDV1:
    return nanoGLDV1(
        numeric_dim=16,
        d_model=32,
        num_heads=4,
        num_transformer_layers=10,
        num_slstm_layers=2,
        t_bars=8,
        patch_len=2,
        patch_stride=2,
        n_classes=3,
        regime_dim=12,
        d_text=64,
        n_news_slots=4,
        dropout=0.0,
        drop_path_max=0.0,
        decomposition_kernel=4,
    )


@pytest.mark.smoke
def test_forward_smoke() -> None:
    m = _build_tiny_model()
    b = 2
    x = torch.randn(b, 8, 16)
    news = torch.randn(b, 4, 64)
    news_mask = torch.ones(b, 4)
    is_news_present = torch.ones(b)
    regime = torch.randn(b, 12)
    out = m(x, news, news_mask, is_news_present, regime)
    assert out["logits_3class"].shape == (b, 3)
    assert out["position_weight"].shape == (b,)


@pytest.mark.smoke
def test_no_view_as_complex_anywhere() -> None:
    """V1 invariant: torch.view_as_complex must not be CALLED anywhere."""
    from nanogld import model as model_pkg

    for _name, mod in inspect.getmembers(model_pkg, inspect.ismodule):
        if not mod.__name__.startswith("nanogld.model"):
            continue
        try:
            src = inspect.getsource(mod)
        except OSError:
            continue
        code = []
        in_doc = False
        for line in src.splitlines():
            s = line.strip()
            if s.startswith('"""') or s.endswith('"""'):
                in_doc = not in_doc if s.count('"""') == 1 else in_doc
                continue
            if in_doc or s.startswith("#"):
                continue
            code.append(line)
        assert "view_as_complex(" not in "\n".join(code)


@pytest.mark.smoke
def test_position_in_unit_interval() -> None:
    m = _build_tiny_model().train(False)
    b = 4
    x = torch.randn(b, 8, 16) * 100.0
    news = torch.randn(b, 4, 64)
    news_mask = torch.ones(b, 4)
    is_news_present = torch.ones(b)
    regime = torch.randn(b, 12)
    out = m(x, news, news_mask, is_news_present, regime)
    assert (out["position_weight"] >= -1.0).all()
    assert (out["position_weight"] <= 1.0).all()


@pytest.mark.smoke
def test_backward_no_nan() -> None:
    torch.manual_seed(0)
    m = _build_tiny_model()
    b = 2
    x = torch.randn(b, 8, 16, requires_grad=True)
    news = torch.randn(b, 4, 64)
    news_mask = torch.ones(b, 4)
    is_news_present = torch.ones(b)
    regime = torch.randn(b, 12)
    out = m(x, news, news_mask, is_news_present, regime)
    loss = out["logits_3class"].sum() + out["position_weight"].sum()
    loss.backward()
    for p in m.parameters():
        if p.grad is not None:
            assert not torch.isnan(p.grad).any(), "NaN in gradient"


@pytest.mark.smoke
def test_param_count_in_range_for_full_v1() -> None:
    """Full V1 config (D=384, 12 layers, 681 channels) param count check."""
    m = nanoGLDV1(
        numeric_dim=681,
        d_model=384,
        num_heads=6,
        num_transformer_layers=10,
        num_slstm_layers=2,
        t_bars=64,
        patch_len=4,
        patch_stride=4,
    )
    n_params = sum(p.numel() for p in m.parameters())
    assert 5_000_000 <= n_params <= 60_000_000, f"got {n_params:,} params"
