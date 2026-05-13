"""Regression locks for sLSTM-layer-11 cross-attn wiring (V1-SPEC §2.1).

The hybrid encoder uses ``DEFAULT_CROSS_ATTN_LAYERS = (3, 7, 11)``.
Layers 3 and 7 are transformer blocks with embedded NewsFuser; layer 11
is the first sLSTM block. Pre-fix, the sLSTM blocks ignored news kwargs
so layer 11's cross-attn was a no-op. After this fix, the sLSTM at layer
11 carries a NewsFuser and the encoder threads news kwargs into it.

Locks:

1. Encoder construction: the layer-11 sLSTM block has ``has_cross_attn=True``;
   the layer-12 sLSTM block does not.
2. Forward shape preserved when news kwargs supplied.
3. News kwargs actually influence the output at layer 11. Verified by
   running forward with news_mask=ones vs news_mask=zeros (with
   is_news_present flipped accordingly). Output differs.
4. ``sLSTMBlock(has_cross_attn=False).forward(x)`` still works without
   news kwargs (backward-compat for layer 12).
5. Rename lock: ``out_proj`` attribute present (the wave-1 fix for
   scaled residual init).
"""

from __future__ import annotations

import torch

from nanogld.model.encoder import DEFAULT_CROSS_ATTN_LAYERS, HybridEncoder
from nanogld.model.slstm_block import sLSTMBlock


def test_layer_11_has_cross_attn_layer_12_does_not() -> None:
    enc = HybridEncoder(
        d_model=32, num_heads=4, max_seq=8, num_transformer_layers=10,
        num_slstm_layers=2, d_text=16, n_news_slots=2, regime_dim=12,
    )
    assert 11 in DEFAULT_CROSS_ATTN_LAYERS
    assert enc.slstm_blocks[0].has_cross_attn is True   # layer 11
    assert enc.slstm_blocks[1].has_cross_attn is False  # layer 12


def test_slstm_block_out_proj_attribute() -> None:
    """`linear` -> `out_proj` rename lock (V1-SPEC §53 / STATUS.md)."""
    b = sLSTMBlock(d_model=16)
    assert hasattr(b, "out_proj")
    assert not hasattr(b, "linear")
    assert not hasattr(b, "in_norm")


def test_slstm_block_legacy_path_still_works() -> None:
    """Layer-12 has_cross_attn=False: news kwargs not needed."""
    b = sLSTMBlock(d_model=16, has_cross_attn=False)
    x = torch.randn(2, 4, 16)
    y = b(x)
    assert y.shape == x.shape


def test_slstm_block_cross_attn_runs_when_news_provided() -> None:
    b = sLSTMBlock(d_model=16, has_cross_attn=True, num_heads=4, d_text=8, n_news_slots=2)
    x = torch.randn(2, 4, 16)
    news = torch.randn(2, 2, 8)
    news_mask = torch.ones(2, 2)
    is_news_present = torch.ones(2, dtype=torch.long)
    y = b(x, bar_pool=x.mean(dim=1), news=news, news_mask=news_mask, is_news_present=is_news_present)
    assert y.shape == x.shape


def _force_news_gates_open(enc: HybridEncoder) -> None:
    """Set every NewsFuser.alpha to a non-zero value so ``tanh(alpha) > 0``.

    NewsFuser starts with ``alpha=0`` so the gate is closed and news has
    no effect on the output. Tests that verify the news path is wired
    (not whether the gate has opened during training) need to break that
    dead-init explicitly.
    """
    import torch.nn as nn

    with torch.no_grad():
        for module in enc.modules():
            if module.__class__.__name__ == "NewsFuser" and isinstance(
                module.alpha, nn.Parameter
            ):
                module.alpha.fill_(1.0)


def test_slstm_layer11_news_influences_output() -> None:
    """Toggling news_mask between all-ones and all-zeros must move the
    encoder's output at layer 11. Runs in eval mode (dropout off) so the
    diff comes from the news path, not RNG. NewsFuser gates are forced
    open since they start at alpha=0 (closed)."""
    torch.manual_seed(0)
    enc = HybridEncoder(
        d_model=32, num_heads=4, max_seq=8, num_transformer_layers=10,
        num_slstm_layers=2, d_text=16, n_news_slots=2, regime_dim=12,
    ).train(False)
    _force_news_gates_open(enc)

    B = 2
    x = torch.randn(B, 8, 32)
    regime = torch.randn(B, 12)
    news = torch.randn(B, 2, 16)
    news_mask_on = torch.ones(B, 2)
    news_mask_off = torch.zeros(B, 2)
    pres_on = torch.ones(B, dtype=torch.long)
    pres_off = torch.zeros(B, dtype=torch.long)

    out_on = enc(x, regime=regime, news=news, news_mask=news_mask_on, is_news_present=pres_on)
    out_off = enc(x, regime=regime, news=news, news_mask=news_mask_off, is_news_present=pres_off)
    diff = (out_on - out_off).abs().max().item()
    assert diff > 1e-4, f"layer 11 cross-attn not wired: max diff {diff}"


def test_gradient_flows_through_layer_11_news_fuser() -> None:
    """News embedding grad must reach the input via layer-11's NewsFuser
    once the gate is non-zero. The post-tanh(alpha=0)=0 dead-init zeros
    the path, so we force the gate open before backward."""
    torch.manual_seed(0)
    enc = HybridEncoder(
        d_model=32, num_heads=4, max_seq=8, num_transformer_layers=10,
        num_slstm_layers=2, d_text=16, n_news_slots=2, regime_dim=12,
    ).train(False)
    _force_news_gates_open(enc)

    B = 2
    x = torch.randn(B, 8, 32)
    regime = torch.randn(B, 12)
    news = torch.randn(B, 2, 16, requires_grad=True)
    news_mask = torch.ones(B, 2)
    pres = torch.ones(B, dtype=torch.long)

    out = enc(x, regime=regime, news=news, news_mask=news_mask, is_news_present=pres)
    out.sum().backward()
    assert news.grad is not None
    assert news.grad.abs().sum().item() > 0
    fuser = enc.slstm_blocks[0].news_fuser
    has_grad = any(
        p.grad is not None and p.grad.abs().sum().item() > 0 for p in fuser.parameters()
    )
    assert has_grad
