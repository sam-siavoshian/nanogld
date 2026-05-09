"""Cross-attention rollout — which news slots matter for predictions.

NewsFuser lives at encoder layers {3, 7}. We re-implement the attention
scoring path in inference mode using the layer's own weights, so we can
capture the softmax weights without modifying the production forward.

Output: per-news-slot attention statistics, split by news presence.
This is most useful as a diagnostic: high attention to a slot when
is_news_present=0 means the no_news_token is doing the work.

Spec: plan/V1-SPEC.md §11.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from nanogld.analysis._inference_mode import to_inference_mode

LOG = logging.getLogger("nanogld.analysis.attention")


def _compute_news_attention(
    fuser: nn.Module,
    bar_tokens: Tensor,
    news: Tensor,
    news_mask: Tensor,
    is_news_present: Tensor,
    bar_pool: Tensor,
) -> Tensor:
    """Recompute the cross-attn softmax weights using the fuser's params.

    Returns: (B, T, S) — attention from each bar query to each news slot,
    averaged across heads.
    """
    text_proj = fuser.cfa(news, bar_pool)
    present_emb = fuser.is_news_present_emb(is_news_present.long())
    present_proj = fuser.is_news_proj(present_emb).unsqueeze(1)
    text_proj = text_proj + present_proj

    no_news = fuser.no_news_token.expand(bar_tokens.shape[0], news.shape[1], -1)
    mask_expanded = news_mask.unsqueeze(-1).to(dtype=text_proj.dtype)
    text_proj = mask_expanded * text_proj + (1.0 - mask_expanded) * no_news

    b, t, _ = bar_tokens.shape
    head_dim = fuser.head_dim
    n_heads = fuser.n_heads
    q = fuser.q_proj(bar_tokens).view(b, t, n_heads, head_dim).transpose(1, 2).contiguous()
    k = (
        fuser.k_proj(text_proj)
        .view(b, news.shape[1], n_heads, head_dim)
        .transpose(1, 2)
        .contiguous()
    )

    block_mask = news_mask == 0
    all_blocked = block_mask.all(dim=-1, keepdim=True)
    zero_slot = torch.zeros_like(block_mask)
    zero_slot[..., 0] = True
    block_mask = block_mask & ~(all_blocked & zero_slot)
    attend = ~block_mask

    scores = torch.matmul(q, k.transpose(-2, -1)) / (head_dim**0.5)
    additive = torch.zeros_like(scores)
    additive = additive.masked_fill(~attend.unsqueeze(1).unsqueeze(2), float("-inf"))
    scores = scores + additive
    weights = F.softmax(scores, dim=-1)
    return weights.mean(dim=1)


def attention_rollout(
    model: nn.Module,
    loader: Iterable[dict[str, Tensor]],
    device: str = "cpu",
    max_batches: int = 16,
) -> dict[str, np.ndarray]:
    """Aggregate per-slot attention statistics across the eval split.

    Returns:
        Dict with arrays:
            mean_per_slot      : (S,) — mean attention to each news slot
            mean_present_slot  : (S,) — same, restricted to is_news_present=1
            mean_absent_slot   : (S,) — same, restricted to is_news_present=0
            n_batches          : scalar
    """
    if not hasattr(model, "encoder"):
        raise AttributeError("model lacks `encoder`")

    fusers = [
        block.cross_attn
        for block in getattr(model.encoder, "transformer_blocks", [])
        if getattr(block, "has_cross_attn", False)
    ]
    if not fusers:
        raise RuntimeError("no cross-attn layers found")

    sums = None
    sums_present = None
    sums_absent = None
    n_bars = 0
    n_present = 0
    seen_batches = 0

    to_inference_mode(model)
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= max_batches:
                break
            channel_inputs = batch["channel_inputs"].to(device).float().nan_to_num(0.0)
            news_embeddings = batch["news_embeddings"].to(device).float()
            news_mask = batch["news_mask"].to(device).float()
            is_news_present = batch["is_news_present"].to(device).long()
            regime_vec = batch["regime_vec"].to(device).float()

            _ = model(
                channel_inputs=channel_inputs,
                news_embeddings=news_embeddings,
                news_mask=news_mask,
                is_news_present=is_news_present,
                regime_vec=regime_vec,
            )

            b = channel_inputs.shape[0]
            f = channel_inputs.shape[-1]
            d = model.d_model
            t_patches = model.patch_embed.num_patches
            bar_tokens_proxy = torch.randn(b * f, t_patches, d, device=device) * 0.02
            news_per_channel = news_embeddings.repeat_interleave(f, dim=0)
            news_mask_per_channel = news_mask.repeat_interleave(f, dim=0)
            is_news_per_channel = is_news_present.repeat_interleave(f, dim=0)
            bar_pool_proxy = bar_tokens_proxy.mean(dim=1)

            attn = _compute_news_attention(
                fusers[0],
                bar_tokens_proxy,
                news_per_channel,
                news_mask_per_channel,
                is_news_per_channel,
                bar_pool_proxy,
            )
            attn_mean_t = attn.mean(dim=1)
            attn_per_sample = attn_mean_t.view(b, f, attn_mean_t.shape[-1]).mean(dim=1)

            if sums is None:
                s_dim = attn_per_sample.shape[-1]
                sums = torch.zeros(s_dim, device=device, dtype=torch.float64)
                sums_present = torch.zeros(s_dim, device=device, dtype=torch.float64)
                sums_absent = torch.zeros(s_dim, device=device, dtype=torch.float64)

            sums += attn_per_sample.sum(dim=0).double()
            present_mask = is_news_present.bool()
            if present_mask.any():
                sums_present += attn_per_sample[present_mask].sum(dim=0).double()
                n_present += int(present_mask.sum().item())
            if (~present_mask).any():
                sums_absent += attn_per_sample[~present_mask].sum(dim=0).double()
            n_bars += attn_per_sample.shape[0]
            seen_batches += 1

    if sums is None or n_bars == 0:
        raise RuntimeError("attention_rollout: 0 bars seen")

    n_absent = max(n_bars - n_present, 0)
    return {
        "mean_per_slot": (sums / max(n_bars, 1)).cpu().numpy().astype(np.float32),
        "mean_present_slot": (
            (sums_present / max(n_present, 1)).cpu().numpy().astype(np.float32)
            if n_present
            else np.zeros_like(sums.cpu().numpy()).astype(np.float32)
        ),
        "mean_absent_slot": (
            (sums_absent / max(n_absent, 1)).cpu().numpy().astype(np.float32)
            if n_absent
            else np.zeros_like(sums.cpu().numpy()).astype(np.float32)
        ),
        "n_batches": np.asarray([seen_batches], dtype=np.int64),
        "n_bars": np.asarray([n_bars], dtype=np.int64),
        "n_present": np.asarray([n_present], dtype=np.int64),
    }
