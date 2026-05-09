"""News fuser — Flamingo-gated cross-attention with CFA + AECF.

Inserted at sparse layers {3, 7, 11} of the encoder (Decision 2.1 +
mPLUG-Owl3 Table 8: sparse > dense at small scale). Bottom 2 layers
are pure-bar (let numerical features form before fusing).

Pipeline (per insertion point):
  news_in:  (B, S, 256)        Qwen3-256d MRL embeddings
  news_mask: (B, S)            1 = source present, 0 = absent
  is_news_present: (B,)        binary: any news visible at this bar
  bar_pool: (B, D)             mean-pooled bar tokens
  bar_tokens: (B, T, D)        bar stream (queries)

  text_proj = CFAProjector(news_in, bar_pool)            # (B, S, D)
  text_proj += is_news_present_emb(is_news_present)      # (B, 1, 8) -> 8 dim concat
  no_news_token: learnable (1, D)                        # broadcast for absent rows
  text_proj = where(news_mask == 0 row-wise, no_news_token, text_proj)

  Q = bar_tokens
  K = V = text_proj
  attn_out = MultiHeadCrossAttention(Q, K, V, mask=news_mask)
  gate = tanh(alpha)                                     # alpha init=0
  return bar_tokens + gate * attn_out

Spec: plan/05-MODEL-TRAINING-CALIBRATION.md V1 news fusion.
Spec: plan/V1-SPEC.md §2 (sparse cross-attn at [3,7,11]).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from nanogld.model.cfa_projector import CFAProjector


class NewsFuser(nn.Module):
    """Per-layer Flamingo-gated cross-attention with CFA projector.

    Args:
        d_model: model hidden dim.
        d_text: news embedding dim (256 from Qwen3+MRL).
        n_heads: number of cross-attn heads.
        n_news_slots: max news slots per bar (default 8).
        d_bottleneck: CFA bottleneck dim (default 64).
        is_news_present_dim: dim of the binary-presence embedding (default 8).
        dropout: attention dropout.
    """

    def __init__(
        self,
        d_model: int = 384,
        d_text: int = 256,
        n_heads: int = 4,
        n_news_slots: int = 8,
        d_bottleneck: int = 64,
        is_news_present_dim: int = 8,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model={d_model} must be divisible by n_heads={n_heads}")
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.dropout = dropout

        self.cfa = CFAProjector(d_text=d_text, d_bottleneck=d_bottleneck, d_model=d_model)
        self.is_news_present_emb = nn.Embedding(2, is_news_present_dim)
        self.is_news_proj = nn.Linear(is_news_present_dim, d_model, bias=False)

        self.no_news_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.no_news_token, std=0.02)

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        bar_tokens: Tensor,
        bar_pool: Tensor,
        news: Tensor,
        news_mask: Tensor,
        is_news_present: Tensor,
    ) -> Tensor:
        """Apply gated cross-attn from news to bar tokens.

        Args:
            bar_tokens: (B, T, d_model) — queries.
            bar_pool: (B, d_model) — pooled bar context for CFA FiLM.
            news: (B, S, d_text) — raw Qwen3 256d embeddings.
            news_mask: (B, S) — 1 = source present.
            is_news_present: (B,) — binary, 1 if any news this bar.

        Returns:
            (B, T, d_model) — bar_tokens + tanh(alpha) * attn_out.
        """
        b, t, _ = bar_tokens.shape

        text_proj = self.cfa(news, bar_pool)
        present_emb = self.is_news_present_emb(is_news_present.long())
        present_proj = self.is_news_proj(present_emb).unsqueeze(1)
        text_proj = text_proj + present_proj

        no_news_broadcast = self.no_news_token.expand(b, news.shape[1], -1)
        mask_expanded = news_mask.unsqueeze(-1).to(dtype=text_proj.dtype)
        text_proj = mask_expanded * text_proj + (1.0 - mask_expanded) * no_news_broadcast

        q = (
            self.q_proj(bar_tokens)
            .view(b, t, self.n_heads, self.head_dim)
            .transpose(1, 2)
            .contiguous()
        )
        k = (
            self.k_proj(text_proj)
            .view(b, news.shape[1], self.n_heads, self.head_dim)
            .transpose(1, 2)
            .contiguous()
        )
        v = (
            self.v_proj(text_proj)
            .view(b, news.shape[1], self.n_heads, self.head_dim)
            .transpose(1, 2)
            .contiguous()
        )

        block_mask = news_mask == 0
        all_blocked = block_mask.all(dim=-1, keepdim=True)
        zero_slot = torch.zeros_like(block_mask)
        zero_slot[..., 0] = True
        block_mask = block_mask & ~(all_blocked & zero_slot)
        attend_mask = (~block_mask).unsqueeze(1).unsqueeze(2)
        attend_mask = attend_mask.expand(-1, self.n_heads, t, -1)
        attn_out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attend_mask, dropout_p=self.dropout if self.training else 0.0
        )

        attn_out = attn_out.transpose(1, 2).contiguous().view(b, t, self.d_model)
        attn_out = self.out_proj(attn_out)

        gate = torch.tanh(self.alpha)
        return bar_tokens + gate * attn_out
