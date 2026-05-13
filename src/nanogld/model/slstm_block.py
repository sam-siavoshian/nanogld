"""Pre-norm sLSTM block (xLSTMTime style) with optional Flamingo cross-attn.

Used for layers 11-12 of the V1 hybrid encoder (Decision 1B). Layer 11
additionally fuses news via :class:`NewsFuser` per V1-SPEC §2.1 (sparse
cross-attn at ``{3, 7, 11}``).

Block layout (per xLSTMTime Alharthi & Mahmood arXiv:2407.10240,
adapted for B=1 safety — BN swapped for GroupNorm(1, d_model); the
trailing InstanceNorm from the V1 sketch was dropped after the wave-2
fix since residual-init is scaled at construction):

    h = GroupNorm(x)
    h = sLSTM(h)
    h = out_proj(h)
    h = dropout(h)
    h = x + h                  # residual
    if has_cross_attn and news provided:
        h = NewsFuser(h, bar_pool, news, news_mask, is_news_present)
    return h

Channel-independent — sLSTM operates per (B, T) sequence, no cross-channel
mixing inside the recurrence. NewsFuser provides bar->news cross-attn
on top of the sLSTM output.

Spec: plan/05-MODEL-TRAINING-CALIBRATION.md V1 backbone Decision 1B.
Spec: plan/V1-SPEC.md §2.1 (sparse cross-attn at {3, 7, 11}).
Spec: plan/STATUS.md §53 (`linear` -> `out_proj` rename + InstanceNorm drop).
"""

from __future__ import annotations

from torch import Tensor, nn

from nanogld.model.news_fuser import NewsFuser
from nanogld.model.slstm import sLSTM


class sLSTMBlock(nn.Module):
    """Pre-norm sLSTM encoder block with optional NewsFuser cross-attn.

    Args:
        d_model: hidden dim.
        dropout: dropout after the linear projection.
        has_cross_attn: if True, instantiate a :class:`NewsFuser` for the
            optional news kwargs path. Default False — when False, the
            block is the pure sLSTM variant for layer 12.
        num_heads: heads for the NewsFuser cross-attn (default 4).
        d_text: news embedding dim (default 256).
        n_news_slots: max news slots per bar (default 8).
    """

    def __init__(
        self,
        d_model: int,
        dropout: float = 0.2,
        has_cross_attn: bool = False,
        num_heads: int = 4,
        d_text: int = 256,
        n_news_slots: int = 8,
    ) -> None:
        super().__init__()
        self.bn = nn.GroupNorm(num_groups=1, num_channels=d_model)
        self.slstm = sLSTM(d_model=d_model)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.has_cross_attn = has_cross_attn
        if has_cross_attn:
            self.news_fuser = NewsFuser(
                d_model=d_model,
                d_text=d_text,
                n_heads=num_heads,
                n_news_slots=n_news_slots,
                dropout=dropout,
            )

    def forward(
        self,
        x: Tensor,
        bar_pool: Tensor | None = None,
        news: Tensor | None = None,
        news_mask: Tensor | None = None,
        is_news_present: Tensor | None = None,
    ) -> Tensor:
        """Run the block; optionally fuse news cross-attn if configured.

        Args:
            x: (B, T, d_model).
            bar_pool: (B, d_model) pooled bar context for CFA FiLM. Required
                if ``has_cross_attn`` is True.
            news: (B, S, d_text). Required if ``has_cross_attn`` is True.
            news_mask: (B, S).
            is_news_present: (B,) binary indicator.

        Returns:
            (B, T, d_model) — residual-added output, post-NewsFuser if
            cross-attn is configured and news kwargs are provided.
        """
        h_bn = self.bn(x.transpose(1, 2)).transpose(1, 2)
        h_lstm = self.slstm(h_bn)
        h_proj = self.dropout(self.out_proj(h_lstm))
        h = x + h_proj

        if self.has_cross_attn and news is not None and news_mask is not None:
            if bar_pool is None:
                bar_pool = h.mean(dim=1)
            if is_news_present is None:
                is_news_present = (news_mask.sum(dim=1) > 0).long()
            h = self.news_fuser(
                bar_tokens=h,
                bar_pool=bar_pool,
                news=news,
                news_mask=news_mask,
                is_news_present=is_news_present,
            )
        return h
