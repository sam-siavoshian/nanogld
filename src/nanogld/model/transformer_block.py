"""Pre-norm transformer block with optional cross-attn + FiLM regime.

Block layout:
    h = x + drop_path(attn(rms_norm_1(x), prev_v))   # self-attn
    if has_cross_attn:
        h = news_fuser(h, ...)                       # cross-attn
    if has_film:
        h = film(h, regime)                          # FiLM modulation
    h = h + drop_path(ffn(rms_norm_2(h)))            # SwiGLU FFN

Per V1-SPEC §3, blocks 1-10 are transformer blocks; blocks 11-12 are
sLSTM blocks (separate module). Cross-attn fires at layers {3, 7, 11};
FiLM fires at layers {2, 4, 6, 8, 10}.

Spec: plan/05-MODEL-TRAINING-CALIBRATION.md V1 backbone.
"""

from __future__ import annotations

from torch import Tensor, nn

from nanogld.model.attention import MultiHeadAttention
from nanogld.model.film import FiLMConditioner
from nanogld.model.news_fuser import NewsFuser
from nanogld.model.rms_norm import RMSNorm
from nanogld.model.swiglu import SwiGLU


class TransformerBlock(nn.Module):
    """Pre-norm transformer encoder block.

    Args:
        d_model: hidden dim.
        num_heads: self-attn head count.
        max_seq: RoPE cache max length.
        dropout: FFN dropout (also self-attn dropout).
        drop_path: stochastic-depth probability for this layer.
        partial_rope_frac: fraction of head_dim to rotate.
        has_cross_attn: insert NewsFuser before FFN.
        has_film: insert FiLMConditioner before FFN.
        regime_dim: regime vector dim (only used if has_film).
        d_text: news embedding dim (only used if has_cross_attn).
        n_news_slots: max news slots (only used if has_cross_attn).
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        max_seq: int,
        dropout: float = 0.2,
        drop_path: float = 0.0,
        partial_rope_frac: float = 0.10,
        has_cross_attn: bool = False,
        has_film: bool = False,
        regime_dim: int = 12,
        d_text: int = 256,
        n_news_slots: int = 8,
    ) -> None:
        super().__init__()
        self.drop_path = drop_path
        self.has_cross_attn = has_cross_attn
        self.has_film = has_film

        self.norm1 = RMSNorm(d_model)
        self.attn = MultiHeadAttention(
            d_model=d_model,
            num_heads=num_heads,
            max_seq=max_seq,
            dropout=dropout,
            partial_rope_frac=partial_rope_frac,
        )
        if has_cross_attn:
            self.cross_attn = NewsFuser(
                d_model=d_model,
                d_text=d_text,
                n_heads=max(1, num_heads // 2),
                n_news_slots=n_news_slots,
                dropout=dropout,
            )
        if has_film:
            self.film = FiLMConditioner(d_model=d_model, regime_dim=regime_dim)

        self.norm2 = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model=d_model, dropout=dropout)

    def _drop_path(self, x: Tensor) -> Tensor:
        if not self.training or self.drop_path == 0.0:
            return x
        keep = 1.0 - self.drop_path
        mask = x.new_empty((x.shape[0], 1, 1)).bernoulli_(keep) / keep
        return x * mask

    def forward(
        self,
        x: Tensor,
        prev_v: Tensor | None = None,
        bar_pool: Tensor | None = None,
        news: Tensor | None = None,
        news_mask: Tensor | None = None,
        is_news_present: Tensor | None = None,
        regime: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Run the block.

        Args:
            x: (B, T, d_model).
            prev_v: optional V from previous block for IMU-1 value residual.
            bar_pool: required if has_cross_attn — (B, d_model).
            news: required if has_cross_attn — (B, S, d_text).
            news_mask: required if has_cross_attn — (B, S).
            is_news_present: required if has_cross_attn — (B,).
            regime: required if has_film — (B, regime_dim).

        Returns:
            (out, v_for_next).
        """
        attn_out, v_for_next = self.attn(self.norm1(x), prev_v=prev_v)
        h = x + self._drop_path(attn_out)

        if self.has_cross_attn:
            assert (
                bar_pool is not None
                and news is not None
                and news_mask is not None
                and is_news_present is not None
            )
            h = self.cross_attn(
                bar_tokens=h,
                bar_pool=bar_pool,
                news=news,
                news_mask=news_mask,
                is_news_present=is_news_present,
            )

        if self.has_film:
            assert regime is not None
            h = self.film(h, regime)

        h = h + self._drop_path(self.ffn(self.norm2(h)))
        return h, v_for_next
