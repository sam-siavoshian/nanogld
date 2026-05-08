"""V1 hybrid encoder — 10 transformer blocks + 2 sLSTM blocks at the head.

Layer plan (1-indexed):
    1, 2:    pure transformer (no cross-attn). Bottom layers form numerical
             features before fusing news.
    3:       transformer + cross-attn (NewsFuser).
    4:       transformer + FiLM regime conditioning.
    5, 6:    transformer + FiLM (every-2 layer pattern).
    7:       transformer + cross-attn.
    8:       transformer + FiLM.
    9, 10:   transformer + FiLM.
    11:      sLSTM block + cross-attn (cross-attn at last attention layer).
    12:      sLSTM block.

FiLM injection layers: {2, 4, 6, 8, 10} (every-2).
Cross-attn injection layers: {3, 7, 11} (mPLUG-Owl3 sparse pattern).
Stochastic depth schedule: linear 0.0 → drop_path_max across all 12 layers.

Spec: plan/05-MODEL-TRAINING-CALIBRATION.md V1 backbone.
Spec: plan/V1-SPEC.md §1.5 + §2.1.
"""

from __future__ import annotations

from torch import Tensor, nn

from nanogld.model.slstm_block import sLSTMBlock
from nanogld.model.transformer_block import TransformerBlock

DEFAULT_FILM_LAYERS: tuple[int, ...] = (2, 4, 6, 8, 10)
DEFAULT_CROSS_ATTN_LAYERS: tuple[int, ...] = (3, 7, 11)
TOTAL_LAYERS = 12


class HybridEncoder(nn.Module):
    """V1 hybrid encoder stack.

    Args:
        d_model: hidden dim.
        num_heads: self-attn head count for transformer blocks.
        max_seq: RoPE cache max length (must be >= num_patches).
        num_transformer_layers: 10 by default (V1 spec).
        num_slstm_layers: 2 by default (V1 spec).
        dropout: per-block dropout.
        drop_path_max: max stochastic-depth probability at deepest layer.
        regime_dim: dim of the regime vector (12 for V1).
        d_text: news embedding dim (256 for Qwen3+MRL).
        n_news_slots: max news slots per bar.
        film_layers: 1-indexed layer indices to apply FiLM.
        cross_attn_layers: 1-indexed layer indices to inject cross-attn.
    """

    def __init__(
        self,
        d_model: int = 384,
        num_heads: int = 6,
        max_seq: int = 16,
        num_transformer_layers: int = 10,
        num_slstm_layers: int = 2,
        dropout: float = 0.2,
        drop_path_max: float = 0.2,
        regime_dim: int = 12,
        d_text: int = 256,
        n_news_slots: int = 8,
        film_layers: tuple[int, ...] = DEFAULT_FILM_LAYERS,
        cross_attn_layers: tuple[int, ...] = DEFAULT_CROSS_ATTN_LAYERS,
    ) -> None:
        super().__init__()
        total = num_transformer_layers + num_slstm_layers
        if total != TOTAL_LAYERS:
            raise ValueError(f"num_transformer + num_slstm must equal {TOTAL_LAYERS}, got {total}")

        self.num_transformer_layers = num_transformer_layers
        self.num_slstm_layers = num_slstm_layers
        self.film_layers = set(film_layers)
        self.cross_attn_layers = set(cross_attn_layers)
        self.d_model = d_model

        drop_path_schedule = [drop_path_max * (i + 1) / TOTAL_LAYERS for i in range(TOTAL_LAYERS)]

        transformer_blocks = []
        for layer_idx in range(num_transformer_layers):
            one_indexed = layer_idx + 1
            transformer_blocks.append(
                TransformerBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    max_seq=max_seq,
                    dropout=dropout,
                    drop_path=drop_path_schedule[layer_idx],
                    has_cross_attn=one_indexed in self.cross_attn_layers,
                    has_film=one_indexed in self.film_layers,
                    regime_dim=regime_dim,
                    d_text=d_text,
                    n_news_slots=n_news_slots,
                )
            )
        self.transformer_blocks = nn.ModuleList(transformer_blocks)

        slstm_blocks = []
        for _ in range(num_slstm_layers):
            slstm_blocks.append(sLSTMBlock(d_model=d_model, dropout=dropout))
        self.slstm_blocks = nn.ModuleList(slstm_blocks)

    def forward(
        self,
        x: Tensor,
        regime: Tensor,
        news: Tensor,
        news_mask: Tensor,
        is_news_present: Tensor,
    ) -> Tensor:
        """Run the full encoder stack.

        Args:
            x: (B', T, d_model) — typically B' = B * num_channels (channel-
               independent reshape from PatchEmbed).
            regime: (B', regime_dim) — regime vector broadcast to per-channel rows.
            news: (B', S, d_text).
            news_mask: (B', S).
            is_news_present: (B',).

        Returns:
            (B', T, d_model) — encoder output.
        """
        h = x
        prev_v: Tensor | None = None
        bar_pool = h.mean(dim=1)

        for layer_idx, block in enumerate(self.transformer_blocks):
            kwargs: dict[str, Tensor | None] = {"prev_v": prev_v}
            if block.has_cross_attn:
                kwargs.update(
                    bar_pool=bar_pool,
                    news=news,
                    news_mask=news_mask,
                    is_news_present=is_news_present,
                )
            if block.has_film:
                kwargs["regime"] = regime
            h, v = block(h, **kwargs)
            prev_v = v
            if (layer_idx + 1) in self.cross_attn_layers or (layer_idx + 1) in self.film_layers:
                bar_pool = h.mean(dim=1)

        for block in self.slstm_blocks:
            h = block(h)

        return h
