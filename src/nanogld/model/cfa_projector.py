"""Constrained Fusion Adapter (CFA) — bar-conditioned text projector.

Sits between the frozen Qwen3 256-d news embedding and the cross-attn
K/V projection. Filters text noise BEFORE the news enters the bar
stream. Lee et al. arXiv:2603.22372 tested 14 TS backbones x 4 text
encoders, ~20K experiments — filtered fusion outperforms unconstrained
fusion at small TS scales.

Pipeline:
    text_in:  (B, S, 256)   raw Qwen3-256d MRL embeddings
    bar_pool: (B, D)        mean-pooled bar tokens conditioning signal

    text_bn = Linear(256, 64)(text_in)               # bottleneck
    text_proj = Linear(64, D)(text_bn)               # back up to model dim

    gamma, beta = Linear(D, 2D)(bar_pool)            # FiLM by bar
    text_filmed = (1 + gamma) * text_proj + beta

    bar_norm_sq = sum(bar_pool^2, dim=-1) + eps
    proj_coeff = sum(text_filmed * bar_pool, dim=-1) / bar_norm_sq
    text_orth = text_filmed - proj_coeff * bar_pool  # orthogonal to bar

    return text_orth  # (B, S, D)

Spec: plan/05-MODEL-TRAINING-CALIBRATION.md V1 news fusion.
Spec: plan/V1-SPEC.md §2.2.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class CFAProjector(nn.Module):
    """Bar-conditioned + orthogonal-residual text projector.

    Args:
        d_text: input news embedding dim (256 from Qwen3+MRL).
        d_bottleneck: low-rank bottleneck dim (64 default).
        d_model: model hidden dim (D=384 default).
        eps: numerical-stability for orthogonal projection.
    """

    def __init__(
        self,
        d_text: int = 256,
        d_bottleneck: int = 64,
        d_model: int = 384,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.eps = eps
        self.text_bn = nn.Linear(d_text, d_bottleneck, bias=False)
        self.text_proj = nn.Linear(d_bottleneck, d_model, bias=False)
        self.film_proj = nn.Linear(d_model, 2 * d_model, bias=True)

        nn.init.zeros_(self.film_proj.weight)
        nn.init.zeros_(self.film_proj.bias)

    def forward(self, text: Tensor, bar_pool: Tensor) -> Tensor:
        """Project + FiLM + orthogonalize.

        Args:
            text: shape (B, S, d_text). S = number of news source slots.
            bar_pool: shape (B, d_model).

        Returns:
            Tensor of shape (B, S, d_model), orthogonal to bar_pool per row.
        """
        text_bn = self.text_bn(text)
        text_proj = self.text_proj(text_bn)

        gamma_beta = self.film_proj(bar_pool)
        gamma, beta = gamma_beta.chunk(2, dim=-1)
        text_filmed = (1.0 + gamma).unsqueeze(1) * text_proj + beta.unsqueeze(1)

        bar_norm_sq = (bar_pool * bar_pool).sum(dim=-1, keepdim=True) + self.eps
        proj_coeff = (text_filmed * bar_pool.unsqueeze(1)).sum(dim=-1, keepdim=True) / bar_norm_sq.unsqueeze(1)
        text_orth = text_filmed - proj_coeff * bar_pool.unsqueeze(1)
        return text_orth
