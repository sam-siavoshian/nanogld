"""SwiGLU FFN — Llama / Qwen consensus feed-forward. No bias.

Math:
    hidden = round_to_multiple(8 * D / 3, 64)
    y = w_down(silu(w_gate(x)) * w_up(x))   # all bias-free
    y = dropout(y)

At D=384, hidden = 1024.

Spec: plan/05-MODEL-TRAINING-CALIBRATION.md V1 backbone.
Spec: plan/V1-SPEC.md §3.
"""

from __future__ import annotations

import torch.nn.functional as F
from torch import Tensor, nn


def swiglu_hidden_dim(d: int, multiple_of: int = 64) -> int:
    """Round (8 * D / 3) up to the nearest multiple."""
    raw = int(8 * d / 3)
    return ((raw + multiple_of - 1) // multiple_of) * multiple_of


class SwiGLU(nn.Module):
    """Bias-free SwiGLU FFN block.

    Args:
        d_model: input/output feature dim.
        hidden_dim: optional override; default = swiglu_hidden_dim(d_model).
        dropout: dropout rate after w_down.
    """

    def __init__(
        self,
        d_model: int,
        hidden_dim: int | None = None,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        h = hidden_dim if hidden_dim is not None else swiglu_hidden_dim(d_model)
        self.w_gate = nn.Linear(d_model, h, bias=False)
        self.w_up = nn.Linear(d_model, h, bias=False)
        self.w_down = nn.Linear(h, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        return self.dropout(self.w_down(F.silu(self.w_gate(x)) * self.w_up(x)))
