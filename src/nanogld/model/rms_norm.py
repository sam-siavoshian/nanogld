"""RMSNorm — Llama / Qwen consensus norm. No mean centering, no bias.

Spec: plan/05-MODEL-TRAINING-CALIBRATION.md V1 backbone.
Spec: plan/V1-SPEC.md §3 (preserved from V1 draft).

Math:
    y = x * rsqrt(mean(x^2, dim=-1, keepdim=True) + eps) * weight

`weight` is a learnable per-feature gain initialized to 1.0. No bias term.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class RMSNorm(nn.Module):
    """Root-mean-square layer norm over the last dimension.

    Args:
        dim: feature dimension (last axis of input).
        eps: numerical-stability floor inside the rsqrt.
    """

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return x * rms * self.weight
