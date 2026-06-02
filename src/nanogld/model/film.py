"""FiLM affine modulation — per-feature gamma/beta from a regime vector.

Used to inject the 12-dim regime conditioning every 2 transformer layers
(layers {2, 4, 6, 8, 10}). Adds ~2 * D params per FiLM block.

Math:
    gamma, beta = Linear_l(regime)  # split into halves
    y = (1 + gamma) * x + beta      # +1 makes init identity-like

Initialization: gamma_l weights to 0 so post-init `1 + gamma = 1` (identity).
beta_l weights to 0 so initial bias is 0. Result: at init, FiLM is the
identity transform, preserving pretrained behavior.

Spec: plan/05-MODEL-TRAINING-CALIBRATION.md V1 backbone.
Spec: plan/V1-SPEC.md §1.5.
"""

from __future__ import annotations

import torch.nn.init as init
from torch import Tensor, nn


class FiLMConditioner(nn.Module):
    """Feature-wise Linear Modulation conditioned on a regime vector.

    Args:
        d_model: model hidden dim.
        regime_dim: dim of the regime vector (12 for V1).
    """

    def __init__(self, d_model: int, regime_dim: int = 12) -> None:
        super().__init__()
        self.proj = nn.Linear(regime_dim, 2 * d_model, bias=True)
        init.zeros_(self.proj.weight)
        init.zeros_(self.proj.bias)

    def forward(self, x: Tensor, regime: Tensor) -> Tensor:
        """Apply FiLM modulation.

        Args:
            x: shape (B, T, D).
            regime: shape (B, regime_dim).

        Returns:
            Tensor of same shape as `x`.
        """
        gamma_beta = self.proj(regime)
        gamma, beta = gamma_beta.chunk(2, dim=-1)
        return (1.0 + gamma).unsqueeze(1) * x + beta.unsqueeze(1)
