"""Variable Selection Network (VSN) — Lim et al. 2021 TFT.

Per-feature softmax-gated selection via a Gated Residual Network (GRN).
The biggest single delta in the Saly-Kaufmann/Wood/Zohren 2026 financial
benchmark: VLSTM (LSTM + VSN) hit 2.40 Sharpe vs plain LSTM 1.48 Sharpe
on daily futures (+0.92 Sharpe).

GRN block: 2-layer MLP with ELU + GLU + LayerNorm.

VSN: per-feature GRN → per-feature scalar gate → softmax across features.
Then `x_gated = (gate * x_proj_per_feature).sum(...)` — but here we keep
the (B, T, F) shape and produce gates that re-weight rather than collapse.

Spec: plan/04-FEATURE-ENGINEERING.md V1 VSN section.
Spec: plan/V1-SPEC.md §4.2.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class GRN(nn.Module):
    """Gated Residual Network from TFT.

    Math:
        a = ELU(W1 x)
        b = W2 a
        gate = sigmoid(W3 a)
        y = LayerNorm(x_residual + gate * b)
    """

    def __init__(
        self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float = 0.2
    ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim, bias=False)
        self.fc2 = nn.Linear(hidden_dim, output_dim, bias=False)
        self.gate = nn.Linear(hidden_dim, output_dim, bias=False)
        self.skip = (
            nn.Linear(input_dim, output_dim, bias=False)
            if input_dim != output_dim
            else nn.Identity()
        )
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(output_dim)

    def forward(self, x: Tensor) -> Tensor:
        a = F.elu(self.fc1(x))
        b = self.dropout(self.fc2(a))
        g = torch.sigmoid(self.gate(a))
        residual = self.skip(x)
        return self.norm(residual + g * b)


class VSN(nn.Module):
    """Variable Selection Network with per-feature softmax gates.

    Input/output shape: (B, T, num_features).

    Per timestep, computes a softmax-weighted gate vector over features,
    then re-scales the input feature-wise. Strictly multiplicative — does
    NOT collapse features into a smaller embedding.

    Args:
        num_features: number of input features (681 for V1).
        hidden_dim: GRN hidden width (64 default).
        dropout: GRN internal dropout.
    """

    def __init__(self, num_features: int, hidden_dim: int = 64, dropout: float = 0.2) -> None:
        super().__init__()
        self.num_features = num_features
        self.gate_grn = GRN(
            input_dim=num_features,
            hidden_dim=hidden_dim,
            output_dim=num_features,
            dropout=dropout,
        )

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Apply per-feature gating.

        Args:
            x: shape (B, T, num_features).

        Returns:
            (x_gated, gate) — both shape (B, T, num_features). gate is
            the softmax-normalized re-weighting applied to x.
        """
        raw = self.gate_grn(x)
        gate = F.softmax(raw, dim=-1)
        gate_scaled = gate * self.num_features
        return x * gate_scaled, gate
