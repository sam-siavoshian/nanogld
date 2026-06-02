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
    """Variable Selection Network with per-feature gating.

    Input/output shape: (B, T, num_features).

    Two gate activations are supported (V1-SPEC §44 + Agent 4 finding):

    - ``"softmax_scaled"`` (default for V1/V2 backward-compat): per-step
      softmax across features, multiplied by ``num_features``. The
      multiplier means mean-gate=1.0 by construction, so the network
      can only *redistribute* mass and CANNOT prune any feature. This
      is the V1 spec's intent (a re-weighting, not a selector).
    - ``"sigmoid"`` (Agent 4 fix): per-feature independent sigmoid in
      ``[0, 1]``. Mean gate is data-dependent; the network can drive
      features to ~0 (true pruning). Pairs with an L1 penalty on the
      gate magnitudes elsewhere in the loss for full sparsity pressure.

    The flag is read from the env var ``NANOGLD_VSN_GATE`` so existing
    checkpoints load cleanly with the original softmax_scaled behavior;
    pass ``NANOGLD_VSN_GATE=sigmoid`` to opt new training into the fix.

    Args:
        num_features: number of input features (681 for V1).
        hidden_dim: GRN hidden width (128 default — V1-SPEC §44 width bump
            from 64 to 128; +130K params, lands at 24.0M floor target).
        dropout: GRN internal dropout.
        gate_activation: ``"softmax_scaled"`` (default) or ``"sigmoid"``.
    """

    def __init__(
        self,
        num_features: int,
        hidden_dim: int = 128,
        dropout: float = 0.2,
        gate_activation: str | None = None,
    ) -> None:
        super().__init__()
        self.num_features = num_features
        self.gate_grn = GRN(
            input_dim=num_features,
            hidden_dim=hidden_dim,
            output_dim=num_features,
            dropout=dropout,
        )
        import os  # noqa: PLC0415

        resolved = gate_activation or os.environ.get(
            "NANOGLD_VSN_GATE", "softmax_scaled"
        )
        if resolved not in ("softmax_scaled", "sigmoid"):
            raise ValueError(
                f"unknown gate_activation {resolved!r}; "
                "expected 'softmax_scaled' or 'sigmoid'"
            )
        self.gate_activation = resolved

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Apply per-feature gating.

        Args:
            x: shape (B, T, num_features).

        Returns:
            (x_gated, gate). Both shape (B, T, num_features). ``gate``
            is the gate vector before any re-scaling — useful for L1
            regularization and for the attribution suite.
        """
        raw = self.gate_grn(x)
        if self.gate_activation == "sigmoid":
            gate = torch.sigmoid(raw)
            return x * gate, gate
        gate = F.softmax(raw, dim=-1)
        return x * (gate * self.num_features), gate
