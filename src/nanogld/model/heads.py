"""V1 dual-head output — Decision 3B (multi-task).

Head A (3-class direction):
    logits = Linear(D, 3, bias=False)(mean_pooled_tokens)
    Loss: focal CE gamma=3 (in training/losses.py).

Head B (position weight):
    raw = Linear(D, 1, bias=False)(mean_pooled_tokens)
    position = tanh(raw)        # in [-1, +1]
    Loss: differentiable -Sharpe (in training/losses.py).

Both heads share the same encoder. Head A outputs feed calibration
(T-scaling, RAPS, AgACI); Head B output is the primary position weight
for sizing.

Spec: plan/05-MODEL-TRAINING-CALIBRATION.md V1 head Decision 3B.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class MultiTaskHead(nn.Module):
    """Joint 3-class CE + position-weight Sharpe head.

    Args:
        d_model: encoder hidden dim.
        n_classes: number of classes for Head A (3 default: DOWN/FLAT/UP).
    """

    def __init__(self, d_model: int, n_classes: int = 3) -> None:
        super().__init__()
        self.cls_head = nn.Linear(d_model, n_classes, bias=False)
        self.pos_head = nn.Linear(d_model, 1, bias=False)

    def forward(self, pooled: Tensor) -> tuple[Tensor, Tensor]:
        """Compute both head outputs.

        Args:
            pooled: shape (B, d_model). Mean-pooled encoder tokens.

        Returns:
            (logits_3class, position_weight) of shapes (B, n_classes), (B,).
            position_weight is in [-1, +1] (tanh applied).
        """
        logits_3class = self.cls_head(pooled)
        pos_raw = self.pos_head(pooled).squeeze(-1)
        position_weight = torch.tanh(pos_raw)
        return logits_3class, position_weight
