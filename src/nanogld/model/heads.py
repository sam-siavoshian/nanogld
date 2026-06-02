"""V1 multi-task output — Decision 3B + DANN era classifier (V1-SPEC §6.7).

Head A (3-class direction):
    logits = Linear(D, 3, bias=False)(mean_pooled_tokens)
    Loss: focal CE gamma=3 (in training/losses.py).

Head B (position weight):
    raw = Linear(D, 1, bias=False)(mean_pooled_tokens)
    position = tanh(raw)        # in [-1, +1]
    Loss: differentiable -Sharpe (in training/losses.py).

DANN head (domain-adversarial era classifier):
    domain_logits = Linear(D, num_eras, bias=False)(grad_reverse(pooled, alpha))
    Loss: CE over year-bucket labels. Lambda ramps 0 -> 0.1 over training.

The DANN head wraps the encoder pooled rep with a gradient reversal so
the encoder is *adversarially* trained to confuse the era classifier
(Feng 2019 +3.11% on stock-movement prediction). Use ``dann_forward``
from the training loop with the current ``alpha`` scalar.

All heads share the same encoder. Head A outputs feed calibration
(T-scaling, RAPS, AgACI); Head B output is the primary position weight
for sizing; the DANN head is training-only (its outputs are not used at
inference).

Spec: plan/05-MODEL-TRAINING-CALIBRATION.md V1 head Decision 3B.
Spec: plan/V1-SPEC.md §6.7 (DANN gradient reversal on era-label).
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

DEFAULT_NUM_ERAS = 4  # year-bucket {2016-2019, 2020-2022, 2023-2024, 2025+}


class MultiTaskHead(nn.Module):
    """Joint 3-class CE + position-weight Sharpe head + DANN era classifier.

    Args:
        d_model: encoder hidden dim.
        n_classes: number of classes for Head A (3 default: DOWN/FLAT/UP).
        num_eras: number of year-bucket eras for the DANN discriminator.
    """

    def __init__(
        self, d_model: int, n_classes: int = 3, num_eras: int = DEFAULT_NUM_ERAS
    ) -> None:
        super().__init__()
        self.cls_head = nn.Linear(d_model, n_classes, bias=False)
        self.pos_head = nn.Linear(d_model, 1, bias=False)
        self.dann_head = nn.Linear(d_model, num_eras, bias=False)

    def forward(self, pooled: Tensor) -> tuple[Tensor, Tensor]:
        """Compute Head A + Head B outputs.

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

    def dann_forward(self, pooled: Tensor, alpha: float) -> Tensor:
        """Domain classifier logits via gradient reversal.

        Args:
            pooled: (B, d_model) encoder pooled rep.
            alpha: gradient reversal scale; 0 disables backprop into encoder.

        Returns:
            (B, num_eras) era logits.
        """
        # Lazy import: heads.py is imported by model/__init__ very early,
        # while nanogld.training.losses imports from nanogld.model.aecf —
        # importing grad_reverse at module top creates a circular cycle.
        from nanogld.training.losses import grad_reverse  # noqa: PLC0415

        return self.dann_head(grad_reverse(pooled, alpha))
