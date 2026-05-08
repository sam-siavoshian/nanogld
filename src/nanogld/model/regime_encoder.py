"""V1 regime vector encoder — assembles the 12-dim regime vector at the
model boundary.

Inputs come either as a pre-computed 12-dim tensor (from the dataloader)
or as a dict of pieces (vix_tercile_one_hot, rv_tercile_one_hot,
fomc_week, year_bucket_one_hot, hmm_p). The model accepts the dict
form for flexibility but the dataloader normally pre-computes the
tensor.

Spec: plan/V1-SPEC.md §1.5.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

REGIME_VECTOR_DIM = 12


class RegimeEncoder(nn.Module):
    """Identity-pass module that validates the 12-dim regime vector shape.

    Kept as a Module (not a function) so future versions can add
    learned mixing without breaking the model's forward signature.

    Args:
        regime_dim: expected last-dim of the regime vector. Default 12.
    """

    def __init__(self, regime_dim: int = REGIME_VECTOR_DIM) -> None:
        super().__init__()
        self.regime_dim = regime_dim

    def forward(self, regime: Tensor) -> Tensor:
        """Validate shape and dtype.

        Args:
            regime: shape (B, regime_dim).

        Returns:
            Same tensor (identity), cast to float32.
        """
        if regime.ndim != 2:
            raise ValueError(f"regime must be 2D (B, {self.regime_dim}), got {regime.shape}")
        if regime.shape[-1] != self.regime_dim:
            raise ValueError(f"regime last dim {regime.shape[-1]} != expected {self.regime_dim}")
        return regime.to(torch.float32)


def compute_regime_vec(
    vix_tercile_one_hot: Tensor,
    rv_tercile_one_hot: Tensor,
    fomc_week: Tensor,
    year_bucket_one_hot: Tensor,
    hmm_p: Tensor,
) -> Tensor:
    """Concat the 5 regime components into the canonical 12-dim vector.

    Args:
        vix_tercile_one_hot: shape (B, 3) int8 or float.
        rv_tercile_one_hot:  shape (B, 3) int8 or float.
        fomc_week:           shape (B,) or (B, 1) int8 or float.
        year_bucket_one_hot: shape (B, 4) int8 or float.
        hmm_p:               shape (B,) or (B, 1) float.

    Returns:
        Float32 tensor of shape (B, 12).
    """
    fomc_week = fomc_week.reshape(-1, 1) if fomc_week.ndim == 1 else fomc_week
    hmm_p = hmm_p.reshape(-1, 1) if hmm_p.ndim == 1 else hmm_p
    parts = [
        vix_tercile_one_hot.to(torch.float32),
        rv_tercile_one_hot.to(torch.float32),
        fomc_week.to(torch.float32),
        year_bucket_one_hot.to(torch.float32),
        hmm_p.to(torch.float32),
    ]
    return torch.cat(parts, dim=-1)
