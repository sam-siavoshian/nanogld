"""Series decomposition — 24-bar moving-avg trend + seasonal split.

Required by xLSTMTime recipe (Alharthi & Mahmood arXiv:2407.10240) and
Autoformer-style pre-norm decomposition. Splits each channel into:

    trend    = causal moving-average over `kernel_size` bars
    seasonal = x - trend

Both streams feed downstream RevIN / VSN / patch-projection, then sum
back inside the encoder.

Causal: uses left-side replication padding so position T's trend value
depends only on positions <= T. No future leakage.

Spec: plan/05-MODEL-TRAINING-CALIBRATION.md V1 backbone.
Spec: plan/V1-SPEC.md §4.3.
"""

from __future__ import annotations

import torch.nn.functional as F
from torch import Tensor, nn


class SeriesDecomposition(nn.Module):
    """Causal moving-average trend/seasonal decomposition.

    Args:
        kernel_size: window size in bars (24 for V1).
    """

    def __init__(self, kernel_size: int = 24) -> None:
        super().__init__()
        if kernel_size < 1:
            raise ValueError(f"kernel_size must be >= 1, got {kernel_size}")
        self.kernel_size = kernel_size

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Split (B, T, F) into (trend, seasonal).

        Args:
            x: shape (B, T, F).

        Returns:
            (trend, seasonal) each shape (B, T, F).
        """
        b, t, f = x.shape
        x_perm = x.transpose(1, 2)  # (B, F, T)
        pad_left = self.kernel_size - 1
        x_padded = F.pad(x_perm, (pad_left, 0), mode="replicate")
        weight = x.new_ones((f, 1, self.kernel_size)) / self.kernel_size
        trend_perm = F.conv1d(x_padded, weight, groups=f)
        trend = trend_perm.transpose(1, 2)
        seasonal = x - trend
        return trend, seasonal
