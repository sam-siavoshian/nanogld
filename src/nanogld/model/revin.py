"""Reversible Instance Normalization (RevIN) per channel.

Kim et al. ICLR 2022: per-instance affine normalization that preserves
the ability to invert at the output. Used to handle non-stationarity
in financial time series.

V1 upgrade from V1-draft: per-CHANNEL (681 instances) instead of
per-group. Huang & Yang ESWA 2026 reports RMSE -50%, MAPE -54% on
cross-market stock data.

Modes:
    "norm"   : compute and store mean/std, return normalized x.
    "denorm" : un-normalize using stored stats.

The mean/std are stored on the module as state (one set per forward),
so caller must invoke `mode="norm"` first to fill stats, then
`mode="denorm"` to invert. Stats are NOT persisted across forward calls
(they are per-instance / per-batch).

Spec: plan/05-MODEL-TRAINING-CALIBRATION.md V1 backbone.
Spec: plan/V1-SPEC.md §4.4.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class RevIN(nn.Module):
    """Per-channel Reversible Instance Norm.

    Args:
        num_features: number of channels (last dim of input).
        eps: numerical-stability floor for the std.
        affine: whether to apply learnable affine transform after norm.
    """

    def __init__(self, num_features: int, eps: float = 1e-5, affine: bool = True) -> None:
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if affine:
            self.affine_weight = nn.Parameter(torch.ones(num_features))
            self.affine_bias = nn.Parameter(torch.zeros(num_features))

        self._mean: Tensor | None = None
        self._stdev: Tensor | None = None

    def forward(self, x: Tensor, mode: str) -> Tensor:
        """Apply normalization in the requested direction.

        Args:
            x: shape (B, T, F).
            mode: "norm" or "denorm".

        Returns:
            Tensor with same shape as `x`.
        """
        if mode == "norm":
            return self._normalize(x)
        if mode == "denorm":
            return self._denormalize(x)
        raise ValueError(f"mode must be 'norm' or 'denorm', got {mode!r}")

    def _normalize(self, x: Tensor) -> Tensor:
        dim_to_reduce = tuple(range(1, x.ndim - 1))
        self._mean = x.mean(dim=dim_to_reduce, keepdim=True).detach()
        self._stdev = torch.sqrt(
            x.var(dim=dim_to_reduce, keepdim=True, unbiased=False) + self.eps
        ).detach()
        y = (x - self._mean) / self._stdev
        if self.affine:
            y = y * self.affine_weight + self.affine_bias
        return y

    def _denormalize(self, x: Tensor) -> Tensor:
        y = x
        if self.affine:
            y = (y - self.affine_bias) / (self.affine_weight + self.eps)
        if self._stdev is None or self._mean is None:
            raise RuntimeError("RevIN.denorm called before norm — no stored stats")
        return y * self._stdev + self._mean
