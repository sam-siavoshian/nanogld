"""DLinear baseline (Zeng AAAI 2023 arXiv:2205.13504).

Simplest possible time-series baseline: split each channel into a trend
moving average + seasonal residual, project both with a single linear
layer to the forecast horizon, sum. We use the *next-bar log return*
as the regression target so we can route the prediction through ``tanh``
into a position weight in ``[-1, +1]``.

Context inputs:

    ctx["train_features_window"]: (N_train, T, F) — lookback windows
    ctx["train_next_log_return"]: (N_train,)
    ctx["test_features_window"]:  (N_test, T, F)

Falls back to all-zero positions when the train arrays aren't supplied
(dry-run path) so the harness still exercises the shape contract.

Spec: plan/V1-SPEC.md §0 (DLinear baseline).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn


class _DLinear(nn.Module):
    """Per-channel decomposition + 2 linear heads + sum, tanh-output."""

    def __init__(self, t_lookback: int, n_features: int, kernel: int = 25) -> None:
        super().__init__()
        self.kernel = kernel
        self.n_features = n_features
        self.t = t_lookback
        self.trend_head = nn.Linear(t_lookback, 1, bias=True)
        self.seasonal_head = nn.Linear(t_lookback, 1, bias=True)
        self.feature_mix = nn.Linear(n_features, 1, bias=True)

    @staticmethod
    def _moving_avg(x: Tensor, kernel: int) -> Tensor:
        pad = (kernel - 1) // 2
        x_perm = x.transpose(1, 2)
        x_padded = F.pad(x_perm, (pad, kernel - 1 - pad), mode="replicate")
        weight = x.new_ones((x.shape[-1], 1, kernel)) / kernel
        trend = F.conv1d(x_padded, weight, groups=x.shape[-1])
        return trend.transpose(1, 2)

    def forward(self, x: Tensor) -> Tensor:
        trend = self._moving_avg(x, self.kernel)
        seasonal = x - trend
        trend_proj = self.trend_head(trend.transpose(1, 2)).squeeze(-1)
        seasonal_proj = self.seasonal_head(seasonal.transpose(1, 2)).squeeze(-1)
        feature_sum = trend_proj + seasonal_proj
        out = self.feature_mix(feature_sum).squeeze(-1)
        return torch.tanh(out)


def dlinear_positions(
    ctx: dict[str, Any],
    *,
    epochs: int = 30,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    seed: int = 0,
) -> np.ndarray:
    """Train DLinear on train windows, emit positions for test windows."""
    test_n = len(ctx["next_log_returns"])
    train_x = ctx.get("train_features_window")
    train_y = ctx.get("train_next_log_return")
    test_x = ctx.get("test_features_window")
    if train_x is None or train_y is None or test_x is None:
        return np.zeros(test_n, dtype=np.float64)

    torch.manual_seed(seed)
    train_x_t = torch.as_tensor(np.asarray(train_x), dtype=torch.float32).nan_to_num_(0.0)
    train_y_t = torch.as_tensor(np.asarray(train_y), dtype=torch.float32).nan_to_num_(0.0)
    test_x_t = torch.as_tensor(np.asarray(test_x), dtype=torch.float32).nan_to_num_(0.0)
    if train_x_t.ndim != 3:
        raise ValueError(f"train_features_window must be (N, T, F); got {train_x_t.shape}")
    if test_x_t.ndim != 3:
        raise ValueError(f"test_features_window must be (N, T, F); got {test_x_t.shape}")

    n_train, t, f_dim = train_x_t.shape
    model = _DLinear(t_lookback=t, n_features=f_dim)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        pred = model(train_x_t)
        loss = F.mse_loss(pred, train_y_t.clamp(-1.0, 1.0))
        loss.backward()
        opt.step()

    model.train(mode=False)
    with torch.no_grad():
        positions = model(test_x_t).cpu().numpy().astype(np.float64)
    if positions.shape[0] != test_n:
        raise RuntimeError(
            f"dlinear_positions: predicted {positions.shape[0]} positions, "
            f"expected {test_n}"
        )
    return positions


__all__ = ["dlinear_positions"]
