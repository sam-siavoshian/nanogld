"""TimeMixer baseline (Wang et al. ICLR 2024 arXiv:2405.14616).

Multi-scale decomposition + per-scale FFN heads, summed at output.
Lightweight variant: build trend at scales {2, 4, 8, 16} via moving
averages, project each through a 2-layer MLP, sum to a single scalar
prediction routed through tanh.

Context contract matches DLinear / TSMixer.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn


def _moving_avg(x: Tensor, kernel: int) -> Tensor:
    if kernel <= 1:
        return x
    pad = (kernel - 1) // 2
    x_perm = x.transpose(1, 2)
    x_padded = F.pad(x_perm, (pad, kernel - 1 - pad), mode="replicate")
    weight = x.new_ones((x.shape[-1], 1, kernel)) / kernel
    trend = F.conv1d(x_padded, weight, groups=x.shape[-1])
    return trend.transpose(1, 2)


class _TimeMixer(nn.Module):
    def __init__(
        self,
        t_dim: int,
        f_dim: int,
        scales: tuple[int, ...] = (2, 4, 8, 16),
        hidden: int = 64,
    ) -> None:
        super().__init__()
        self.scales = scales
        self.heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(t_dim * f_dim, hidden),
                    nn.GELU(),
                    nn.Linear(hidden, 1),
                )
                for _ in scales
            ]
        )
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x: Tensor) -> Tensor:
        acc = self.bias.expand(x.shape[0])
        for scale, head in zip(self.scales, self.heads, strict=True):
            scaled = _moving_avg(x, scale)
            flat = scaled.flatten(start_dim=1)
            acc = acc + head(flat).squeeze(-1)
        return torch.tanh(acc / float(len(self.scales)))


def timemixer_positions(
    ctx: dict[str, Any],
    *,
    epochs: int = 30,
    lr: float = 1e-3,
    seed: int = 0,
) -> np.ndarray:
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
    _, t, f_dim = train_x_t.shape

    model = _TimeMixer(t_dim=t, f_dim=f_dim)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
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
    return positions


__all__ = ["timemixer_positions"]
