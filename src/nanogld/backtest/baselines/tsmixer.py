"""TSMixer baseline (Chen et al. ICLR 2023 arXiv:2303.06053).

MLP-Mixer for time series: alternating per-time MLP (mixes across time
steps for each channel) and per-feature MLP (mixes across features for
each time step). 2 layers each. Output: scalar position prediction
through tanh.

Context contract matches DLinear: ``train_features_window``,
``train_next_log_return``, ``test_features_window``. Falls back to
zeros when train data missing.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn


class _MixerBlock(nn.Module):
    def __init__(
        self,
        t_dim: int,
        f_dim: int,
        hidden_t: int,
        hidden_f: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.norm_t = nn.LayerNorm([t_dim, f_dim])
        self.mlp_t = nn.Sequential(
            nn.Linear(t_dim, hidden_t),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_t, t_dim),
            nn.Dropout(dropout),
        )
        self.norm_f = nn.LayerNorm([t_dim, f_dim])
        self.mlp_f = nn.Sequential(
            nn.Linear(f_dim, hidden_f),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_f, f_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        # Time-mixing
        h = self.norm_t(x).transpose(1, 2)
        h = self.mlp_t(h).transpose(1, 2)
        x = x + h
        # Feature-mixing
        h = self.norm_f(x)
        h = self.mlp_f(h)
        return x + h


class _TSMixer(nn.Module):
    def __init__(
        self,
        t_dim: int,
        f_dim: int,
        n_blocks: int = 2,
        hidden_t: int = 32,
        hidden_f: int = 32,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [_MixerBlock(t_dim, f_dim, hidden_t, hidden_f, dropout) for _ in range(n_blocks)]
        )
        self.head = nn.Linear(t_dim * f_dim, 1)

    def forward(self, x: Tensor) -> Tensor:
        for block in self.blocks:
            x = block(x)
        flat = x.flatten(start_dim=1)
        return torch.tanh(self.head(flat).squeeze(-1))


def tsmixer_positions(
    ctx: dict[str, Any],
    *,
    epochs: int = 30,
    lr: float = 1e-3,
    hidden_t: int = 32,
    hidden_f: int = 32,
    n_blocks: int = 2,
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
    n_train, t, f_dim = train_x_t.shape

    model = _TSMixer(
        t_dim=t, f_dim=f_dim, n_blocks=n_blocks, hidden_t=hidden_t, hidden_f=hidden_f
    )
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


__all__ = ["tsmixer_positions"]
