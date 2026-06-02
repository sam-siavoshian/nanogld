"""xLSTMTime baseline (Alharthi & Mahmood arXiv:2407.10240).

Reuses the in-tree :class:`nanogld.model.slstm_block.sLSTMBlock` rather
than the external ``xlstm`` package (see plan deviation note). Pipeline:

    x (B, T, F)
    -> mean across feature dim -> (B, T, 1)
    -> per-channel project to d_model
    -> sLSTMBlock
    -> mean across T
    -> linear head
    -> tanh -> position weight

Channel-independent variant; not aiming to beat the full xLSTMTime
paper recipe — just provide the recurrent-state alternative to the
attention-only baselines for Gate 3 comparison.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from nanogld.model.slstm_block import sLSTMBlock


class _XLSTMTimeBaseline(nn.Module):
    def __init__(self, n_features: int, d_model: int = 32) -> None:
        super().__init__()
        self.in_proj = nn.Linear(n_features, d_model, bias=False)
        self.slstm = sLSTMBlock(d_model=d_model, dropout=0.1, has_cross_attn=False)
        self.out_head = nn.Linear(d_model, 1, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, T, F) -> (B, T, d_model)
        h = self.in_proj(x)
        h = self.slstm(h)
        pooled = h.mean(dim=1)
        out = self.out_head(pooled).squeeze(-1)
        return torch.tanh(out)


def xlstm_time_positions(
    ctx: dict[str, Any],
    *,
    d_model: int = 32,
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
    n_features = train_x_t.shape[-1]

    model = _XLSTMTimeBaseline(n_features=n_features, d_model=d_model)
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


__all__ = ["xlstm_time_positions"]
