"""VLSTM baseline (Saly-Kaufmann/Wood/Zohren CVPR 2026 arXiv:2603.01820).

LSTM + Variable Selection Network feature gate. Saly-Kaufmann benchmark
reports VLSTM 2.40 Sharpe vs plain LSTM 1.48 on daily futures — the
VSN gate is the +0.92 Sharpe delta and is the bar nanoGLD must clear
on Gate 3.

Pipeline:

    x (B, T, F)
    -> VSN feature gate (per-time multiplicative re-weight)
    -> LSTM (hidden=d_hidden, num_layers=1)
    -> mean across T
    -> linear head
    -> tanh -> position

Reuses ``nanogld.model.vsn.VSN``. Channel-independent.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from nanogld.model.vsn import VSN


class _VLSTM(nn.Module):
    def __init__(self, n_features: int, d_hidden: int = 32) -> None:
        super().__init__()
        self.vsn = VSN(num_features=n_features, hidden_dim=64, dropout=0.1)
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=d_hidden,
            num_layers=1,
            batch_first=True,
        )
        self.out_head = nn.Linear(d_hidden, 1, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        gated, _gate = self.vsn(x)
        out, _ = self.lstm(gated)
        pooled = out.mean(dim=1)
        scalar = self.out_head(pooled).squeeze(-1)
        return torch.tanh(scalar)


def vlstm_positions(
    ctx: dict[str, Any],
    *,
    d_hidden: int = 32,
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

    model = _VLSTM(n_features=n_features, d_hidden=d_hidden)
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


__all__ = ["vlstm_positions"]
