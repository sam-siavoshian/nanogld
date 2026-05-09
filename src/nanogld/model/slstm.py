"""sLSTM cell from xLSTMTime (Alharthi & Mahmood arXiv:2407.10240).

Used in the final 2 layers of the V1 hybrid encoder (Decision 1B).
Channel-independent operation — same cell applied per-token sequence.

This is the SIMPLEST sLSTM variant: standard input/forget/cell gates
plus an exponential output gate. Per V1 spec: implement simplest first;
document any deviation from xLSTMTime if param count drifts > 2x baseline.

Math (per timestep t, with previous state (h, c)):
    z = tanh(W_z x_t + R_z h + b_z)         # cell candidate
    i = sigmoid(W_i x_t + R_i h + b_i)      # input gate
    f = sigmoid(W_f x_t + R_f h + b_f)      # forget gate
    o = sigmoid(W_o x_t + R_o h + b_o)      # output gate (sigmoid; xLSTMTime
                                            #              uses exp on stabilized
                                            #              variant; we match
                                            #              vanilla LSTM here)
    c_t = f * c + i * z
    h_t = o * tanh(c_t)

Spec: plan/05-MODEL-TRAINING-CALIBRATION.md V1 backbone Decision 1B.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class sLSTMCell(nn.Module):
    """Single-step sLSTM cell.

    Args:
        d_model: hidden dim (must equal input dim — square cell).
    """

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.w_ih = nn.Linear(d_model, 4 * d_model, bias=True)
        self.w_hh = nn.Linear(d_model, 4 * d_model, bias=False)

    def forward(
        self, x: Tensor, state: tuple[Tensor, Tensor]
    ) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        """Single timestep.

        Args:
            x: shape (B, d_model).
            state: (h, c) each shape (B, d_model).

        Returns:
            (h_new, (h_new, c_new)).
        """
        h, c = state
        gates = self.w_ih(x) + self.w_hh(h)
        z, i, f, o = gates.chunk(4, dim=-1)
        z = torch.tanh(z)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        c_new = f * c + i * z
        h_new = o * torch.tanh(c_new)
        return h_new, (h_new, c_new)


class sLSTM(nn.Module):
    """Sequence-level sLSTM wrapper. Channel-independent.

    Args:
        d_model: hidden dim.
    """

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.cell = sLSTMCell(d_model=d_model)

    def forward(self, x: Tensor) -> Tensor:
        """Run the cell over a sequence.

        Args:
            x: shape (B, T, d_model).

        Returns:
            (B, T, d_model) — h_t at every step.
        """
        b, t, d = x.shape
        h = x.new_zeros((b, d))
        c = x.new_zeros((b, d))
        outputs = []
        for step in range(t):
            h, (h_new, c_new) = self.cell(x[:, step, :], (h, c))
            h, c = h_new, c_new
            outputs.append(h)
        return torch.stack(outputs, dim=1)
