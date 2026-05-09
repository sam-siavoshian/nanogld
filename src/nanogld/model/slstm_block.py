"""Pre-norm sLSTM block (xLSTMTime style).

Used for layers 11-12 of the V1 hybrid encoder (Decision 1B).

Block layout (per xLSTMTime Alharthi & Mahmood arXiv:2407.10240,
adapted for B=1 safety: BN swapped for GroupNorm(1, d_model)):
    h = GroupNorm(x)
    h = sLSTM(h)
    h = Linear(h)
    h = InstanceNorm1d(h)
    return x + h    # residual

Channel-independent — sLSTM operates per (B, T) sequence, no cross-channel
mixing.

Spec: plan/05-MODEL-TRAINING-CALIBRATION.md V1 backbone Decision 1B.
"""

from __future__ import annotations

from torch import Tensor, nn

from nanogld.model.slstm import sLSTM


class sLSTMBlock(nn.Module):
    """Pre-norm sLSTM encoder block.

    Args:
        d_model: hidden dim.
        dropout: dropout after the linear projection.
    """

    def __init__(self, d_model: int, dropout: float = 0.2) -> None:
        super().__init__()
        self.bn = nn.GroupNorm(num_groups=1, num_channels=d_model)
        self.slstm = sLSTM(d_model=d_model)
        self.linear = nn.Linear(d_model, d_model, bias=False)
        self.in_norm = nn.InstanceNorm1d(d_model, affine=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        """Run the block.

        Args:
            x: (B, T, d_model).

        Returns:
            (B, T, d_model) — residual-added output.
        """
        h_bn = self.bn(x.transpose(1, 2)).transpose(1, 2)
        h_lstm = self.slstm(h_bn)
        h_proj = self.dropout(self.linear(h_lstm))
        h_in = self.in_norm(h_proj.transpose(1, 2)).transpose(1, 2)
        return x + h_in
