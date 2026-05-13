"""Channel-independent patching (PatchTST-style) — Decision 2B.

Splits a (B, T_bars, num_channels) input into patches of length P,
treats each channel independently (no cross-channel mixing in main
backbone), projects each patch to D via a shared Linear(P, D).

Output shape: (B * num_channels, num_patches, D).

The backbone runs on this reshaped tensor as if each channel were its
own batch element. Channel mixing is recovered later via:
  - VSN feature gate at input (per-bar feature importance)
  - FiLM regime conditioning (cross-feature via regime vector)

Saly-Kaufmann/Wood/Zohren 2026 (arXiv:2603.01820): LPatchTST = 2.31
Sharpe vs iTransformer (channel-as-token) = 0.38 Sharpe on daily
futures. Channel-independence is the right inductive bias for
non-stationary financial data at 75K samples.

Spec: plan/05-MODEL-TRAINING-CALIBRATION.md V1 backbone.
Spec: plan/V1-SPEC.md §1 Decision 2B.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class PatchEmbed(nn.Module):
    """Channel-independent patching with shared linear projection.

    Args:
        patch_len: number of bars per patch (P=4 default).
        patch_stride: stride between patches (S=4 default; non-overlapping).
        t_bars: lookback length in bars (T=64 default → 16 patches).
        d_model: output embedding dim (D=384 default).
    """

    def __init__(
        self,
        patch_len: int = 4,
        patch_stride: int = 4,
        t_bars: int = 64,
        d_model: int = 384,
    ) -> None:
        super().__init__()
        if t_bars % patch_stride != 0:
            raise ValueError(
                f"t_bars={t_bars} must be divisible by patch_stride={patch_stride}"
            )
        self.patch_len = patch_len
        self.patch_stride = patch_stride
        self.num_patches = t_bars // patch_stride
        self.d_model = d_model

        self.proj = nn.Linear(patch_len, d_model, bias=False)

        pos = self._sinusoidal_pos_emb(self.num_patches, d_model)
        self.register_buffer("pos_emb", pos.unsqueeze(0), persistent=False)

    @staticmethod
    def _sinusoidal_pos_emb(num_patches: int, d_model: int) -> Tensor:
        position = torch.arange(num_patches, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-(torch.log(torch.tensor(10000.0)) / d_model))
        )
        pe = torch.zeros(num_patches, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe

    def forward(self, x: Tensor) -> Tensor:
        """Reshape and project.

        Args:
            x: shape (B, T_bars, num_channels).

        Returns:
            Tensor of shape (B * num_channels, num_patches, d_model).
        """
        b, t, c = x.shape
        expected_t = self.num_patches * self.patch_stride
        if t != expected_t:
            raise ValueError(
                f"input T={t} does not match configured T_bars={expected_t}"
            )
        x_perm = x.permute(0, 2, 1)
        x_patched = x_perm.unfold(dimension=-1, size=self.patch_len, step=self.patch_stride)
        x_flat = x_patched.reshape(b * c, self.num_patches, self.patch_len)
        embedded = self.proj(x_flat)
        return embedded + self.pos_emb
