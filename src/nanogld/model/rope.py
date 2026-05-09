"""Real-form Rotary Position Embedding (RoPE), partial 10% application.

CRITICAL — V1 invariant 9: NEVER use `torch.view_as_complex`. The complex
form of RoPE is silently broken on Apple MPS and produces wrong results.
We use the real-form rotation only:

    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    rotated = stack([x1*cos - x2*sin, x1*sin + x2*cos]).flatten(-2)

This same kernel runs identically on CUDA, MPS, and CPU.

Partial RoPE: only the first `frac * head_dim` features get rotated; the
remaining features pass through unchanged. arXiv:2603.11611 finds 10%
partial application matches full RoPE on most tasks at lower compute.

Spec: plan/05-MODEL-TRAINING-CALIBRATION.md V1 backbone.
Spec: plan/V1-SPEC.md §3.
"""

from __future__ import annotations

import torch
from torch import Tensor


def precompute_rope_cache(
    head_dim: int, max_seq: int, theta: float = 10000.0
) -> tuple[Tensor, Tensor]:
    """Compute cached cos/sin tables for RoPE.

    Args:
        head_dim: per-head feature dimension; must be even.
        max_seq: maximum sequence length to support.
        theta: base for the inverse-frequency schedule.

    Returns:
        (cos, sin) each of shape (max_seq, head_dim // 2), dtype float32.
    """
    if head_dim % 2 != 0:
        raise ValueError(f"head_dim must be even, got {head_dim}")
    half = head_dim // 2
    inv_freq = 1.0 / (theta ** (torch.arange(0, half, dtype=torch.float32) / half))
    positions = torch.arange(max_seq, dtype=torch.float32)
    freqs = torch.outer(positions, inv_freq)  # (max_seq, half)
    return freqs.cos(), freqs.sin()


def apply_partial_rope(x: Tensor, cos: Tensor, sin: Tensor, frac: float = 0.10) -> Tensor:
    """Apply RoPE to the first `frac * head_dim` features of x.

    Args:
        x: tensor of shape (..., seq, head_dim). The last two dims are
            sequence and head feature dim.
        cos, sin: cached tables from `precompute_rope_cache`. Each shape
            `(max_seq, head_dim // 2)`. Slice to needed seq length by caller.
        frac: fraction of head_dim to rotate. Remaining `(1 - frac) * head_dim`
            passes through unchanged.

    Returns:
        Tensor with same shape as `x`.
    """
    head_dim = x.shape[-1]
    seq = x.shape[-2]
    rot_pairs = max(1, int((frac * head_dim) // 2))
    rot_dim = 2 * rot_pairs

    cos_slice = cos[:seq, :rot_pairs].to(dtype=x.dtype, device=x.device)
    sin_slice = sin[:seq, :rot_pairs].to(dtype=x.dtype, device=x.device)

    x_rot = x[..., :rot_dim]
    x_pass = x[..., rot_dim:]

    x1 = x_rot[..., 0::2]
    x2 = x_rot[..., 1::2]

    rotated_even = x1 * cos_slice - x2 * sin_slice
    rotated_odd = x1 * sin_slice + x2 * cos_slice

    rotated = torch.stack([rotated_even, rotated_odd], dim=-1).flatten(-2)
    return torch.cat([rotated, x_pass], dim=-1)
