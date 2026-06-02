"""Multi-Head Attention with QK-Norm, partial RoPE, per-head gating,
and IMU-1 value residual. Encoder-only (is_causal=False).

Block stack per V1-SPEC §3:
  qkv = Linear(D, 3D, bias=False)(x)
  q, k, v = split
  q = RMSNorm(q)  # QK-Norm BEFORE RoPE
  k = RMSNorm(k)
  q = apply_partial_rope(q, frac=0.10)
  k = apply_partial_rope(k, frac=0.10)
  v = v + value_residual_proj(prev_v)   # IMU-1 if prev_v provided
  attn_out = SDPA(q.contiguous(), k.contiguous(), v.contiguous(), is_causal=False)
  attn_out = sigmoid(per_head_gate) * attn_out   # IMU-1 gating
  out = w_o(attn_out)

V1 invariants honored:
  - QK-Norm BEFORE RoPE (line order above)
  - .contiguous() on Q/K/V before SDPA (PyTorch #181133)
  - is_causal=False (encoder)
  - no bias anywhere
  - real-form RoPE (never view_as_complex)

Spec: plan/05-MODEL-TRAINING-CALIBRATION.md V1 backbone.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from nanogld.model.rms_norm import RMSNorm
from nanogld.model.rope import apply_partial_rope, precompute_rope_cache


class MultiHeadAttention(nn.Module):
    """Encoder-only multi-head attention with V1 enhancements.

    Args:
        d_model: model hidden dim (D=384 default).
        num_heads: number of attention heads (6 default → head_dim=64).
        max_seq: maximum sequence length for RoPE cache.
        dropout: attention dropout probability.
        partial_rope_frac: fraction of head_dim to rotate (0.10 default).
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        max_seq: int,
        dropout: float = 0.0,
        partial_rope_frac: float = 0.10,
    ) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(f"d_model={d_model} must be divisible by num_heads={num_heads}")
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.d_model = d_model
        self.dropout = dropout
        self.partial_rope_frac = partial_rope_frac

        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.value_residual_proj = nn.Linear(d_model, d_model, bias=False)

        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)

        self.head_gate = nn.Parameter(torch.zeros(num_heads))

        cos, sin = precompute_rope_cache(self.head_dim, max_seq)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    def forward(
        self,
        x: Tensor,
        prev_v: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Run attention.

        Args:
            x: shape (B, T, D).
            prev_v: optional shape (B, T, D) — V from previous attention
                layer for IMU-1 value residual. Pass None on first layer.

        Returns:
            (out, v_for_next) — out is shape (B, T, D), v_for_next is the
            V tensor (after value-residual blending) of shape (B, T, D),
            for the next layer's value residual.
        """
        b, t, _ = x.shape
        qkv = self.qkv_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(b, t, self.num_heads, self.head_dim)
        k = k.view(b, t, self.num_heads, self.head_dim)
        v = v.view(b, t, self.num_heads, self.head_dim)

        q = self.q_norm(q)
        k = self.k_norm(k)

        q = apply_partial_rope(q, self.rope_cos, self.rope_sin, self.partial_rope_frac)
        k = apply_partial_rope(k, self.rope_cos, self.rope_sin, self.partial_rope_frac)

        if prev_v is not None:
            prev_v_resid = self.value_residual_proj(prev_v)
            prev_v_resid = prev_v_resid.view(b, t, self.num_heads, self.head_dim)
            v = v + prev_v_resid

        q = q.transpose(1, 2).contiguous()
        k = k.transpose(1, 2).contiguous()
        v_attn = v.transpose(1, 2).contiguous()

        attn_out = F.scaled_dot_product_attention(
            q, k, v_attn, dropout_p=self.dropout if self.training else 0.0, is_causal=False
        )

        gate = torch.sigmoid(self.head_gate).view(1, self.num_heads, 1, 1)
        attn_out = attn_out * gate

        attn_out = attn_out.transpose(1, 2).contiguous().view(b, t, self.d_model)
        out = self.out_proj(attn_out)

        v_for_next = v.reshape(b, t, self.d_model)
        return out, v_for_next
