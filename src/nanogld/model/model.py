"""nanoGLDV1 — top-level multimodal transformer for 30-min GLD direction.

Forward pipeline:
    Input bars (B, T, F=681)
    → RevIN per-channel (norm)
    → VSN feature gate
    → Channel-independent patching (P=4, S=4 → 16 patches/channel)
    → Hybrid encoder stack (10 transformer + 2 sLSTM, FiLM at {2,4,6,8,10},
       cross-attn at {3,7})
    → Mean-pool over patches → reshape back to (B, F, D), pool over channels
    → MultiTaskHead: (logits_3class, position_weight)

Note: SeriesDecomposition module is instantiated for forward-compat with
the V1-SPEC two-stream wiring (deferred). Currently NOT called in forward;
input passes through directly. Two-stream wiring tracked as separate task.

Output dict keys:
    logits_3class:    (B, 3) — focal CE target
    position_weight:  (B,)   — diff -Sharpe target

Spec: plan/V1-SPEC.md (full).
Spec: plan/05-MODEL-TRAINING-CALIBRATION.md V1.
"""

from __future__ import annotations

from torch import Tensor, nn

from nanogld.model.decomposition import SeriesDecomposition
from nanogld.model.encoder import HybridEncoder
from nanogld.model.heads import MultiTaskHead
from nanogld.model.patch_embed import PatchEmbed
from nanogld.model.regime_encoder import REGIME_VECTOR_DIM, RegimeEncoder
from nanogld.model.revin import RevIN
from nanogld.model.vsn import VSN


def _scaled_residual_init_std(num_layers: int) -> float:
    return 0.02 / (2.0 * num_layers) ** 0.5


class nanoGLDV1(nn.Module):
    """V1 top-level model.

    Args:
        numeric_dim: number of input numerical features (681 for V1).
        d_model: encoder hidden dim (D=384).
        num_heads: self-attn head count (6 → head_dim=64).
        num_transformer_layers: 10 by default.
        num_slstm_layers: 2 by default.
        t_bars: lookback length in bars (T=64).
        patch_len: patch length (P=4).
        patch_stride: patch stride (S=4).
        n_classes: classification head class count (3).
        regime_dim: regime vector dim (12).
        d_text: news embedding dim (256 from Qwen3+MRL).
        n_news_slots: max news slots per bar (8).
        dropout: per-block dropout.
        drop_path_max: max stochastic-depth probability.
        decomposition_kernel: trend MA kernel (24 default).
    """

    def __init__(
        self,
        numeric_dim: int = 681,
        d_model: int = 384,
        num_heads: int = 6,
        num_transformer_layers: int = 10,
        num_slstm_layers: int = 2,
        t_bars: int = 64,
        patch_len: int = 4,
        patch_stride: int = 4,
        n_classes: int = 3,
        regime_dim: int = REGIME_VECTOR_DIM,
        d_text: int = 256,
        n_news_slots: int = 8,
        dropout: float = 0.2,
        drop_path_max: float = 0.2,
        decomposition_kernel: int = 24,
    ) -> None:
        super().__init__()
        self.numeric_dim = numeric_dim
        self.d_model = d_model
        self.t_bars = t_bars
        self.regime_dim = regime_dim

        self.decomposition = SeriesDecomposition(kernel_size=decomposition_kernel)
        self.revin = RevIN(num_features=numeric_dim, affine=True)
        self.vsn = VSN(num_features=numeric_dim, hidden_dim=64, dropout=dropout)
        self.patch_embed = PatchEmbed(
            patch_len=patch_len,
            patch_stride=patch_stride,
            t_bars=t_bars,
            d_model=d_model,
        )
        self.regime_encoder = RegimeEncoder(regime_dim=regime_dim)
        self.encoder = HybridEncoder(
            d_model=d_model,
            num_heads=num_heads,
            max_seq=self.patch_embed.num_patches,
            num_transformer_layers=num_transformer_layers,
            num_slstm_layers=num_slstm_layers,
            dropout=dropout,
            drop_path_max=drop_path_max,
            regime_dim=regime_dim,
            d_text=d_text,
            n_news_slots=n_news_slots,
        )
        self.head = MultiTaskHead(d_model=d_model, n_classes=n_classes)

        total_layers = num_transformer_layers + num_slstm_layers
        self._init_weights(num_layers=total_layers)

    def _init_weights(self, num_layers: int) -> None:
        residual_std = _scaled_residual_init_std(num_layers)
        residual_suffixes = ("out_proj", "w_down", "value_residual_proj")
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                if any(name.endswith(s) for s in residual_suffixes):
                    nn.init.trunc_normal_(module.weight, std=residual_std)
                else:
                    nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.trunc_normal_(module.weight, std=0.02)

    def forward(
        self,
        channel_inputs: Tensor,
        news_embeddings: Tensor,
        news_mask: Tensor,
        is_news_present: Tensor,
        regime_vec: Tensor,
        return_pooled: bool = False,
    ) -> dict[str, Tensor]:
        """Full forward pass.

        Args:
            channel_inputs: (B, T, numeric_dim).
            news_embeddings: (B, S, d_text).
            news_mask: (B, S) — 1 = source present.
            is_news_present: (B,) — binary.
            regime_vec: (B, regime_dim) — per-bar regime vector.
            return_pooled: if True, also return the encoder pooled
                representation as `pooled` (B, d_model). Used by SSL
                pretrain so the recon loss has a real signal.

        Returns:
            Dict with `logits_3class` (B, n_classes) and `position_weight` (B,).
            If return_pooled=True, also `pooled` (B, d_model).
        """
        b, t, f = channel_inputs.shape

        x = channel_inputs

        x = self.revin(x, mode="norm")
        x_gated, _gate = self.vsn(x)

        patched = self.patch_embed(x_gated)

        regime_vec = self.regime_encoder(regime_vec)
        regime_per_channel = regime_vec.repeat_interleave(f, dim=0)

        n_slots = news_embeddings.shape[1]
        news_per_channel = news_embeddings.repeat_interleave(f, dim=0)
        news_mask_per_channel = news_mask.repeat_interleave(f, dim=0)
        is_news_per_channel = is_news_present.repeat_interleave(f, dim=0)

        if news_per_channel.shape[1] != n_slots:
            raise RuntimeError("news shape mismatch after channel broadcast")

        encoded = self.encoder(
            patched,
            regime=regime_per_channel,
            news=news_per_channel,
            news_mask=news_mask_per_channel,
            is_news_present=is_news_per_channel,
        )

        encoded = encoded.view(b, f, self.patch_embed.num_patches, self.d_model)
        pooled_per_channel = encoded.mean(dim=2)
        pooled = pooled_per_channel.mean(dim=1)

        logits, pos = self.head(pooled)
        out: dict[str, Tensor] = {"logits_3class": logits, "position_weight": pos}
        if return_pooled:
            out["pooled"] = pooled
        return out
