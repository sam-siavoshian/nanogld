"""Smoke test for modality ablation against a tiny synthetic model."""

from __future__ import annotations

import pytest
import torch
from torch import Tensor, nn

from nanogld.analysis.modality_ablation import modality_ablation


pytestmark = pytest.mark.smoke


class _TinyModel(nn.Module):
    """Toy model that mimics nanoGLDV1's forward signature."""

    def __init__(self, n_classes: int = 3) -> None:
        super().__init__()
        self.proj = nn.Linear(681, 16)
        self.cls = nn.Linear(16, n_classes)
        self.pos = nn.Linear(16, 1)

    def forward(
        self,
        channel_inputs: Tensor,
        news_embeddings: Tensor,
        news_mask: Tensor,
        is_news_present: Tensor,
        regime_vec: Tensor,
        return_pooled: bool = False,
    ) -> dict[str, Tensor]:
        # ignore aux modalities — they only need to be accepted
        del news_embeddings, news_mask, is_news_present, regime_vec, return_pooled
        h = self.proj(channel_inputs.mean(dim=1))
        return {
            "logits_3class": self.cls(h),
            "position_weight": torch.tanh(self.pos(h)).squeeze(-1),
        }


def _batch(b: int = 4) -> dict[str, Tensor]:
    return {
        "channel_inputs": torch.randn(b, 64, 681),
        "news_embeddings": torch.randn(b, 8, 256),
        "news_mask": torch.ones(b, 8),
        "is_news_present": torch.ones(b, dtype=torch.long),
        "regime_vec": torch.zeros(b, 12),
        "label_3class": torch.randint(0, 3, (b,)),
        "next_log_return": torch.randn(b),
    }


def test_modality_ablation_returns_all_keys() -> None:
    torch.manual_seed(0)
    model = _TinyModel()
    loader = [_batch(4) for _ in range(3)]
    out = modality_ablation(model, loader, device="cpu", max_batches=3)
    for name in ("none", "bars", "news", "regime", "bars_news"):
        assert name in out
        assert "focal" in out[name]
        assert "sharpe" in out[name]
        assert "sharpe_present" in out[name]
        assert "sharpe_absent" in out[name]


def test_modality_ablation_empty_loader_raises() -> None:
    model = _TinyModel()
    with pytest.raises(RuntimeError):
        modality_ablation(model, [], device="cpu")
