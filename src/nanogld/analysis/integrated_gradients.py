"""Integrated Gradients for nanoGLD V1 feature attribution.

Uses captum's IntegratedGradients on the model's `logits_3class` head
with respect to the `channel_inputs` tensor only (other modalities
held at their batch values to isolate numeric-feature attribution).

Path integral: alpha * x + (1 - alpha) * baseline, alpha in [0, 1].
Baseline: zero or per-feature mean across the eval set.

Spec: plan/V1-SPEC.md §11.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable

import numpy as np
import torch
from torch import Tensor, nn

from nanogld.analysis._inference_mode import to_inference_mode

LOG = logging.getLogger("nanogld.analysis.ig")


class _IGModelWrapper(nn.Module):
    """Wrap nanoGLDV1 so its forward returns just `logits_3class`."""

    def __init__(
        self,
        model: nn.Module,
        news_embeddings: Tensor,
        news_mask: Tensor,
        is_news_present: Tensor,
        regime_vec: Tensor,
    ) -> None:
        super().__init__()
        self.model = model
        self.news_embeddings = news_embeddings
        self.news_mask = news_mask
        self.is_news_present = is_news_present
        self.regime_vec = regime_vec

    def forward(self, channel_inputs: Tensor) -> Tensor:
        out = self.model(
            channel_inputs=channel_inputs,
            news_embeddings=self.news_embeddings,
            news_mask=self.news_mask,
            is_news_present=self.is_news_present,
            regime_vec=self.regime_vec,
        )
        return out["logits_3class"]


def integrated_gradients(
    model: nn.Module,
    loader: Iterable[dict[str, Tensor]],
    device: str = "cpu",
    n_samples: int = 256,
    n_steps: int = 32,
    baseline_mode: str = "zero",
) -> dict[str, np.ndarray]:
    """Run captum IG over the eval loader."""
    try:
        from captum.attr import IntegratedGradients  # noqa: PLC0415
    except ImportError as exc:
        raise RuntimeError(
            "captum is required for integrated_gradients; install via `uv add captum>=0.7,<1`"
        ) from exc

    to_inference_mode(model)
    n_classes = 3

    sum_abs = None
    sum_signed = None
    sum_per_class = None
    seen = 0
    f_dim = None

    iter_loader = iter(loader)

    while seen < n_samples:
        try:
            batch = next(iter_loader)
        except StopIteration:
            break

        channel_inputs = (
            batch["channel_inputs"].to(device).float().nan_to_num(0.0).requires_grad_(True)
        )
        news_embeddings = batch["news_embeddings"].to(device).float()
        news_mask = batch["news_mask"].to(device).float()
        is_news_present = batch["is_news_present"].to(device).long()
        regime_vec = batch["regime_vec"].to(device).float()

        wrapper = _IGModelWrapper(model, news_embeddings, news_mask, is_news_present, regime_vec)
        ig = IntegratedGradients(wrapper)

        baseline = torch.zeros_like(channel_inputs) if baseline_mode == "mean" else None

        if f_dim is None:
            f_dim = int(channel_inputs.shape[-1])
            sum_abs = torch.zeros(f_dim, device=device, dtype=torch.float64)
            sum_signed = torch.zeros(f_dim, device=device, dtype=torch.float64)
            sum_per_class = torch.zeros(n_classes, f_dim, device=device, dtype=torch.float64)

        for cls in range(n_classes):
            attr = ig.attribute(
                channel_inputs,
                baselines=baseline,
                target=cls,
                n_steps=n_steps,
            )
            attr_per_feat = attr.detach().mean(dim=1)
            sum_per_class[cls] += attr_per_feat.sum(dim=0).double()
            if cls == 0:
                sum_abs += attr_per_feat.abs().sum(dim=0).double()
                sum_signed += attr_per_feat.sum(dim=0).double()

        seen += channel_inputs.shape[0]

    if seen == 0 or sum_abs is None:
        raise RuntimeError("IG: no samples attributed")

    mean_abs = (sum_abs / seen).cpu().numpy().astype(np.float32)
    mean_signed = (sum_signed / seen).cpu().numpy().astype(np.float32)
    per_class_mean = (sum_per_class / seen).cpu().numpy().astype(np.float32)

    LOG.info(
        "IG aggregated over %d samples; top-3 by |IG|: %s",
        seen,
        np.argsort(-mean_abs)[:3].tolist(),
    )

    return {
        "mean_abs": mean_abs,
        "mean_signed": mean_signed,
        "per_class_mean": per_class_mean,
        "n_samples_seen": np.asarray([seen], dtype=np.int64),
    }
