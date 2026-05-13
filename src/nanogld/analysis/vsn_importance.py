"""Native feature importance from the VSN softmax gate.

The Variable Selection Network in `nanogld.model.vsn` outputs a per-bar
softmax distribution over the 681 input features (sums to 1). Mean over
an eval split gives a free, calibration-quality importance signal —
nothing else to compute.

This module:
    - Hooks the VSN forward to capture gate tensors.
    - Aggregates over an eval DataLoader split.
    - Splits per news-presence bucket (V1 invariant 18).
    - Saves a parquet with columns:
        feature_idx, mean_gate, std_gate, mean_present, mean_absent, n_bars

Spec: plan/V1-SPEC.md §11.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable

import numpy as np
import torch
from torch import Tensor, nn

from nanogld.analysis._inference_mode import to_inference_mode

LOG = logging.getLogger("nanogld.analysis.vsn")


class _GateCapture:
    """Forward hook that stores the most recent VSN gate output."""

    def __init__(self) -> None:
        self.last_gate: Tensor | None = None

    def __call__(self, module: nn.Module, inputs: tuple, output: tuple) -> None:  # noqa: ANN001
        if isinstance(output, tuple) and len(output) == 2:
            self.last_gate = output[1].detach()


def collect_vsn_gates(
    model: nn.Module,
    loader: Iterable[dict[str, Tensor]],
    device: str = "cpu",
) -> dict[str, np.ndarray]:
    """Run the model over `loader` and aggregate VSN gate stats."""
    if not hasattr(model, "vsn"):
        raise AttributeError("model lacks `vsn` attribute; cannot run VSN importance")

    capture = _GateCapture()
    handle = model.vsn.register_forward_hook(capture)
    to_inference_mode(model)

    sums = None
    sum_sq = None
    sums_present = None
    sums_absent = None
    n_bars = 0
    n_present = 0

    try:
        with torch.no_grad():
            for batch in loader:
                channel_inputs = batch["channel_inputs"].to(device).float().nan_to_num(0.0)
                news_embeddings = batch["news_embeddings"].to(device).float()
                news_mask = batch["news_mask"].to(device).float()
                is_news_present = batch["is_news_present"].to(device).long()
                regime_vec = batch["regime_vec"].to(device).float()

                _ = model(
                    channel_inputs=channel_inputs,
                    news_embeddings=news_embeddings,
                    news_mask=news_mask,
                    is_news_present=is_news_present,
                    regime_vec=regime_vec,
                )
                gate = capture.last_gate
                if gate is None:
                    continue

                gate_per_sample = gate.mean(dim=1)
                if sums is None:
                    f_dim = gate_per_sample.shape[-1]
                    sums = torch.zeros(f_dim, device=device, dtype=torch.float64)
                    sum_sq = torch.zeros(f_dim, device=device, dtype=torch.float64)
                    sums_present = torch.zeros(f_dim, device=device, dtype=torch.float64)
                    sums_absent = torch.zeros(f_dim, device=device, dtype=torch.float64)

                sums += gate_per_sample.sum(dim=0).double()
                sum_sq += (gate_per_sample.double() ** 2).sum(dim=0)
                present_mask = is_news_present.bool()
                if present_mask.any():
                    sums_present += gate_per_sample[present_mask].sum(dim=0).double()
                    n_present += int(present_mask.sum().item())
                if (~present_mask).any():
                    sums_absent += gate_per_sample[~present_mask].sum(dim=0).double()

                n_bars += gate_per_sample.shape[0]
    finally:
        handle.remove()
        capture.last_gate = None

    if sums is None or n_bars == 0:
        raise RuntimeError("VSN gate aggregation got 0 bars")

    mean = (sums / max(n_bars, 1)).cpu().numpy()
    var = (sum_sq / max(n_bars, 1)).cpu().numpy() - mean**2
    var = np.maximum(var, 0.0)
    std = np.sqrt(var)
    n_absent = max(n_bars - n_present, 0)
    mean_present = (
        (sums_present / n_present).cpu().numpy() if n_present > 0 else np.zeros_like(mean)
    )
    mean_absent = (
        (sums_absent / n_absent).cpu().numpy() if n_absent > 0 else np.zeros_like(mean)
    )

    LOG.info(
        "VSN gate aggregated over %d bars (%d present / %d absent); top-3 by mean: %s",
        n_bars,
        n_present,
        n_absent,
        np.argsort(-mean)[:3].tolist(),
    )

    return {
        "mean_gate": mean.astype(np.float32),
        "std_gate": std.astype(np.float32),
        "mean_present": mean_present.astype(np.float32),
        "mean_absent": mean_absent.astype(np.float32),
        "n_bars": np.asarray([n_bars], dtype=np.int64),
        "n_present": np.asarray([n_present], dtype=np.int64),
    }
