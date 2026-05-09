"""Permutation feature importance — model-agnostic ground truth.

For each feature `f`, the eval set is run with `f`'s column shuffled
across the (B, T) axis (per-sample shuffle of the time series), then
the focal loss / Sharpe drop relative to the unshuffled baseline is
recorded. Repeated `n_perm_repeats` times, then averaged.

Slow but architecture-agnostic. We cap to `max_features` ranked by the
cheaper VSN gate signal so the suite finishes within budget.

Spec: plan/V1-SPEC.md §11.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from nanogld.analysis._inference_mode import to_inference_mode

LOG = logging.getLogger("nanogld.analysis.permutation")


def _eval_focal_and_sharpe(
    model: nn.Module,
    batches: list[dict[str, Tensor]],
    device: str,
    feature_idx: int | None,
    rng: np.random.Generator,
) -> tuple[float, float]:
    """Forward over cached batches; if `feature_idx` is set, shuffle that
    feature column across the batch axis before each forward.

    Returns: (focal_loss, sharpe).
    """
    losses = []
    pnls = []
    to_inference_mode(model)
    with torch.no_grad():
        for batch in batches:
            channel_inputs = batch["channel_inputs"].to(device).float().nan_to_num(0.0).clone()
            if feature_idx is not None:
                b = channel_inputs.shape[0]
                order = rng.permutation(b)
                channel_inputs[:, :, feature_idx] = channel_inputs[order, :, feature_idx]

            news_embeddings = batch["news_embeddings"].to(device).float()
            news_mask = batch["news_mask"].to(device).float()
            is_news_present = batch["is_news_present"].to(device).long()
            regime_vec = batch["regime_vec"].to(device).float()
            labels = batch["label_3class"].to(device).long()
            nlr = batch["next_log_return"].to(device).float()

            out = model(
                channel_inputs=channel_inputs,
                news_embeddings=news_embeddings,
                news_mask=news_mask,
                is_news_present=is_news_present,
                regime_vec=regime_vec,
            )
            logits = out["logits_3class"]
            pos = out["position_weight"]
            focal_per_sample = F.cross_entropy(logits, labels, reduction="none")
            losses.append(focal_per_sample.detach().cpu().numpy())
            pnls.append((pos.detach() * nlr).cpu().numpy())

    all_loss = np.concatenate(losses) if losses else np.zeros(1)
    all_pnl = np.concatenate(pnls) if pnls else np.zeros(1)
    focal = float(all_loss.mean()) if all_loss.size else 0.0
    if all_pnl.size < 2 or all_pnl.std(ddof=1) == 0:
        sharpe = 0.0
    else:
        sharpe = float(all_pnl.mean() / all_pnl.std(ddof=1) * np.sqrt(3276))
    return focal, sharpe


def permutation_importance(
    model: nn.Module,
    loader: Iterable[dict[str, Tensor]],
    feature_indices: list[int],
    device: str = "cpu",
    n_repeats: int = 3,
    seed: int = 42,
    max_batches: int = 32,
) -> dict[str, np.ndarray]:
    """Permutation importance for the requested features."""
    rng = np.random.default_rng(seed)
    cached: list[dict[str, Tensor]] = []
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        cached.append(batch)
    if not cached:
        raise RuntimeError("permutation_importance: empty loader")

    base_focal, base_sharpe = _eval_focal_and_sharpe(
        model, cached, device, feature_idx=None, rng=rng
    )

    feature_indices = sorted({int(i) for i in feature_indices})
    n_feat = len(feature_indices)
    df_mean = np.zeros(n_feat, dtype=np.float64)
    df_std = np.zeros(n_feat, dtype=np.float64)
    ds_mean = np.zeros(n_feat, dtype=np.float64)
    ds_std = np.zeros(n_feat, dtype=np.float64)

    for i, fidx in enumerate(feature_indices):
        focal_deltas = []
        sharpe_deltas = []
        for r in range(n_repeats):
            rep_rng = np.random.default_rng(seed + r * 1009 + fidx)
            f, s = _eval_focal_and_sharpe(model, cached, device, feature_idx=fidx, rng=rep_rng)
            focal_deltas.append(f - base_focal)
            sharpe_deltas.append(base_sharpe - s)
        df_mean[i] = float(np.mean(focal_deltas))
        df_std[i] = float(np.std(focal_deltas, ddof=0))
        ds_mean[i] = float(np.mean(sharpe_deltas))
        ds_std[i] = float(np.std(sharpe_deltas, ddof=0))
        if i % 20 == 0 and i > 0:
            LOG.info("permutation: %d / %d features done", i, n_feat)

    LOG.info("baseline focal=%.4f sharpe=%.4f", base_focal, base_sharpe)

    return {
        "feature_idx": np.asarray(feature_indices, dtype=np.int64),
        "delta_focal_mean": df_mean.astype(np.float32),
        "delta_focal_std": df_std.astype(np.float32),
        "delta_sharpe_mean": ds_mean.astype(np.float32),
        "delta_sharpe_std": ds_std.astype(np.float32),
        "baseline_focal": np.asarray([base_focal], dtype=np.float32),
        "baseline_sharpe": np.asarray([base_sharpe], dtype=np.float32),
        "n_batches": np.asarray([len(cached)], dtype=np.int64),
    }
