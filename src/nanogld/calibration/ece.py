"""Adaptive ECE — equal-mass binning, classwise + per-bucket variants.

Standard ECE bins by equal-WIDTH probability, which over-weights bins
with no data. Adaptive ECE uses equal-MASS bins (each bin holds the
same number of samples).

V1 reports:
  marginal ECE                  — overall calibration quality
  classwise AdaECE              — 1-vs-rest macro across all classes
  per-bucket ECE                — {news-present, news-absent, both}
                                   (V1 hard requirement, invariant 18)

Spec: plan/V1-SPEC.md §5 + §9.1.
"""

from __future__ import annotations

import torch
from torch import Tensor

DEFAULT_N_BINS = 15


def adaptive_ece(probs: Tensor, labels: Tensor, n_bins: int = DEFAULT_N_BINS) -> float:
    """Equal-mass-bin ECE on top-class probability.

    Args:
        probs: (B, C) softmax probs.
        labels: (B,) int true labels.
        n_bins: number of equal-mass bins.

    Returns:
        Scalar ECE in [0, 1]. Lower is better.
    """
    if probs.numel() == 0:
        return 0.0
    confidences, predictions = probs.max(dim=-1)
    correct = (predictions == labels).float()

    sorted_conf, sort_idx = confidences.sort()
    sorted_correct = correct[sort_idx]

    n = sorted_conf.shape[0]
    bin_size = max(1, n // n_bins)
    ece_total = 0.0
    for b in range(n_bins):
        start = b * bin_size
        end = (b + 1) * bin_size if b < n_bins - 1 else n
        if end <= start:
            continue
        bin_conf = sorted_conf[start:end].mean().item()
        bin_acc = sorted_correct[start:end].mean().item()
        bin_weight = (end - start) / n
        ece_total += bin_weight * abs(bin_conf - bin_acc)
    return float(ece_total)


def classwise_ada_ece(
    probs: Tensor,
    labels: Tensor,
    n_bins: int = DEFAULT_N_BINS,
) -> tuple[float, float]:
    """One-vs-rest AdaECE per class. Returns (macro_mean, worst).

    For each class c, treats the multi-class problem as binary
    (c-vs-rest) using probs[:, c] as the score and (labels == c) as
    the binary target.
    """
    n_classes = probs.shape[-1]
    eces = []
    for c in range(n_classes):
        score = probs[:, c]
        target = (labels == c).long()
        prob_c = torch.stack([1.0 - score, score], dim=-1)
        eces.append(adaptive_ece(prob_c, target, n_bins=n_bins))
    if not eces:
        return 0.0, 0.0
    return float(sum(eces) / len(eces)), float(max(eces))


def per_bucket_ece(
    probs: Tensor,
    labels: Tensor,
    bucket_mask: Tensor,
    n_bins: int = DEFAULT_N_BINS,
) -> dict[str, float]:
    """Compute ECE for {news-present, news-absent, both} buckets.

    Args:
        probs: (B, C).
        labels: (B,).
        bucket_mask: (B,) bool — True = news-present.

    Returns:
        Dict with keys "present", "absent", "both" mapping to ECE values.
    """
    out: dict[str, float] = {}
    out["both"] = adaptive_ece(probs, labels, n_bins=n_bins)
    if int(bucket_mask.sum()) > 0:
        out["present"] = adaptive_ece(probs[bucket_mask], labels[bucket_mask], n_bins=n_bins)
    else:
        out["present"] = 0.0
    absent_mask = ~bucket_mask
    if int(absent_mask.sum()) > 0:
        out["absent"] = adaptive_ece(probs[absent_mask], labels[absent_mask], n_bins=n_bins)
    else:
        out["absent"] = 0.0
    return out


def macro_brier(probs: Tensor, labels: Tensor) -> float:
    """Macro Brier score for diagnostic reporting.

    Brier_i = mean((p_i - 1{y == i})^2) per class, then averaged.
    """
    one_hot = torch.zeros_like(probs)
    one_hot.scatter_(dim=-1, index=labels.unsqueeze(-1), value=1.0)
    return float(((probs - one_hot) ** 2).mean().item())
