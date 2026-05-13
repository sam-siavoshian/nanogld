"""Regularized Adaptive Prediction Sets (RAPS).

Angelopoulos et al. arXiv:2009.14193. Adaptive prediction sets with a
size penalty:
    score(x, y) = 1 - p_y_at_rank(x) + lambda * max(0, rank_y(x) - kreg)

where rank_y(x) is the position of y in the descending-sorted softmax
probabilities for x. The penalty discourages excessively large sets.

Mondrian per-class quantiles: fit a separate q_hat per true class.
Quantile = order-statistic at k = ceil((n+1)*(1-alpha)) (clamped to
[1, n]) per Angelopoulos 2020 finite-sample bound.

If a class has fewer than min_class_n=20 calibration samples in
Mondrian mode, its q_hat falls back to the pooled (cross-class)
quantile to avoid an inf-quantile pathology that would exclude the
class entirely.

Spec: plan/V1-SPEC.md §5.2.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor


def raps_score(
    probs: Tensor,
    labels: Tensor,
    lambda_reg: float = 0.01,
    k_reg: int = 1,
) -> Tensor:
    """Compute RAPS score per (probs, labels) pair.

    Args:
        probs: (B, C) softmax probabilities.
        labels: (B,) int true labels.
        lambda_reg: size-penalty weight.
        k_reg: rank threshold below which the penalty is zero.

    Returns:
        (B,) score tensor. Smaller = better (true label closer to top).
    """
    sorted_idx = probs.argsort(dim=-1, descending=True)
    rank_of_label = (sorted_idx == labels.unsqueeze(-1)).int().argmax(dim=-1) + 1
    p_label = probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
    penalty = lambda_reg * torch.clamp(rank_of_label.float() - float(k_reg), min=0.0)
    return 1.0 - p_label + penalty


def fit_raps_quantile(
    cal_probs: Tensor,
    cal_labels: Tensor,
    alpha: float = 0.10,
    mondrian: bool = True,
    lambda_reg: float = 0.01,
    k_reg: int = 1,
) -> dict[int, float]:
    """Fit per-class q_hat on calibration set.

    Args:
        cal_probs: (B, C) calibration softmax probs.
        cal_labels: (B,) true labels.
        alpha: target miscoverage rate (0.10 = 90% coverage).
        mondrian: if True, fit one q_hat per class; else single q_hat.
        lambda_reg, k_reg: RAPS hyperparams.

    Returns:
        Dict mapping class index to q_hat. If mondrian=False, dict has
        one entry under key -1.
    """
    scores = raps_score(cal_probs, cal_labels, lambda_reg=lambda_reg, k_reg=k_reg)
    out: dict[int, float] = {}

    def _order_quantile(s: Tensor, n: int, alpha_: float) -> float:
        k = int(np.ceil((n + 1) * (1 - alpha_)))
        k = max(1, min(n, k))
        return float(torch.kthvalue(s, k).values.item())

    if not mondrian:
        n = cal_labels.shape[0]
        out[-1] = _order_quantile(scores, n, alpha)
        return out

    n_classes = cal_probs.shape[-1]
    n_total = cal_labels.shape[0]
    pooled_q = _order_quantile(scores, n_total, alpha)
    min_class_n = 20
    for c in range(n_classes):
        mask = cal_labels == c
        n_c = int(mask.sum().item())
        if n_c < min_class_n:
            out[c] = pooled_q
            continue
        scores_c = scores[mask]
        out[c] = _order_quantile(scores_c, n_c, alpha)
    return out


def raps_set(
    probs: Tensor,
    q_hats: dict[int, float],
    lambda_reg: float = 0.01,
    k_reg: int = 1,
) -> Tensor:
    """Build the prediction set per row.

    Args:
        probs: (B, C) softmax probs.
        q_hats: per-class quantiles from `fit_raps_quantile`. Use `-1` key
            for the non-Mondrian case.
        lambda_reg, k_reg: RAPS hyperparams used at calibration time.

    Returns:
        (B, C) bool tensor — True means class is in the prediction set.
    """
    sorted_idx = probs.argsort(dim=-1, descending=True)
    n_classes = probs.shape[-1]
    pos = torch.empty_like(sorted_idx)
    arange = torch.arange(n_classes, device=probs.device).unsqueeze(0).expand_as(sorted_idx)
    pos.scatter_(dim=-1, index=sorted_idx, src=arange)
    rank_per_class = pos.float() + 1.0

    in_set = torch.zeros_like(probs, dtype=torch.bool)
    for c in range(n_classes):
        score_c = 1.0 - probs[:, c] + lambda_reg * torch.clamp(rank_per_class[:, c] - float(k_reg), min=0.0)
        q = q_hats.get(c, q_hats.get(-1, float("inf")))
        in_set[:, c] = score_c <= q
    return in_set
