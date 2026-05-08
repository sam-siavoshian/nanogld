"""V1 training losses.

Six loss terms total:

1. focal_loss — focal CE with gamma=3 for Head A. Replaces vanilla CE
   per Xi 2024 (T-scaling on CE harms APS coverage; focal fixes it).

2. sharpe_loss — differentiable -Sharpe on Head B position weight.
   Cost-aware variant penalizes turnover. Saly-Kaufmann 2026 trains
   directly on -Sharpe to hit 2.40 daily-futures Sharpe.

3. dann_loss — Domain-Adversarial Neural Network with gradient reversal
   on era-label. Lambda ramps 0 -> 0.1. Feng 2019 +3.11% on stock
   movement prediction.

4. simmtm_loss — multi-mask similarity-weighted reconstruction (Dong
   NeurIPS 2023). Replaces plain MAE in V1 SSL.

5. clip_infonce — CLIP-style contrastive between bar reps and news
   embeddings within +/- 5min of the bar. Free semantic priors.

6. aecf_entropy_reg — re-exported from model.aecf.

Spec: plan/05-MODEL-TRAINING-CALIBRATION.md V1 losses.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from nanogld.model.aecf import aecf_entropy_reg as aecf_entropy_reg

DEFAULT_FOCAL_GAMMA = 3.0
DEFAULT_INFONCE_TAU = 0.07


def focal_loss(
    logits: Tensor,
    targets: Tensor,
    gamma: float = DEFAULT_FOCAL_GAMMA,
    reduction: str = "mean",
) -> Tensor:
    """Focal loss for multi-class classification.

    Math:
        p_t = softmax(logits)_target
        L_focal = -(1 - p_t)^gamma * log(p_t)

    Reduces to vanilla cross-entropy when gamma == 0.

    Args:
        logits: (B, C).
        targets: (B,) int.
        gamma: focusing parameter. 0.0 -> CE; 3.0 -> V1 default.
        reduction: "mean", "sum", or "none".
    """
    log_probs = F.log_softmax(logits, dim=-1)
    log_p_t = log_probs.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
    p_t = log_p_t.exp()
    loss = -((1.0 - p_t) ** gamma) * log_p_t
    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    return loss


def sharpe_loss(
    position: Tensor,
    next_log_return: Tensor,
    prev_position: Tensor | None = None,
    cost_bps: float = 2.0,
    eps: float = 1e-8,
) -> Tensor:
    """Differentiable -Sharpe for Head B (cost-aware optional).

    Math:
        pnl = position * next_log_return - cost * |position - prev_position|
        L = -mean(pnl) / (std(pnl) + eps)

    Cost is per-unit-turnover in basis points, scaled by 1e-4 to match
    log-return units.

    Args:
        position: (B,) in [-1, +1] (typically tanh of Head B logit).
        next_log_return: (B,) realized log return of the next bar.
        prev_position: (B,) optional previous position for turnover penalty.
            Pass None to skip cost (training step 0, or leak-prone setup).
        cost_bps: round-trip cost in basis points.
        eps: numerical floor on the std denominator.
    """
    cost_frac = cost_bps / 10_000.0
    if prev_position is None:
        pnl = position * next_log_return
    else:
        turnover = (position - prev_position).abs()
        pnl = position * next_log_return - cost_frac * turnover
    return -pnl.mean() / (pnl.std(unbiased=False) + eps)


class GradientReversalLayer(torch.autograd.Function):
    """Gradient-reversal layer (Ganin & Lempitsky 2015)."""

    @staticmethod
    def forward(ctx, x: Tensor, alpha: float) -> Tensor:  # noqa: ANN001
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> tuple[Tensor, None]:  # noqa: ANN001
        return -ctx.alpha * grad_output, None


def grad_reverse(x: Tensor, alpha: float) -> Tensor:
    """Helper to apply GRL with a scalar alpha."""
    return GradientReversalLayer.apply(x, alpha)


def dann_loss(
    domain_logits: Tensor,
    era_labels: Tensor,
) -> Tensor:
    """Domain-adversarial CE on era-bucket logits.

    Caller is responsible for routing the encoder representation through
    `grad_reverse(z, alpha)` BEFORE the domain classifier. This function
    only computes the CE loss on the classifier's output.

    Args:
        domain_logits: (B, num_eras).
        era_labels: (B,) int.
    """
    return F.cross_entropy(domain_logits, era_labels)


def simmtm_loss(
    views: Tensor,
    targets: Tensor,
    lambda_sim: float = 0.5,
) -> Tensor:
    """SimMTM-style multi-mask reconstruction loss.

    For K masked views per sample, compute pairwise cosine similarities
    between view representations, then use those similarities as weights
    for blending reconstruction targets.

    This implementation is a reduced form: MSE recon weighted by softmax
    of pairwise cosine similarity between views.

    Args:
        views: (B, K, D) — K view representations per sample.
        targets: (B, K, D) — reconstruction targets per view.
        lambda_sim: weight on the similarity-blend term.
    """
    b, k, _ = views.shape
    views_flat = views.reshape(b, k, -1)
    sim = F.cosine_similarity(views_flat.unsqueeze(2), views_flat.unsqueeze(1), dim=-1)
    weights = F.softmax(sim, dim=-1)
    blended = torch.einsum("bkj,bjd->bkd", weights, views_flat)
    recon_mse = F.mse_loss(views_flat, targets.reshape(b, k, -1))
    sim_blend_mse = F.mse_loss(blended, targets.reshape(b, k, -1))
    return recon_mse + lambda_sim * sim_blend_mse


def clip_infonce(
    z_bar: Tensor,
    z_news: Tensor,
    tau: float = DEFAULT_INFONCE_TAU,
) -> Tensor:
    """Symmetric InfoNCE between bar reps and news reps.

    Args:
        z_bar: (B, D) — L2-normalized bar representations.
        z_news: (B, D) — L2-normalized news representations from the same
            bar window (positive pairs).
        tau: temperature.
    """
    z_bar = F.normalize(z_bar, dim=-1)
    z_news = F.normalize(z_news, dim=-1)
    logits = z_bar @ z_news.t() / tau
    targets = torch.arange(z_bar.shape[0], device=z_bar.device)
    loss_b = F.cross_entropy(logits, targets)
    loss_n = F.cross_entropy(logits.t(), targets)
    return 0.5 * (loss_b + loss_n)
