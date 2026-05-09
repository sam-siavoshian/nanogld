"""AECF — Adaptive Entropy-gated Curriculum Fusion.

Chlon et al. arXiv:2505.15417: replaces V1-draft constant 15% modality
dropout with a per-batch sampled mask probability + entropy regularizer
on the gate distribution. PAC-bound on calibration across 2^M-1
modality subsets.

Two pieces:

1. AECFMask: samples a per-batch mask probability from `Uniform(p_min,
   p_max)`, with optional curriculum schedule that ramps from 0 to
   `p_max` over training steps. Returns a Bernoulli mask of shape
   (B,) or (B, M) for M modalities.

2. aecf_entropy_reg: scalar regularizer on the gate distribution. Sums
   the entropy of each per-sample gate and scales by lambda(x).

V1 has two modalities (bars, news) but only news is droppable; bars
are always present. So the mask shape is (B,) for the news modality
indicator.

Spec: plan/05-MODEL-TRAINING-CALIBRATION.md V1 news fusion.
Spec: plan/V1-SPEC.md §2.3.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class AECFMask(nn.Module):
    """Sample a per-batch curriculum mask probability.

    Args:
        p_min: lower bound on the per-batch mask probability (0.0 default).
        p_max: upper bound (0.9 default — matches empirical 51% news absence).
        curriculum_steps: number of training steps over which p_max is
            reached. During step `s` of training, the effective p_max
            is `min(1.0, s / curriculum_steps) * p_max`.
        seed: optional torch.Generator seed for reproducibility.
    """

    def __init__(
        self,
        p_min: float = 0.0,
        p_max: float = 0.9,
        curriculum_steps: int = 10_000,
    ) -> None:
        super().__init__()
        if not (0.0 <= p_min <= p_max <= 1.0):
            raise ValueError(f"need 0 <= p_min <= p_max <= 1, got {p_min}, {p_max}")
        self.p_min = p_min
        self.p_max = p_max
        self.curriculum_steps = max(1, curriculum_steps)

    def sample_p(self, training_step: int = 0) -> float:
        """Sample one scalar mask probability for the current batch."""
        ramp = min(1.0, float(training_step) / float(self.curriculum_steps))
        effective_max = self.p_min + ramp * (self.p_max - self.p_min)
        if effective_max <= self.p_min:
            return self.p_min
        return float(torch.empty(1).uniform_(self.p_min, effective_max).item())

    def sample_mask(
        self,
        batch_size: int,
        training_step: int = 0,
        device: torch.device | str = "cpu",
    ) -> Tensor:
        """Return a Bernoulli mask of shape (batch_size,) — 1 = present.

        At eval, returns all-ones (no modality dropout at inference).
        """
        if not self.training:
            return torch.ones(batch_size, dtype=torch.float32, device=device)
        p_drop = self.sample_p(training_step=training_step)
        keep = 1.0 - p_drop
        return (torch.rand(batch_size, device=device) < keep).to(torch.float32)


def aecf_entropy_reg(gate_dist: Tensor, lambda_x: float = 0.01) -> Tensor:
    """Entropy regularizer on a probability distribution.

    Args:
        gate_dist: shape (B, M) where each row is a non-negative
            distribution over M modalities. Need not be normalized;
            we re-normalize internally.
        lambda_x: regularization strength.

    Returns:
        Scalar tensor — `lambda_x * mean(-sum(p * log(p)))`. Always
        non-negative; encourages spread-out gate distributions.
    """
    eps = 1e-8
    p = gate_dist / (gate_dist.sum(dim=-1, keepdim=True) + eps)
    entropy_per_sample = -(p * torch.log(p + eps)).sum(dim=-1)
    return lambda_x * entropy_per_sample.mean()
