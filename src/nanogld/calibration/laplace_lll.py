"""Laplace last-layer (LLLA) approximation.

Daxberger et al. NeurIPS 2021 arXiv:2106.14806. Fits a Bayesian posterior
over the LAST layer of a trained model via a Hessian-based Laplace
approximation. Posterior variance feeds into Kelly-sizing as an
epistemic uncertainty signal.

Replaces V1-draft MC dropout T=20 forward passes (slower, weaker
signal in transformers per Gal's original CNN-only validation).

Lazy import of `laplace-torch`; falls back to a no-op if unavailable.

Spec: plan/V1-SPEC.md §5.3.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class LaplaceLLLA:
    """Laplace last-layer wrapper with predict-time epistemic variance.

    Args:
        head_module: the last-layer Linear module to fit.
        hessian_structure: "kron" (default) or "diag".
    """

    def __init__(
        self,
        head_module: nn.Module,
        hessian_structure: str = "kron",
    ) -> None:
        self.head_module = head_module
        self.hessian_structure = hessian_structure
        self._la = None

    def fit(self, train_loader) -> None:  # noqa: ANN001
        """Fit Laplace on the head using the training loader."""
        try:
            from laplace import Laplace  # noqa: PLC0415
        except ImportError:
            return
        la = Laplace(
            self.head_module,
            "classification",
            subset_of_weights="last_layer",
            hessian_structure=self.hessian_structure,
        )
        la.fit(train_loader)
        self._la = la

    def optimize_prior_precision(self) -> None:
        if self._la is None:
            return
        if hasattr(self._la, "optimize_prior_precision"):
            self._la.optimize_prior_precision(method="marglik")

    def predict_variance(self, x: Tensor) -> Tensor:
        """Return per-sample epistemic variance proxy (max-class variance).

        Falls back to zeros if Laplace was not fit.
        """
        if self._la is None:
            b = x.shape[0]
            return torch.zeros(b, dtype=torch.float32, device=x.device)
        try:
            mean, var = self._la(x, link_approx="probit")
        except (TypeError, AttributeError):
            return torch.zeros(x.shape[0], dtype=torch.float32, device=x.device)
        return var.amax(dim=-1).to(torch.float32)


def kelly_multiplier(
    epistemic_var: Tensor,
    sigma_target: float = 0.05,
    floor: float = 0.0,
    ceil: float = 1.0,
) -> Tensor:
    """Convert posterior variance into a Kelly multiplier in [floor, ceil].

    `multiplier = clamp(sigma_target / sqrt(var), floor, ceil)`. Uniform
    1.0 when variance is at sigma_target^2.
    """
    sigma = torch.sqrt(epistemic_var.clamp(min=1e-12))
    raw = sigma_target / sigma
    return raw.clamp(min=floor, max=ceil)
