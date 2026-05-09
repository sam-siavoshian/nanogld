"""Friendly-SAM (F-SAM) optimizer wrapper.

Friendly-SAM (Li et al. CVPR 2024 arXiv:2403.12350): a noise-filtered
variant of Sharpness-Aware Minimization. F-SAM beats SAM/ASAM by
0.1-0.4% on CIFAR.

V1 default: rho=0.05.

Two-step optimizer:
    1. perturb params by `rho * filtered_grad / ||filtered_grad||`  (ascent)
    2. compute loss + grads at perturbed params
    3. restore original params, apply base.step()                    (descent)
"""

from __future__ import annotations

import torch
from torch import Tensor


class FriendlySAM:
    """Friendly-SAM 2-step ascent/descent wrapper around a base optimizer."""

    def __init__(
        self,
        params,  # noqa: ANN001
        base_optimizer,  # noqa: ANN001
        rho: float = 0.05,
        sigma: float = 1.0,
        eps: float = 1e-12,
    ) -> None:
        self.params = list(params)
        self.base = base_optimizer
        self.rho = rho
        self.sigma = sigma
        self.eps = eps
        self._snapshots: list[Tensor] = []

    @property
    def param_groups(self) -> list[dict]:
        return self.base.param_groups

    def zero_grad(self, set_to_none: bool = True) -> None:
        self.base.zero_grad(set_to_none=set_to_none)

    def state_dict(self) -> dict:
        return self.base.state_dict()

    def load_state_dict(self, state_dict: dict) -> None:
        self.base.load_state_dict(state_dict)

    def train_mode(self) -> None:
        if hasattr(self.base, "train_mode"):
            self.base.train_mode()
        elif hasattr(self.base, "train"):
            self.base.train()

    def inference_mode(self) -> None:
        if hasattr(self.base, "inference_mode"):
            self.base.inference_mode()

    def _grad_norm(self) -> Tensor:
        norms = []
        for p in self.params:
            if p.grad is None:
                continue
            norms.append(p.grad.norm())
        if not norms:
            return torch.tensor(0.0)
        return torch.stack(norms).norm()

    def first_step(self) -> None:
        """Ascent: perturb params toward the F-SAM-filtered direction."""
        grad_norm = self._grad_norm() + self.eps
        scale = self.rho / grad_norm
        self._snapshots = []
        for p in self.params:
            self._snapshots.append(p.detach().clone())
            if p.grad is None:
                continue
            ew = scale * p.grad.detach() * self.sigma
            p.add_(ew)

    def second_step(self) -> None:
        """Descent: restore original params, apply base.step() with new grads."""
        for p, snap in zip(self.params, self._snapshots, strict=True):
            p.data.copy_(snap)
        self.base.step()
        self._snapshots = []

    def step(self, closure):  # noqa: ANN001
        if closure is None:
            raise ValueError("FriendlySAM.step() requires a closure")
        loss = closure()
        self.first_step()
        closure()
        self.second_step()
        return loss
