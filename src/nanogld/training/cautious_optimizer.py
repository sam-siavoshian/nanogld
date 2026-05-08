"""Cautious update mask wrapper for Schedule-Free AdamW.

Liang 2024 (arXiv:2411.16085): zero out parameter updates where the
sign of the update disagrees with the sign of the gradient. Acts as a
trust-region filter that masks "noisy" steps.

1.47x sample efficiency on AdamW-class optimizers, no new hparams,
~5 LOC patch on top of any base optimizer.
"""

from __future__ import annotations

import torch
from torch import Tensor


class CautiousMask:
    """Wraps a base optimizer and applies the cautious update mask.

    Usage:
        from schedulefree import AdamWScheduleFree
        base = AdamWScheduleFree(params, lr=1e-4)
        opt = CautiousMask(base)

        opt.train_mode()
        loss.backward()
        opt.step()
        opt.zero_grad()

    The mask is applied AFTER `base.step()` mutates parameters: for each
    Tensor `p`, we compute the per-element update `p_new - p_old`, zero
    out elements where `sign(update) != sign(-grad)`, then add the masked
    update to `p_old`.
    """

    def __init__(self, base_optimizer) -> None:  # noqa: ANN001
        self.base = base_optimizer

    @property
    def param_groups(self) -> list[dict]:
        return self.base.param_groups

    def zero_grad(self, set_to_none: bool = True) -> None:
        self.base.zero_grad(set_to_none=set_to_none)

    def train_mode(self) -> None:
        """Switch base optimizer to training mode (Schedule-Free averaging off)."""
        if hasattr(self.base, "train"):
            self.base.train()

    def inference_mode(self) -> None:
        """Switch base optimizer to inference mode (Schedule-Free averaging on)."""
        for method_name in ("eval",):
            method = getattr(self.base, method_name, None)
            if method is not None:
                method()

    def state_dict(self) -> dict:
        return self.base.state_dict()

    def load_state_dict(self, state_dict: dict) -> None:
        self.base.load_state_dict(state_dict)

    def step(self, closure=None) -> Tensor | None:  # noqa: ANN001
        snapshots: dict[int, Tensor] = {}
        grads: dict[int, Tensor] = {}
        for group in self.base.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    snapshots[id(p)] = p.detach().clone()
                    grads[id(p)] = p.grad.detach().clone()

        loss = self.base.step(closure)

        for group in self.base.param_groups:
            for p in group["params"]:
                pid = id(p)
                if pid not in snapshots:
                    continue
                old = snapshots[pid]
                grad = grads[pid]
                update = p.detach() - old
                mask = (torch.sign(update) == torch.sign(-grad)).to(update.dtype)
                p.data.copy_(old + mask * update)
        return loss
