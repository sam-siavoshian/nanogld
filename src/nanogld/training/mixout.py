"""Mixout regularizer (Lee, Cho, Kang ICLR 2020 arXiv:1909.11299).

Bernoulli-mix between current parameters and a frozen anchor (e.g. SSL
checkpoint) at every forward pass during stage 3 fine-tune. Acts as
L2-toward-pretrained with adaptive coefficient `p / (1 - p)`.

V1 default: p=0.7.

Usage:
    anchor = {k: v.detach().clone() for k, v in model.state_dict().items()}
    mixout = Mixout(anchor, p=0.7)
    mixout.apply(model)            # applies in-place mix to current params
    # ... forward + backward + step ...
"""

from __future__ import annotations

import torch
from torch import nn


class Mixout:
    """Stateful Mixout regularizer.

    Args:
        anchor_state_dict: snapshot of pre-finetune (e.g. SSL) params.
        p: probability of replacing each param element with the anchor.
            p=0 means no mixing (keep current); p=1 means full anchor restore.
    """

    def __init__(self, anchor_state_dict: dict[str, torch.Tensor], p: float = 0.7) -> None:
        if not (0.0 <= p <= 1.0):
            raise ValueError(f"Mixout p must be in [0, 1], got {p}")
        self.anchor = {k: v.detach().clone() for k, v in anchor_state_dict.items()}
        self.p = p

    def apply(self, model: nn.Module) -> None:
        """Mix in-place: each param element is anchor with prob p, current with prob 1-p."""
        if self.p == 0.0:
            return
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name not in self.anchor:
                    continue
                a = self.anchor[name].to(device=param.device, dtype=param.dtype)
                if a.shape != param.shape:
                    continue
                mask = torch.empty_like(param).bernoulli_(self.p)
                param.data.mul_(1.0 - mask).add_(a * mask)
