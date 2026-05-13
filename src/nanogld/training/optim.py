"""Canonical V1 optimizer stack: ``Cautious(FriendlySAM(ScheduleFreeAdamW))``.

Stack rationale (per V1-SPEC §8.1 + Decision 1B):

- **ScheduleFreeAdamW** (Defazio 2024): implicit Polyak averaging, removes
  the need for an LR schedule. Mode-switching matters: ``.train()`` for the
  forward+backward passes that produce gradients, ``.eval()`` to swap to
  the averaged weights at inference time.
- **FriendlySAM** wrap (Li 2024, rho=0.05): two-step optimizer that perturbs
  parameters toward the noise-filtered gradient direction, recomputes the
  loss at the perturbed point, then takes the actual descent step from the
  ORIGINAL parameters using the perturbed gradient. Promotes flat minima.
  Requires a ``closure`` callable from the caller because ``step()`` must
  run the loss + grads TWICE per parameter update.
- **CautiousMask** outermost (Liang 2024): after each step, zero out the
  per-element parameter update where the sign disagrees with the negative
  gradient (the "trust" direction). Drop-in 1.47x sample-efficiency gain
  with no new hyperparameters.

Composition order matters:

    CautiousMask( FriendlySAM( ScheduleFreeAdamW( ... ) ) )

CautiousMask observes the net parameter delta from one step, so it must
wrap the entire SAM 2-pass operation. FriendlySAM in turn delegates the
actual descent step to ScheduleFree, which produces the canonical update.

All three stages in V1 (SSL pretrain, linear probe, LLRD fine-tune) use
the same stack. Each caller must pass a closure that does
``zero_grad -> forward -> loss -> backward -> grad_clip``, returning the
loss tensor. The closure may run twice per step (SAM ascent + descent).
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from schedulefree import AdamWScheduleFree
from torch import Tensor

from nanogld.training.cautious_optimizer import CautiousMask
from nanogld.training.friendly_sam import FriendlySAM

ParamSource = Iterable[Tensor] | Iterable[dict[str, Any]]


def build_optimizer(
    params_or_groups: ParamSource,
    *,
    lr: float = 1e-4,
    betas: tuple[float, float] = (0.9, 0.95),
    weight_decay: float = 0.1,
    warmup_steps: int = 300,
    fsam_rho: float = 0.05,
    fsam_sigma: float = 1.0,
) -> CautiousMask:
    """Build ``Cautious(FriendlySAM(ScheduleFreeAdamW))``.

    The returned object behaves like a torch optimizer: ``.zero_grad()``,
    ``.step(closure)``, ``.state_dict()``, ``.load_state_dict()``,
    ``.train_mode()``, ``.inference_mode()``. ``step`` REQUIRES a closure
    because FSAM computes the loss twice per update.

    Args:
        params_or_groups: either ``model.parameters()`` or per-group dicts
            from LLRD (e.g. ``[{"params": [...], "lr": ...}, ...]``).
        lr: base learning rate; per-group ``lr`` overrides this.
        betas: AdamW momentum coefficients.
        weight_decay: decoupled weight decay.
        warmup_steps: ScheduleFree warmup horizon.
        fsam_rho: Friendly-SAM perturbation radius.
        fsam_sigma: Friendly-SAM noise-filter scale.

    Returns:
        ``CautiousMask`` instance wrapping the full stack.
    """
    groups_list = list(params_or_groups)
    base = AdamWScheduleFree(
        groups_list,
        lr=lr,
        betas=betas,
        weight_decay=weight_decay,
        warmup_steps=warmup_steps,
    )
    base.train()

    fsam_params: list[Tensor] = []
    for entry in groups_list:
        if isinstance(entry, dict):
            fsam_params.extend(entry["params"])
        else:
            fsam_params.append(entry)

    fsam = FriendlySAM(
        fsam_params,
        base_optimizer=base,
        rho=fsam_rho,
        sigma=fsam_sigma,
    )
    return CautiousMask(fsam)


__all__ = ["build_optimizer", "ParamSource"]
