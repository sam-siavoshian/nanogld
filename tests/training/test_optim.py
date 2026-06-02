"""Tests for ``nanogld.training.optim.build_optimizer`` and the wrapped stack.

Locks two bugs found while wiring this:

1. ``CautiousMask.step`` used to snapshot grads BEFORE ``base.step`` runs,
   which yielded empty snapshots in the closure-style flow (the closure
   only populates ``p.grad`` later, during the SAM ascent). The mask was
   silently never applied. After the fix, params still snapshot before
   step but grads are read off ``p.grad`` AFTER step returns.
2. ``FriendlySAM.first_step`` / ``second_step`` mutated leaf parameters
   in-place outside ``no_grad``, which raises
   ``RuntimeError: a leaf Variable that requires grad is being used in an
   in-place operation`` for grad-tracking params. Both methods now run
   inside ``@torch.no_grad()``.

Plus a baseline smoke test that the full stack drives loss down on a tiny
classification toy problem.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from nanogld.training.cautious_optimizer import CautiousMask
from nanogld.training.friendly_sam import FriendlySAM
from nanogld.training.optim import build_optimizer


def _toy_model_data(seed: int = 0) -> tuple[nn.Module, torch.Tensor, torch.Tensor]:
    torch.manual_seed(seed)
    model = nn.Sequential(nn.Linear(10, 16), nn.GELU(), nn.Linear(16, 3))
    x = torch.randn(32, 10)
    y = torch.randint(0, 3, (32,))
    return model, x, y


def test_build_optimizer_returns_full_stack() -> None:
    m = nn.Linear(4, 2)
    opt = build_optimizer(m.parameters(), lr=1e-3)
    assert isinstance(opt, CautiousMask)
    assert isinstance(opt.base, FriendlySAM)


def test_build_optimizer_accepts_param_groups() -> None:
    m = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 2))
    groups = [
        {"params": list(m[0].parameters()), "lr": 1e-4},
        {"params": list(m[1].parameters()), "lr": 1e-3},
    ]
    opt = build_optimizer(groups, lr=1e-4)
    assert len(opt.param_groups) == 2
    assert opt.param_groups[0]["lr"] == 1e-4
    assert opt.param_groups[1]["lr"] == 1e-3


def test_closure_step_runs_and_reduces_loss() -> None:
    model, x, y = _toy_model_data()
    opt = build_optimizer(model.parameters(), lr=5e-3)

    def closure() -> torch.Tensor:
        opt.zero_grad()
        loss = F.cross_entropy(model(x), y)
        loss.backward()
        return loss

    first = float(opt.step(closure).detach())
    for _ in range(40):
        opt.step(closure)
    last = float(opt.step(closure).detach())
    assert last < first


def test_cautious_mask_applied_in_closure_path() -> None:
    """Mask should zero updates where sign(update) != sign(-grad).

    Build a tiny model where we can hand-craft a gradient and verify
    masking. After step, no parameter element should have moved in a
    direction OPPOSITE to ``-grad`` (i.e. moved with the grad rather
    than against it). The mask zeroes those elements.
    """
    torch.manual_seed(0)
    m = nn.Linear(4, 2, bias=False)
    opt = build_optimizer(m.parameters(), lr=1e-3)

    snapshots = {id(p): p.detach().clone() for p in m.parameters()}

    def closure() -> torch.Tensor:
        opt.zero_grad()
        x = torch.randn(8, 4)
        y = torch.randint(0, 2, (8,))
        loss = F.cross_entropy(m(x), y)
        loss.backward()
        return loss

    opt.step(closure)

    for p in m.parameters():
        old = snapshots[id(p)]
        update = p.detach() - old
        grad = p.grad.detach() if p.grad is not None else torch.zeros_like(old)
        # Wherever the update is non-zero, it must point opposite to grad.
        moved = update.abs() > 0
        signs_ok = torch.sign(update[moved]) == torch.sign(-grad[moved])
        assert signs_ok.all(), "CautiousMask did not zero opposed updates"


def test_friendly_sam_no_grad_inplace() -> None:
    """Regression lock: in-place perturbation of grad-tracking leaves must succeed."""
    model, x, y = _toy_model_data()
    opt = build_optimizer(model.parameters(), lr=1e-3)

    def closure() -> torch.Tensor:
        opt.zero_grad()
        loss = F.cross_entropy(model(x), y)
        loss.backward()
        return loss

    # If we hit the leaf-inplace bug, this would raise.
    opt.step(closure)


def test_train_and_inference_mode_propagate() -> None:
    """``train_mode()`` and ``inference_mode()`` must traverse Cautious + FSAM
    down to ScheduleFreeAdamW.

    We check the base ScheduleFreeAdamW's ``param_groups[0]["train_mode"]``
    state which the library uses internally.
    """
    m = nn.Linear(4, 2)
    opt = build_optimizer(m.parameters(), lr=1e-3)

    # FSAM is the immediate base; SF AdamW is FSAM.base.
    sf_adamw = opt.base.base
    assert sf_adamw.param_groups[0].get("train_mode", False) is True

    opt.inference_mode()
    assert sf_adamw.param_groups[0].get("train_mode", True) is False

    opt.train_mode()
    assert sf_adamw.param_groups[0].get("train_mode", False) is True


def test_step_requires_closure_in_full_stack() -> None:
    """Without a closure, FSAM.step raises (closure-required contract)."""
    import pytest

    m = nn.Linear(4, 2)
    opt = build_optimizer(m.parameters(), lr=1e-3)
    with pytest.raises(ValueError, match="closure"):
        opt.step()


def test_state_dict_roundtrip() -> None:
    m = nn.Linear(4, 2)
    opt = build_optimizer(m.parameters(), lr=1e-3)

    def closure() -> torch.Tensor:
        opt.zero_grad()
        x = torch.randn(2, 4)
        y = torch.randint(0, 2, (2,))
        loss = F.cross_entropy(m(x), y)
        loss.backward()
        return loss

    opt.step(closure)
    sd = opt.state_dict()
    assert isinstance(sd, dict)
    opt.load_state_dict(sd)  # should not raise


def test_zero_grad_clears_grads() -> None:
    model, x, y = _toy_model_data()
    opt = build_optimizer(model.parameters(), lr=1e-3)
    F.cross_entropy(model(x), y).backward()
    any_grad = any(
        p.grad is not None and p.grad.abs().sum().item() > 0 for p in model.parameters()
    )
    assert any_grad
    opt.zero_grad()
    cleared = all(p.grad is None or p.grad.abs().sum().item() == 0 for p in model.parameters())
    assert cleared
