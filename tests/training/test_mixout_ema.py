"""Unit tests for Mixout and EMA wrappers."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from nanogld.training.ema import make_ema
from nanogld.training.mixout import Mixout


@pytest.mark.smoke
def test_mixout_p1_full_anchor_restore() -> None:
    m = nn.Linear(8, 4)
    anchor = {k: v.detach().clone() for k, v in m.state_dict().items()}
    with torch.no_grad():
        for p in m.parameters():
            p.add_(10.0)
    mix = Mixout(anchor, p=1.0)
    mix.apply(m)
    for name, p in m.named_parameters():
        torch.testing.assert_close(p, anchor[name])


@pytest.mark.smoke
def test_mixout_p0_no_change() -> None:
    m = nn.Linear(8, 4)
    anchor = {k: v.detach().clone() for k, v in m.state_dict().items()}
    pre = {k: v.detach().clone() for k, v in m.state_dict().items()}
    mix = Mixout(anchor, p=0.0)
    mix.apply(m)
    for name, p in m.named_parameters():
        torch.testing.assert_close(p, pre[name])


@pytest.mark.smoke
def test_mixout_invalid_p() -> None:
    with pytest.raises(ValueError):
        Mixout({}, p=1.5)


@pytest.mark.smoke
def test_ema_decay_one_freezes_after_first_copy() -> None:
    """After first copy, decay=1.0 keeps EMA frozen on subsequent updates."""
    m = nn.Linear(4, 4)
    ema = make_ema(m, decay=1.0)
    ema.update_parameters(m)
    snapshot = {k: v.detach().clone() for k, v in ema.module.state_dict().items()}
    with torch.no_grad():
        for p in m.parameters():
            p.fill_(99.0)
    ema.update_parameters(m)
    for k, v in ema.module.state_dict().items():
        if k in snapshot:
            torch.testing.assert_close(v, snapshot[k])


@pytest.mark.smoke
def test_ema_decay_zero_follows_model() -> None:
    m = nn.Linear(4, 4)
    ema = make_ema(m, decay=0.0)
    with torch.no_grad():
        for p in m.parameters():
            p.fill_(7.0)
    ema.update_parameters(m)
    for k, v in ema.module.state_dict().items():
        if "weight" in k:
            assert torch.allclose(v, torch.full_like(v, 7.0))
