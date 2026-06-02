"""Unit tests for the V1 training orchestrator."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from nanogld.training.train import (
    TrainConfig,
    attach_ema,
    build_optimizer_stack,
    llrd_param_groups,
    setup_determinism,
)


@pytest.mark.smoke
def test_train_config_to_dict_serializable() -> None:
    cfg = TrainConfig(fold_idx=0)
    d = cfg.to_dict()
    assert d["base_lr"] == 1e-4
    assert d["fsam_rho"] == 0.05
    assert d["llrd_decay"] == 0.85
    assert isinstance(d["output_dir"], str)


@pytest.mark.smoke
def test_train_config_frozen() -> None:
    cfg = TrainConfig()
    from dataclasses import FrozenInstanceError

    with pytest.raises(FrozenInstanceError):
        cfg.base_lr = 5e-4  # type: ignore[misc]


@pytest.mark.smoke
def test_build_optimizer_stack_returns_fsam() -> None:
    pytest.importorskip("schedulefree")
    model = nn.Linear(8, 4)
    cfg = TrainConfig()
    opt = build_optimizer_stack(model, cfg)
    assert hasattr(opt, "first_step")
    assert hasattr(opt, "second_step")


@pytest.mark.smoke
def test_llrd_param_groups_decay() -> None:
    """Per-block LR is base_lr * decay^(N-l-1)."""

    class FakeBlock(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lin = nn.Linear(4, 4)

    class FakeEncoder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.transformer_blocks = nn.ModuleList([FakeBlock() for _ in range(3)])

    class FakeModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = FakeEncoder()
            self.head = nn.Linear(4, 1)

    model = FakeModel()
    groups = llrd_param_groups(model, base_lr=1e-3, decay=0.5, num_layers=3)
    assert groups[0]["lr"] == pytest.approx(1e-3 * 0.5**2)
    assert groups[1]["lr"] == pytest.approx(1e-3 * 0.5**1)
    assert groups[2]["lr"] == pytest.approx(1e-3 * 0.5**0)


@pytest.mark.smoke
def test_setup_determinism_does_not_crash() -> None:
    setup_determinism(seed=123)
    a = torch.randn(4)
    setup_determinism(seed=123)
    b = torch.randn(4)
    torch.testing.assert_close(a, b)


@pytest.mark.smoke
def test_attach_ema_returns_averaged_model() -> None:
    model = nn.Linear(4, 4)
    ema = attach_ema(model, decay=0.999)
    assert hasattr(ema, "update_parameters")
    assert hasattr(ema, "module")
