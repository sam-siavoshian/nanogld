"""Unit tests for V1 training losses."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from nanogld.training.losses import (
    clip_infonce,
    dann_loss,
    focal_loss,
    grad_reverse,
    sharpe_loss,
    simmtm_loss,
)


@pytest.mark.smoke
def test_focal_loss_zero_gamma_equals_ce() -> None:
    torch.manual_seed(0)
    logits = torch.randn(8, 3)
    targets = torch.randint(0, 3, (8,))
    fl = focal_loss(logits, targets, gamma=0.0)
    ce = F.cross_entropy(logits, targets)
    torch.testing.assert_close(fl, ce, atol=1e-5, rtol=1e-5)


@pytest.mark.smoke
def test_focal_loss_gamma3_smaller_for_correct() -> None:
    """Higher gamma down-weights well-classified examples."""
    logits = torch.tensor([[10.0, 0.0, 0.0]])
    targets = torch.tensor([0])
    fl0 = focal_loss(logits, targets, gamma=0.0)
    fl3 = focal_loss(logits, targets, gamma=3.0)
    assert fl3 < fl0


@pytest.mark.smoke
def test_sharpe_loss_handles_constant_pnl() -> None:
    pos = torch.tensor([0.5, 0.5, 0.5])
    ret = torch.tensor([0.001, 0.001, 0.001])
    loss = sharpe_loss(pos, ret, prev_position=None, cost_bps=0.0)
    assert torch.isfinite(loss)


@pytest.mark.smoke
def test_sharpe_loss_cost_aware_penalizes_turnover() -> None:
    pos = torch.tensor([1.0, -1.0, 1.0])
    prev = torch.tensor([0.0, 0.0, 0.0])
    ret = torch.tensor([0.001, 0.001, 0.001])
    no_cost = sharpe_loss(pos, ret, prev_position=None, cost_bps=0.0)
    with_cost = sharpe_loss(pos, ret, prev_position=prev, cost_bps=10.0)
    assert with_cost > no_cost


@pytest.mark.smoke
def test_grad_reverse_inverts_gradient() -> None:
    x = torch.randn(4, requires_grad=True)
    y = grad_reverse(x, alpha=2.0)
    y.sum().backward()
    expected_grad = -2.0 * torch.ones_like(x)
    torch.testing.assert_close(x.grad, expected_grad)


@pytest.mark.smoke
def test_dann_loss_basic() -> None:
    logits = torch.randn(6, 4)
    labels = torch.randint(0, 4, (6,))
    loss = dann_loss(logits, labels)
    assert loss.item() > 0


@pytest.mark.smoke
def test_simmtm_loss_runs_and_finite() -> None:
    views = torch.randn(2, 3, 16)
    targets = torch.randn(2, 3, 16)
    loss = simmtm_loss(views, targets)
    assert torch.isfinite(loss)


@pytest.mark.smoke
def test_clip_infonce_positive_pair_lower_loss() -> None:
    z_bar = torch.randn(8, 16)
    z_news_pos = z_bar.clone()
    z_news_neg = torch.randn(8, 16)
    loss_pos = clip_infonce(z_bar, z_news_pos)
    loss_neg = clip_infonce(z_bar, z_news_neg)
    assert loss_pos < loss_neg
