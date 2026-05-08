"""Unit tests for RevIN."""

from __future__ import annotations

import pytest
import torch

from nanogld.model.revin import RevIN


@pytest.mark.smoke
def test_round_trip_recovers_input() -> None:
    n = RevIN(num_features=681, affine=False)
    x = torch.randn(2, 64, 681) * 5.0 + 100.0
    y = n(x, mode="norm")
    x_back = n(y, mode="denorm")
    torch.testing.assert_close(x_back, x, atol=1e-4, rtol=1e-4)


@pytest.mark.smoke
def test_per_channel_affine_count() -> None:
    n = RevIN(num_features=681, affine=True)
    n_params = sum(p.numel() for p in n.parameters())
    assert n_params == 2 * 681


@pytest.mark.smoke
def test_invalid_mode_raises() -> None:
    n = RevIN(num_features=10)
    x = torch.randn(1, 4, 10)
    with pytest.raises(ValueError):
        n(x, mode="invalid")


@pytest.mark.smoke
def test_norm_centers_and_scales() -> None:
    n = RevIN(num_features=4, affine=False)
    x = torch.randn(2, 16, 4) * 3.0 + 10.0
    y = n(x, mode="norm")
    mean = y.mean(dim=1)
    std = y.std(dim=1, unbiased=False)
    torch.testing.assert_close(mean, torch.zeros_like(mean), atol=1e-5, rtol=1e-3)
    torch.testing.assert_close(std, torch.ones_like(std), atol=1e-2, rtol=1e-2)
