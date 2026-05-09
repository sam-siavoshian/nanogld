"""Unit tests for series decomposition."""

from __future__ import annotations

import pytest
import torch

from nanogld.model.decomposition import SeriesDecomposition


@pytest.mark.smoke
def test_trend_plus_seasonal_recovers_input() -> None:
    d = SeriesDecomposition(kernel_size=24)
    x = torch.randn(2, 64, 681)
    trend, seasonal = d(x)
    torch.testing.assert_close(trend + seasonal, x, atol=1e-5, rtol=1e-5)


@pytest.mark.smoke
def test_shape_preservation() -> None:
    d = SeriesDecomposition(kernel_size=24)
    x = torch.randn(2, 64, 7)
    trend, seasonal = d(x)
    assert trend.shape == x.shape
    assert seasonal.shape == x.shape


@pytest.mark.smoke
def test_causal_no_future_leak() -> None:
    """Truncating after position T must not change trend at positions <= T."""
    d = SeriesDecomposition(kernel_size=24)
    x = torch.randn(1, 80, 4)
    trend_full, _ = d(x)
    trend_short, _ = d(x[:, :40, :])
    torch.testing.assert_close(trend_full[:, :40, :], trend_short, atol=1e-6, rtol=1e-6)


@pytest.mark.smoke
def test_kernel_one_is_identity() -> None:
    d = SeriesDecomposition(kernel_size=1)
    x = torch.randn(1, 16, 4)
    trend, seasonal = d(x)
    torch.testing.assert_close(trend, x)
    torch.testing.assert_close(seasonal, torch.zeros_like(x), atol=1e-6, rtol=1e-6)


@pytest.mark.smoke
def test_invalid_kernel_size() -> None:
    with pytest.raises(ValueError):
        SeriesDecomposition(kernel_size=0)
