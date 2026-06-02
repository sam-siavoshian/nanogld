"""Unit tests for V1 augmentations."""

from __future__ import annotations

import pytest
import torch

from nanogld.training.augmentation import manifold_mixup, simpsi_jitter, wave_mask


@pytest.mark.smoke
def test_simpsi_shape_preservation() -> None:
    x = torch.randn(2, 64, 8)
    y = simpsi_jitter(x, sigma=0.02)
    assert y.shape == x.shape


@pytest.mark.smoke
def test_simpsi_zero_sigma_is_identity() -> None:
    x = torch.randn(2, 32, 4)
    y = simpsi_jitter(x, sigma=0.0)
    torch.testing.assert_close(y, x)


@pytest.mark.smoke
def test_wave_mask_shape_or_passes_through() -> None:
    x = torch.randn(1, 32, 2)
    y = wave_mask(x, mask_prob=0.3)
    assert y.shape == x.shape


@pytest.mark.smoke
def test_wave_mask_zero_prob_is_identity_when_pywt_missing() -> None:
    x = torch.randn(1, 32, 2)
    y = wave_mask(x, mask_prob=0.0)
    torch.testing.assert_close(y, x)


@pytest.mark.smoke
def test_manifold_mixup_shape() -> None:
    h = torch.randn(8, 16, 32)
    labels = torch.randint(0, 3, (8,))
    mixed, la, lb, lam = manifold_mixup(h, labels, alpha=0.2)
    assert mixed.shape == h.shape
    assert la.shape == labels.shape
    assert lb.shape == labels.shape
    assert 0.0 <= lam <= 1.0


@pytest.mark.smoke
def test_manifold_mixup_alpha_zero_passes_through() -> None:
    h = torch.randn(4, 8, 16)
    labels = torch.randint(0, 3, (4,))
    mixed, la, lb, lam = manifold_mixup(h, labels, alpha=0.0)
    torch.testing.assert_close(mixed, h)
    assert lam == 1.0
    torch.testing.assert_close(la, lb)
