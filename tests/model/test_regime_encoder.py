"""Unit tests for the regime-vector encoder."""

from __future__ import annotations

import pytest
import torch

from nanogld.model.regime_encoder import REGIME_VECTOR_DIM, RegimeEncoder, compute_regime_vec


@pytest.mark.smoke
def test_regime_dim_is_12() -> None:
    assert REGIME_VECTOR_DIM == 12


@pytest.mark.smoke
def test_encoder_passes_through() -> None:
    e = RegimeEncoder()
    r = torch.randn(4, 12)
    out = e(r)
    assert out.shape == r.shape
    assert out.dtype == torch.float32


@pytest.mark.smoke
def test_encoder_wrong_dim_raises() -> None:
    e = RegimeEncoder(regime_dim=12)
    bad = torch.randn(4, 10)
    with pytest.raises(ValueError):
        e(bad)


@pytest.mark.smoke
def test_compute_regime_vec_concatenates_to_12() -> None:
    b = 5
    vix = torch.randint(0, 2, (b, 3))
    rv = torch.randint(0, 2, (b, 3))
    fomc = torch.randint(0, 2, (b,))
    year = torch.randint(0, 2, (b, 4))
    hmm = torch.rand(b)
    v = compute_regime_vec(vix, rv, fomc, year, hmm)
    assert v.shape == (b, 12)
    assert v.dtype == torch.float32


@pytest.mark.smoke
def test_compute_regime_vec_handles_2d_fomc_hmm() -> None:
    b = 3
    vix = torch.zeros(b, 3)
    rv = torch.zeros(b, 3)
    fomc = torch.zeros(b, 1)
    year = torch.zeros(b, 4)
    hmm = torch.zeros(b, 1)
    v = compute_regime_vec(vix, rv, fomc, year, hmm)
    assert v.shape == (b, 12)
