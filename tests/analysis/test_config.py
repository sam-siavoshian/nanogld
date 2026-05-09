"""Unit tests for AnalysisConfig hashing + immutability."""

from __future__ import annotations

from pathlib import Path

import pytest

from nanogld.analysis.config import AnalysisConfig

pytestmark = pytest.mark.smoke


def _cfg(**overrides: object) -> AnalysisConfig:
    base = {
        "checkpoint_path": Path("/tmp/ckpt.pt"),
        "unified_path": Path("/tmp/unified.pt"),
        "sidecar_path": Path("/tmp/sidecar.pt"),
        "fold_idx": 0,
    }
    base.update(overrides)
    return AnalysisConfig(**base)


def test_run_hash_deterministic() -> None:
    a = _cfg()
    b = _cfg()
    assert a.run_hash() == b.run_hash()


def test_run_hash_changes_with_fold() -> None:
    a = _cfg(fold_idx=0)
    b = _cfg(fold_idx=1)
    assert a.run_hash() != b.run_hash()


def test_run_hash_ignores_paths() -> None:
    a = _cfg(checkpoint_path=Path("/x.pt"))
    b = _cfg(checkpoint_path=Path("/y.pt"))
    assert a.run_hash() == b.run_hash()


def test_run_hash_changes_with_seed() -> None:
    a = _cfg(seed=1)
    b = _cfg(seed=2)
    assert a.run_hash() != b.run_hash()


def test_immutable() -> None:
    cfg = _cfg()
    from dataclasses import FrozenInstanceError

    with pytest.raises(FrozenInstanceError):
        cfg.fold_idx = 99  # type: ignore[misc]
