"""Tests for ``nanogld._manifest``."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from nanogld._manifest import build_manifest

ALLOW_DIRTY = "NANOGLD_ALLOW_DIRTY_MANIFEST"
GIT_SHA = "NANOGLD_GIT_SHA"


@pytest.fixture(autouse=True)
def _allow_dirty(monkeypatch: pytest.MonkeyPatch) -> None:
    """Most tests run outside a git repo; allow dirty mode."""
    monkeypatch.setenv(ALLOW_DIRTY, "1")


def test_required_fields_present() -> None:
    m = build_manifest()
    required = {
        "git_sha",
        "python_version",
        "torch_version",
        "cuda_version",
        "platform",
        "hostname",
        "started_at_utc",
    }
    assert required <= set(m.keys())


def test_optional_fields_absent_when_not_requested() -> None:
    m = build_manifest()
    for key in ("dataset_sha256", "sidecar_sha256", "hparams_hash", "hparams", "extras"):
        assert key not in m


def test_extras_pass_through() -> None:
    m = build_manifest(extras={"stage": "ssl", "fold": 0})
    assert m["extras"] == {"stage": "ssl", "fold": 0}


def test_hparams_hash_stable_and_present(tmp_path: Path) -> None:
    hp = {"lr": 1e-4, "betas": [0.9, 0.95], "weight_decay": 0.1}
    m1 = build_manifest(hparams=hp)
    m2 = build_manifest(hparams=hp)
    assert m1["hparams_hash"] == m2["hparams_hash"]
    assert m1["hparams"] == hp


def test_hparams_hash_order_independent() -> None:
    hp_a = {"a": 1, "b": 2, "c": 3}
    hp_b = {"c": 3, "b": 2, "a": 1}
    m_a = build_manifest(hparams=hp_a)
    m_b = build_manifest(hparams=hp_b)
    assert m_a["hparams_hash"] == m_b["hparams_hash"]


def test_dataset_sha256_deterministic(tmp_path: Path) -> None:
    f = tmp_path / "data.bin"
    f.write_bytes(b"the quick brown fox jumps over the lazy dog")
    m_a = build_manifest(dataset_path=f)
    m_b = build_manifest(dataset_path=f)
    assert m_a["dataset_sha256"] == m_b["dataset_sha256"]
    assert (
        m_a["dataset_sha256"]
        == "05c6e08f1d9fdafa03147fcb8f82f124c76d2f70e3d989dc8aadb5e7d7450bec"
    )


def test_sidecar_sha256_changes_with_content(tmp_path: Path) -> None:
    f = tmp_path / "side.bin"
    f.write_bytes(b"v1")
    m_v1 = build_manifest(sidecar_path=f)
    f.write_bytes(b"v2")
    m_v2 = build_manifest(sidecar_path=f)
    assert m_v1["sidecar_sha256"] != m_v2["sidecar_sha256"]


def test_missing_dataset_path_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        build_manifest(dataset_path=tmp_path / "does-not-exist.pt")


def test_git_sha_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(GIT_SHA, "deadbeef" * 5)
    m = build_manifest()
    assert m["git_sha"] == "deadbeef" * 5


def test_fails_closed_without_env_or_repo(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv(ALLOW_DIRTY, raising=False)
    monkeypatch.delenv(GIT_SHA, raising=False)
    # Point HOME + cwd somewhere that has no git ancestry so the rev-parse fails.
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "nanogld._manifest._resolve_git_sha",
        lambda repo_root=None: _explode(),
    )
    with pytest.raises(RuntimeError, match="git SHA"):
        build_manifest()


def _explode() -> str:
    raise RuntimeError(
        "Cannot resolve git SHA: not a git repo and NANOGLD_GIT_SHA env not set."
    )


def test_iso_timestamp_format() -> None:
    m = build_manifest()
    # ISO 8601 with timezone, e.g. 2026-05-11T17:38:16.762289+00:00
    ts = m["started_at_utc"]
    assert "T" in ts
    assert ts.endswith("+00:00") or ts.endswith("Z")


def test_platform_field_has_machine() -> None:
    m = build_manifest()
    parts = m["platform"].split("/")
    assert len(parts) == 2
    assert parts[0] in {"darwin", "linux", "win32"} or parts[0].startswith("linux")
    assert parts[1] in {"arm64", "x86_64", "aarch64", "AMD64"}


def test_cuda_version_none_or_string() -> None:
    m = build_manifest()
    assert isinstance(m["cuda_version"], str)
    # "none" on Mac, e.g. "12.4" on CUDA host.
    assert m["cuda_version"] != ""


def test_dataset_and_sidecar_both_supplied(tmp_path: Path) -> None:
    a = tmp_path / "a.bin"
    a.write_bytes(b"dataset")
    b = tmp_path / "b.bin"
    b.write_bytes(b"sidecar")
    m = build_manifest(dataset_path=a, sidecar_path=b, hparams={"lr": 1e-4})
    assert m["dataset_sha256"]
    assert m["sidecar_sha256"]
    assert m["dataset_sha256"] != m["sidecar_sha256"]
    assert m["hparams"]["lr"] == 1e-4
