"""Regression locks for ``data.integrity`` SHA-256 manifest (V1-SPEC §45)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from nanogld.data.integrity import (
    MANIFEST_NAME,
    build_manifest,
    file_sha256,
    verify_artifacts,
    write_manifest,
)


def test_file_sha256_deterministic(tmp_path: Path) -> None:
    p = tmp_path / "a.bin"
    p.write_bytes(b"the quick brown fox")
    assert file_sha256(p) == file_sha256(p)


def test_file_sha256_changes_with_content(tmp_path: Path) -> None:
    p = tmp_path / "a.bin"
    p.write_bytes(b"v1")
    h1 = file_sha256(p)
    p.write_bytes(b"v2")
    h2 = file_sha256(p)
    assert h1 != h2


def test_file_sha256_missing_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        file_sha256(tmp_path / "nope.bin")


def test_build_manifest_lists_files(tmp_path: Path) -> None:
    (tmp_path / "a.pt").write_bytes(b"AAA")
    (tmp_path / "b.pt").write_bytes(b"BBBB")
    (tmp_path / "c.tmp").write_bytes(b"junk")  # should be excluded
    manifest = build_manifest(tmp_path)
    assert manifest["schema_version"] == "v1_manifest.1"
    assert set(manifest["files"].keys()) == {"a.pt", "b.pt"}
    for name in ("a.pt", "b.pt"):
        assert manifest["files"][name]["size_bytes"] == (tmp_path / name).stat().st_size


def test_write_manifest_atomic(tmp_path: Path) -> None:
    (tmp_path / "x.pt").write_bytes(b"hello")
    out = write_manifest(tmp_path)
    assert out.name == MANIFEST_NAME
    assert out.exists()
    payload = json.loads(out.read_text())
    assert "x.pt" in payload["files"]
    # Atomic temp file should be gone.
    assert not (tmp_path / f"{MANIFEST_NAME}.tmp").exists()


def test_verify_artifacts_happy_path(tmp_path: Path) -> None:
    (tmp_path / "x.pt").write_bytes(b"hello")
    write_manifest(tmp_path)
    verified = verify_artifacts(tmp_path)
    assert "x.pt" in verified


def test_verify_artifacts_catches_tampering(tmp_path: Path) -> None:
    (tmp_path / "x.pt").write_bytes(b"hello")
    write_manifest(tmp_path)
    # Mutate file AFTER manifest written.
    (tmp_path / "x.pt").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="sha256 mismatch"):
        verify_artifacts(tmp_path)


def test_verify_artifacts_required_missing(tmp_path: Path) -> None:
    (tmp_path / "x.pt").write_bytes(b"hello")
    write_manifest(tmp_path)
    with pytest.raises(FileNotFoundError, match="required files not in manifest"):
        verify_artifacts(tmp_path, require=["missing.pt"])


def test_verify_artifacts_no_manifest(tmp_path: Path) -> None:
    (tmp_path / "x.pt").write_bytes(b"hello")
    with pytest.raises(FileNotFoundError, match="MANIFEST.json missing"):
        verify_artifacts(tmp_path)


def test_verify_artifacts_required_file_missing_on_disk(tmp_path: Path) -> None:
    (tmp_path / "x.pt").write_bytes(b"hello")
    write_manifest(tmp_path)
    (tmp_path / "x.pt").unlink()
    with pytest.raises(FileNotFoundError, match="missing on disk"):
        verify_artifacts(tmp_path, require=["x.pt"])


def test_dataset_verify_integrity_smoke(tmp_path: Path) -> None:
    """Hooks at dataset.__init__ should not crash when manifest absent."""
    import torch

    from nanogld.data.dataset import NanoGLDDataset

    # Build a tiny synthetic unified.pt that the dataset can load. Real
    # NanoGLDDataset needs many keys; this only checks the verify path
    # short-circuits cleanly. We trigger the FileNotFoundError on missing
    # unified to confirm error path is reached.
    with pytest.raises(FileNotFoundError):
        NanoGLDDataset(unified_path=tmp_path / "missing.pt", verify_integrity=True)
