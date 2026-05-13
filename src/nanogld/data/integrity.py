"""SHA-256 manifest + verify-on-load (V1-SPEC §45).

A ``MANIFEST.json`` lives next to every artifact directory and carries
the sha256 + byte size of each file. Loaders call
:func:`verify_artifacts` before consuming a file so a partially-downloaded
or tampered checkpoint fails closed at load time, not deep inside a
training step.

Layout::

    data/processed/
        training_v1_unified.pt
        training_v1_sidecar_fold_0.pt
        training_v1_sidecar_fold_1.pt
        MANIFEST.json   <- maps each filename to {sha256, size_bytes, built_at_utc}

The manifest is keyed by the basename, so the same MANIFEST.json travels
with the directory regardless of absolute path.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
from pathlib import Path
from typing import Any

from nanogld._atomic import atomic_write_json

MANIFEST_NAME = "MANIFEST.json"
_CHUNK = 1 << 20


def file_sha256(path: Path | str) -> str:
    """Streaming SHA-256 of a file (1 MiB chunks)."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"sha256: file not found {p}")
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(_CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def build_manifest(artifact_dir: Path | str, *, include: list[str] | None = None) -> dict[str, Any]:
    """Compute MANIFEST.json contents for every file in ``artifact_dir``.

    Args:
        artifact_dir: directory containing artifacts.
        include: optional whitelist of filenames; if None, every regular
            file in the directory (excluding ``MANIFEST.json`` itself and
            ``*.tmp`` / ``*.bak`` siblings) is included.

    Returns:
        Manifest dict with one entry per file.
    """
    base = Path(artifact_dir)
    if not base.exists() or not base.is_dir():
        raise FileNotFoundError(f"artifact_dir missing or not a dir: {base}")

    files: list[Path] = []
    if include is not None:
        files = [base / name for name in include]
    else:
        for p in sorted(base.iterdir()):
            if not p.is_file():
                continue
            if p.name == MANIFEST_NAME:
                continue
            if p.suffix in {".tmp", ".bak"}:
                continue
            if p.name.endswith(".tmp") or p.name.endswith(".bak"):
                continue
            files.append(p)

    entries: dict[str, dict[str, Any]] = {}
    for f in files:
        if not f.exists():
            raise FileNotFoundError(f"manifest: declared file missing {f}")
        entries[f.name] = {
            "sha256": file_sha256(f),
            "size_bytes": f.stat().st_size,
        }
    return {
        "schema_version": "v1_manifest.1",
        "built_at_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "files": entries,
    }


def write_manifest(artifact_dir: Path | str, *, include: list[str] | None = None) -> Path:
    """Build + atomically write ``MANIFEST.json`` in ``artifact_dir``."""
    base = Path(artifact_dir)
    manifest = build_manifest(base, include=include)
    out = base / MANIFEST_NAME
    atomic_write_json(out, manifest)
    return out


def verify_artifacts(
    artifact_dir: Path | str, *, require: list[str] | None = None
) -> dict[str, str]:
    """Verify each file under ``artifact_dir`` against ``MANIFEST.json``.

    Args:
        artifact_dir: directory containing the artifacts + MANIFEST.json.
        require: optional list of filenames that MUST be present + verified.
            If any required file is missing or has a different sha256
            than the manifest declares, raises.

    Returns:
        Dict ``{filename: verified_sha256}`` for every file in the manifest
        that exists on disk.

    Raises:
        FileNotFoundError: if MANIFEST.json or a required file is missing.
        ValueError: if a file's sha256 disagrees with the manifest.
    """
    base = Path(artifact_dir)
    manifest_path = base / MANIFEST_NAME
    if not manifest_path.exists():
        raise FileNotFoundError(f"MANIFEST.json missing in {base}")
    with manifest_path.open() as f:
        manifest = json.load(f)
    declared = manifest.get("files", {})
    verified: dict[str, str] = {}
    if require:
        missing = [name for name in require if name not in declared]
        if missing:
            raise FileNotFoundError(
                f"verify_artifacts: required files not in manifest: {missing}"
            )
    for name, info in declared.items():
        target = base / name
        if not target.exists():
            if require and name in require:
                raise FileNotFoundError(
                    f"verify_artifacts: required file {target} declared in "
                    f"MANIFEST.json but missing on disk"
                )
            continue
        actual = file_sha256(target)
        if actual != info["sha256"]:
            raise ValueError(
                f"verify_artifacts: sha256 mismatch for {name}: "
                f"manifest={info['sha256']} actual={actual}"
            )
        verified[name] = actual
    return verified


__all__ = [
    "MANIFEST_NAME",
    "build_manifest",
    "file_sha256",
    "verify_artifacts",
    "write_manifest",
]
