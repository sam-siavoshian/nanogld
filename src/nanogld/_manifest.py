"""Reproducibility manifest builder.

Used by every torch.save() in training and calibration to embed
ground-truth metadata: code version (git SHA), data version (sha256),
runtime (Python / torch / CUDA versions, hostname), start time, and
arbitrary extras (hparams).

Fails closed on missing required fields unless ``NANOGLD_ALLOW_DIRTY_MANIFEST=1``
is set in the environment (for local smoke tests). Production training
and calibration runs MUST resolve a git SHA and artifact hashes.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import platform
import socket
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

_GIT_SHA_ENV = "NANOGLD_GIT_SHA"
_ALLOW_DIRTY_ENV = "NANOGLD_ALLOW_DIRTY_MANIFEST"


def _resolve_git_sha(repo_root: Path | None = None) -> str:
    """Resolve git SHA via env var, then ``git rev-parse HEAD``, else raise.

    Args:
        repo_root: optional explicit repo root. Defaults to two parents up
            from this module (i.e. project root containing ``src/``).

    Returns:
        40-char lower-case hex SHA, env-supplied override, or ``"dirty-no-git"``
        when ``NANOGLD_ALLOW_DIRTY_MANIFEST=1`` is set.

    Raises:
        RuntimeError: if neither env nor git lookup succeeds and
            ALLOW_DIRTY env is not set.
    """
    env = os.environ.get(_GIT_SHA_ENV)
    if env:
        return env
    try:
        root = repo_root or Path(__file__).resolve().parent.parent.parent
        out = (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=str(root),
                stderr=subprocess.DEVNULL,
                timeout=2,
            )
            .decode()
            .strip()
        )
        if len(out) == 40 and all(c in "0123456789abcdef" for c in out):
            return out
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        pass
    if os.environ.get(_ALLOW_DIRTY_ENV) == "1":
        return "dirty-no-git"
    raise RuntimeError(
        "Cannot resolve git SHA: not a git repo and "
        f"{_GIT_SHA_ENV} env not set. "
        f"Set {_ALLOW_DIRTY_ENV}=1 for local dev to override."
    )


def _file_sha256(path: Path | str) -> str:
    """Compute SHA-256 of a file's bytes (streaming, 1 MiB chunks)."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Cannot hash missing file: {p}")
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _hparams_hash(hparams: Mapping[str, Any]) -> str:
    """Stable SHA-256 of canonical-form hparams dict (sorted keys, no spaces)."""
    canonical = json.dumps(
        dict(hparams), sort_keys=True, separators=(",", ":"), default=str
    )
    return hashlib.sha256(canonical.encode()).hexdigest()


def build_manifest(
    *,
    dataset_path: Path | str | None = None,
    sidecar_path: Path | str | None = None,
    hparams: Mapping[str, Any] | None = None,
    extras: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a reproducibility manifest dict.

    Required fields (always present; raises on lookup failure):
        - ``git_sha``: git rev-parse HEAD or ``NANOGLD_GIT_SHA`` env override
        - ``python_version``: ``major.minor.micro``
        - ``torch_version``: ``torch.__version__``
        - ``cuda_version``: ``torch.version.cuda`` or ``"none"``
        - ``platform``: ``sys.platform/machine``
        - ``hostname``: ``socket.gethostname()``
        - ``started_at_utc``: ISO-8601 UTC timestamp at call time

    Optional fields (present only if argument supplied):
        - ``dataset_sha256`` — when ``dataset_path`` given.
        - ``sidecar_sha256`` — when ``sidecar_path`` given.
        - ``hparams_hash`` and ``hparams`` — when ``hparams`` given.
        - ``extras`` — pass-through key/values.

    Args:
        dataset_path: path to dataset artifact; sha256 computed if supplied.
        sidecar_path: path to sidecar artifact; sha256 computed if supplied.
        hparams: hparams dict; hashed and stored verbatim.
        extras: pass-through key/values (e.g. ``{"stage": "ssl"}``).

    Returns:
        Manifest dict suitable for embedding in ``torch.save`` payloads.

    Raises:
        RuntimeError: if git SHA cannot resolve and ALLOW_DIRTY env unset.
        FileNotFoundError: if a path argument is supplied but the file is missing.
    """
    m: dict[str, Any] = {
        "git_sha": _resolve_git_sha(),
        "python_version": (
            f"{sys.version_info.major}.{sys.version_info.minor}."
            f"{sys.version_info.micro}"
        ),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda or "none",
        "platform": f"{sys.platform}/{platform.machine()}",
        "hostname": socket.gethostname(),
        "started_at_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
    }
    if dataset_path is not None:
        m["dataset_sha256"] = _file_sha256(dataset_path)
    if sidecar_path is not None:
        m["sidecar_sha256"] = _file_sha256(sidecar_path)
    if hparams is not None:
        m["hparams_hash"] = _hparams_hash(hparams)
        m["hparams"] = dict(hparams)
    if extras:
        m["extras"] = dict(extras)
    return m


__all__ = ["build_manifest"]
