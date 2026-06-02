"""Atomic write helpers.

All artifact writes go through these helpers so a crashed process never
leaves a half-written checkpoint, JSON, or directory of files on disk.

The contract for every helper:
- Writes go first to a temporary sibling path (``<path>.tmp`` or
  ``<final_dir>.tmp/`` for directory-level commits).
- On success: a single ``os.replace`` swaps the temp into place. POSIX
  guarantees rename atomicity within the same filesystem.
- On failure: the temp is removed and the final path is untouched.

Use ``atomic_save_torch`` / ``atomic_write_json`` / ``atomic_write_bytes``
for single-file outputs. Use ``atomic_dir_commit`` (or the
``atomic_dir_writer`` context manager) to bundle several related files
(e.g. one calibration fold's ``t_scaler.pt`` + ``raps_quantiles.json`` +
``agaci_state.json`` + ``laplace.pt`` + ``meta.json``) into a single
atomic-or-rollback transaction.
"""

from __future__ import annotations

import json
import os
import shutil
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import torch


def _tmp_sibling(path: Path) -> Path:
    """Return ``<path>.tmp`` as a sibling for swap."""
    return path.with_suffix(path.suffix + ".tmp")


def atomic_save_torch(payload: Any, path: Path | str) -> None:
    """Write ``payload`` to ``path`` via ``torch.save`` atomically.

    Args:
        payload: anything ``torch.save`` accepts.
        path: final destination (parent directory must already exist).
    """
    p = Path(path)
    if not p.parent.exists():
        raise FileNotFoundError(f"Parent directory missing: {p.parent}")
    tmp = _tmp_sibling(p)
    try:
        with tmp.open("wb") as f:
            torch.save(payload, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, p)
    except Exception:
        if tmp.exists():
            tmp.unlink(missing_ok=True)
        raise


def atomic_write_json(
    path: Path | str, payload: dict[str, Any], *, indent: int | None = 2
) -> None:
    """Write JSON to ``path`` atomically.

    Args:
        path: final destination.
        payload: JSON-serializable dict.
        indent: passed to ``json.dump`` (default 2 for readability).
    """
    p = Path(path)
    if not p.parent.exists():
        raise FileNotFoundError(f"Parent directory missing: {p.parent}")
    tmp = _tmp_sibling(p)
    try:
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=indent, sort_keys=True, default=str)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, p)
    except Exception:
        if tmp.exists():
            tmp.unlink(missing_ok=True)
        raise


def atomic_write_bytes(path: Path | str, data: bytes) -> None:
    """Write raw bytes to ``path`` atomically."""
    p = Path(path)
    if not p.parent.exists():
        raise FileNotFoundError(f"Parent directory missing: {p.parent}")
    tmp = _tmp_sibling(p)
    try:
        with tmp.open("wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, p)
    except Exception:
        if tmp.exists():
            tmp.unlink(missing_ok=True)
        raise


def atomic_dir_commit(tmp_dir: Path | str, final_dir: Path | str) -> None:
    """Atomically swap ``tmp_dir`` to ``final_dir``.

    POSIX ``rename`` is atomic only when source and destination share a
    filesystem AND the destination is either nonexistent or a directory.
    Strategy: if ``final_dir`` exists we swap it aside to a ``.bak``
    sibling first, then rename ``tmp_dir`` into place, then remove the
    backup. On failure we restore the backup.

    Args:
        tmp_dir: scratch directory containing the new artifacts.
        final_dir: destination directory; replaced if it exists.

    Raises:
        FileNotFoundError: if ``tmp_dir`` does not exist.
        NotADirectoryError: if ``tmp_dir`` is not a directory.
    """
    tmp = Path(tmp_dir)
    final = Path(final_dir)
    if not tmp.exists():
        raise FileNotFoundError(f"tmp_dir missing: {tmp}")
    if not tmp.is_dir():
        raise NotADirectoryError(f"tmp_dir not a directory: {tmp}")
    if not final.parent.exists():
        raise FileNotFoundError(f"Parent of final_dir missing: {final.parent}")

    backup: Path | None = None
    if final.exists():
        backup = final.with_name(final.name + ".bak")
        if backup.exists():
            shutil.rmtree(backup)
        os.rename(final, backup)
    try:
        os.rename(tmp, final)
    except Exception:
        if backup is not None and backup.exists():
            if final.exists():
                shutil.rmtree(final, ignore_errors=True)
            os.rename(backup, final)
        raise
    if backup is not None and backup.exists():
        shutil.rmtree(backup, ignore_errors=True)


@contextmanager
def atomic_dir_writer(final_dir: Path | str) -> Generator[Path, None, None]:
    """Context manager yielding a temp dir, committing on clean exit.

    Usage::

        with atomic_dir_writer(out_dir) as tmp:
            atomic_save_torch(state, tmp / "state.pt")
            atomic_write_json(tmp / "meta.json", meta)
        # On clean exit, tmp -> out_dir via atomic swap.

    If the ``with`` block raises, the temp dir is removed and
    ``final_dir`` is left untouched.
    """
    final = Path(final_dir)
    tmp = final.with_name(final.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True, exist_ok=True)
    try:
        yield tmp
    except Exception:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    atomic_dir_commit(tmp, final)


__all__ = [
    "atomic_save_torch",
    "atomic_write_json",
    "atomic_write_bytes",
    "atomic_dir_commit",
    "atomic_dir_writer",
]
