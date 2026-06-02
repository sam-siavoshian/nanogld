"""Regression locks for ``training.observability`` (V1-SPEC §47).

W&B init is a no-op without ``WANDB_API_KEY`` + the wandb package — but
must never raise. The heartbeat thread is a critical liveness signal
for external watchdogs (Spark wrapper scripts, cron checks); we verify
it touches the sentinel file on the expected cadence and stops cleanly
on context exit (even when the body raises).
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from nanogld.training.observability import (
    HeartbeatThread,
    finish_wandb,
    heartbeat,
    init_wandb,
)


def test_init_wandb_no_project_returns_none() -> None:
    assert init_wandb() is None


def test_init_wandb_handles_missing_package(monkeypatch: pytest.MonkeyPatch) -> None:
    """When the wandb import fails, init_wandb returns None silently."""
    import builtins

    real_import = builtins.__import__

    def _blocked(name: str, *a: object, **kw: object):  # noqa: ANN202
        if name == "wandb":
            raise ImportError("blocked")
        return real_import(name, *a, **kw)

    monkeypatch.setattr(builtins, "__import__", _blocked)
    assert init_wandb(project="nanogld-v1") is None


def test_finish_wandb_with_none_is_safe() -> None:
    finish_wandb(None)  # no exception


def test_heartbeat_writes_sentinel(tmp_path: Path) -> None:
    sentinel = tmp_path / ".heartbeat"
    with heartbeat(sentinel, interval_seconds=0.05):
        time.sleep(0.2)
    assert sentinel.exists()
    # File must be recent (within last second).
    age = time.time() - sentinel.stat().st_mtime
    assert age < 1.0


def test_heartbeat_stops_on_exit(tmp_path: Path) -> None:
    sentinel = tmp_path / ".heartbeat"
    with heartbeat(sentinel, interval_seconds=0.05) as hb:
        time.sleep(0.1)
    assert not hb.is_alive()


def test_heartbeat_stops_on_exception(tmp_path: Path) -> None:
    sentinel = tmp_path / ".heartbeat"
    with pytest.raises(RuntimeError, match="boom"):
        with heartbeat(sentinel, interval_seconds=0.05) as hb:
            time.sleep(0.1)
            raise RuntimeError("boom")
    assert not hb.is_alive()


def test_heartbeat_creates_parent_dir(tmp_path: Path) -> None:
    sentinel = tmp_path / "subdir" / ".heartbeat"
    with heartbeat(sentinel, interval_seconds=0.05):
        time.sleep(0.1)
    assert sentinel.exists()


def test_heartbeat_thread_is_daemon(tmp_path: Path) -> None:
    t = HeartbeatThread(tmp_path / ".heartbeat", interval_seconds=1.0)
    assert t.daemon is True
    # Don't actually start; verifying class attr is enough.
