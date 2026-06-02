"""W&B init + heartbeat sentinel (V1-SPEC §47).

Two lightweight side effects long training loops need:

1. **W&B init**: optional. Off by default — set ``WANDB_API_KEY`` and
   pass ``wandb_project`` to enable. We do NOT pip-install wandb here;
   if the import fails, the call is a no-op.
2. **Heartbeat sentinel**: a background daemon thread that touches
   ``fold_out/.heartbeat`` every ``heartbeat_interval`` seconds. An
   external orchestrator (Spark wrapper script or cron watchdog)
   watches the mtime of this file to detect stalled training processes
   (e.g., NaN-loss hang or CUDA OOM). A stale heartbeat triggers a
   fold re-launch.

Both are intentionally optional. The training pipeline runs fine
without them; the orchestrator just loses out-of-process observability
when they're disabled.
"""

from __future__ import annotations

import logging
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

LOG = logging.getLogger("nanogld.training.observability")


def init_wandb(
    *,
    project: str | None = None,
    run_name: str | None = None,
    config: dict[str, Any] | None = None,
    tags: list[str] | None = None,
) -> Any | None:
    """Initialize a W&B run if the optional dependency is installed.

    Returns the wandb run object on success, ``None`` on any failure
    (import error, no API key, network issue). All failures are logged
    but never raise — training must not be blocked by telemetry.

    Args:
        project: W&B project name (e.g. ``"nanogld-v1"``).
        run_name: human-readable run name; default falls through to
            wandb's auto-generated name.
        config: hparams dict to log under run.config.
        tags: optional tag list.
    """
    if project is None:
        return None
    try:
        import wandb  # noqa: PLC0415
    except ImportError:
        LOG.info("wandb not installed — skipping init")
        return None
    try:
        run = wandb.init(
            project=project,
            name=run_name,
            config=config or {},
            tags=tags or [],
        )
        LOG.info("wandb run initialized: %s", getattr(run, "name", "?"))
        return run
    except Exception as exc:  # noqa: BLE001
        LOG.warning("wandb init failed (non-fatal): %s", exc)
        return None


def finish_wandb(run: Any | None) -> None:
    """Tear down a W&B run if one was created."""
    if run is None:
        return
    try:
        import wandb  # noqa: PLC0415

        wandb.finish()
    except Exception as exc:  # noqa: BLE001
        LOG.warning("wandb finish failed (non-fatal): %s", exc)


class HeartbeatThread(threading.Thread):
    """Background daemon thread that touches a sentinel file every N s.

    Use via :func:`heartbeat` context manager rather than instantiating
    directly. The thread is daemon=True so it does not block process
    exit; the context manager also signals an explicit stop.
    """

    def __init__(self, sentinel: Path, interval_seconds: float) -> None:
        super().__init__(daemon=True, name="nanogld-heartbeat")
        self.sentinel = sentinel
        self.interval = float(interval_seconds)
        # NB: cannot name this ``_stop`` — threading.Thread has its own
        # ``_stop()`` method called from inside ``.join()``; shadowing it
        # raises ``TypeError: 'Event' object is not callable`` on exit.
        self._stop_event = threading.Event()

    def run(self) -> None:
        while not self._stop_event.is_set():
            try:
                self.sentinel.parent.mkdir(parents=True, exist_ok=True)
                self.sentinel.touch()
            except OSError as exc:  # noqa: BLE001
                LOG.warning("heartbeat touch failed: %s", exc)
            self._stop_event.wait(self.interval)

    def stop(self) -> None:
        self._stop_event.set()


@contextmanager
def heartbeat(sentinel: Path | str, *, interval_seconds: float = 60.0):
    """Context manager: starts a heartbeat thread, stops it on exit.

    Usage::

        with heartbeat(fold_out / ".heartbeat", interval_seconds=60):
            run_long_stage(...)

    Even if ``run_long_stage`` raises, the thread is signaled to stop
    and joined with a short timeout so the process can exit cleanly.
    """
    thread = HeartbeatThread(Path(sentinel), interval_seconds=interval_seconds)
    thread.start()
    try:
        yield thread
    finally:
        thread.stop()
        thread.join(timeout=2.0)


__all__ = [
    "HeartbeatThread",
    "finish_wandb",
    "heartbeat",
    "init_wandb",
]
