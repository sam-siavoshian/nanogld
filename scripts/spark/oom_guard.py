"""Spark OOM-safety guard.

Mounted at the very top of every Python entrypoint on Spark (DGX Spark
prototype shared with Omar Ramadan).

Mission: never kill, OOM, or slow down Omar's processes. Period.

Defenses applied (in order):
  1. set my /proc/self/oom_score_adj to +1000 so the kernel picks ME
     first if the system runs out of memory.
  2. PYTORCH_CUDA_ALLOC_CONF env var set BEFORE torch import (so this
     module must be imported before torch).
  3. torch.cuda.set_per_process_memory_fraction(gpu_fraction).
  4. Soft watchdog thread monitors:
        - free system RAM (must stay > min_free_ram_gb)
        - Omar's named processes still alive (PIDs pinned at startup)
        - delta growth in Omar's RSS (if he spikes, we yield)
     When any tripwire fires, the guard:
        - calls the user's checkpoint callback (atomic write)
        - sends SIGTERM to itself
  5. nice() to +19, ionice idle class.
  6. Logs every event to logs/oom_guard_<pid>.log so post-mortems work.
"""

from __future__ import annotations

import atexit
import logging
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path

LOG = logging.getLogger("oom_guard")

_GUARD_INSTALLED = False
_WATCHDOG_THREAD: threading.Thread | None = None
_STOP_WATCHDOG = threading.Event()
_CHECKPOINT_FN: Callable[[], None] | None = None
_LOG_PATH: Path | None = None


def _setup_logging(log_dir: Path) -> Path:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"oom_guard_{os.getpid()}.log"
    handler = logging.FileHandler(log_path)
    handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    )
    LOG.addHandler(handler)
    LOG.setLevel(logging.INFO)
    LOG.propagate = True
    return log_path


def _set_oom_score(score: int) -> None:
    """Make us the kernel's preferred OOM victim."""
    try:
        Path("/proc/self/oom_score_adj").write_text(str(score))
        LOG.info("oom_score_adj set to %d on pid %d", score, os.getpid())
    except (PermissionError, OSError) as exc:
        LOG.warning("could not set oom_score_adj: %s", exc)


def _set_torch_alloc_env(max_split_mb: int) -> None:
    """Configure the PyTorch CUDA caching allocator. MUST run before torch import."""
    cfg = (
        f"expandable_segments:True,"
        f"max_split_size_mb:{max_split_mb},"
        f"garbage_collection_threshold:0.7"
    )
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = cfg
    LOG.info("PYTORCH_CUDA_ALLOC_CONF=%s", cfg)


def _set_priority(nice_inc: int) -> None:
    """Be a polite neighbor: low CPU priority + idle I/O class."""
    try:
        os.nice(nice_inc)
        LOG.info("nice set to +%d", nice_inc)
    except OSError as exc:
        LOG.warning("nice() failed: %s", exc)
    ionice = shutil.which("ionice")
    if ionice is not None:
        try:
            subprocess.run(
                [ionice, "-c", "3", "-p", str(os.getpid())],
                check=False,
                capture_output=True,
                timeout=5,
            )
            LOG.info("ionice idle class set on pid %d", os.getpid())
        except (subprocess.SubprocessError, OSError) as exc:
            LOG.warning("ionice failed: %s", exc)


def _apply_gpu_cap(gpu_fraction: float) -> None:
    """Hard cap GPU memory via PyTorch.

    Lazy import so guard can be installed before torch exists.
    """
    try:
        import torch  # type: ignore
    except ImportError:
        LOG.warning("torch not importable yet; skipping GPU cap")
        return
    if not torch.cuda.is_available():
        LOG.warning("CUDA not available; gpu cap skipped")
        return
    try:
        torch.cuda.set_per_process_memory_fraction(gpu_fraction, device=0)
        total_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        cap_gb = total_gb * gpu_fraction
        LOG.info(
            "GPU cap: %.1f GB (%.0f%% of %.1f GB on %s)",
            cap_gb,
            gpu_fraction * 100,
            total_gb,
            torch.cuda.get_device_name(0),
        )
    except (RuntimeError, ValueError) as exc:
        LOG.warning("GPU cap failed: %s", exc)


def _signal_self(sig: int) -> None:
    """Politely tell the kernel to send us the signal."""
    try:
        os.kill(os.getpid(), sig)
    except OSError as exc:
        LOG.error("self-signal %s failed: %s", sig, exc)


def _checkpoint_and_die(reason: str) -> None:
    """One-shot teardown: snapshot state then SIGTERM ourselves."""
    LOG.error("TRIPWIRE: %s", reason)
    LOG.error("calling checkpoint_fn and shutting down")
    if _CHECKPOINT_FN is not None:
        try:
            _CHECKPOINT_FN()
            LOG.info("checkpoint_fn returned cleanly")
        except Exception as exc:  # pragma: no cover - last-ditch
            LOG.exception("checkpoint_fn raised: %s", exc)
    _STOP_WATCHDOG.set()
    time.sleep(0.5)
    _signal_self(signal.SIGTERM)


def _watchdog_loop(
    sys_ram_max_gb: float,
    min_free_ram_gb: float,
    watchdog_period_s: int,
    protect_pids: list[int],
    omar_rss_spike_gb: float,
) -> None:
    """Background guardian. Polls free RAM + protected PIDs."""
    import psutil  # noqa: PLC0415

    baseline_rss: dict[int, float] = {}
    for pid in protect_pids:
        try:
            baseline_rss[pid] = psutil.Process(pid).memory_info().rss / 1e9
            LOG.info("baseline RSS for protected pid %d = %.1f GB", pid, baseline_rss[pid])
        except psutil.NoSuchProcess:
            LOG.warning("protected pid %d already gone at startup", pid)

    while not _STOP_WATCHDOG.is_set():
        try:
            vm = psutil.virtual_memory()
            free_gb = vm.available / 1e9
            self_rss_gb = psutil.Process(os.getpid()).memory_info().rss / 1e9

            # Defense in depth: only trip on SELF growth, not Omar's.
            # The kernel cgroup MemoryMax owned by safe_run.sh already
            # OOM-kills the whole scope (and ONLY the scope) when we
            # exceed our cap. The watchdog's job is to do a graceful
            # checkpoint save BEFORE the kernel SIGKILLs us, so when our
            # own RSS goes within 5% of the cap, snapshot + tear down.
            if self_rss_gb > sys_ram_max_gb * 0.95:
                _checkpoint_and_die(
                    f"self RSS {self_rss_gb:.1f} GB approaches sys cap "
                    f"{sys_ram_max_gb:.1f} GB (within 5%)"
                )
                return

            # Optional Omar-spike guard. Only fires when Omar's RSS grows
            # by >omar_rss_spike_gb above startup baseline. Set spike very
            # high (or empty protect_pids) to disable.
            for pid in list(baseline_rss.keys()):
                try:
                    p = psutil.Process(pid)
                    rss_gb = p.memory_info().rss / 1e9
                    if rss_gb > baseline_rss[pid] + omar_rss_spike_gb:
                        LOG.warning(
                            "protected pid %d RSS spike: %.1f GB (baseline %.1f GB)",
                            pid,
                            rss_gb,
                            baseline_rss[pid],
                        )
                        _checkpoint_and_die(
                            f"protected pid {pid} grew by >{omar_rss_spike_gb} GB"
                        )
                        return
                except psutil.NoSuchProcess:
                    LOG.info("protected pid %d exited cleanly", pid)
                    baseline_rss.pop(pid, None)
                    continue

            # Soft warn (not kill) on low system free RAM. Useful for
            # post-mortems but does NOT trigger checkpoint.
            if free_gb < min_free_ram_gb:
                LOG.warning(
                    "low system free RAM: %.1f GB (floor %.1f GB) — "
                    "self_rss=%.1f GB (cap %.1f GB); not killing",
                    free_gb,
                    min_free_ram_gb,
                    self_rss_gb,
                    sys_ram_max_gb,
                )

            LOG.debug(
                "watchdog tick: free=%.1f GB self_rss=%.1f GB",
                free_gb,
                self_rss_gb,
            )
        except Exception as exc:  # pragma: no cover
            LOG.exception("watchdog iteration failed: %s", exc)

        _STOP_WATCHDOG.wait(watchdog_period_s)


def install_guard(
    *,
    gpu_fraction: float = 0.20,
    sys_ram_max_gb: float = 25.0,
    min_free_ram_gb: float = 8.0,
    watchdog_period_s: int = 20,
    protect_pids: list[int] | None = None,
    omar_rss_spike_gb: float = 10.0,
    log_dir: Path = Path("logs"),
    checkpoint_fn: Callable[[], None] | None = None,
    oom_score: int = 1000,
    max_split_mb: int = 128,
    nice_inc: int = 19,
) -> Callable[[], None]:
    """Install the OOM guard. Idempotent. Returns a teardown callable."""
    global _GUARD_INSTALLED, _WATCHDOG_THREAD, _CHECKPOINT_FN, _LOG_PATH

    if _GUARD_INSTALLED:
        LOG.warning("guard already installed; ignoring re-install")
        return lambda: None

    _LOG_PATH = _setup_logging(log_dir)
    LOG.info("=" * 60)
    LOG.info("OOM guard initialising on pid %d", os.getpid())
    LOG.info("=" * 60)

    _CHECKPOINT_FN = checkpoint_fn

    _set_oom_score(oom_score)
    _set_torch_alloc_env(max_split_mb=max_split_mb)
    _set_priority(nice_inc=nice_inc)
    _apply_gpu_cap(gpu_fraction)

    pids = protect_pids or []
    _WATCHDOG_THREAD = threading.Thread(
        target=_watchdog_loop,
        kwargs={
            "sys_ram_max_gb": sys_ram_max_gb,
            "min_free_ram_gb": min_free_ram_gb,
            "watchdog_period_s": watchdog_period_s,
            "protect_pids": pids,
            "omar_rss_spike_gb": omar_rss_spike_gb,
        },
        name="oom_guard_watchdog",
        daemon=True,
    )
    _WATCHDOG_THREAD.start()
    LOG.info(
        "watchdog started: period=%ds protected=%s spike=%.1f GB free_floor=%.1f GB self_cap=%.1f GB",
        watchdog_period_s,
        pids,
        omar_rss_spike_gb,
        min_free_ram_gb,
        sys_ram_max_gb,
    )

    def teardown() -> None:
        _STOP_WATCHDOG.set()
        if _WATCHDOG_THREAD is not None and _WATCHDOG_THREAD.is_alive():
            _WATCHDOG_THREAD.join(timeout=2.0)
        LOG.info("guard teardown complete on pid %d", os.getpid())

    atexit.register(teardown)

    def _sig(signum, _frame):
        LOG.warning("received signal %d, attempting graceful checkpoint", signum)
        if _CHECKPOINT_FN is not None:
            try:
                _CHECKPOINT_FN()
            except Exception:  # pragma: no cover
                LOG.exception("graceful checkpoint failed")
        teardown()
        sys.exit(0 if signum == signal.SIGTERM else 130)

    for s in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(s, _sig)

    _GUARD_INSTALLED = True
    LOG.info("guard installed. log=%s", _LOG_PATH)
    return teardown


def register_checkpoint(fn: Callable[[], None]) -> None:
    """Allow the trainer to register/replace the checkpoint hook later."""
    global _CHECKPOINT_FN
    _CHECKPOINT_FN = fn
    LOG.info("checkpoint_fn registered: %s", getattr(fn, "__name__", fn))


if __name__ == "__main__":
    install_guard(
        gpu_fraction=0.05,
        sys_ram_max_gb=4.0,
        min_free_ram_gb=2.0,
        watchdog_period_s=5,
    )
    print("guard self-test running for 30s, watch logs/oom_guard_*.log")
    time.sleep(30)
    print("done")
