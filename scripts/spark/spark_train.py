"""Spark-side trainer entrypoint with OOM guard mounted first.

This wrapper:
  1. Imports oom_guard BEFORE torch so PYTORCH_CUDA_ALLOC_CONF is set.
  2. Reads protected PIDs from $SAAM_NANOGLD_PROTECT_PIDS (a comma-
     separated list of Omar's GPU users captured at launch).
  3. Installs the watchdog with a checkpoint hook that the trainer
     registers later (see `oom_guard.register_checkpoint`).
  4. Delegates to `nanogld.training.__main__` which already handles
     per-stage sentinels for resume.

Run as: `python spark_train.py --config <path> --fold N --output-dir <path> --device cuda`
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def _bootstrap_guard() -> None:
    """Mount oom_guard with Spark defaults BEFORE importing torch."""
    here = Path(__file__).resolve().parent
    if str(here) not in sys.path:
        sys.path.insert(0, str(here))
    from oom_guard import install_guard  # noqa: PLC0415

    protect_env = os.environ.get("SAAM_NANOGLD_PROTECT_PIDS", "")
    protect_pids: list[int] = []
    for tok in protect_env.split(","):
        tok = tok.strip()
        if not tok:
            continue
        try:
            pid = int(tok)
        except ValueError:
            continue
        if pid != os.getpid():
            protect_pids.append(pid)

    log_dir = Path("logs")
    install_guard(
        # Spark contention is gone (no Omar/Saam jobs). Bump to 0.80 =
        # 97 GB GPU cap. Allows B=16 LLRD at peak ~80 GB.
        gpu_fraction=float(os.environ.get("SAAM_NANOGLD_GPU_FRACTION", "0.80")),
        sys_ram_max_gb=float(os.environ.get("SAAM_NANOGLD_SYS_RAM_MAX_GB", "8.0")),
        # Soft warning floor; does not trigger kill in current guard.
        min_free_ram_gb=float(os.environ.get("SAAM_NANOGLD_MIN_FREE_GB", "1.0")),
        watchdog_period_s=int(os.environ.get("SAAM_NANOGLD_WATCHDOG_PERIOD", "30")),
        protect_pids=protect_pids,
        omar_rss_spike_gb=float(os.environ.get("SAAM_NANOGLD_RSS_SPIKE_GB", "10.0")),
        log_dir=log_dir,
        checkpoint_fn=None,
        oom_score=int(os.environ.get("SAAM_NANOGLD_OOM_SCORE", "1000")),
    )


def main() -> int:
    _bootstrap_guard()

    # PYTHONPATH should already include src/. Defensive add anyway so the
    # script works when invoked directly without the launcher.
    src_dir = (Path(__file__).resolve().parent.parent.parent / "src").resolve()
    if src_dir.is_dir() and str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

    # Hand off to the existing trainer CLI. It owns the per-stage
    # sentinel resume logic, so SIGTERM/SIGKILL/OOM/SSH-disconnect
    # safely resumes from the last `stage.done` on relaunch.
    from nanogld.training.__main__ import main as trainer_main  # noqa: PLC0415

    # Forward our argv. Insert "run" subcommand if not present (the
    # launcher passes --config/--fold flags directly).
    argv = sys.argv[1:]
    if "run" not in argv and "--help" not in argv and "-h" not in argv:
        argv = ["run", *argv]
    return trainer_main(argv)


if __name__ == "__main__":
    sys.exit(main())
