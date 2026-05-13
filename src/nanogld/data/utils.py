"""Shared utilities for the data + features pipeline.

Exposes:
    get_logger(name): stdlib logger with consistent format.
    ET: US/Eastern timezone (NYSE RTH).
    raw_dir(): pathlib.Path to data/raw, anchored at the repo root.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")

_LOG_FORMAT = "%(asctime)s | %(name)s | %(levelname)s | %(message)s"


def get_logger(name: str) -> logging.Logger:
    """Return a logger with the project's standard format.

    Idempotent: re-calling with the same name does not duplicate handlers.
    """
    logger = logging.getLogger(name)
    if not logger.handlers and not logging.getLogger().handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(_LOG_FORMAT))
        logger.addHandler(handler)
    if logger.level == logging.NOTSET:
        logger.setLevel(logging.INFO)
    return logger


def repo_root() -> Path:
    """Resolve the repo root by env var override or known marker files."""
    override = os.environ.get("NANOGLD_REPO_ROOT")
    if override:
        return Path(override).resolve()
    here = Path(__file__).resolve()
    for parent in (here, *here.parents):
        if (parent / "pyproject.toml").exists() or (parent / ".git").is_dir():
            return parent
    return Path.cwd()


def raw_dir() -> Path:
    """Return the data/raw directory under the repo root, creating it if missing."""
    override = os.environ.get("NANOGLD_RAW_DIR")
    if override:
        path = Path(override)
    else:
        path = repo_root() / "data" / "raw"
    path.mkdir(parents=True, exist_ok=True)
    return path
