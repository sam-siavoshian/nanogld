"""Pytest top-level conftest.

Sets ``NANOGLD_ALLOW_DIRTY_MANIFEST=1`` for the whole test session because
the local working directory is not a git repo (the canonical repo lives
on the remote host at ``~/Desktop/nanogld``). Production training jobs
either run inside the git repo OR set ``NANOGLD_GIT_SHA`` explicitly.

Without this, every call to :func:`nanogld._manifest.build_manifest`
from inside the test process would raise ``RuntimeError``.
"""

from __future__ import annotations

import os


os.environ.setdefault("NANOGLD_ALLOW_DIRTY_MANIFEST", "1")
