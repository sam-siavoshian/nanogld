"""Module entrypoint: ``python -m nanogld.calibration <args>``."""

from __future__ import annotations

import sys

from nanogld.calibration.cli import main

if __name__ == "__main__":
    sys.exit(main())
