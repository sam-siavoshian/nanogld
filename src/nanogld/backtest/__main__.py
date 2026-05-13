"""Module entrypoint: ``python -m nanogld.backtest <args>``."""

from __future__ import annotations

import sys

from nanogld.backtest.cli import main

if __name__ == "__main__":
    sys.exit(main())
