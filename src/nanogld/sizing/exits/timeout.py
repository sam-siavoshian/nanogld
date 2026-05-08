"""30-day timeout exit (F2F machinery).

390 bars = 30 days * 13 RTH bars/day. Per Wright F2F.
"""

from __future__ import annotations

from dataclasses import dataclass

DEFAULT_MAX_BARS = 390


@dataclass
class TimeoutExit:
    """Counts bars from entry, fires at `max_bars`."""

    max_bars: int = DEFAULT_MAX_BARS
    _bars_held: int = 0

    def step(self) -> bool:
        """Increment bar counter and return True if timeout fired."""
        self._bars_held += 1
        return self._bars_held >= self.max_bars
