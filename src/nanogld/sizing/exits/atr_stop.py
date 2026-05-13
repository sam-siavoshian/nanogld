"""ATR-14 hard stop + trailing stop (F2F machinery).

Hard stop: 2.0 * ATR-14 from entry price.
Trailing stop: 1.5 * ATR-14, ratchets only (never relaxes back).

Stateful per-position. Caller calls `update(current_price)` each bar
and acts on the returned action.

Spec: plan/V1-SPEC.md §10.1.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


Action = Literal["hold", "exit"]


@dataclass
class ATRStop:
    """ATR-scaled stop with hard + trailing levels.

    Args:
        entry_price: filled entry price.
        entry_atr: ATR-14 at entry.
        side: +1 for long, -1 for short.
        hard_mult: hard-stop multiplier of ATR (default 2.0).
        trail_mult: trailing-stop multiplier of ATR (default 1.5).
    """

    entry_price: float
    entry_atr: float
    side: int = 1
    hard_mult: float = 2.0
    trail_mult: float = 1.5
    _high_water_long: float = 0.0
    _low_water_short: float = 0.0
    _initialized: bool = False

    def update(self, current_price: float, current_atr: float | None = None) -> Action:
        """Return 'exit' if either stop fires, 'hold' otherwise.

        Args:
            current_price: latest observed price.
            current_atr: live ATR-14 at this bar; used for the trailing stop
                width per V1-SPEC §10.1 (live, not entry). When None, falls
                back to entry_atr (V1-draft behavior, kept for backwards-compat).
        """
        if not self._initialized:
            self._high_water_long = self.entry_price
            self._low_water_short = self.entry_price
            self._initialized = True

        live_atr = float(current_atr) if current_atr is not None else self.entry_atr
        hard_stop_long = self.entry_price - self.hard_mult * self.entry_atr
        hard_stop_short = self.entry_price + self.hard_mult * self.entry_atr

        if self.side > 0:
            if current_price > self._high_water_long:
                self._high_water_long = current_price
            trail_active = self._high_water_long > self.entry_price
            trail = self._high_water_long - self.trail_mult * live_atr
            stop = max(hard_stop_long, trail) if trail_active else hard_stop_long
            return "exit" if current_price < stop else "hold"

        if current_price < self._low_water_short:
            self._low_water_short = current_price
        trail_active = self._low_water_short < self.entry_price
        trail = self._low_water_short + self.trail_mult * live_atr
        stop = min(hard_stop_short, trail) if trail_active else hard_stop_short
        return "exit" if current_price > stop else "hold"
