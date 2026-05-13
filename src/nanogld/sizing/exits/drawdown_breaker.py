"""Drawdown circuit breaker.

Stateful sizing-modifier that watches cumulative equity drawdown:
   -5% drawdown  -> halve size
   -10% drawdown -> quarter size
   -15% drawdown -> halt (size = 0)

Recovery to within -10% of peak OR 65 bars elapsed re-enables full sizing.

Spec: plan/V1-SPEC.md §10.1.
"""

from __future__ import annotations

from dataclasses import dataclass


HALVE_THRESHOLD = -0.05
QUARTER_THRESHOLD = -0.10
HALT_THRESHOLD = -0.15
RECOVERY_THRESHOLD = -0.10
RECOVERY_BARS = 65


@dataclass
class DrawdownCircuitBreaker:
    """Tracks cumulative equity drawdown and gates size.

    Halt timeout fires after RECOVERY_BARS total bars in halt (cumulative,
    persists across halt→quarter→halt cycles). Counter resets only on
    full recovery.
    """

    peak_equity: float = 1.0
    _state: str = "full"
    _bars_in_state: int = 0
    _halt_total_bars: int = 0

    def step(self, current_equity: float) -> float:
        """Update with current cumulative equity; return size multiplier."""
        if current_equity > self.peak_equity:
            self.peak_equity = current_equity

        drawdown = (current_equity - self.peak_equity) / max(self.peak_equity, 1e-9)

        # Inclusive boundary: drawdown of exactly -5% must trigger halve.
        # Using `+ 1e-9` makes the comparison FP-robust on the inclusive side
        # (e.g. drawdown computed from 1.0 -> 0.95 lands at -0.05 modulo FP).
        if drawdown <= HALT_THRESHOLD + 1e-9:
            new_state = "halt"
        elif drawdown <= QUARTER_THRESHOLD + 1e-9:
            new_state = "quarter"
        elif drawdown <= HALVE_THRESHOLD + 1e-9:
            new_state = "halve"
        else:
            new_state = "full"

        if self._state == "halt" and self._halt_total_bars >= RECOVERY_BARS:
            new_state = "full"

        if new_state == self._state:
            self._bars_in_state += 1
        else:
            self._state = new_state
            self._bars_in_state = 0

        if self._state == "halt":
            self._halt_total_bars += 1
        elif self._state == "full":
            self._halt_total_bars = 0

        return {
            "full": 1.0,
            "halve": 0.5,
            "quarter": 0.25,
            "halt": 0.0,
        }[self._state]
