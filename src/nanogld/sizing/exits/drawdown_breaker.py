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
    """Tracks cumulative equity drawdown and gates size."""

    peak_equity: float = 1.0
    _state: str = "full"
    _bars_in_state: int = 0

    def step(self, current_equity: float) -> float:
        """Update with current cumulative equity; return size multiplier."""
        if current_equity > self.peak_equity:
            self.peak_equity = current_equity

        drawdown = (current_equity - self.peak_equity) / max(self.peak_equity, 1e-9)

        if drawdown <= HALT_THRESHOLD:
            new_state = "halt"
        elif drawdown <= QUARTER_THRESHOLD:
            new_state = "quarter"
        elif drawdown <= HALVE_THRESHOLD:
            new_state = "halve"
        else:
            new_state = "full"

        if self._state in ("quarter", "halt") and drawdown > RECOVERY_THRESHOLD:
            new_state = "halve" if new_state == "halve" else "full"
        if self._state in ("halt",) and self._bars_in_state >= RECOVERY_BARS:
            new_state = "full"

        if new_state == self._state:
            self._bars_in_state += 1
        else:
            self._state = new_state
            self._bars_in_state = 0

        return {
            "full": 1.0,
            "halve": 0.5,
            "quarter": 0.25,
            "halt": 0.0,
        }[self._state]
