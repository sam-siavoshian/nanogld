"""Conformal floor — defensive position-shutoff.

If the APS lower-bound on the top-class probability falls below
`threshold`, force position to 0.0. This shuts the trade off when
the conformal prediction set is too wide to commit capital.

V1 default: threshold = 0.40.

Spec: plan/V1-SPEC.md §10.1.
"""

from __future__ import annotations

import numpy as np

DEFAULT_FLOOR_THRESHOLD = 0.40


def apply_conformal_floor(
    size: float | np.ndarray,
    aps_lower_bound: float | np.ndarray,
    threshold: float = DEFAULT_FLOOR_THRESHOLD,
) -> np.ndarray:
    """Zero out `size` where APS lower-bound is below `threshold`."""
    size_arr = np.asarray(size, dtype=np.float64)
    lb = np.asarray(aps_lower_bound, dtype=np.float64)
    return np.where(lb < threshold, 0.0, size_arr)
