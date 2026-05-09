"""Conformal floor — defensive position-shutoff.

If the APS lower-bound on the top-class probability falls below
`threshold`, force position to 0.0. This shuts the trade off when
the conformal prediction set is too wide to commit capital.

NaN or non-finite lower-bound also forces size to 0.0 (defensive).

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
    """Zero out `size` where APS lower-bound is below `threshold` (or NaN)."""
    size_arr = np.asarray(size, dtype=np.float64)
    lb = np.asarray(aps_lower_bound, dtype=np.float64)
    block = ~np.isfinite(lb) | (lb < threshold)
    return np.where(block, 0.0, size_arr)
