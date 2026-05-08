"""Deflated Sharpe Ratio (DSR).

Bailey & López de Prado 2014: penalize raw Sharpe for multi-trial
selection bias and skew/kurtosis adjustments.

Math:
    SR0 ~ N(0, 1) under the null (raw Sharpe of zero strategy).
    Expected max under N trials with variance V(SR):
        E[max SR0] ~= sqrt(V(SR)) * (
            (1 - gamma) * Phi^{-1}(1 - 1/N) +
            gamma * Phi^{-1}(1 - 1/(N*e))
        )
    where gamma = Euler-Mascheroni = 0.5772.

    DSR = Phi(
        (SR_observed - E[max SR0]) * sqrt(T - 1) /
        sqrt(1 - skew * SR + ((kurt - 1) / 4) * SR^2)
    )

Spec: plan/06-BACKTEST.md V1 DSR.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

EULER_MASCHERONI = 0.5772156649015329


def deflated_sharpe(
    sharpe_observed: float,
    n_trials: int,
    n_obs: int,
    skew: float = 0.0,
    kurt: float = 3.0,
) -> tuple[float, float]:
    """Compute DSR p-value and the deflated Sharpe.

    Args:
        sharpe_observed: realized (annualized) Sharpe.
        n_trials: number of distinct strategies/configs tried.
        n_obs: number of return observations in the test period.
        skew: realized skew of returns.
        kurt: realized kurt (Pearson; 3 = normal).

    Returns:
        (p_value, deflated_sharpe). p_value > 0.95 → result not pure noise.
    """
    if n_trials <= 1:
        n_trials = 2
    if n_obs <= 1:
        return 0.0, 0.0

    sr_var = 1.0
    expected_max = np.sqrt(sr_var) * (
        (1.0 - EULER_MASCHERONI) * norm.ppf(1.0 - 1.0 / n_trials)
        + EULER_MASCHERONI * norm.ppf(1.0 - 1.0 / (n_trials * np.e))
    )

    sr = float(sharpe_observed)
    denom = np.sqrt(max(1e-12, 1.0 - skew * sr + ((kurt - 1.0) / 4.0) * sr * sr))

    z = (sr - expected_max) * np.sqrt(max(0.0, n_obs - 1.0)) / denom
    p_value = float(norm.cdf(z))
    return p_value, float(sr - expected_max)
