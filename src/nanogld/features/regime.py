"""V1 deterministic regime features (11 of the 12-dim regime vector).

The full 12-dim regime vector is:
  [VIX-tercile (3), RV-tercile (3), FOMC-week (1), year-bucket (4),
   HMM P(high-vol) scalar (1)]

This module emits the deterministic 11 dims (everything except the HMM
scalar). HMM is fit + applied separately in features/hmm_regime.py and
joined into the final 12-dim vector at sidecar build time.

Tercile cuts are computed on the train split only and frozen, so val/
test see the SAME thresholds. This is critical to avoid leakage.

Spec: plan/04-FEATURE-ENGINEERING.md V1 regime section.
Spec: plan/V1-SPEC.md §1.5.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from nanogld.data.utils import get_logger

LOG = get_logger("nanogld.features.regime")

DEFAULT_RV_LOOKBACK = 60
YEAR_BUCKET_NAMES = ("y_2016_2019", "y_2020_2022", "y_2023_2024", "y_2025_plus")
TERCILE_NAMES = ("low", "mid", "high")


@dataclass(frozen=True)
class RegimeThresholds:
    """Frozen thresholds fit on train split, reused at val/test."""

    vix_tercile: tuple[float, float]
    rv_tercile: tuple[float, float]


def fit_regime_thresholds(
    train_df: pd.DataFrame,
    *,
    vix_col: str = "vix_level",
    close_col: str = "gld_close",
    rv_lookback: int = DEFAULT_RV_LOOKBACK,
) -> RegimeThresholds:
    """Fit tercile thresholds on a train-split DataFrame.

    Returns frozen thresholds that should be reused for val/test.
    """
    if vix_col in train_df.columns:
        vix_quantiles = train_df[vix_col].quantile([1 / 3, 2 / 3])
        vix_terc = (float(vix_quantiles.iloc[0]), float(vix_quantiles.iloc[1]))
    else:
        LOG.warning("%s missing in train_df — VIX tercile will be neutral", vix_col)
        vix_terc = (np.nan, np.nan)

    if close_col in train_df.columns:
        log_ret = np.log(train_df[close_col] / train_df[close_col].shift(1))
        rv = log_ret.rolling(rv_lookback, min_periods=rv_lookback // 2).std()
        rv_quantiles = rv.dropna().quantile([1 / 3, 2 / 3])
        rv_terc = (float(rv_quantiles.iloc[0]), float(rv_quantiles.iloc[1]))
    else:
        LOG.warning("%s missing in train_df — RV tercile will be neutral", close_col)
        rv_terc = (np.nan, np.nan)

    LOG.info("regime thresholds: vix=%s rv=%s", vix_terc, rv_terc)
    return RegimeThresholds(vix_tercile=vix_terc, rv_tercile=rv_terc)


def _tercile_one_hot(values: pd.Series, thresholds: tuple[float, float]) -> np.ndarray:
    """Return (N, 3) int8 one-hot of {low, mid, high}.

    NaN inputs map to all-zero rows. Threshold pair is (1/3, 2/3) cuts.
    """
    n = len(values)
    out = np.zeros((n, 3), dtype=np.int8)
    if any(np.isnan(t) for t in thresholds):
        return out
    arr = values.to_numpy(dtype=np.float64)
    valid = ~np.isnan(arr)
    out[valid & (arr <= thresholds[0]), 0] = 1
    out[valid & (arr > thresholds[0]) & (arr <= thresholds[1]), 1] = 1
    out[valid & (arr > thresholds[1]), 2] = 1
    return out


def _year_bucket_one_hot(timestamps_utc: pd.Series) -> np.ndarray:
    """Return (N, 4) int8 one-hot of {2016-2019, 2020-2022, 2023-2024, 2025+}."""
    ts = pd.to_datetime(timestamps_utc, utc=True)
    years = ts.dt.year.to_numpy()
    out = np.zeros((len(years), 4), dtype=np.int8)
    out[(years >= 2016) & (years <= 2019), 0] = 1
    out[(years >= 2020) & (years <= 2022), 1] = 1
    out[(years >= 2023) & (years <= 2024), 2] = 1
    out[years >= 2025, 3] = 1
    return out


def add_regime_columns(
    df: pd.DataFrame,
    *,
    thresholds: RegimeThresholds,
    bar_close_col: str = "bar_close_utc",
    vix_col: str = "vix_level",
    close_col: str = "gld_close",
    fomc_week_col: str = "is_fomc_week",
    rv_lookback: int = DEFAULT_RV_LOOKBACK,
) -> pd.DataFrame:
    """Append the 11 deterministic regime columns.

    Columns added (all int8):
      regime_vix_low, regime_vix_mid, regime_vix_high
      regime_rv_low, regime_rv_mid, regime_rv_high
      regime_fomc_week
      regime_year_2016_2019, regime_year_2020_2022,
      regime_year_2023_2024, regime_year_2025_plus

    HMM P(high-vol) column is added separately by hmm_regime.add_hmm_column.
    """
    out = df.copy()
    if bar_close_col not in out.columns:
        raise KeyError(f"{bar_close_col} required for regime features")
    out[bar_close_col] = pd.to_datetime(out[bar_close_col], utc=True)

    if vix_col in out.columns:
        vix_oh = _tercile_one_hot(out[vix_col], thresholds.vix_tercile)
    else:
        vix_oh = np.zeros((len(out), 3), dtype=np.int8)
    out["regime_vix_low"] = vix_oh[:, 0]
    out["regime_vix_mid"] = vix_oh[:, 1]
    out["regime_vix_high"] = vix_oh[:, 2]

    if close_col in out.columns:
        log_ret = np.log(out[close_col] / out[close_col].shift(1))
        rv = log_ret.rolling(rv_lookback, min_periods=rv_lookback // 2).std()
        rv_oh = _tercile_one_hot(rv, thresholds.rv_tercile)
    else:
        rv_oh = np.zeros((len(out), 3), dtype=np.int8)
    out["regime_rv_low"] = rv_oh[:, 0]
    out["regime_rv_mid"] = rv_oh[:, 1]
    out["regime_rv_high"] = rv_oh[:, 2]

    if fomc_week_col in out.columns:
        out["regime_fomc_week"] = out[fomc_week_col].fillna(0).astype(np.int8)
    else:
        out["regime_fomc_week"] = np.zeros(len(out), dtype=np.int8)

    year_oh = _year_bucket_one_hot(out[bar_close_col])
    for i, name in enumerate(YEAR_BUCKET_NAMES):
        out[f"regime_year_{name.split('_', 1)[1]}"] = year_oh[:, i]

    LOG.info("regime columns added (11 deterministic dims; HMM scalar appended later)")
    return out


def regime_vector_columns() -> list[str]:
    """Return the canonical ordered list of 12-dim regime column names.

    Caller must ensure column 12 (`regime_hmm_p_high_vol`) is appended by
    `hmm_regime.add_hmm_column` before stacking into a tensor.
    """
    return [
        "regime_vix_low",
        "regime_vix_mid",
        "regime_vix_high",
        "regime_rv_low",
        "regime_rv_mid",
        "regime_rv_high",
        "regime_fomc_week",
        "regime_year_2016_2019",
        "regime_year_2020_2022",
        "regime_year_2023_2024",
        "regime_year_2025_plus",
        "regime_hmm_p_high_vol",
    ]
