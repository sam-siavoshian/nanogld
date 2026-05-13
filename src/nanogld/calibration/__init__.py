"""nanoGLD V1 calibration package — public API."""

from nanogld.calibration.agaci import AgACI
from nanogld.calibration.calibrate import (
    CalibrationArtifacts,
    CalibrationConfig,
    calibrate,
    load_calibration,
)
from nanogld.calibration.ece import (
    adaptive_ece,
    classwise_ada_ece,
    macro_brier,
    per_bucket_ece,
)
from nanogld.calibration.inference import CalibratedPrediction, predict_calibrated
from nanogld.calibration.laplace_lll import LaplaceLLLA, kelly_multiplier
from nanogld.calibration.raps import fit_raps_quantile, raps_score, raps_set
from nanogld.calibration.temperature_scaling import TemperatureScaler

__all__ = [
    "AgACI",
    "CalibratedPrediction",
    "CalibrationArtifacts",
    "CalibrationConfig",
    "LaplaceLLLA",
    "TemperatureScaler",
    "adaptive_ece",
    "calibrate",
    "classwise_ada_ece",
    "fit_raps_quantile",
    "kelly_multiplier",
    "load_calibration",
    "macro_brier",
    "per_bucket_ece",
    "predict_calibrated",
    "raps_score",
    "raps_set",
]
