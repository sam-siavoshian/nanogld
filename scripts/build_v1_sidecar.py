"""Build the V1 sidecar tensor for nanoGLD training.

Produces `data/processed/training_v1_sidecar.pt` containing:
  next_log_return    (N,)    float32
  gld_h5_log_return  (N,)    float32  (Gao 2014 prior; NaN outside RTH-5)
  gld_h5_x_vol_high  (N,)    float32
  gld_spread_bps_t   (N,)    float32  (5-min trailing avg or proxy)
  gld_atr_14         (N,)    float32  (price units; for stop-sizing)
  barrier_up         (N,)    float32  (positive log-return magnitude)
  barrier_down       (N,)    float32  (positive)
  regime_vec         (N, 12) float32
  era_label          (N,)    int8

Inputs:
  - training_v1_unified.pt (provides bar_close_utc_ns + features index)
  - data/raw/alpaca_bars_GLD_30min.parquet (OHLC source)
  - optional data/raw/alpaca_quotes_GLD_5min.parquet (spread; falls back to proxy)

Reuses patterns from src/nanogld/embed/precompute.py:
  - frozen-config dataclass
  - atomic write via os.replace
  - run-hash content-addressing of inputs

Usage:
    python scripts/build_v1_sidecar.py [--output PATH]

Spec: plan/V1-SPEC.md §0/4.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from nanogld.data.integrity import write_manifest
from nanogld.data.utils import data_root, get_logger, raw_dir
from nanogld.data.walk_forward_splits import compute_fold_boundaries
from nanogld.features.atr import add_atr_and_barriers
from nanogld.features.h5 import add_h5_features, fit_h5_vol_threshold
from nanogld.features.hmm_regime import add_hmm_column, fit_hmm, save_hmm
from nanogld.features.regime import (
    add_regime_columns,
    fit_regime_thresholds,
    regime_vector_columns,
)
from nanogld.features.spread import add_spread_feature

LOG = get_logger("nanogld.scripts.build_v1_sidecar")

UNIFIED_NAME = "training_v1_unified.pt"
SIDECAR_NAME = "training_v1_sidecar.pt"
HMM_NAME = "v1_hmm.joblib"
GLD_BARS_NAME = "alpaca_bars_GLD_30min.parquet"


@dataclass(frozen=True)
class BuildConfig:
    """Inputs and output paths for the sidecar build.

    When ``fold_idx`` is None the sidecar is built using the WHOLE
    ``splits == "train"`` mask (legacy global behavior). When it is
    set the script fits HMM + h5_threshold + regime_thresholds on the
    fold's train slice only, closing the V1-SPEC §32 leak.
    """

    unified_path: Path
    bars_path: Path
    output_path: Path
    hmm_path: Path
    rv_lookback: int = 60
    atr_period: int = 14
    fold_idx: int | None = None


def _load_unified(path: Path) -> dict:
    LOG.info("loading unified .pt from %s", path)
    return torch.load(path, weights_only=False, map_location="cpu")


def _load_bars(path: Path) -> pd.DataFrame:
    LOG.info("loading GLD OHLC bars from %s", path)
    df = pd.read_parquet(path)
    if "bar_open_utc" not in df.columns and "timestamp" in df.columns:
        df = df.rename(columns={"timestamp": "bar_open_utc"})
    if "bar_open_utc" in df.columns:
        df["bar_open_utc"] = pd.to_datetime(df["bar_open_utc"], utc=True)
    if "close" in df.columns and "gld_close" not in df.columns:
        df = df.rename(columns={"close": "gld_close", "high": "gld_high", "low": "gld_low"})
    df = df.sort_values("bar_open_utc").reset_index(drop=True)
    df["bar_close_utc"] = df["bar_open_utc"] + pd.Timedelta(minutes=30)
    return df


def _align_bars_to_unified(bars: pd.DataFrame, unified: dict) -> pd.DataFrame:
    """Reindex `bars` so each row's bar_close_utc matches unified[bar_close_utc_ns]."""
    target_ns = np.asarray(unified["bar_close_utc_ns"], dtype=np.int64)
    target_ts = pd.to_datetime(target_ns, utc=True)

    target_df = pd.DataFrame({"bar_close_utc": target_ts})
    merged = pd.merge_asof(
        target_df,
        bars,
        on="bar_close_utc",
        direction="backward",
        tolerance=pd.Timedelta(minutes=30),
    )
    LOG.info(
        "aligned bars: %d/%d rows matched within 30min tolerance",
        int(merged["gld_close"].notna().sum()),
        len(target_df),
    )
    merged["bar_open_utc"] = merged["bar_close_utc"] - pd.Timedelta(minutes=30)
    return merged


def _add_is_fomc_week_stub(df: pd.DataFrame) -> pd.DataFrame:
    """Stub `is_fomc_week` if not present (real one comes from doc 04 calendar).

    Sidecar build is a downstream consumer of the calendar feature; if the
    upstream pipeline didn't produce it, fall back to all-zero.
    """
    if "is_fomc_week" not in df.columns:
        df["is_fomc_week"] = 0
    return df


def _next_log_return(df: pd.DataFrame, close_col: str = "gld_close") -> pd.Series:
    safe = df[close_col].where(df[close_col] > 0)
    next_close = safe.shift(-1)
    return np.log(next_close / safe).replace([np.inf, -np.inf], np.nan)


def _era_label_from_year(years: np.ndarray) -> np.ndarray:
    out = np.zeros(years.shape, dtype=np.int8)
    out[years <= 2019] = 0
    out[(years >= 2020) & (years <= 2022)] = 1
    out[(years >= 2023) & (years <= 2024)] = 2
    out[years >= 2025] = 3
    return out


def build_sidecar(cfg: BuildConfig) -> Path:
    """Build the V1 sidecar tensor and write atomically to cfg.output_path."""
    unified = _load_unified(cfg.unified_path)
    bars = _load_bars(cfg.bars_path)
    aligned = _align_bars_to_unified(bars, unified)
    aligned = _add_is_fomc_week_stub(aligned)

    aligned["next_log_return"] = _next_log_return(aligned).astype("float32")

    aligned = add_atr_and_barriers(aligned, period=cfg.atr_period)
    aligned = add_spread_feature(aligned)

    splits_arr = np.asarray(unified["splits"])
    if cfg.fold_idx is None:
        train_mask = splits_arr == "train"
        train_df = aligned[train_mask].reset_index(drop=True)
        train_window_label = "global"
    else:
        # Per-fold leak fix (V1-SPEC §32): fit thresholds + HMM on this
        # fold's train slice only, then APPLY globally to produce a
        # fold-specific sidecar.
        fold_boundaries = compute_fold_boundaries(
            np.asarray(unified["bar_close_utc_ns"], dtype=np.int64)
        )
        if cfg.fold_idx >= len(fold_boundaries):
            raise ValueError(
                f"requested fold {cfg.fold_idx} but only {len(fold_boundaries)} folds fit "
                f"the dataset span"
            )
        fb = fold_boundaries[cfg.fold_idx]
        train_mask = np.zeros(len(aligned), dtype=bool)
        train_mask[fb.train_start : fb.train_end] = True
        train_df = aligned[train_mask].reset_index(drop=True)
        train_window_label = (
            f"fold {cfg.fold_idx} train=[{fb.train_start},{fb.train_end})"
        )
        LOG.info(
            "per-fold sidecar: %s (%d train bars, fits HMM/h5/regime on this slice only)",
            train_window_label,
            int(train_mask.sum()),
        )

    h5_threshold = fit_h5_vol_threshold(train_df, vol_lookback=cfg.rv_lookback)
    LOG.info("fit h5 vol threshold (train-only): %.6e", h5_threshold)
    aligned = add_h5_features(
        aligned, high_vol_threshold=h5_threshold, vol_lookback=cfg.rv_lookback
    )

    regime_thresholds = fit_regime_thresholds(train_df, rv_lookback=cfg.rv_lookback)
    aligned = add_regime_columns(
        aligned, thresholds=regime_thresholds, rv_lookback=cfg.rv_lookback
    )

    hmm_model = fit_hmm(train_df, rv_lookback=cfg.rv_lookback)
    save_hmm(hmm_model, cfg.hmm_path)
    aligned = add_hmm_column(aligned, model=hmm_model, rv_lookback=cfg.rv_lookback)

    regime_cols = regime_vector_columns()
    regime_vec = aligned[regime_cols].to_numpy(dtype=np.float32)

    years = pd.to_datetime(aligned["bar_close_utc"], utc=True).dt.year.to_numpy()
    era_label = _era_label_from_year(years)

    payload = {
        "next_log_return": aligned["next_log_return"].to_numpy(dtype=np.float32),
        "gld_h5_log_return": aligned["gld_h5_log_return"].to_numpy(dtype=np.float32),
        "gld_h5_x_vol_high": aligned["gld_h5_x_vol_high"].to_numpy(dtype=np.float32),
        "gld_spread_bps_t": aligned["gld_spread_bps_t"].to_numpy(dtype=np.float32),
        f"gld_atr_{cfg.atr_period}": aligned[f"gld_atr_{cfg.atr_period}"].to_numpy(
            dtype=np.float32
        ),
        "barrier_up": aligned["barrier_up"].to_numpy(dtype=np.float32),
        "barrier_down": aligned["barrier_down"].to_numpy(dtype=np.float32),
        "regime_vec": regime_vec,
        "era_label": era_label,
        "meta": {
            "schema_version": "v1_sidecar.2",
            "unified_path": str(cfg.unified_path),
            "bars_path": str(cfg.bars_path),
            "rv_lookback": cfg.rv_lookback,
            "atr_period": cfg.atr_period,
            "h5_vol_threshold": float(h5_threshold)
            if not np.isnan(h5_threshold)
            else None,
            "fold_idx": cfg.fold_idx,
            "train_window": train_window_label,
        },
    }

    tmp_path = cfg.output_path.with_suffix(cfg.output_path.suffix + ".tmp")
    cfg.output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, tmp_path)
    os.replace(tmp_path, cfg.output_path)
    LOG.info("sidecar written: %s (%d bytes)", cfg.output_path, cfg.output_path.stat().st_size)
    return cfg.output_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--unified",
        type=Path,
        default=data_root() / "processed" / UNIFIED_NAME,
        help="path to training_v1_unified.pt",
    )
    parser.add_argument(
        "--bars",
        type=Path,
        default=raw_dir() / GLD_BARS_NAME,
        help="path to alpaca_bars_GLD_30min.parquet",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=data_root() / "processed" / SIDECAR_NAME,
        help="output sidecar .pt path (used when --per-fold not set)",
    )
    parser.add_argument(
        "--hmm",
        type=Path,
        default=data_root() / "processed" / HMM_NAME,
        help="output HMM joblib path (used when --per-fold not set)",
    )
    parser.add_argument("--rv_lookback", type=int, default=60)
    parser.add_argument("--atr_period", type=int, default=14)
    parser.add_argument(
        "--per-fold",
        action="store_true",
        help=(
            "build one sidecar per walk-forward fold (V1-SPEC §32 leak fix). "
            "HMM + h5_threshold + regime thresholds fit on each fold's train "
            "slice only. Outputs training_v1_sidecar_fold_N.pt + v1_hmm_fold_N.joblib."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=data_root() / "processed",
        help="output directory for per-fold sidecars (used with --per-fold)",
    )
    args = parser.parse_args()

    if not args.per_fold:
        cfg = BuildConfig(
            unified_path=args.unified,
            bars_path=args.bars,
            output_path=args.output,
            hmm_path=args.hmm,
            rv_lookback=args.rv_lookback,
            atr_period=args.atr_period,
        )
        build_sidecar(cfg)
        # Write MANIFEST.json so dataset.__init__ can verify on load (§45).
        manifest_path = write_manifest(cfg.output_path.parent)
        print(
            json.dumps(
                {
                    "status": "ok",
                    "output": str(cfg.output_path),
                    "manifest": str(manifest_path),
                },
                indent=2,
            )
        )
        return 0

    # Per-fold mode: build one sidecar per fold.
    unified = _load_unified(args.unified)
    fold_boundaries = compute_fold_boundaries(
        np.asarray(unified["bar_close_utc_ns"], dtype=np.int64)
    )
    if not fold_boundaries:
        LOG.error("compute_fold_boundaries returned 0 folds; dataset span too short")
        return 2

    outputs: list[str] = []
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for fb in fold_boundaries:
        out_path = args.output_dir / f"training_v1_sidecar_fold_{fb.fold_idx}.pt"
        hmm_path = args.output_dir / f"v1_hmm_fold_{fb.fold_idx}.joblib"
        cfg = BuildConfig(
            unified_path=args.unified,
            bars_path=args.bars,
            output_path=out_path,
            hmm_path=hmm_path,
            rv_lookback=args.rv_lookback,
            atr_period=args.atr_period,
            fold_idx=fb.fold_idx,
        )
        build_sidecar(cfg)
        outputs.append(str(out_path))
    manifest_path = write_manifest(args.output_dir)
    print(
        json.dumps(
            {
                "status": "ok",
                "n_folds": len(outputs),
                "outputs": outputs,
                "manifest": str(manifest_path),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
