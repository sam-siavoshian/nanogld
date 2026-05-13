"""Production model-inference path for the backtest CLI.

Wires the artifacts the Block 5 training pipeline produces (per-fold
``llrd_final.pt`` checkpoints, per-fold sidecar.pt from the §32
refactor, and per-fold calibration dirs) into the walk-forward harness
so ``python -m nanogld.backtest run --checkpoints ... --sidecars ...``
runs the model on each fold's test window and writes a real ship
report.

Per fold the flow is:

  1. Load model checkpoint + sidecar.
  2. Open ``NanoGLDDataset`` on the test slice (using the fold's per-fold
     sidecar) with the V1-SPEC §9.5 1-week embargo already encoded by the
     fold boundary helper.
  3. Forward pass per batch; collect logits + position_weight + bar
     timestamps + next_log_return + is_news_present.
  4. Load the calibration dir for the fold, run ``predict_calibrated``
     to get conformal aps_lower_bound; multiply ``position_weight`` by
     the conformal floor (zero out if lower bound < 0.40 per §10.1).
  5. Hand the resulting per-bar positions + returns + news mask back
     to the walk-forward harness as a fold context.

Inference happens once per fold (no extra strategy call) — the model's
positions are baked into the fold context under the key
``_production_model_positions`` and the CLI's model strategy reads them
from there.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from nanogld.calibration import predict_calibrated
from nanogld.calibration.calibrate import load_calibration
from nanogld.data.dataset import NanoGLDDataset
from nanogld.data.walk_forward_splits import FoldBoundary, compute_fold_boundaries
from nanogld.model import nanoGLDV1

LOG = logging.getLogger("nanogld.backtest.production")

DEFAULT_CONFORMAL_FLOOR = 0.40


def _build_model(cfg: dict[str, Any]) -> torch.nn.Module:
    m = cfg["model"]
    return nanoGLDV1(
        numeric_dim=int(m["numeric_dim"]),
        d_model=int(m["d_model"]),
        num_heads=int(m["num_heads"]),
        num_transformer_layers=int(m["num_transformer_layers"]),
        num_slstm_layers=int(m["num_slstm_layers"]),
        t_bars=int(m["t_bars"]),
        patch_len=int(m["patch_len"]),
        patch_stride=int(m["patch_stride"]),
        n_classes=int(m["n_classes"]),
        regime_dim=int(m["regime_dim"]),
        d_text=int(m["d_text"]),
        n_news_slots=int(m["n_news_slots"]),
        dropout=float(m["dropout"]),
        drop_path_max=float(m["drop_path_max"]),
        decomposition_kernel=int(m["decomposition_kernel"]),
    )


def _slice_dataset_to_fold(
    dataset: NanoGLDDataset, fb: FoldBoundary
) -> NanoGLDDataset:
    """Mutate the dataset's ``_valid_indices`` to the fold's test slice only.

    NanoGLDDataset's normal splits {train, val, test} are based on the
    static unified.pt split labels. The walk-forward fold geometry needs
    the *fold-specific* test window (which slides 3mo per fold), so we
    intersect the dataset's valid_indices with the fold's [test_start,
    test_end) interval. The dataset is otherwise reused unchanged.
    """
    keep = (dataset._valid_indices >= fb.test_start) & (
        dataset._valid_indices < fb.test_end
    )
    dataset._valid_indices = dataset._valid_indices[keep]
    return dataset


def _bar_idx_for_dataset_indices(dataset: NanoGLDDataset) -> np.ndarray:
    return dataset._valid_indices.copy()


def run_model_on_fold(
    *,
    config_path: Path,
    checkpoint_path: Path,
    unified_path: Path,
    sidecar_path: Path,
    fold_boundary: FoldBoundary,
    calibration_dir: Path | None = None,
    device: str = "cpu",
    batch_size: int = 32,
    conformal_floor: float = DEFAULT_CONFORMAL_FLOOR,
) -> dict[str, np.ndarray]:
    """Run the model on one fold's test window; return per-bar arrays.

    Args:
        config_path: V1 training YAML config (used for model construction).
        checkpoint_path: ``fold_<n>/llrd/llrd_final.pt``.
        unified_path: global ``training_v1_unified.pt``.
        sidecar_path: ``training_v1_sidecar_fold_<n>.pt`` from §32.
        fold_boundary: the fold's index boundaries.
        calibration_dir: optional ``calibration_<n>/`` directory; when
            present, ``predict_calibrated`` runs and the conformal floor
            gates ``position_weight``.
        device: torch device.
        batch_size: dataloader batch size.
        conformal_floor: APS lower-bound cutoff for the sizer floor.

    Returns:
        Dict with per-bar arrays for the walk-forward harness.
    """
    with open(config_path) as f:
        cfg_yaml = yaml.safe_load(f)

    model = _build_model(cfg_yaml).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = ckpt.get("model_state") if isinstance(ckpt, dict) else None
    if state is None:
        raise RuntimeError(
            f"checkpoint {checkpoint_path} has no 'model_state' key; "
            f"keys present: {list(ckpt.keys()) if isinstance(ckpt, dict) else 'not a dict'}"
        )
    model.load_state_dict(state)
    model.train(mode=False)

    dl_cfg = cfg_yaml["dataloader"]
    dataset = NanoGLDDataset(
        unified_path=unified_path,
        sidecar_path=sidecar_path,
        split="test",
        lookback_T=int(dl_cfg["lookback_T"]),
        n_news_slots=int(dl_cfg["n_news_slots"]),
        label_mode=dl_cfg["label_mode"],
    )
    dataset = _slice_dataset_to_fold(dataset, fold_boundary)
    if len(dataset) == 0:
        raise RuntimeError(
            f"fold {fold_boundary.fold_idx}: empty test slice "
            f"[{fold_boundary.test_start},{fold_boundary.test_end})"
        )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.startswith("cuda"),
    )

    logits_all: list[torch.Tensor] = []
    pos_raw_all: list[torch.Tensor] = []
    nlr_all: list[torch.Tensor] = []
    news_present_all: list[torch.Tensor] = []
    with torch.no_grad():
        for batch in loader:
            channel_inputs = batch["channel_inputs"].to(device).float().nan_to_num_(0.0)
            news_embeddings = batch["news_embeddings"].to(device).float()
            news_mask = batch["news_mask"].to(device).float()
            is_news_present = batch["is_news_present"].to(device).long()
            regime_vec = batch["regime_vec"].to(device).float()
            out = model(
                channel_inputs=channel_inputs,
                news_embeddings=news_embeddings,
                news_mask=news_mask,
                is_news_present=is_news_present,
                regime_vec=regime_vec,
            )
            logits_all.append(out["logits_3class"].detach().cpu())
            pos_raw_all.append(out["position_weight"].detach().cpu())
            nlr_all.append(batch["next_log_return"].cpu())
            news_present_all.append(is_news_present.detach().cpu())

    logits = torch.cat(logits_all, dim=0)
    pos_raw = torch.cat(pos_raw_all, dim=0).float()
    nlr = torch.cat(nlr_all, dim=0).float()
    news_present = torch.cat(news_present_all, dim=0).bool()

    if calibration_dir is not None and Path(calibration_dir).exists():
        calib = load_calibration(calibration_dir)
        cal = predict_calibrated(logits, calib)
        floor_mask = cal.aps_lower_bound >= conformal_floor
        gated = pos_raw * floor_mask.to(pos_raw.dtype)
    else:
        LOG.warning(
            "no calibration_dir supplied — skipping conformal floor; "
            "model positions used raw (V1-SPEC §10.1 not enforced)"
        )
        gated = pos_raw

    n_bars = int(gated.shape[0])
    return {
        "positions": gated.numpy().astype(np.float64),
        "next_log_returns": nlr.numpy().astype(np.float64),
        "is_news_present": news_present.numpy().astype(bool),
        "close": np.zeros(n_bars, dtype=np.float64),
        "h5_log_return": np.zeros(n_bars, dtype=np.float64),
        "is_last_bar_of_day": np.zeros(n_bars, dtype=bool),
        "is_high_vol": np.zeros(n_bars, dtype=bool),
    }


def build_production_contexts(
    *,
    config_path: Path,
    unified_path: Path,
    checkpoints: Iterable[Path],
    sidecars: Iterable[Path],
    calibration_dirs: Iterable[Path | None] | None = None,
    device: str = "cpu",
    batch_size: int = 32,
    conformal_floor: float = DEFAULT_CONFORMAL_FLOOR,
) -> list[dict[str, Any]]:
    """Produce one fold context per (checkpoint, sidecar) pair.

    The returned contexts are walk-forward-harness-ready: each context
    carries the model's per-bar positions under
    ``_production_model_positions`` plus the next_log_returns +
    is_news_present arrays the harness expects.
    """
    ckpts = list(checkpoints)
    side_paths = list(sidecars)
    if len(ckpts) != len(side_paths):
        raise ValueError(
            f"checkpoints / sidecars length mismatch: {len(ckpts)} vs {len(side_paths)}"
        )
    cal_dirs = list(calibration_dirs) if calibration_dirs is not None else [None] * len(ckpts)
    if len(cal_dirs) != len(ckpts):
        raise ValueError(
            f"calibration_dirs length mismatch: {len(cal_dirs)} vs {len(ckpts)}"
        )

    unified = torch.load(unified_path, map_location="cpu", weights_only=False)
    fold_boundaries = compute_fold_boundaries(
        np.asarray(unified["bar_close_utc_ns"], dtype=np.int64)
    )
    if len(ckpts) > len(fold_boundaries):
        raise ValueError(
            f"more checkpoints ({len(ckpts)}) than fold boundaries ({len(fold_boundaries)}); "
            f"check unified.pt span"
        )

    contexts: list[dict[str, Any]] = []
    for idx, (ckpt, side, cal_dir) in enumerate(zip(ckpts, side_paths, cal_dirs, strict=True)):
        fb = fold_boundaries[idx]
        out = run_model_on_fold(
            config_path=config_path,
            checkpoint_path=Path(ckpt),
            unified_path=Path(unified_path),
            sidecar_path=Path(side),
            fold_boundary=fb,
            calibration_dir=Path(cal_dir) if cal_dir is not None else None,
            device=device,
            batch_size=batch_size,
            conformal_floor=conformal_floor,
        )
        ctx: dict[str, Any] = {
            "fold_idx": fb.fold_idx,
            "_production_model_positions": out["positions"],
            **{k: v for k, v in out.items() if k != "positions"},
        }
        contexts.append(ctx)
    return contexts


__all__ = [
    "build_production_contexts",
    "run_model_on_fold",
]
