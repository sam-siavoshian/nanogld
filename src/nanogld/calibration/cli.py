"""Stage 4 calibration CLI — per-fold orchestrator for V1-SPEC §5 + §10.1.

Without this, ``--calibration-dirs`` on the backtest CLI has no producer
and the conformal floor (``aps_lower_bound >= 0.40`` per §10.1) silently
disables. This module closes that gap.

Per fold:
    1. Load LLRD checkpoint + per-fold sidecar.
    2. Open ``val_b`` + ``val_c`` slices via ``NanoGLDDataset`` (the
       dataset already bisects the static "val" split into val_b/val_c).
    3. Forward pass with ``no_grad`` to collect logits, labels, pooled
       reps, news-presence mask.
    4. Build a Laplace fit loader of ``(pooled, label)`` pairs from
       val_b for ``LaplaceLLLA.fit``.
    5. Call ``calibration.calibrate.calibrate(...)`` with
       ``fit_laplace=True`` and the head module.
    6. ``write_manifest`` next to the output so the backtest
       ``verify_artifacts`` gate passes.

Usage::

    python -m nanogld.calibration run \\
        --config configs/v1_main.yaml \\
        --fold 0 \\
        --checkpoint checkpoints/v1/fold_0/llrd/llrd_final.pt \\
        --output-dir checkpoints/v1/fold_0 \\
        [--device cuda] [--batch-size 32] [--no-laplace]

Output goes under ``<output_dir>/calibration_<fold>/`` with the canonical
file set: ``t_scaler.pt``, ``raps_quantiles.json``, ``agaci_state.json``,
``laplace.pt`` (when ``fit_laplace=True``), ``meta.json``, and a
``MANIFEST.json`` for sha256 verify-on-load.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

import torch
import yaml
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset

from nanogld.calibration.calibrate import (
    CalibrationArtifacts,
    CalibrationConfig,
    calibrate,
)
from nanogld.data.dataset import NanoGLDDataset
from nanogld.data.integrity import write_manifest
from nanogld.model import nanoGLDV1

LOG = logging.getLogger("nanogld.calibration.cli")


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


def _collect(
    model: torch.nn.Module, loader: DataLoader, device: str
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Forward pass; return (logits, labels, pooled, is_news_present_bool)."""
    logits_all: list[Tensor] = []
    labels_all: list[Tensor] = []
    pooled_all: list[Tensor] = []
    news_all: list[Tensor] = []
    with torch.no_grad():
        for batch in loader:
            channel_inputs = batch["channel_inputs"].to(device).float().nan_to_num_(0.0)
            news_embeddings = batch["news_embeddings"].to(device).float()
            news_mask = batch["news_mask"].to(device).float()
            is_news_present = batch["is_news_present"].to(device).long()
            regime_vec = batch["regime_vec"].to(device).float()
            labels = batch["label_3class"].to(device).long()
            out = model(
                channel_inputs=channel_inputs,
                news_embeddings=news_embeddings,
                news_mask=news_mask,
                is_news_present=is_news_present,
                regime_vec=regime_vec,
                return_pooled=True,
            )
            logits_all.append(out["logits_3class"].cpu())
            labels_all.append(labels.cpu())
            pooled_all.append(out["pooled"].cpu())
            news_all.append(is_news_present.cpu())
    return (
        torch.cat(logits_all, dim=0),
        torch.cat(labels_all, dim=0),
        torch.cat(pooled_all, dim=0),
        torch.cat(news_all, dim=0).bool(),
    )


def run(
    *,
    config_path: Path,
    fold: int,
    checkpoint_path: Path,
    output_dir: Path,
    device: str = "cuda",
    batch_size: int = 32,
    fit_laplace: bool = True,
) -> CalibrationArtifacts:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    paths = cfg["paths"]
    dl_cfg = cfg["dataloader"]

    model = _build_model(cfg).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if not isinstance(ckpt, dict) or "model_state" not in ckpt:
        raise RuntimeError(
            f"checkpoint {checkpoint_path} has no 'model_state' key; "
            "expected a dict produced by training/llrd_finetune.py"
        )
    model.load_state_dict(ckpt["model_state"])
    model.train(mode=False)

    sidecar_default = Path(paths["sidecar"])
    sidecar_per_fold = sidecar_default.parent / f"training_v1_sidecar_fold_{fold}.pt"
    sidecar_path = sidecar_per_fold if sidecar_per_fold.exists() else sidecar_default
    LOG.info("fold %d sidecar: %s", fold, sidecar_path)

    common_kwargs: dict[str, Any] = dict(
        unified_path=Path(paths["unified"]),
        sidecar_path=sidecar_path,
        lookback_T=int(dl_cfg["lookback_T"]),
        n_news_slots=int(dl_cfg["n_news_slots"]),
        label_mode=dl_cfg["label_mode"],
    )
    val_b_ds = NanoGLDDataset(split="val_b", **common_kwargs)
    val_c_ds = NanoGLDDataset(split="val_c", **common_kwargs)
    if len(val_b_ds) == 0 or len(val_c_ds) == 0:
        raise RuntimeError(
            f"fold {fold} empty val slice (val_b={len(val_b_ds)}, val_c={len(val_c_ds)})"
        )

    loader_kwargs: dict[str, Any] = dict(
        batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=device.startswith("cuda")
    )
    val_b_loader = DataLoader(val_b_ds, **loader_kwargs)
    val_c_loader = DataLoader(val_c_ds, **loader_kwargs)

    LOG.info("collecting val_b logits + pooled (%d bars) ...", len(val_b_ds))
    val_b_logits, val_b_labels, val_b_pooled, _ = _collect(model, val_b_loader, device)
    LOG.info("collecting val_c logits + pooled (%d bars) ...", len(val_c_ds))
    val_c_logits, val_c_labels, _, val_c_news = _collect(model, val_c_loader, device)

    laplace_loader: DataLoader | None = None
    if fit_laplace:
        ds = TensorDataset(val_b_pooled, val_b_labels)
        laplace_loader = DataLoader(ds, batch_size=max(8, batch_size), shuffle=False)

    cfg_cal = CalibrationConfig(
        fold_idx=fold,
        output_dir=output_dir,
        fit_laplace=fit_laplace,
    )
    LOG.info("running calibrate() -> %s/calibration_%d/", output_dir, fold)
    artifacts = calibrate(
        cfg_cal,
        val_b_logits=val_b_logits,
        val_b_labels=val_b_labels,
        val_c_logits=val_c_logits,
        val_c_labels=val_c_labels,
        val_c_news_present_mask=val_c_news,
        model_head=model.head.cls_head if fit_laplace else None,
        val_b_laplace_loader=laplace_loader,
    )
    final_dir = artifacts.meta_path.parent
    write_manifest(final_dir)
    LOG.info("calibration done. fold=%d output_dir=%s", fold, final_dir)
    return artifacts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="nanogld.calibration",
        description="Stage 4: T-scaling + RAPS + AgACI + Laplace per fold.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("run", help="run calibration for one fold")
    p.add_argument("--config", type=Path, required=True, help="V1 training YAML")
    p.add_argument("--fold", type=int, required=True, help="0..N-1")
    p.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="path to llrd_final.pt for this fold",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        dest="output_dir",
        help="dir under which calibration_<fold>/ is written",
    )
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--batch-size", type=int, default=32, dest="batch_size")
    p.add_argument(
        "--no-laplace",
        action="store_true",
        help="skip Laplace last-layer fit (faster; no epistemic variance signal)",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    )
    if args.cmd != "run":
        parser.print_help()
        return 1
    run(
        config_path=args.config,
        fold=args.fold,
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        device=args.device,
        batch_size=args.batch_size,
        fit_laplace=not args.no_laplace,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
