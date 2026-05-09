"""CLI entrypoint for V1 training.

Wires NanoGLDDataset → 3-stage pipeline (SSL pretrain → linear probe → LLRD
fine-tune) → checkpoints under `output_dir`. Loads a YAML config built per
configs/v1_main.yaml.

Usage:
    python -m nanogld.training run --config configs/v1_main.yaml --fold 0

Spec: plan/V1-SPEC.md §8.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from nanogld.data.dataset import NanoGLDDataset
from nanogld.model import nanoGLDV1
from nanogld.training.linear_probe import LinearProbeConfig, train_linear_probe
from nanogld.training.llrd_finetune import LLRDConfig, llrd_finetune
from nanogld.training.simmtm_pretrain import SimMTMConfig, pretrain_simmtm
from nanogld.training.train import setup_determinism

LOG = logging.getLogger("nanogld.training.cli")


def _build_dataloader(
    cfg: dict, split: str, output_dir: Path, device: str = "cpu", seed: int = 42
) -> DataLoader:
    paths = cfg["paths"]
    dl_cfg = cfg["dataloader"]
    ds = NanoGLDDataset(
        unified_path=Path(paths["unified"]),
        sidecar_path=Path(paths["sidecar"]) if Path(paths["sidecar"]).exists() else None,
        split=split,
        lookback_T=int(dl_cfg["lookback_T"]),
        n_news_slots=int(dl_cfg["n_news_slots"]),
        label_mode=dl_cfg["label_mode"],
    )
    num_workers = int(dl_cfg["num_workers"])

    def _worker_init(worker_id: int) -> None:
        import random as _py_random  # noqa: PLC0415

        import numpy as np  # noqa: PLC0415

        worker_seed = seed + worker_id
        np.random.seed(worker_seed)
        _py_random.seed(worker_seed)
        torch.manual_seed(worker_seed)

    shuffle_gen = torch.Generator()
    shuffle_gen.manual_seed(seed)
    return DataLoader(
        ds,
        batch_size=int(dl_cfg["batch_size"]),
        num_workers=num_workers,
        shuffle=(split == "train"),
        pin_memory=device.startswith("cuda"),
        persistent_workers=(num_workers > 0),
        worker_init_fn=_worker_init if num_workers > 0 else None,
        generator=shuffle_gen if split == "train" else None,
    )


def _build_model(cfg: dict) -> torch.nn.Module:
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


def run(config_path: Path, fold: int, output_dir: Path, device: str = "cpu") -> int:
    """Run the full 3-stage pipeline for one fold."""
    if device not in {"cpu", "cuda", "mps"}:
        raise ValueError(f"device must be one of cpu/cuda/mps, got {device!r}")
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    seed = int(cfg["reproducibility"].get("seed", 42)) + fold
    setup_determinism(seed=seed)

    fold_out = output_dir / f"fold_{fold}"
    fold_out.mkdir(parents=True, exist_ok=True)

    LOG.info("building model + dataloaders for fold %d on device=%s", fold, device)
    model = _build_model(cfg).to(device)
    train_loader = _build_dataloader(
        cfg, split="train", output_dir=fold_out, device=device, seed=seed
    )

    LOG.info("Stage 1: SSL pretrain")
    ssl_cfg = SimMTMConfig(
        epochs=int(cfg["ssl"]["epochs"]),
        mask_ratio=float(cfg["ssl"]["mask_ratio"]),
        k_views=int(cfg["ssl"]["k_views"]),
        lambda_sim=float(cfg["ssl"]["lambda_sim"]),
        lambda_clip=float(cfg["ssl"]["lambda_clip"]),
        grad_clip_max_norm=float(cfg["ssl"]["grad_clip_max_norm"]),
        log_every_n_steps=int(cfg["ssl"]["log_every_n_steps"]),
        output_dir=fold_out / "ssl",
    )
    ssl_optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, betas=(0.9, 0.95))
    ssl_metrics = pretrain_simmtm(model, ssl_optimizer, train_loader, ssl_cfg, device=device)
    ssl_anchor_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    LOG.info("Stage 1 done: %s", ssl_metrics)

    LOG.info("Stage 2: linear probe")
    probe_cfg = LinearProbeConfig(
        epochs=int(cfg["probe"]["epochs"]),
        lr=float(cfg["probe"]["lr"]),
        grad_clip_max_norm=float(cfg["probe"]["grad_clip_max_norm"]),
        focal_gamma=float(cfg["probe"]["focal_gamma"]),
        output_dir=fold_out / "probe",
    )
    probe_metrics = train_linear_probe(model, train_loader, probe_cfg, device=device)
    LOG.info("Stage 2 done: %s", probe_metrics)

    LOG.info("Stage 3: LLRD fine-tune")
    llrd_cfg = LLRDConfig(
        epochs=int(cfg["llrd"]["epochs"]),
        base_lr=float(cfg["llrd"]["base_lr"]),
        decay=float(cfg["llrd"]["decay"]),
        mixout_p=float(cfg["llrd"]["mixout_p"]),
        ema_decay=float(cfg["llrd"]["ema_decay"]),
        grad_clip_max_norm=float(cfg["llrd"]["grad_clip_max_norm"]),
        focal_gamma=float(cfg["llrd"]["focal_gamma"]),
        cost_bps=float(cfg["llrd"]["cost_bps"]),
        focal_weight=float(cfg["llrd"]["focal_weight"]),
        sharpe_weight=float(cfg["llrd"]["sharpe_weight"]),
        aecf_weight=float(cfg["llrd"]["aecf_weight"]),
        freelb_K=int(cfg["llrd"]["freelb_K"]),
        freelb_epsilon=float(cfg["llrd"]["freelb_epsilon"]),
        output_dir=fold_out / "llrd",
    )
    llrd_metrics = llrd_finetune(model, ssl_anchor_state, train_loader, llrd_cfg, device=device)
    LOG.info("Stage 3 done: %s", llrd_metrics)

    paths = cfg["paths"]
    LOG.info(
        "next: feature attribution → "
        "uv run python -m nanogld.analysis run "
        "--checkpoint %s --unified %s --sidecar %s "
        "--fold %d --split val_c --output-dir %s --device %s",
        fold_out / "llrd" / "llrd_final.pt",
        paths["unified"],
        paths["sidecar"],
        fold,
        fold_out / "analysis",
        device,
    )

    return 0


def _autodetect_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="nanogld.training", description="V1 training pipeline")
    sub = parser.add_subparsers(dest="cmd", required=True)
    run_p = sub.add_parser("run", help="run full 3-stage pipeline for one fold")
    run_p.add_argument("--config", type=Path, required=True)
    run_p.add_argument("--fold", type=int, required=True)
    run_p.add_argument(
        "--output-dir",
        dest="output_dir",
        type=Path,
        default=Path("checkpoints/v1"),
    )
    run_p.add_argument("--device", type=str, default="auto")
    run_p.add_argument(
        "--allow-cpu",
        action="store_true",
        help="allow running on CPU; without this flag CPU is rejected when CUDA/MPS unavailable",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s | %(name)s | %(levelname)s | %(message)s"
    )

    if args.cmd == "run":
        device = args.device
        if device == "auto":
            device = _autodetect_device()
        if device == "cpu" and not args.allow_cpu:
            LOG.error(
                "device=cpu rejected by default (would burn paid GPU hours silently). "
                "Pass --allow-cpu to override, or --device cuda|mps."
            )
            return 2
        return run(
            config_path=args.config,
            fold=args.fold,
            output_dir=args.output_dir,
            device=device,
        )
    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
