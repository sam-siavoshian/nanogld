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
from contextlib import contextmanager
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from nanogld._manifest import build_manifest
from nanogld.data.dataset import NanoGLDDataset
from nanogld.data.integrity import verify_artifacts
from nanogld.model import nanoGLDV1
from nanogld.training.linear_probe import LinearProbeConfig, train_linear_probe
from nanogld.training.llrd_finetune import LLRDConfig, llrd_finetune
from nanogld.training.observability import (
    finish_wandb,
    gpu_temp_watchdog,
    heartbeat,
    init_wandb,
)
from nanogld.training.optim import build_optimizer
from nanogld.training.simmtm_pretrain import SimMTMConfig, pretrain_simmtm
from nanogld.training.train import setup_determinism

LOG = logging.getLogger("nanogld.training.cli")

_LOG_FORMATTER = logging.Formatter(
    "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
)

OOM_EXIT_CODE = 137  # matches Linux OOM-killer signal convention


def _write_oom_sentinel(stage_dir: Path, message: str) -> None:
    """Mark a stage as OOM-aborted so the next run distinguishes it from
    a normal crash. The orchestrator can then refuse to retry until the
    operator clears the sentinel (after lowering batch_size or freeing memory)."""
    stage_dir.mkdir(parents=True, exist_ok=True)
    (stage_dir / "stage.oom").write_text(message + "\n", encoding="utf-8")


def _refuse_if_oom(stage_dir: Path, stage_name: str) -> bool:
    """Return True if a stage.oom sentinel is present (caller must abort).

    On a fresh retry after an OOM, the orchestrator should NOT silently
    re-run the failing stage — the operator first has to clear the
    sentinel after lowering batch_size or freeing memory.
    """
    oom_path = stage_dir / "stage.oom"
    if not oom_path.exists():
        return False
    LOG.error(
        "%s has stage.oom sentinel at %s — lower batch_size in "
        "configs/v1_main.yaml and delete the sentinel before retrying. "
        "Sentinel content:\n%s",
        stage_name,
        oom_path,
        oom_path.read_text(),
    )
    return True


@contextmanager
def _stage_log_file(fold_out: Path, stage: str):
    """Add a FileHandler that writes `fold_out/<stage>.log` for the lifetime of the block.

    Stdout still receives the same log lines (via the root handler set up
    by ``logging.basicConfig`` in ``main``). The file gives post-mortem
    evidence when tmux scrollback is gone or the process crashed.
    """
    fold_out.mkdir(parents=True, exist_ok=True)
    log_path = fold_out / f"{stage}.log"
    handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    handler.setLevel(logging.INFO)
    handler.setFormatter(_LOG_FORMATTER)
    root = logging.getLogger()
    root.addHandler(handler)
    LOG.info("stage log file: %s", log_path)
    try:
        yield log_path
    finally:
        handler.flush()
        handler.close()
        root.removeHandler(handler)


def _build_dataloader(cfg: dict, split: str, output_dir: Path, device: str = "cpu", seed: int = 42) -> DataLoader:
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

    paths = cfg["paths"]

    # Pre-flight (V1-SPEC §45): verify SHA-256 of unified.pt + per-fold
    # sidecar against MANIFEST.json BEFORE building model / dataloader.
    # A truncated rsync or corrupted artifact fails fast here, not 5
    # minutes into training. MANIFEST.json is written by
    # build_v1_sidecar.py and rsync'd by spark_sync.sh.
    unified_path = Path(paths["unified"])
    sidecar_path = Path(paths["sidecar"])
    artifact_dir = unified_path.parent
    require = []
    if unified_path.exists():
        require.append(unified_path.name)
    if sidecar_path.exists():
        require.append(sidecar_path.name)
    if (artifact_dir / "MANIFEST.json").exists() and require:
        verify_artifacts(artifact_dir, require=require)
        LOG.info("pre-flight sha256 verify OK: %s", require)
    else:
        LOG.warning(
            "no MANIFEST.json next to %s — skipping sha256 verify "
            "(run scripts/build_v1_sidecar.py --per-fold to produce one)",
            artifact_dir,
        )

    LOG.info("building model + dataloaders for fold %d on device=%s", fold, device)
    model = _build_model(cfg).to(device)
    train_loader = _build_dataloader(
        cfg, split="train", output_dir=fold_out, device=device, seed=seed
    )
    run_manifest = build_manifest(
        dataset_path=Path(paths["unified"]) if Path(paths["unified"]).exists() else None,
        sidecar_path=Path(paths["sidecar"]) if Path(paths["sidecar"]).exists() else None,
        hparams={"config_path": str(config_path), "fold": fold, "seed": seed},
        extras={"run_kind": "v1_train"},
    )
    LOG.info("run manifest: git_sha=%s host=%s", run_manifest["git_sha"], run_manifest["hostname"])

    # Observability (V1-SPEC §47). Both are optional: W&B is a no-op
    # without WANDB_API_KEY + wandb package installed; the heartbeat
    # thread starts unconditionally so an external watchdog can detect
    # stalled processes via fold_out/.heartbeat mtime.
    obs_cfg = cfg.get("observability", {}) or {}
    wandb_run = init_wandb(
        project=obs_cfg.get("wandb_project"),
        run_name=f"v1_fold_{fold}",
        config={"fold": fold, "seed": seed, "git_sha": run_manifest["git_sha"]},
        tags=obs_cfg.get("wandb_tags", []),
    )
    heartbeat_interval = float(obs_cfg.get("heartbeat_seconds", 60.0))

    # Stage 1: SSL pretrain (skip if `stage.done` sentinel exists — see #41).
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
    if _refuse_if_oom(ssl_cfg.output_dir, "Stage 1 SSL"):
        return OOM_EXIT_CODE
    ssl_done = (ssl_cfg.output_dir / "stage.done").exists()
    if ssl_done:
        LOG.info("Stage 1 SSL skipped — found %s", ssl_cfg.output_dir / "stage.done")
        ssl_metrics: dict[str, float] = {"resumed": 1.0}
    else:
        LOG.info("Stage 1: SSL pretrain")
        ssl_optimizer = build_optimizer(
            model.parameters(),
            lr=float(cfg["ssl"].get("lr", 1e-4)),
            betas=(0.9, 0.95),
            weight_decay=float(cfg["ssl"].get("weight_decay", 0.1)),
            warmup_steps=int(cfg["ssl"].get("warmup_steps", 300)),
            # SSL's SimMTM closure computes loss OUTSIDE the closure (K=3
            # view forward + recon head + CLIP head are stateful + lazy);
            # FSAM's two-pass would backward through a freed graph. Drop
            # FSAM here; Stage 3 LLRD has a proper closure-rebuild path
            # and keeps the full Cautious(FSAM(SF)) stack.
            use_fsam=False,
        )
        try:
            with _stage_log_file(fold_out, "ssl"), heartbeat(
                fold_out / ".heartbeat", interval_seconds=heartbeat_interval
            ), gpu_temp_watchdog():
                ssl_metrics = pretrain_simmtm(
                    model,
                    ssl_optimizer,
                    train_loader,
                    ssl_cfg,
                    device=device,
                    manifest=run_manifest,
                    wandb_run=wandb_run,
                )
        except torch.cuda.OutOfMemoryError as exc:
            LOG.exception("OOM during SSL pretrain; aborting fold %d", fold)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            _write_oom_sentinel(ssl_cfg.output_dir, f"fold={fold} stage=ssl exc={exc}")
            return OOM_EXIT_CODE

    # SSL anchor MUST be the averaged-form weights, not the in-memory z-form
    # the optimizer maintains during training. pretrain_simmtm saves the
    # averaged form to disk; load it always (also restores model state when
    # resuming from sentinel). See plan/STATUS.md §52.
    ssl_anchor_ckpt = torch.load(
        ssl_cfg.output_dir / "ssl_anchor.pt", map_location="cpu", weights_only=False
    )
    ssl_anchor_state = ssl_anchor_ckpt["model_state"]
    model.load_state_dict(ssl_anchor_state)
    model.to(device)
    LOG.info("Stage 1 done: %s", ssl_metrics)

    # Stage 2: linear probe (skip if sentinel exists).
    probe_cfg = LinearProbeConfig(
        epochs=int(cfg["probe"]["epochs"]),
        lr=float(cfg["probe"]["lr"]),
        grad_clip_max_norm=float(cfg["probe"]["grad_clip_max_norm"]),
        focal_gamma=float(cfg["probe"]["focal_gamma"]),
        output_dir=fold_out / "probe",
    )
    if _refuse_if_oom(probe_cfg.output_dir, "Stage 2 probe"):
        return OOM_EXIT_CODE
    if (probe_cfg.output_dir / "stage.done").exists():
        LOG.info("Stage 2 probe skipped — found %s", probe_cfg.output_dir / "stage.done")
        probe_ckpt = torch.load(
            probe_cfg.output_dir / "probe.pt", map_location="cpu", weights_only=False
        )
        model.load_state_dict(probe_ckpt["model_state"])
        model.to(device)
        probe_metrics: dict[str, float] = {"resumed": 1.0}
    else:
        LOG.info("Stage 2: linear probe")
        try:
            with _stage_log_file(fold_out, "probe"), heartbeat(
                fold_out / ".heartbeat", interval_seconds=heartbeat_interval
            ), gpu_temp_watchdog():
                probe_metrics = train_linear_probe(
                    model,
                    train_loader,
                    probe_cfg,
                    device=device,
                    manifest=run_manifest,
                    wandb_run=wandb_run,
                )
        except torch.cuda.OutOfMemoryError as exc:
            LOG.exception("OOM during probe; aborting fold %d", fold)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            _write_oom_sentinel(probe_cfg.output_dir, f"fold={fold} stage=probe exc={exc}")
            return OOM_EXIT_CODE
    LOG.info("Stage 2 done: %s", probe_metrics)

    # Stage 3: LLRD fine-tune (skip if sentinel exists).
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
    if _refuse_if_oom(llrd_cfg.output_dir, "Stage 3 LLRD"):
        return OOM_EXIT_CODE
    if (llrd_cfg.output_dir / "stage.done").exists():
        LOG.info("Stage 3 LLRD skipped — found %s", llrd_cfg.output_dir / "stage.done")
        llrd_metrics: dict[str, float] = {"resumed": 1.0}
    else:
        LOG.info("Stage 3: LLRD fine-tune")
        try:
            with _stage_log_file(fold_out, "llrd"), heartbeat(
                fold_out / ".heartbeat", interval_seconds=heartbeat_interval
            ), gpu_temp_watchdog():
                llrd_metrics = llrd_finetune(
                    model,
                    ssl_anchor_state,
                    train_loader,
                    llrd_cfg,
                    device=device,
                    manifest=run_manifest,
                    wandb_run=wandb_run,
                )
        except torch.cuda.OutOfMemoryError as exc:
            LOG.exception("OOM during LLRD; aborting fold %d", fold)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            _write_oom_sentinel(llrd_cfg.output_dir, f"fold={fold} stage=llrd exc={exc}")
            return OOM_EXIT_CODE
    LOG.info("Stage 3 done: %s", llrd_metrics)

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

    finish_wandb(wandb_run)
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

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(name)s | %(levelname)s | %(message)s")

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
