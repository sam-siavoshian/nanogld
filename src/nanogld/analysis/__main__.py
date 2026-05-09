"""CLI entrypoint for post-train feature analysis.

Usage:
    python -m nanogld.analysis run \
        --checkpoint checkpoints/v1/fold_0/llrd/llrd_final.pt \
        --unified data/processed/training_v1_unified.pt \
        --sidecar data/processed/training_v1_sidecar.pt \
        --fold 0 \
        --split val_c \
        --output-dir reports/analysis/fold_0 \
        --device auto
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from nanogld.analysis.attention_rollout import attention_rollout
from nanogld.analysis.config import AnalysisConfig
from nanogld.analysis.integrated_gradients import integrated_gradients
from nanogld.analysis.modality_ablation import modality_ablation
from nanogld.analysis.permutation import permutation_importance
from nanogld.analysis.report import write_report
from nanogld.analysis.vsn_importance import collect_vsn_gates
from nanogld.data.dataset import NanoGLDDataset
from nanogld.model import nanoGLDV1

LOG = logging.getLogger("nanogld.analysis.cli")


def _autodetect_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _build_dataset(cfg: AnalysisConfig) -> NanoGLDDataset:
    return NanoGLDDataset(
        unified_path=cfg.unified_path,
        sidecar_path=cfg.sidecar_path if cfg.sidecar_path.exists() else None,
        split=cfg.split,
    )


def _load_model(checkpoint_path: Path, device: str) -> nanoGLDV1:
    """Reconstruct nanoGLDV1 from checkpoint state_dict."""
    model = nanoGLDV1()
    state = torch.load(checkpoint_path, weights_only=False, map_location="cpu")
    if "model_state" in state:
        sd = state["model_state"]
    elif "state_dict" in state:
        sd = state["state_dict"]
    else:
        sd = state
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        LOG.warning("missing keys (%d): %s", len(missing), missing[:5])
    if unexpected:
        LOG.warning("unexpected keys (%d): %s", len(unexpected), unexpected[:5])
    model.to(device)
    return model


def _feature_names_for(n: int) -> list[str]:
    """Synthetic feature names when the dataset doesn't ship a schema.

    Future: when unified.pt grows a `feature_names` field, prefer that.
    """
    return [f"f_{i:04d}" for i in range(n)]


def run(cfg: AnalysisConfig) -> int:
    """Run the full feature-analysis pipeline."""
    if cfg.device not in {"cpu", "cuda", "mps"}:
        raise ValueError(f"device must be cpu/cuda/mps, got {cfg.device!r}")

    LOG.info("loading dataset (split=%s)", cfg.split)
    ds = _build_dataset(cfg)
    if len(ds) == 0:
        raise RuntimeError(f"split {cfg.split!r} has 0 samples; cannot run analysis")
    loader = DataLoader(ds, batch_size=8, num_workers=0, shuffle=False)

    LOG.info("loading model from %s", cfg.checkpoint_path)
    model = _load_model(cfg.checkpoint_path, device=cfg.device)
    f_dim = ds.meta.n_features
    feature_names = _feature_names_for(f_dim)

    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    LOG.info("[1/5] VSN gate importance ...")
    vsn = collect_vsn_gates(model, loader, device=cfg.device)

    LOG.info("[2/5] Modality ablation ...")
    ablation = modality_ablation(model, loader, device=cfg.device, max_batches=64)

    LOG.info("[3/5] Integrated Gradients (n=%d, steps=%d) ...", cfg.n_samples_ig, cfg.n_steps_ig)
    try:
        ig = integrated_gradients(
            model,
            loader,
            device=cfg.device,
            n_samples=cfg.n_samples_ig,
            n_steps=cfg.n_steps_ig,
            baseline_mode=cfg.attribution_baseline,
        )
    except RuntimeError as exc:
        LOG.warning("IG skipped: %s", exc)
        ig = None

    LOG.info("[4/5] Permutation importance (top-%d by VSN) ...", cfg.max_features_perm)
    top_idx = torch.from_numpy(-vsn["mean_gate"]).argsort()[: cfg.max_features_perm].tolist()
    permutation = permutation_importance(
        model,
        loader,
        feature_indices=top_idx,
        device=cfg.device,
        n_repeats=cfg.n_perm_repeats,
        seed=cfg.seed,
    )

    LOG.info("[5/5] Cross-attention rollout ...")
    try:
        attention = attention_rollout(model, loader, device=cfg.device)
    except (AttributeError, RuntimeError) as exc:
        LOG.warning("attention rollout skipped: %s", exc)
        attention = None

    md_path = write_report(
        cfg=cfg,
        feature_names=feature_names,
        vsn=vsn,
        ig=ig,
        permutation=permutation,
        ablation=ablation,
        attention=attention,
    )
    LOG.info("analysis complete → %s", md_path)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="nanogld.analysis", description="Feature attribution")
    sub = parser.add_subparsers(dest="cmd", required=True)
    run_p = sub.add_parser("run", help="run all 6 attribution methods on one fold")
    run_p.add_argument("--checkpoint", type=Path, required=True)
    run_p.add_argument("--unified", type=Path, required=True)
    run_p.add_argument("--sidecar", type=Path, required=True)
    run_p.add_argument("--fold", type=int, required=True)
    run_p.add_argument("--split", type=str, default="val_c")
    run_p.add_argument(
        "--output-dir", dest="output_dir", type=Path, default=Path("reports/analysis")
    )
    run_p.add_argument("--device", type=str, default="auto")
    run_p.add_argument("--n-samples-ig", dest="n_samples_ig", type=int, default=256)
    run_p.add_argument("--n-steps-ig", dest="n_steps_ig", type=int, default=32)
    run_p.add_argument("--n-perm-repeats", dest="n_perm_repeats", type=int, default=3)
    run_p.add_argument("--max-features-perm", dest="max_features_perm", type=int, default=100)
    run_p.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    )
    if args.cmd != "run":
        parser.print_help()
        return 1

    device = args.device if args.device != "auto" else _autodetect_device()
    cfg = AnalysisConfig(
        checkpoint_path=args.checkpoint,
        unified_path=args.unified,
        sidecar_path=args.sidecar,
        fold_idx=args.fold,
        split=args.split,
        output_dir=args.output_dir,
        n_samples_ig=args.n_samples_ig,
        n_steps_ig=args.n_steps_ig,
        n_perm_repeats=args.n_perm_repeats,
        max_features_perm=args.max_features_perm,
        seed=args.seed,
        device=device,
    )
    return run(cfg)


if __name__ == "__main__":
    sys.exit(main())
