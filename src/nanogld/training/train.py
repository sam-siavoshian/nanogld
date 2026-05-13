"""V1 training orchestrator — 3-stage pipeline (SSL → linear probe → LLRD).

This is a skeleton orchestrator. Full SSL/probe/LLRD loops are wired
once `nanogld.data.dataset.NanoGLDDataset` lands (Block 2 completion).

Stage 1: SSL (SimMTM + CLIP + DANN + AECF) — `pretrain_simmtm`
Stage 2: linear probe (encoder frozen, focal CE only) — `linear_probe`
Stage 3: LLRD fine-tune (multi-task) — `llrd_finetune`

Each stage saves a checkpoint with `{git_sha, dataset_sha256, hparams,
runtime_versions}` manifest for reproducibility.

Spec: plan/05-MODEL-TRAINING-CALIBRATION.md V1 training plan.
Spec: plan/V1-SPEC.md §8.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
from torch import nn

from nanogld.training.cautious_optimizer import CautiousMask
from nanogld.training.ema import make_ema
from nanogld.training.friendly_sam import FriendlySAM


@dataclass(frozen=True)
class TrainConfig:
    """V1 training configuration."""

    base_lr: float = 1e-4
    betas_first: float = 0.9
    betas_second: float = 0.95
    weight_decay: float = 0.1
    warmup_steps: int = 300
    fsam_rho: float = 0.05
    ema_decay: float = 0.999
    grad_clip_max_norm: float = 1.0
    focal_gamma: float = 3.0

    ssl_epochs: int = 15
    probe_epochs: int = 5
    llrd_epochs: int = 10
    llrd_decay: float = 0.85
    mixout_p: float = 0.7

    freelb_K: int = 2
    freelb_epsilon: float = 0.5

    aecf_p_max: float = 0.9
    aecf_curriculum_steps: int = 10_000

    sharpe_loss_weight: float = 0.5
    focal_loss_weight: float = 0.5
    dann_loss_weight: float = 0.05
    aecf_reg_weight: float = 0.05

    cost_bps: float = 2.0

    seed: int = 42
    deterministic: bool = True
    fold_idx: int = 0
    output_dir: Path = field(default_factory=lambda: Path("checkpoints/v1"))

    def to_dict(self) -> dict:
        d = asdict(self)
        d["output_dir"] = str(d["output_dir"])
        return d


def build_optimizer_stack(
    model: nn.Module,
    cfg: TrainConfig,
):  # noqa: ANN201 — return is FriendlySAM(CautiousMask(SF-AdamW))
    """Wire Schedule-Free AdamW + Cautious mask + Friendly-SAM.

    Returns the outermost optimizer wrapper (FriendlySAM). Caller invokes
    `opt.step(closure)` per step.
    """
    from schedulefree import AdamWScheduleFree  # noqa: PLC0415

    decay_params: list[torch.Tensor] = []
    no_decay_params: list[torch.Tensor] = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        is_no_decay = "bias" in name or "norm" in name.lower() or "pos_emb" in name
        if is_no_decay:
            no_decay_params.append(p)
        else:
            decay_params.append(p)

    base = AdamWScheduleFree(
        [
            {"params": decay_params, "weight_decay": cfg.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=cfg.base_lr,
        betas=(cfg.betas_first, cfg.betas_second),
        warmup_steps=cfg.warmup_steps,
    )
    cautious = CautiousMask(base)
    fsam = FriendlySAM(model.parameters(), base_optimizer=cautious, rho=cfg.fsam_rho)
    return fsam


def llrd_param_groups(
    model: nn.Module,
    base_lr: float,
    decay: float,
    num_layers: int,
) -> list[dict]:
    """Layer-wise LR decay groups for Stage 3 fine-tune.

    For layer `l` (0-indexed from input → output), `lr_l = base_lr * decay^(N-l-1)`.
    Lower layers train slower (preserve general features).

    Args:
        model: top-level model with `encoder.transformer_blocks` attribute.
        base_lr: LR for the head and last layer.
        decay: per-layer multiplicative decay (0.85 V1).
        num_layers: total layers in the encoder stack.

    Returns:
        List of param-group dicts ready to pass to torch.optim.
    """
    groups: list[dict] = []
    encoder = getattr(model, "encoder", None)
    if encoder is None or not hasattr(encoder, "transformer_blocks"):
        return [{"params": list(model.parameters()), "lr": base_lr}]

    blocks = list(encoder.transformer_blocks)
    if hasattr(encoder, "slstm_blocks"):
        blocks = blocks + list(encoder.slstm_blocks)

    total = len(blocks)
    for i, block in enumerate(blocks):
        layer_lr = base_lr * (decay ** (total - i - 1))
        groups.append({"params": list(block.parameters()), "lr": layer_lr})

    head_params = []
    seen = {id(p) for grp in groups for p in grp["params"]}
    for p in model.parameters():
        if id(p) not in seen and p.requires_grad:
            head_params.append(p)
    if head_params:
        groups.append({"params": head_params, "lr": base_lr})
    return groups


def setup_determinism(seed: int = 42) -> None:
    """Seed every RNG and force deterministic algorithms.

    Python's ``hash()`` is salted at interpreter-startup using
    ``PYTHONHASHSEED``. Setting the env var INSIDE the interpreter only
    affects child processes, not this one. To get truly deterministic
    dict iteration order we need the env var set in the shell BEFORE
    ``python`` starts — ``scripts/spark_train.sh`` exports it; we log
    a clear warning here when it's missing so the owner can spot the
    miss in the per-stage log file.
    """
    import logging  # noqa: PLC0415
    import os  # noqa: PLC0415
    import random  # noqa: PLC0415

    import numpy as np  # noqa: PLC0415

    log = logging.getLogger("nanogld.training.setup_determinism")
    if os.environ.get("PYTHONHASHSEED") is None:
        log.warning(
            "PYTHONHASHSEED not set in env at process start — "
            "dict iteration order is not reproducible. "
            "Export PYTHONHASHSEED=%d in your shell before launching "
            "(scripts/spark_train.sh does this automatically).",
            seed,
        )
    os.environ["PYTHONHASHSEED"] = str(seed)  # best-effort for subprocesses
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def attach_ema(model: nn.Module, decay: float = 0.999):  # noqa: ANN201
    """Convenience wrapper around `make_ema`."""
    return make_ema(model, decay=decay)
