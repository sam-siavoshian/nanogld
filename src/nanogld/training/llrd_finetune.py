"""Stage 3: LLRD fine-tune (Layer-wise Learning Rate Decay).

Per V1-SPEC §8.3 + §6.1:
  - Layer-wise LR decay 0.85: lr_l = base_lr * 0.85 ** (n_layers - l - 1)
  - Mixout p=0.7 anchored to SSL checkpoint
  - Schedule-Free AdamW base (Friendly-SAM + Cautious wrap deferred)
  - FreeLB on news embeddings (K=2, ε=0.5)
  - EMA decay 0.999
  - Multi-task loss = 0.5 * L_focal + 0.5 * L_sharpe + L_aecf
    (L_DANN deferred until domain classifier is wired)

Spec: plan/V1-SPEC.md §8.3.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from nanogld._atomic import atomic_save_torch
from nanogld.model.aecf import AECFMask
from nanogld.training.ema import make_ema
from nanogld.training.freelb import FreeLB
from nanogld.training.losses import dann_loss, focal_loss, sharpe_loss
from nanogld.training.mixout import Mixout
from nanogld.training.optim import build_optimizer

LOG = logging.getLogger("nanogld.training.llrd_finetune")


@dataclass(frozen=True)
class LLRDConfig:
    """Stage 3 config."""

    epochs: int = 10
    base_lr: float = 1e-4
    decay: float = 0.85
    mixout_p: float = 0.7
    ema_decay: float = 0.999
    grad_clip_max_norm: float = 1.0
    focal_gamma: float = 3.0
    cost_bps: float = 2.0
    focal_weight: float = 0.5
    sharpe_weight: float = 0.5
    aecf_weight: float = 0.05
    freelb_K: int = 2
    freelb_epsilon: float = 0.5
    log_every_n_steps: int = 100
    snapshot_every_n_steps: int = 0
    snapshot_keep: int = 3
    output_dir: Path = field(default_factory=lambda: Path("checkpoints/v1/llrd"))
    aecf_p_min: float = 0.1
    aecf_p_max: float = 0.9
    aecf_curriculum_steps: int = 2_000
    dann_weight: float = 0.05
    dann_max_alpha: float = 0.1
    dann_warmup_steps: int = 2_000


def _build_llrd_param_groups(
    model: nn.Module, base_lr: float, decay: float
) -> list[dict]:
    """Per-block LR decay for the encoder; head + non-encoder use base_lr."""
    encoder = getattr(model, "encoder", None)
    if encoder is None:
        return [{"params": list(model.parameters()), "lr": base_lr}]

    blocks = []
    if hasattr(encoder, "transformer_blocks"):
        blocks.extend(list(encoder.transformer_blocks))
    if hasattr(encoder, "slstm_blocks"):
        blocks.extend(list(encoder.slstm_blocks))
    total = len(blocks)

    groups: list[dict] = []
    seen_ids: set[int] = set()
    for i, block in enumerate(blocks):
        layer_lr = base_lr * (decay ** (total - i - 1))
        params = [p for p in block.parameters() if p.requires_grad]
        for p in params:
            seen_ids.add(id(p))
        groups.append({"params": params, "lr": layer_lr})

    other = [
        p
        for p in model.parameters()
        if p.requires_grad and id(p) not in seen_ids
    ]
    if other:
        groups.append({"params": other, "lr": base_lr})
    return groups


def llrd_finetune(
    model: nn.Module,
    ssl_anchor_state: dict,
    train_loader: Iterable[dict[str, Tensor]],
    cfg: LLRDConfig,
    device: str = "cpu",
    manifest: dict[str, Any] | None = None,
    wandb_run: Any | None = None,
) -> dict[str, float]:
    """Run Stage 3 multi-task LLRD fine-tune.

    Args:
        model: nanoGLDV1 with SSL-pretrained encoder.
        ssl_anchor_state: state_dict of the SSL checkpoint (Mixout anchor).
        train_loader: iterable of batches.
        cfg: LLRDConfig.
        device: torch device.

    Returns:
        {"final_loss", "n_steps"}.
    """
    for p in model.parameters():
        p.requires_grad_(True)

    groups = _build_llrd_param_groups(model, base_lr=cfg.base_lr, decay=cfg.decay)
    opt = build_optimizer(
        groups,
        lr=cfg.base_lr,
        betas=(0.9, 0.95),
        weight_decay=0.1,
        warmup_steps=300,
    )

    mixout = Mixout(ssl_anchor_state, p=cfg.mixout_p)
    ema = make_ema(model, decay=cfg.ema_decay)
    freelb = FreeLB(K=cfg.freelb_K, epsilon=cfg.freelb_epsilon)
    aecf_mask_module = AECFMask(
        p_min=cfg.aecf_p_min,
        p_max=cfg.aecf_p_max,
        curriculum_steps=cfg.aecf_curriculum_steps,
    )
    aecf_mask_module.train()

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    try:
        loader_len = len(train_loader)  # type: ignore[arg-type]
    except TypeError:
        loader_len = -1
    if loader_len == 0:
        raise RuntimeError("llrd_finetune received an empty train_loader; refusing to run")
    n_steps = 0
    final_loss = float("nan")

    model.train()

    # Pre-allocated tensors for Mixout state-transfer. Avoids per-step
    # allocator churn (one of the wave-1 fixes; see plan/STATUS.md §61).
    snap_unmixed: dict[str, torch.Tensor] = {}
    snap_mixed: dict[str, torch.Tensor] = {}

    for epoch in range(cfg.epochs):
        for batch in train_loader:
            channel_inputs = batch["channel_inputs"].to(device).float().nan_to_num_(0.0)
            news_embeddings = batch["news_embeddings"].to(device).float()
            news_mask = batch["news_mask"].to(device).float()
            is_news_present = batch["is_news_present"].to(device).long()
            regime_vec = batch["regime_vec"].to(device).float()
            labels = batch["label_3class"].to(device).long()
            next_log_return = batch["next_log_return"].to(device).float()
            era_label = batch["era_label"].to(device).long()

            b_size = channel_inputs.shape[0]
            aecf_keep = aecf_mask_module.sample_mask(
                batch_size=b_size, training_step=n_steps, device=channel_inputs.device
            )
            news_mask = news_mask * aecf_keep.unsqueeze(-1)
            is_news_present = (is_news_present.float() * aecf_keep).to(is_news_present.dtype)

            # Snapshot pristine pre-mixout params (so we can transfer the
            # SF AdamW delta from "mixed" coords back onto unmixed coords).
            with torch.no_grad():
                for name, p in model.named_parameters():
                    if name not in snap_unmixed:
                        snap_unmixed[name] = p.data.detach().clone()
                    else:
                        snap_unmixed[name].copy_(p.data)

            mixout.apply(model)

            with torch.no_grad():
                for name, p in model.named_parameters():
                    if name not in snap_mixed:
                        snap_mixed[name] = p.data.detach().clone()
                    else:
                        snap_mixed[name].copy_(p.data)

            def model_forward(  # noqa: ANN001, B023
                news_embeddings,
                _ci=channel_inputs,
                _nm=news_mask,
                _inp=is_news_present,
                _rv=regime_vec,
                **rest,
            ):
                return model(
                    channel_inputs=_ci,
                    news_embeddings=news_embeddings,
                    news_mask=_nm,
                    is_news_present=_inp,
                    regime_vec=_rv,
                    return_pooled=True,
                )

            # alpha ramps 0 -> dann_max_alpha; weight stays at dann_weight.
            dann_alpha = (
                min(1.0, float(n_steps) / max(1.0, float(cfg.dann_warmup_steps)))
                * cfg.dann_max_alpha
            )

            def loss_fn(  # noqa: ANN001, B023
                out,
                _batch,
                _labels=labels,
                _nlr=next_log_return,
                _eras=era_label,
                _alpha=dann_alpha,
            ):
                logits = out["logits_3class"]
                pos = out["position_weight"]
                pooled = out["pooled"]
                l_focal = focal_loss(logits, _labels, gamma=cfg.focal_gamma)
                l_sharpe = sharpe_loss(
                    pos,
                    _nlr,
                    prev_position=None,
                    cost_bps=cfg.cost_bps,
                )
                dann_logits = model.head.dann_forward(pooled, _alpha)
                l_dann = dann_loss(dann_logits, _eras)
                return (
                    cfg.focal_weight * l_focal
                    + cfg.sharpe_weight * l_sharpe
                    + cfg.dann_weight * l_dann
                )

            # Closure-style step for Cautious(FSAM(SF)). FSAM may call the
            # closure twice (ascent + descent); FreeLB runs K=2 embedding-
            # ascent steps inside each pass.
            last_grad_norm: list[float] = []

            def closure(
                _ne: Tensor = news_embeddings,
                _grad_clip: float = cfg.grad_clip_max_norm,
                _gn: list[float] = last_grad_norm,
            ) -> Tensor:
                opt.zero_grad()
                loss_local = freelb.compute_loss(
                    model_forward, {"news_embeddings": _ne}, loss_fn
                )
                if not torch.isfinite(loss_local):
                    raise RuntimeError(
                        f"non-finite loss at llrd step {n_steps}: "
                        f"{float(loss_local):.4f}; aborting fold"
                    )
                loss_local.backward()
                if _grad_clip > 0:
                    gn = torch.nn.utils.clip_grad_norm_(model.parameters(), _grad_clip)
                    _gn.append(float(gn))
                return loss_local

            loss = opt.step(closure)
            if loss is None:
                raise RuntimeError("optimizer.step returned None — closure misconfigured")

            # Transfer the SF AdamW delta (computed against mixed coords)
            # onto the unmixed params: params_new = unmixed + (mixed_after_step - mixed_before_step).
            with torch.no_grad():
                for name, p in model.named_parameters():
                    if name not in snap_mixed:
                        continue
                    mixed_after = p.data
                    delta = mixed_after - snap_mixed[name].to(p.device, dtype=p.dtype)
                    p.data.copy_(snap_unmixed[name].to(p.device, dtype=p.dtype) + delta)

            ema.update_parameters(model)

            n_steps += 1
            final_loss = float(loss.detach().cpu().item())
            import os as _os  # noqa: PLC0415
            _smoke = int(_os.environ.get("NANOGLD_MAX_STEPS", "0") or 0)
            if _smoke > 0 and n_steps >= _smoke:
                LOG.info("llrd smoke-break: NANOGLD_MAX_STEPS=%d reached", _smoke)
                break
            if n_steps % cfg.log_every_n_steps == 0:
                LOG.info("llrd epoch %d step %d loss=%.4f", epoch, n_steps, final_loss)
                if wandb_run is not None:
                    # Per-group LRs surface LLRD's layer-wise decay
                    # schedule so the dashboard shows the slope.
                    lr_avg = sum(g["lr"] for g in opt.param_groups) / max(1, len(opt.param_groups))
                    payload = {
                        "llrd/loss": final_loss,
                        "llrd/lr_avg": lr_avg,
                        "llrd/epoch": epoch,
                        "step": n_steps,
                    }
                    if last_grad_norm:
                        payload["llrd/grad_norm"] = last_grad_norm[-1]
                    wandb_run.log(payload)

            if cfg.snapshot_every_n_steps > 0 and n_steps % cfg.snapshot_every_n_steps == 0:
                from nanogld.training.simmtm_pretrain import _write_snapshot
                _write_snapshot(cfg.output_dir, "llrd", n_steps, model, cfg.snapshot_keep)

    if n_steps == 0:
        raise RuntimeError("llrd_finetune produced no steps; refusing to write checkpoint")

    # Swap to averaged weights before snapshotting state_dict for the
    # final checkpoint (same reason as the SSL anchor fix at §52).
    _inference_mode = getattr(opt, "inference_mode", None)
    if callable(_inference_mode):
        _inference_mode()

    ckpt: dict[str, Any] = {
        "model_state": model.state_dict(),
        "ema_state": ema.state_dict(),
        "n_steps": n_steps,
        "final_loss": final_loss,
        "stage": "llrd",
    }
    if manifest is not None:
        ckpt["manifest"] = manifest
    out_path = cfg.output_dir / "llrd_final.pt"
    atomic_save_torch(ckpt, out_path)
    (cfg.output_dir / "stage.done").write_text("ok\n")

    _train_mode = getattr(opt, "train_mode", None)
    if callable(_train_mode):
        _train_mode()

    LOG.info(
        "llrd fine-tune done: %d steps, final_loss=%.4f, ckpt=%s",
        n_steps,
        final_loss,
        out_path,
    )
    return {"final_loss": final_loss, "n_steps": float(n_steps)}
