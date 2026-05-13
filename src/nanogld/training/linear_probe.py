"""Stage 2: linear probe.

Encoder is frozen; only the classification head trains. Loss = focal CE
gamma=3 on the 3-class direction labels. Sanity check that the SSL
features are linearly separable for direction prediction.

Per V1-SPEC §8.3: 3-10 epochs, focal_3class loss only.

Spec: plan/V1-SPEC.md §8.3.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from nanogld._atomic import atomic_save_torch
from nanogld.model.aecf import AECFMask
from nanogld.training.losses import focal_loss
from nanogld.training.optim import build_optimizer

LOG = logging.getLogger("nanogld.training.linear_probe")


@dataclass(frozen=True)
class LinearProbeConfig:
    """Stage 2 config."""

    epochs: int = 5
    lr: float = 1e-3
    grad_clip_max_norm: float = 1.0
    focal_gamma: float = 3.0
    log_every_n_steps: int = 100
    snapshot_every_n_steps: int = 0
    snapshot_keep: int = 3
    output_dir: Path = Path("checkpoints/v1/probe")
    aecf_p_min: float = 0.1
    aecf_p_max: float = 0.9
    aecf_curriculum_steps: int = 1_000


def _freeze_encoder(model: nn.Module) -> None:
    """Set requires_grad=False on every module except the head."""
    head = getattr(model, "head", None)
    if head is None:
        raise AttributeError("model.head missing — linear probe needs a head")
    head_params = {id(p) for p in head.parameters()}
    for p in model.parameters():
        if id(p) not in head_params:
            p.requires_grad_(False)


def _unfreeze_all(model: nn.Module) -> None:
    for p in model.parameters():
        p.requires_grad_(True)


def train_linear_probe(
    model: nn.Module,
    train_loader: Iterable[dict[str, Tensor]],
    cfg: LinearProbeConfig,
    device: str = "cpu",
    manifest: dict[str, Any] | None = None,
    wandb_run: Any | None = None,
) -> dict[str, float]:
    """Run Stage 2 linear probe.

    Args:
        model: nanoGLDV1 with SSL-pretrained encoder.
        train_loader: iterable of batches.
        cfg: LinearProbeConfig.
        device: torch device.
        manifest: optional reproducibility manifest dict (embedded in ckpt).

    Returns:
        {"final_loss", "n_steps", "accuracy_last_batch"}.
    """
    _freeze_encoder(model)
    model.train()

    head_params = list(model.head.parameters())
    opt = build_optimizer(
        head_params,
        lr=cfg.lr,
        betas=(0.9, 0.95),
        weight_decay=0.0,
        warmup_steps=0,
    )

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    aecf_mask_module = AECFMask(
        p_min=cfg.aecf_p_min,
        p_max=cfg.aecf_p_max,
        curriculum_steps=cfg.aecf_curriculum_steps,
    )
    aecf_mask_module.train()

    n_steps = 0
    final_loss = float("nan")
    last_acc = 0.0

    for epoch in range(cfg.epochs):
        for batch in train_loader:
            channel_inputs = batch["channel_inputs"].to(device).float().nan_to_num_(0.0)
            news_embeddings = batch["news_embeddings"].to(device).float()
            news_mask = batch["news_mask"].to(device).float()
            is_news_present = batch["is_news_present"].to(device).long()
            regime_vec = batch["regime_vec"].to(device).float()
            labels = batch["label_3class"].to(device).long()

            b_size = channel_inputs.shape[0]
            aecf_keep = aecf_mask_module.sample_mask(
                batch_size=b_size, training_step=n_steps, device=channel_inputs.device
            )
            news_mask = news_mask * aecf_keep.unsqueeze(-1)
            is_news_present = (is_news_present.float() * aecf_keep).to(is_news_present.dtype)

            # Capture the latest forward output for stats (logits/preds) by
            # writing into a 1-slot list from inside the closure. The closure
            # may run twice (FSAM ascent + descent); the LAST call's logits
            # land in latest[0].
            latest: list[Tensor] = []
            last_grad_norm: list[float] = []

            def closure(
                _ci: Tensor = channel_inputs,
                _ne: Tensor = news_embeddings,
                _nm: Tensor = news_mask,
                _inp: Tensor = is_news_present,
                _rv: Tensor = regime_vec,
                _labels: Tensor = labels,
                _hp: list[Tensor] = head_params,
                _latest: list[Tensor] = latest,
                _gn: list[float] = last_grad_norm,
            ) -> Tensor:
                opt.zero_grad()
                out = model(
                    channel_inputs=_ci,
                    news_embeddings=_ne,
                    news_mask=_nm,
                    is_news_present=_inp,
                    regime_vec=_rv,
                )
                logits = out["logits_3class"]
                _latest.clear()
                _latest.append(logits.detach())
                loss = focal_loss(logits, _labels, gamma=cfg.focal_gamma)
                loss.backward()
                if cfg.grad_clip_max_norm > 0:
                    gn = torch.nn.utils.clip_grad_norm_(_hp, cfg.grad_clip_max_norm)
                    _gn.append(float(gn))
                return loss

            loss = opt.step(closure)
            if loss is None or not torch.isfinite(loss):
                raise RuntimeError(
                    f"non-finite probe loss at step {n_steps}: {loss}; aborting fold"
                )

            n_steps += 1
            final_loss = float(loss.detach().cpu().item())
            import os as _os  # noqa: PLC0415
            _smoke = 0
            for _k in ("NANOGLD_PROBE_MAX_STEPS", "NANOGLD_MAX_STEPS"):
                _raw = _os.environ.get(_k, "")
                if _raw:
                    try:
                        _smoke = max(0, int(_raw))
                        break
                    except ValueError:
                        continue
            if _smoke > 0 and n_steps >= _smoke:
                LOG.info("probe step-cap: %d reached", _smoke)
                break
            if latest:
                preds = latest[0].argmax(dim=-1)
                last_acc = float((preds == labels).float().mean().item())
            if n_steps % cfg.log_every_n_steps == 0:
                LOG.info(
                    "probe epoch %d step %d loss=%.4f acc=%.3f",
                    epoch,
                    n_steps,
                    final_loss,
                    last_acc,
                )
                if wandb_run is not None:
                    payload = {
                        "probe/loss": final_loss,
                        "probe/acc": last_acc,
                        "probe/epoch": epoch,
                        "step": n_steps,
                    }
                    if last_grad_norm:
                        payload["probe/grad_norm"] = last_grad_norm[-1]
                    wandb_run.log(payload)

            if cfg.snapshot_every_n_steps > 0 and n_steps % cfg.snapshot_every_n_steps == 0:
                from nanogld.training.simmtm_pretrain import _write_snapshot
                _write_snapshot(cfg.output_dir, "probe", n_steps, model, cfg.snapshot_keep)

    if n_steps == 0:
        raise RuntimeError("train_linear_probe produced no steps; refusing to write checkpoint")

    _inference_mode = getattr(opt, "inference_mode", None)
    if callable(_inference_mode):
        _inference_mode()

    ckpt: dict[str, Any] = {
        "model_state": model.state_dict(),
        "n_steps": n_steps,
        "stage": "probe",
        "final_loss": final_loss,
        "accuracy_last_batch": last_acc,
    }
    if manifest is not None:
        ckpt["manifest"] = manifest
    out_path = cfg.output_dir / "probe.pt"
    atomic_save_torch(ckpt, out_path)
    (cfg.output_dir / "stage.done").write_text("ok\n")

    LOG.info(
        "linear probe done: %d steps, final_loss=%.4f, ckpt=%s",
        n_steps,
        final_loss,
        out_path,
    )
    return {
        "final_loss": final_loss,
        "n_steps": float(n_steps),
        "accuracy_last_batch": last_acc,
    }
