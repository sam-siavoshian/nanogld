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

import torch
from torch import Tensor, nn

from nanogld.training.losses import focal_loss

LOG = logging.getLogger("nanogld.training.linear_probe")


@dataclass(frozen=True)
class LinearProbeConfig:
    """Stage 2 config."""

    epochs: int = 5
    lr: float = 1e-3
    grad_clip_max_norm: float = 1.0
    focal_gamma: float = 3.0
    log_every_n_steps: int = 100
    output_dir: Path = Path("checkpoints/v1/probe")


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
) -> dict[str, float]:
    """Run Stage 2 linear probe.

    Args:
        model: nanoGLDV1 with SSL-pretrained encoder.
        train_loader: iterable of batches.
        cfg: LinearProbeConfig.
        device: torch device.

    Returns:
        {"final_loss", "n_steps", "accuracy_last_batch"}.
    """
    _freeze_encoder(model)
    model.train()

    head_params = list(model.head.parameters())
    opt = torch.optim.Adam(head_params, lr=cfg.lr)

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
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

            opt.zero_grad()
            out = model(
                channel_inputs=channel_inputs,
                news_embeddings=news_embeddings,
                news_mask=news_mask,
                is_news_present=is_news_present,
                regime_vec=regime_vec,
            )
            logits = out["logits_3class"]
            loss = focal_loss(logits, labels, gamma=cfg.focal_gamma)
            loss.backward()
            if cfg.grad_clip_max_norm > 0:
                torch.nn.utils.clip_grad_norm_(head_params, cfg.grad_clip_max_norm)
            opt.step()

            n_steps += 1
            final_loss = float(loss.detach().cpu().item())
            preds = logits.argmax(dim=-1)
            last_acc = float((preds == labels).float().mean().item())
            if n_steps % cfg.log_every_n_steps == 0:
                LOG.info(
                    "probe epoch %d step %d loss=%.4f acc=%.3f",
                    epoch,
                    n_steps,
                    final_loss,
                    last_acc,
                )

    out_path = cfg.output_dir / "probe.pt"
    torch.save({"model_state": model.state_dict(), "n_steps": n_steps}, out_path)
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
