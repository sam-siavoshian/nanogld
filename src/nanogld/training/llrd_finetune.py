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

import torch
from torch import Tensor, nn

from nanogld.training.ema import make_ema
from nanogld.training.freelb import FreeLB
from nanogld.training.losses import focal_loss, sharpe_loss
from nanogld.training.mixout import Mixout

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
    output_dir: Path = field(default_factory=lambda: Path("checkpoints/v1/llrd"))


def _build_llrd_param_groups(model: nn.Module, base_lr: float, decay: float) -> list[dict]:
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

    other = [p for p in model.parameters() if p.requires_grad and id(p) not in seen_ids]
    if other:
        groups.append({"params": other, "lr": base_lr})
    return groups


def llrd_finetune(
    model: nn.Module,
    ssl_anchor_state: dict,
    train_loader: Iterable[dict[str, Tensor]],
    cfg: LLRDConfig,
    device: str = "cpu",
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
    try:
        from schedulefree import AdamWScheduleFree  # noqa: PLC0415

        opt = AdamWScheduleFree(
            groups,
            lr=cfg.base_lr,
            betas=(0.9, 0.95),
            weight_decay=0.1,
            warmup_steps=300,
        )
        if hasattr(opt, "train"):
            opt.train()
    except ImportError:
        opt = torch.optim.AdamW(groups, betas=(0.9, 0.95), weight_decay=0.1)

    mixout = Mixout(ssl_anchor_state, p=cfg.mixout_p)
    snapshot_state: dict[str, torch.Tensor] = {}
    ema = make_ema(model, decay=cfg.ema_decay)
    freelb = FreeLB(K=cfg.freelb_K, epsilon=cfg.freelb_epsilon)

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

    for epoch in range(cfg.epochs):
        for batch in train_loader:
            channel_inputs = batch["channel_inputs"].to(device).float().nan_to_num_(0.0)
            news_embeddings = batch["news_embeddings"].to(device).float()
            news_mask = batch["news_mask"].to(device).float()
            is_news_present = batch["is_news_present"].to(device).long()
            regime_vec = batch["regime_vec"].to(device).float()
            labels = batch["label_3class"].to(device).long()
            next_log_return = batch["next_log_return"].to(device).float()

            if not snapshot_state:
                for _name, _p in model.named_parameters():
                    snapshot_state[_name] = _p.data.detach().clone()
            else:
                with torch.no_grad():
                    for _name, _p in model.named_parameters():
                        snapshot_state[_name].copy_(_p.data)
            mixout.apply(model)
            opt.zero_grad()

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
                )

            def loss_fn(  # noqa: ANN001, B023
                out,
                _batch,
                _labels=labels,
                _nlr=next_log_return,
            ):
                logits = out["logits_3class"]
                pos = out["position_weight"]
                l_focal = focal_loss(logits, _labels, gamma=cfg.focal_gamma)
                l_sharpe = sharpe_loss(
                    pos,
                    _nlr,
                    prev_position=None,
                    cost_bps=cfg.cost_bps,
                )
                return cfg.focal_weight * l_focal + cfg.sharpe_weight * l_sharpe

            loss = freelb.compute_loss(model_forward, {"news_embeddings": news_embeddings}, loss_fn)
            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"non-finite loss at llrd step {n_steps}: {float(loss):.4f}; aborting fold"
                )
            loss.backward()
            if cfg.grad_clip_max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_max_norm)

            with torch.no_grad():
                for name, param in model.named_parameters():
                    if name in snapshot_state:
                        param.data.copy_(snapshot_state[name].to(param.device))

            opt.step()

            ema.update_parameters(model)

            n_steps += 1
            final_loss = float(loss.detach().cpu().item())
            if n_steps % cfg.log_every_n_steps == 0:
                LOG.info("llrd epoch %d step %d loss=%.4f", epoch, n_steps, final_loss)

    if n_steps == 0:
        raise RuntimeError("llrd_finetune produced no steps; refusing to write checkpoint")
    _opt_to_infer = getattr(opt, "eval", None)
    if _opt_to_infer is not None:
        _opt_to_infer()
    out_path = cfg.output_dir / "llrd_final.pt"
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    torch.save(
        {
            "model_state": model.state_dict(),
            "ema_state": ema.state_dict(),
            "n_steps": n_steps,
        },
        tmp_path,
    )
    import os as _os  # noqa: PLC0415

    _os.replace(tmp_path, out_path)
    _opt_to_train = getattr(opt, "train", None)
    if _opt_to_train is not None:
        _opt_to_train()
    LOG.info(
        "llrd fine-tune done: %d steps, final_loss=%.4f, ckpt=%s",
        n_steps,
        final_loss,
        out_path,
    )
    return {"final_loss": final_loss, "n_steps": float(n_steps)}
