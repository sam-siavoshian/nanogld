"""Stage 1: SimMTM SSL pretraining loop.

Per V1-SPEC §7:
  - K=3 masked views per sample
  - Mask ratio 0.40
  - Combined loss = simmtm_recon + 0.5*L_clip + 0.05*L_DANN + 1.0*L_aecf
  - 15-20 epochs (~25-30% of total compute budget)

Implementation note: this is a self-supervised reconstruction objective
that DOES NOT require labels. The model.forward() returns logits; we
take the encoder representation (mean-pooled tokens) and run two heads:
  1. Reconstruction head — predict the masked bar values.
  2. CLIP head — contrastive between bar-rep and news-rep.

For V1, we use the model's pooled output as `z_bar` and the mean of
news_embeddings (filtered by news_mask) as `z_news`. The recon head is
a small Linear we add inline (not a permanent model component).

Spec: plan/V1-SPEC.md §7 + §8.3 (Stage 1 SSL).
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor, nn

from nanogld.training.losses import aecf_entropy_reg, clip_infonce, simmtm_loss

LOG = logging.getLogger("nanogld.training.simmtm_pretrain")

DEFAULT_MASK_RATIO = 0.40
DEFAULT_K_VIEWS = 3


@dataclass(frozen=True)
class SimMTMConfig:
    """SSL stage config."""

    epochs: int = 15
    mask_ratio: float = DEFAULT_MASK_RATIO
    k_views: int = DEFAULT_K_VIEWS
    lambda_sim: float = 0.5
    lambda_clip: float = 0.5
    lambda_aecf: float = 1.0
    grad_clip_max_norm: float = 1.0
    log_every_n_steps: int = 100
    output_dir: Path = Path("checkpoints/v1/ssl")


def _generate_masked_views(
    channel_inputs: Tensor,
    k_views: int,
    mask_ratio: float,
) -> tuple[Tensor, Tensor]:
    """Build K masked views per sample.

    Args:
        channel_inputs: (B, T, F).
        k_views: number of views per sample.
        mask_ratio: per-element mask probability.

    Returns:
        (views, targets) each of shape (B, K, T, F). `views` are masked
        (zeroed where masked); `targets` are the original values.
    """
    b, t, f = channel_inputs.shape
    targets = channel_inputs.unsqueeze(1).expand(b, k_views, t, f).contiguous()

    mask = torch.rand((b, k_views, t, f), device=channel_inputs.device) > mask_ratio
    views = targets * mask.to(targets.dtype)
    return views, targets


class SimMTMReconHead(nn.Module):
    """Small linear head that predicts bar values from pooled encoder output."""

    def __init__(self, d_model: int, target_dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, target_dim, bias=False)

    def forward(self, pooled: Tensor) -> Tensor:
        return self.proj(pooled)


def pretrain_simmtm(
    model: nn.Module,
    optimizer,  # noqa: ANN001 — wraps any of our optimizer wrappers
    train_loader: Iterable[dict[str, Tensor]],
    cfg: SimMTMConfig,
    device: str = "cpu",
) -> dict[str, float]:
    """Run the SSL pretrain loop.

    Args:
        model: nanoGLDV1 instance.
        optimizer: opt-wrapper supporting .zero_grad() / .step(closure).
        train_loader: iterable yielding training batches.
        cfg: SimMTMConfig.
        device: torch device string.

    Returns:
        Dict with {"final_loss": float, "n_steps": int}.
    """
    model.train()
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    target_dim = None
    recon_head = None
    optimizer_aux = None

    n_steps = 0
    final_loss = float("nan")

    for epoch in range(cfg.epochs):
        for batch in train_loader:
            channel_inputs = batch["channel_inputs"].to(device).float().nan_to_num_(0.0)
            news_embeddings = batch["news_embeddings"].to(device).float()
            news_mask = batch["news_mask"].to(device).float()
            is_news_present = batch["is_news_present"].to(device).long()
            regime_vec = batch["regime_vec"].to(device).float()

            if target_dim is None:
                target_dim = channel_inputs.shape[-1]
                recon_head = SimMTMReconHead(d_model=model.d_model, target_dim=target_dim).to(
                    device
                )
                optimizer_aux = torch.optim.Adam(recon_head.parameters(), lr=1e-3)

            views, targets = _generate_masked_views(
                channel_inputs, k_views=cfg.k_views, mask_ratio=cfg.mask_ratio
            )

            view_pooled_list = []
            for k in range(cfg.k_views):
                v_k = views[:, k]
                out = model(
                    channel_inputs=v_k,
                    news_embeddings=news_embeddings,
                    news_mask=news_mask,
                    is_news_present=is_news_present,
                    regime_vec=regime_vec,
                )
                logits = out["logits_3class"]
                pooled_proxy = logits.mean(dim=-1, keepdim=True).expand(-1, model.d_model)
                view_pooled_list.append(pooled_proxy)

            views_pooled = torch.stack(view_pooled_list, dim=1)
            target_per_view_proj = recon_head(views_pooled)

            l_simmtm = simmtm_loss(
                views=views_pooled,
                targets=target_per_view_proj,
                lambda_sim=cfg.lambda_sim,
            )
            mask_for_news = news_mask.bool()
            mask_count = mask_for_news.sum(dim=1, keepdim=True).clamp(min=1)
            news_pool = (news_embeddings * mask_for_news.unsqueeze(-1)).sum(dim=1) / mask_count
            bar_pool = views_pooled.mean(dim=1)
            l_clip = clip_infonce(bar_pool, news_pool, tau=0.07)
            l_aecf = aecf_entropy_reg(
                gate_dist=torch.softmax(bar_pool, dim=-1).clamp(min=1e-8),
                lambda_x=cfg.lambda_aecf,
            )

            loss = l_simmtm + cfg.lambda_clip * l_clip + l_aecf

            _aux = optimizer_aux
            _loss = loss

            def closure(_aux=_aux, _loss=_loss):  # noqa: B023
                optimizer.zero_grad()
                if _aux is not None:
                    _aux.zero_grad()
                _loss.backward(retain_graph=False)
                if cfg.grad_clip_max_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_max_norm)
                return _loss

            try:
                optimizer.step(closure)
            except TypeError:
                closure()
                optimizer.step()
            if _aux is not None:
                _aux.step()

            n_steps += 1
            final_loss = float(loss.detach().cpu().item())
            if n_steps % cfg.log_every_n_steps == 0:
                LOG.info(
                    "ssl epoch %d step %d loss=%.4f (simmtm=%.4f clip=%.4f aecf=%.4f)",
                    epoch,
                    n_steps,
                    final_loss,
                    float(l_simmtm.detach().item()),
                    float(l_clip.detach().item()),
                    float(l_aecf.detach().item()),
                )

    ckpt = {
        "model_state": model.state_dict(),
        "config": {"mask_ratio": cfg.mask_ratio, "k_views": cfg.k_views, "epochs": cfg.epochs},
        "final_loss": final_loss,
        "n_steps": n_steps,
    }
    out_path = cfg.output_dir / "ssl_anchor.pt"
    torch.save(ckpt, out_path)
    LOG.info("ssl pretrain done: %d steps, final_loss=%.4f, ckpt=%s", n_steps, final_loss, out_path)
    return {"final_loss": final_loss, "n_steps": float(n_steps)}
