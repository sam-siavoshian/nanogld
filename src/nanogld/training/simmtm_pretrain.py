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
from typing import Any

import torch
from torch import Tensor, nn

from nanogld._atomic import atomic_save_torch
from nanogld.model.aecf import AECFMask
from nanogld.training.losses import clip_infonce, dann_loss, simmtm_loss

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
    # AECF curriculum mask (V1-SPEC §2.3 / §2.5). Stage 1 SSL ramps from
    # p_drop=0.0 to p_drop=0.9 over `aecf_curriculum_steps` iterations.
    aecf_p_min: float = 0.0
    aecf_p_max: float = 0.9
    aecf_curriculum_steps: int = 10_000
    # DANN (V1-SPEC §6.7): adversarial era classifier with gradient reversal.
    # Alpha ramps 0 -> dann_max_alpha linearly over dann_warmup_steps.
    dann_weight: float = 0.05
    dann_max_alpha: float = 0.1
    dann_warmup_steps: int = 10_000


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
    """Auxiliary heads for SSL: bar reconstruction + news projection.

    - `recon(pooled)`: maps (B, K, d_model) → (B, K, target_dim) for SimMTM
      reconstruction against time-pooled actual bars.
    - `news_proj(news_pool)`: maps (B, d_text) → (B, d_model) so CLIP can
      compute z_bar @ z_news.t() with matching dims.
    """

    def __init__(self, d_model: int, target_dim: int, d_news: int = 256) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, target_dim, bias=False)
        self.news_proj = nn.Linear(d_news, d_model, bias=False)

    def forward(self, pooled: Tensor) -> Tensor:
        return self.proj(pooled)

    def project_news(self, news_pool: Tensor) -> Tensor:
        return self.news_proj(news_pool)


def pretrain_simmtm(
    model: nn.Module,
    optimizer,  # noqa: ANN001 — wraps any of our optimizer wrappers
    train_loader: Iterable[dict[str, Tensor]],
    cfg: SimMTMConfig,
    device: str = "cpu",
    manifest: dict[str, Any] | None = None,
    wandb_run: Any | None = None,
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
    try:
        loader_len = len(train_loader)  # type: ignore[arg-type]
    except TypeError:
        loader_len = -1
    if loader_len == 0:
        raise RuntimeError("pretrain_simmtm received an empty train_loader; refusing to run")

    target_dim = None
    recon_head = None
    optimizer_aux = None

    aecf_mask_module = AECFMask(
        p_min=cfg.aecf_p_min,
        p_max=cfg.aecf_p_max,
        curriculum_steps=cfg.aecf_curriculum_steps,
    )
    aecf_mask_module.train()

    n_steps = 0
    final_loss = float("nan")

    for epoch in range(cfg.epochs):
        for batch in train_loader:
            channel_inputs = batch["channel_inputs"].to(device).float().nan_to_num_(0.0)
            news_embeddings = batch["news_embeddings"].to(device).float()
            news_mask = batch["news_mask"].to(device).float()
            is_news_present = batch["is_news_present"].to(device).long()
            regime_vec = batch["regime_vec"].to(device).float()
            era_label = batch["era_label"].to(device).long()

            # AECF curriculum modality dropout: per-batch Bernoulli over
            # samples. When 0, news for that sample is fully dropped via
            # mask zero-out and is_news_present flipped to 0.
            b_size = channel_inputs.shape[0]
            aecf_keep = aecf_mask_module.sample_mask(
                batch_size=b_size, training_step=n_steps, device=channel_inputs.device
            )
            news_mask = news_mask * aecf_keep.unsqueeze(-1)
            is_news_present = (is_news_present.float() * aecf_keep).to(is_news_present.dtype)

            if target_dim is None:
                target_dim = channel_inputs.shape[-1]
                recon_head = SimMTMReconHead(
                    d_model=model.d_model,
                    target_dim=target_dim,
                    d_news=news_embeddings.shape[-1],
                ).to(device)
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
                    return_pooled=True,
                )
                view_pooled_list.append(out["pooled"])

            views_pooled = torch.stack(view_pooled_list, dim=1)
            pred_per_view = recon_head(views_pooled)
            target_per_view = targets.mean(dim=2)

            l_simmtm = simmtm_loss(
                views=pred_per_view,
                targets=target_per_view,
                lambda_sim=cfg.lambda_sim,
            )
            mask_for_news = news_mask.bool()
            mask_count = mask_for_news.sum(dim=1, keepdim=True).clamp(min=1)
            news_pool_raw = (
                news_embeddings * mask_for_news.unsqueeze(-1)
            ).sum(dim=1) / mask_count
            news_pool = recon_head.project_news(news_pool_raw)
            bar_pool = views_pooled.mean(dim=1)
            l_clip = clip_infonce(bar_pool, news_pool, tau=0.07)

            # DANN: classify era from the encoder pooled rep via grad-reverse.
            # alpha ramps 0 -> dann_max_alpha over dann_warmup_steps.
            dann_alpha = min(
                1.0, float(n_steps) / max(1.0, float(cfg.dann_warmup_steps))
            ) * cfg.dann_max_alpha
            dann_logits = model.head.dann_forward(bar_pool, dann_alpha)
            l_dann = dann_loss(dann_logits, era_label)

            loss = (
                l_simmtm
                + cfg.lambda_clip * l_clip
                + cfg.dann_weight * l_dann
            )
            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"non-finite SSL loss at step {n_steps}: {float(loss):.4f}; aborting"
                )

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

            # Closure-required path: build_optimizer returns
            # Cautious(FriendlySAM(SF AdamW)) which mandates a closure.
            optimizer.step(closure)
            if _aux is not None:
                _aux.step()

            n_steps += 1
            final_loss = float(loss.detach().cpu().item())
            if n_steps % cfg.log_every_n_steps == 0:
                LOG.info(
                    "ssl epoch %d step %d loss=%.4f (simmtm=%.4f clip=%.4f)",
                    epoch,
                    n_steps,
                    final_loss,
                    float(l_simmtm.detach().item()),
                    float(l_clip.detach().item()),
                )
                if wandb_run is not None:
                    wandb_run.log(
                        {
                            "ssl/loss": final_loss,
                            "ssl/simmtm_loss": float(l_simmtm.detach().item()),
                            "ssl/clip_loss": float(l_clip.detach().item()),
                            "ssl/epoch": epoch,
                            "step": n_steps,
                        }
                    )

    if n_steps == 0:
        raise RuntimeError("pretrain_simmtm produced no steps; refusing to write checkpoint")

    # Swap to averaged weights BEFORE snapshotting state_dict. Schedule-Free's
    # inference-mode hook replaces the bare in-memory params with the running
    # average; that is the "anchor" we want Mixout to regularize toward at
    # Stage 3. See plan/STATUS.md §52.
    _inference_mode = getattr(optimizer, "inference_mode", None)
    if callable(_inference_mode):
        _inference_mode()

    ckpt: dict[str, Any] = {
        "model_state": model.state_dict(),
        "config": {"mask_ratio": cfg.mask_ratio, "k_views": cfg.k_views, "epochs": cfg.epochs},
        "final_loss": final_loss,
        "n_steps": n_steps,
        "stage": "ssl",
    }
    if manifest is not None:
        ckpt["manifest"] = manifest
    out_path = cfg.output_dir / "ssl_anchor.pt"
    atomic_save_torch(ckpt, out_path)

    # Sentinel marking this stage's clean completion (used by resume logic).
    (cfg.output_dir / "stage.done").write_text("ok\n")

    # Restore training mode so callers can keep iterating with the SAME
    # optimizer object if they want (e.g. fine-tune continuing from SSL).
    _train_mode = getattr(optimizer, "train_mode", None)
    if callable(_train_mode):
        _train_mode()

    LOG.info("ssl pretrain done: %d steps, final_loss=%.4f, ckpt=%s", n_steps, final_loss, out_path)
    return {"final_loss": final_loss, "n_steps": float(n_steps)}
