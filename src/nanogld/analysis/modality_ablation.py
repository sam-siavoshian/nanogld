"""Modality ablation — quantifies what each input stream contributes.

Strategy: re-run the eval loader with one input stream zeroed at a time
and report the focal-loss + Sharpe delta vs the full-pipeline baseline.

Streams ablated:
    bars       : channel_inputs zeroed
    news       : news_embeddings + news_mask zeroed, is_news_present=0
    regime     : regime_vec zeroed
    bars_news  : both zeroed (lower bound)

Per V1 invariant 18, results are also split by news-present vs
news-absent buckets when reporting.

Spec: plan/V1-SPEC.md §11.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from nanogld.analysis._inference_mode import to_inference_mode

LOG = logging.getLogger("nanogld.analysis.ablation")


_ABLATIONS: tuple[str, ...] = ("none", "bars", "news", "regime", "bars_news")


def _apply_ablation(
    name: str,
    channel_inputs: Tensor,
    news_embeddings: Tensor,
    news_mask: Tensor,
    is_news_present: Tensor,
    regime_vec: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    ci, ne, nm, inp, rv = channel_inputs, news_embeddings, news_mask, is_news_present, regime_vec
    if name in ("bars", "bars_news"):
        ci = torch.zeros_like(ci)
    if name in ("news", "bars_news"):
        ne = torch.zeros_like(ne)
        nm = torch.zeros_like(nm)
        inp = torch.zeros_like(inp)
    if name == "regime":
        rv = torch.zeros_like(rv)
    return ci, ne, nm, inp, rv


def _sharpe_arr(arr: np.ndarray) -> float:
    if arr.size < 2:
        return 0.0
    s = arr.std(ddof=1)
    if not np.isfinite(s) or s == 0:
        return 0.0
    return float(arr.mean() / s * np.sqrt(3276))


def _eval_metrics(
    model: nn.Module,
    batches: list[dict[str, Tensor]],
    device: str,
    ablation: str,
) -> dict[str, float]:
    losses = []
    pnls = []
    presence = []
    to_inference_mode(model)
    with torch.no_grad():
        for batch in batches:
            channel_inputs = batch["channel_inputs"].to(device).float().nan_to_num(0.0)
            news_embeddings = batch["news_embeddings"].to(device).float()
            news_mask = batch["news_mask"].to(device).float()
            is_news_present = batch["is_news_present"].to(device).long()
            regime_vec = batch["regime_vec"].to(device).float()

            ci, ne, nm, inp, rv = _apply_ablation(
                ablation, channel_inputs, news_embeddings, news_mask, is_news_present, regime_vec
            )
            out = model(
                channel_inputs=ci,
                news_embeddings=ne,
                news_mask=nm,
                is_news_present=inp,
                regime_vec=rv,
            )
            logits = out["logits_3class"]
            pos = out["position_weight"]
            labels = batch["label_3class"].to(device).long()
            nlr = batch["next_log_return"].to(device).float()
            losses.append(F.cross_entropy(logits, labels, reduction="none").detach().cpu().numpy())
            pnls.append((pos.detach() * nlr).cpu().numpy())
            presence.append(is_news_present.detach().cpu().numpy())

    all_loss = np.concatenate(losses) if losses else np.zeros(1)
    all_pnl = np.concatenate(pnls) if pnls else np.zeros(1)
    all_pres = np.concatenate(presence) if presence else np.zeros(1, dtype=np.int64)
    p_mask = all_pres.astype(bool)
    return {
        "focal": float(all_loss.mean()) if all_loss.size else 0.0,
        "sharpe": _sharpe_arr(all_pnl),
        "focal_present": float(all_loss[p_mask].mean()) if p_mask.any() else 0.0,
        "focal_absent": float(all_loss[~p_mask].mean()) if (~p_mask).any() else 0.0,
        "sharpe_present": _sharpe_arr(all_pnl[p_mask]),
        "sharpe_absent": _sharpe_arr(all_pnl[~p_mask]),
    }


def modality_ablation(
    model: nn.Module,
    loader: Iterable[dict[str, Tensor]],
    device: str = "cpu",
    max_batches: int = 64,
) -> dict[str, dict[str, float]]:
    cached: list[dict[str, Tensor]] = []
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        cached.append(batch)
    if not cached:
        raise RuntimeError("modality_ablation: empty loader")

    results: dict[str, dict[str, float]] = {}
    for name in _ABLATIONS:
        metrics = _eval_metrics(model, cached, device, ablation=name)
        results[name] = metrics
        LOG.info(
            "ablation=%s focal=%.4f sharpe=%.3f (present=%.3f / absent=%.3f)",
            name,
            metrics["focal"],
            metrics["sharpe"],
            metrics["sharpe_present"],
            metrics["sharpe_absent"],
        )
    return results
