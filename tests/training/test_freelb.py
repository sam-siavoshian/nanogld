"""Unit tests for FreeLB adversarial wrapper."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from nanogld.training.freelb import FreeLB


@pytest.mark.smoke
def test_freelb_k_zero_returns_clean_loss() -> None:
    fl = FreeLB(K=0, epsilon=0.5)

    def model_forward(news_embeddings, **kw):
        return {"logits": news_embeddings.sum(dim=(1, 2))}

    def loss_fn(outputs, batch):
        return outputs["logits"].mean()

    batch = {"news_embeddings": torch.randn(2, 4, 8)}
    loss = fl.compute_loss(model_forward, batch, loss_fn)
    assert torch.isfinite(loss)


@pytest.mark.smoke
def test_freelb_runs_k2_no_crash() -> None:
    fl = FreeLB(K=2, epsilon=0.5, alpha=0.25)
    linear = torch.nn.Linear(8, 3)
    target = torch.randint(0, 3, (4,))

    def model_forward(news_embeddings, **kw):
        return {"logits": linear(news_embeddings.mean(dim=1))}

    def loss_fn(outputs, batch):
        return F.cross_entropy(outputs["logits"], target)

    batch = {"news_embeddings": torch.randn(4, 5, 8, requires_grad=False)}
    loss = fl.compute_loss(model_forward, batch, loss_fn)
    assert torch.isfinite(loss)
    assert loss.requires_grad
