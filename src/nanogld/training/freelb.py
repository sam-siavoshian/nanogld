"""FreeLB adversarial perturbation on news embeddings.

Zhu et al. ICLR 2020 (arXiv:1909.11764): K-step PGD ascent in the
embedding space, gradient-shared across steps so total cost is ~Kx
without K independent backward passes.

V1 applies FreeLB ONLY to the 256-d Qwen3 news embeddings (NEVER to
bars — PIT correctness risk). K=2, ε=0.5 per V1-SPEC.

Usage pattern:
    freelb = FreeLB(K=2, epsilon=0.5)
    loss = freelb.compute(model, batch, loss_fn)

The wrapper expects `batch` to be a dict with at least `news_embeddings`
key. The model is called via `loss_fn(model_outputs, batch)`.

Spec: plan/05-MODEL-TRAINING-CALIBRATION.md V1 regularization.
Spec: plan/V1-SPEC.md §6.6.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import Tensor


class FreeLB:
    """K-step adversarial perturbation on news embeddings.

    Args:
        K: number of inner ascent steps (2 default).
        epsilon: max perturbation L-inf norm (0.5 default).
        alpha: per-step learning rate. None -> epsilon / K.
    """

    def __init__(
        self,
        K: int = 2,
        epsilon: float = 0.5,
        alpha: float | None = None,
    ) -> None:
        self.K = K
        self.epsilon = epsilon
        self.alpha = alpha if alpha is not None else epsilon / max(1, K)

    def compute_loss(
        self,
        model_forward: Callable[..., dict[str, Tensor]],
        batch: dict[str, Tensor],
        loss_fn: Callable[[dict[str, Tensor], dict[str, Tensor]], Tensor],
    ) -> Tensor:
        """Run K-step FreeLB on the news embedding and return averaged loss.

        Args:
            model_forward: callable taking the batch (with possibly perturbed
                news_embeddings) and returning a dict of model outputs.
            batch: dict with at least `news_embeddings` (B, S, D_text).
            loss_fn: callable(outputs, batch) -> scalar loss.

        Returns:
            scalar Tensor — mean of K-step losses, with gradients accumulated
            over all K steps. Caller still needs to call `loss.backward()`
            and step the optimizer.
        """
        if self.K <= 0:
            outputs = model_forward(**batch)
            return loss_fn(outputs, batch)

        original_news = batch["news_embeddings"]
        delta = torch.zeros_like(original_news, requires_grad=True)
        loss_acc: Tensor = torch.tensor(0.0, device=original_news.device)

        for step in range(self.K):
            perturbed_batch = dict(batch)
            perturbed_batch["news_embeddings"] = original_news + delta

            outputs = model_forward(**perturbed_batch)
            step_loss = loss_fn(outputs, batch) / self.K
            loss_acc = loss_acc + step_loss

            if step < self.K - 1:
                delta_grad = torch.autograd.grad(
                    step_loss, delta, retain_graph=True, create_graph=False
                )[0]
                with torch.no_grad():
                    delta = delta + self.alpha * delta_grad.sign()
                    delta = torch.clamp(delta, min=-self.epsilon, max=self.epsilon)
                delta = delta.detach().requires_grad_(True)

        return loss_acc
