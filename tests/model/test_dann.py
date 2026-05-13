"""Regression locks for DANN domain classifier wiring (V1-SPEC §6.7).

The DANN head is now attached to ``MultiTaskHead`` and wired into both
SSL (simmtm_pretrain) and Stage 3 LLRD (llrd_finetune) losses with
gradient reversal. These tests pin the contract at the module level:

1. ``MultiTaskHead.dann_head`` exists with shape ``(d_model, num_eras=4)``.
2. ``dann_forward`` applies ``grad_reverse`` — gradient sign flips on the
   pooled rep with respect to the dann_head's loss.
3. SSL + LLRD imports include ``dann_loss``.
4. Default DANN config values match V1-SPEC §6.7 (weight 0.05, max alpha 0.1).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from nanogld.model.heads import DEFAULT_NUM_ERAS, MultiTaskHead


def test_dann_head_exists_and_shape() -> None:
    head = MultiTaskHead(d_model=32, n_classes=3)
    assert hasattr(head, "dann_head")
    assert head.dann_head.weight.shape == (DEFAULT_NUM_ERAS, 32)


def test_dann_forward_basic_shape() -> None:
    head = MultiTaskHead(d_model=32, n_classes=3)
    pooled = torch.randn(8, 32, requires_grad=True)
    logits = head.dann_forward(pooled, alpha=0.5)
    assert logits.shape == (8, DEFAULT_NUM_ERAS)


def test_grad_reverse_flips_pooled_gradient() -> None:
    """With grad reversal, the gradient on pooled w.r.t. CE loss must point
    AWAY from the classifier's preferred direction.

    Concrete check: compute domain CE loss without GRL (alpha=0) vs with
    GRL (alpha=1). The gradient on `pooled` must change sign.
    """
    torch.manual_seed(0)
    head = MultiTaskHead(d_model=8, n_classes=3)
    pooled = torch.randn(4, 8, requires_grad=True)
    era_label = torch.tensor([0, 1, 2, 3], dtype=torch.long)

    # alpha=0: no reversal — gradient sign matches normal classifier descent.
    logits_no_grl = head.dann_forward(pooled, alpha=0.0)
    loss_no_grl = F.cross_entropy(logits_no_grl, era_label)
    grad_no_grl = torch.autograd.grad(loss_no_grl, pooled, retain_graph=False)[0]
    # alpha=0 produces zero grad (multiplied by 0). Skip and verify alpha=1.

    # Reset and use alpha=1.
    pooled2 = pooled.detach().clone().requires_grad_(True)
    logits_grl = head.dann_forward(pooled2, alpha=1.0)
    loss_grl = F.cross_entropy(logits_grl, era_label)
    grad_grl = torch.autograd.grad(loss_grl, pooled2, retain_graph=False)[0]

    # Without GRL on the SAME network: compute grad by bypassing dann_forward.
    pooled3 = pooled.detach().clone().requires_grad_(True)
    logits_plain = head.dann_head(pooled3)
    loss_plain = F.cross_entropy(logits_plain, era_label)
    grad_plain = torch.autograd.grad(loss_plain, pooled3, retain_graph=False)[0]

    # GRL flips the gradient sign on pooled.
    assert torch.allclose(grad_grl, -grad_plain, atol=1e-6)
    # alpha=0 means zero gradient flow
    assert grad_no_grl.abs().max().item() < 1e-6


def test_dann_imported_in_ssl_and_llrd() -> None:
    """Static import check."""
    import nanogld.training.llrd_finetune as llrd
    import nanogld.training.simmtm_pretrain as ssl

    assert hasattr(ssl, "dann_loss")
    assert hasattr(llrd, "dann_loss")


def test_config_defaults_match_spec() -> None:
    """V1-SPEC §6.7: dann_weight=0.05, dann_max_alpha=0.1."""
    from nanogld.training.llrd_finetune import LLRDConfig
    from nanogld.training.simmtm_pretrain import SimMTMConfig

    ssl_cfg = SimMTMConfig()
    llrd_cfg = LLRDConfig()
    assert ssl_cfg.dann_weight == 0.05
    assert ssl_cfg.dann_max_alpha == 0.1
    assert llrd_cfg.dann_weight == 0.05
    assert llrd_cfg.dann_max_alpha == 0.1
