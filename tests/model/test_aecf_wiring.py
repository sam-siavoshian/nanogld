"""Regression locks for AECF curriculum mask wiring (V1-SPEC §2.3 / §2.5).

The AECFMask module is now instantiated inside each training stage
(simmtm_pretrain, linear_probe, llrd_finetune). These tests don't run
the full stages (those need data fixtures); they pin the module-level
contract that the stages depend on:

1. ``sample_mask`` returns shape (B,) with values in {0, 1} during
   training, all-ones outside training mode.
2. The curriculum ramp respects ``p_min`` (Stage 1: 0.0; Stages 2/3: 0.1)
   and reaches ``p_max=0.9`` by ``curriculum_steps``.
3. Sample-level zeroing fully suppresses news for that sample (news_mask
   zeroed AND is_news_present flipped). Verifies the multiply pattern
   used by every stage.
"""

from __future__ import annotations

import torch

from nanogld.model.aecf import AECFMask


def test_inference_mode_returns_all_ones() -> None:
    aecf = AECFMask(p_min=0.0, p_max=0.9, curriculum_steps=10)
    aecf.train(mode=False)
    mask = aecf.sample_mask(batch_size=32, training_step=10)
    assert torch.all(mask == 1.0)


def test_training_mask_binary() -> None:
    aecf = AECFMask(p_min=0.0, p_max=0.9, curriculum_steps=10)
    aecf.train()
    mask = aecf.sample_mask(batch_size=256, training_step=100)
    assert mask.dtype == torch.float32
    assert mask.shape == (256,)
    assert torch.all((mask == 0.0) | (mask == 1.0))


def test_curriculum_ramps_p_max() -> None:
    aecf = AECFMask(p_min=0.0, p_max=0.9, curriculum_steps=100)
    # At step 0 the effective p_max equals p_min (no drop).
    assert aecf.sample_p(training_step=0) == 0.0
    # At step >= curriculum_steps, effective_max == 0.9.
    samples = [aecf.sample_p(training_step=10_000) for _ in range(2000)]
    assert max(samples) <= 0.9 + 1e-6
    # Some samples should be in the upper half.
    assert sum(1 for s in samples if s > 0.45) > 100


def test_stage1_p_min_zero() -> None:
    """Stage 1 SSL: p_min=0.0 — can fully keep news."""
    aecf = AECFMask(p_min=0.0, p_max=0.9, curriculum_steps=10)
    samples = [aecf.sample_p(training_step=1_000) for _ in range(500)]
    assert min(samples) < 0.1


def test_stage23_p_min_floor() -> None:
    """Stages 2/3: p_min=0.1 — never fully keep news (minimum 10% drop)."""
    aecf = AECFMask(p_min=0.1, p_max=0.9, curriculum_steps=10)
    samples = [aecf.sample_p(training_step=1_000) for _ in range(500)]
    assert min(samples) >= 0.1 - 1e-6
    assert max(samples) <= 0.9 + 1e-6


def test_sample_level_zeroing_pattern() -> None:
    """Verifies the multiply pattern used by every stage to drop news."""
    aecf = AECFMask(p_min=0.0, p_max=1.0, curriculum_steps=10)
    aecf.train()
    torch.manual_seed(0)
    batch_size = 8
    news_mask = torch.ones(batch_size, 4)  # (B, S=4 slots)
    is_news_present = torch.ones(batch_size, dtype=torch.long)

    aecf_keep = aecf.sample_mask(batch_size=batch_size, training_step=10_000)
    new_news_mask = news_mask * aecf_keep.unsqueeze(-1)
    new_is_news_present = (is_news_present.float() * aecf_keep).to(is_news_present.dtype)

    for i in range(batch_size):
        if aecf_keep[i] == 0.0:
            assert torch.all(new_news_mask[i] == 0.0)
            assert new_is_news_present[i] == 0
        else:
            assert torch.all(new_news_mask[i] == 1.0)
            assert new_is_news_present[i] == 1


def test_aecf_imported_in_all_three_stages() -> None:
    """Static import check — catches accidental future removal."""
    import nanogld.training.linear_probe as probe
    import nanogld.training.llrd_finetune as llrd
    import nanogld.training.simmtm_pretrain as ssl

    assert hasattr(ssl, "AECFMask")
    assert hasattr(probe, "AECFMask")
    assert hasattr(llrd, "AECFMask")
