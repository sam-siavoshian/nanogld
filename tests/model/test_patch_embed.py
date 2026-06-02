"""Unit tests for PatchEmbed (channel-independent patching)."""

from __future__ import annotations

import pytest
import torch

from nanogld.model.patch_embed import PatchEmbed


@pytest.mark.smoke
def test_output_shape() -> None:
    p = PatchEmbed(patch_len=4, patch_stride=4, t_bars=64, d_model=384)
    x = torch.randn(2, 64, 681)
    y = p(x)
    assert y.shape == (2 * 681, 16, 384)


@pytest.mark.smoke
def test_pos_emb_buffer_shape() -> None:
    p = PatchEmbed(patch_len=4, patch_stride=4, t_bars=64, d_model=384)
    assert p.pos_emb.shape == (1, 16, 384)


@pytest.mark.smoke
def test_proj_bias_free() -> None:
    p = PatchEmbed(patch_len=4, patch_stride=4, t_bars=64, d_model=384)
    assert p.proj.bias is None


@pytest.mark.smoke
def test_invalid_t_bars_raises() -> None:
    with pytest.raises(ValueError):
        PatchEmbed(patch_len=4, patch_stride=4, t_bars=63, d_model=384)


@pytest.mark.smoke
def test_input_t_mismatch_raises() -> None:
    p = PatchEmbed(patch_len=4, patch_stride=4, t_bars=64, d_model=384)
    x = torch.randn(2, 60, 681)
    with pytest.raises(ValueError):
        p(x)


@pytest.mark.smoke
def test_channel_independence() -> None:
    """Output for channel c at batch b sits at row b*num_channels + c."""
    p = PatchEmbed(patch_len=4, patch_stride=4, t_bars=8, d_model=16)
    x = torch.zeros(1, 8, 3)
    x[:, :, 1] = torch.arange(8, dtype=torch.float32)
    y = p(x)
    ch0 = y[0]
    ch1 = y[1]
    ch2 = y[2]
    assert not torch.allclose(ch0, ch1)
    torch.testing.assert_close(ch0, ch2)
