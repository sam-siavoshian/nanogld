"""Unit tests for real-form RoPE."""

from __future__ import annotations

import pytest
import torch

from nanogld.model.rope import apply_partial_rope, precompute_rope_cache


@pytest.mark.smoke
def test_cache_shape() -> None:
    cos, sin = precompute_rope_cache(head_dim=64, max_seq=128)
    assert cos.shape == (128, 32)
    assert sin.shape == (128, 32)


@pytest.mark.smoke
def test_apply_preserves_shape() -> None:
    cos, sin = precompute_rope_cache(head_dim=64, max_seq=16)
    x = torch.randn(2, 4, 16, 64)
    y = apply_partial_rope(x, cos, sin, frac=0.10)
    assert y.shape == x.shape


@pytest.mark.smoke
def test_partial_leaves_remainder_untouched() -> None:
    """At frac=0.10 of head_dim=64, only first 6 dims (3 pairs) rotate."""
    cos, sin = precompute_rope_cache(head_dim=64, max_seq=8)
    x = torch.randn(2, 4, 8, 64)
    y = apply_partial_rope(x, cos, sin, frac=0.10)
    rot_pairs = max(1, int((0.10 * 64) // 2))
    rot_dim = 2 * rot_pairs
    torch.testing.assert_close(y[..., rot_dim:], x[..., rot_dim:])


@pytest.mark.smoke
def test_view_as_complex_not_called() -> None:
    """V1 invariant: never CALL torch.view_as_complex on MPS-bound code."""
    import inspect
    from nanogld.model import rope as rope_mod

    src = inspect.getsource(rope_mod)
    code_lines = []
    in_docstring = False
    for line in src.splitlines():
        stripped = line.strip()
        if stripped.startswith('"""') or stripped.endswith('"""'):
            in_docstring = not in_docstring if stripped.count('"""') == 1 else in_docstring
            continue
        if in_docstring:
            continue
        if stripped.startswith("#"):
            continue
        code_lines.append(line)
    code_only = "\n".join(code_lines)
    assert "view_as_complex(" not in code_only


@pytest.mark.smoke
def test_zero_frac_uses_at_least_one_pair() -> None:
    """`frac=0` should still rotate at least 1 pair (max(1, ...) guard)."""
    cos, sin = precompute_rope_cache(head_dim=64, max_seq=8)
    x = torch.randn(1, 1, 8, 64)
    y = apply_partial_rope(x, cos, sin, frac=0.0)
    assert not torch.equal(y[..., :2], x[..., :2])


@pytest.mark.smoke
def test_odd_head_dim_raises() -> None:
    with pytest.raises(ValueError):
        precompute_rope_cache(head_dim=63, max_seq=8)
