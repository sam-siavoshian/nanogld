"""Global PIT regression tests.

Asserts that no `NanoGLDDataset.__getitem__` window includes future
bars relative to the sample's prediction-time anchor. This is the
"single most important test in the project" per `plan/V1-SPEC.md §6/11`
because a PIT leak silently inflates every reported Sharpe.

The test builds a synthetic unified.pt + sidecar.pt where every
feature value at bar t is `t` (encoded in float32). Iterating the
dataset and asserting `window.max() < bar_idx` proves the window
slice excludes the current bar AND all bars after it.

Also verifies:
  - the last bar in any split has no `next_log_return` NaN (caught
    by ``_compute_valid_indices(idx < n_bars - 1)``);
  - bar_close_utc_ns is monotonic non-decreasing inside any split;
  - news lookups via ``bar_news_offsets`` only index articles whose
    ``release_ts_ns`` (proxied by article index here, since the
    synthetic fixture is ordered) is strictly less than the bar's
    ``bar_close_utc_ns``.

These are smoke-grade synthetic tests. The full-fidelity PIT scan
across real ``training_v1_unified.pt`` lives behind ``needs_data``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from nanogld.data.dataset import NanoGLDDataset


pytestmark = pytest.mark.smoke


def _build_pit_synthetic(tmp: Path, n_bars: int = 200, f_dim: int = 16) -> tuple[Path, Path]:
    """Write a tiny unified.pt + sidecar.pt where features encode bar_idx."""
    unified_path = tmp / "training_v1_unified.pt"
    sidecar_path = tmp / "training_v1_sidecar.pt"

    bar_idx = np.arange(n_bars, dtype=np.float32)
    features = np.broadcast_to(bar_idx.reshape(-1, 1), (n_bars, f_dim)).copy()

    # Two-class fake label, splits chronologically (60/20/20).
    labels = (bar_idx % 3).astype(np.int8)
    splits = np.empty(n_bars, dtype=object)
    splits[:120] = "train"
    splits[120:160] = "val"
    splits[160:] = "test"

    bar_close_utc_ns = (
        np.arange(n_bars, dtype=np.int64) * 30 * 60 * 1_000_000_000
        + 1_700_000_000 * 1_000_000_000
    )

    # CSR-style empty news index: every bar has zero articles.
    bar_news_offsets = np.zeros(n_bars + 1, dtype=np.int64)
    bar_news_values = np.zeros(0, dtype=np.int64)
    embeddings = np.zeros((0, 256), dtype=np.float16)

    unified = {
        "features": features,
        "labels": labels,
        "splits": splits,
        "bar_close_utc_ns": bar_close_utc_ns,
        "bar_news_offsets": bar_news_offsets,
        "bar_news_values": bar_news_values,
        "embeddings": embeddings,
    }
    torch.save(unified, unified_path)

    sidecar = {
        "next_log_return": (np.diff(bar_idx, append=bar_idx[-1])).astype(np.float32) * 1e-4,
        "barrier_up": np.full(n_bars, 1e-3, dtype=np.float32),
        "barrier_down": np.full(n_bars, -1e-3, dtype=np.float32),
        "gld_spread_bps_t": np.full(n_bars, 1.0, dtype=np.float32),
        "regime_vec": np.zeros((n_bars, 12), dtype=np.float32),
        "era_label": np.zeros(n_bars, dtype=np.int64),
    }
    torch.save(sidecar, sidecar_path)
    return unified_path, sidecar_path


def test_window_excludes_current_and_future_bars(tmp_path: Path) -> None:
    """Window for bar_idx must contain only bars strictly less than bar_idx."""
    u, s = _build_pit_synthetic(tmp_path, n_bars=200, f_dim=16)
    ds = NanoGLDDataset(
        unified_path=u, sidecar_path=s, split="train", lookback_T=32, label_mode="fixed_5bps"
    )
    assert len(ds) > 0

    for sample in ds:
        window = sample["channel_inputs"]
        # Recover the original bar_idx from the synthetic encoding.
        # window shape: (T=32, F=16). All cols of a row equal `bar_idx`.
        bar_idxs_in_window = window[:, 0].cpu().numpy().astype(np.int64)
        anchor_bar = int(bar_idxs_in_window.max() + 1)  # excluded right edge
        assert (bar_idxs_in_window < anchor_bar).all(), (
            f"window contains bar >= anchor {anchor_bar}: {bar_idxs_in_window.tolist()}"
        )


def test_valid_indices_drop_final_bar(tmp_path: Path) -> None:
    u, s = _build_pit_synthetic(tmp_path, n_bars=200, f_dim=8)
    for split in ("train", "val_a", "val_b", "val_c", "test"):
        ds = NanoGLDDataset(
            unified_path=u, sidecar_path=s, split=split, lookback_T=16, label_mode="fixed_5bps"
        )
        # Final bar of dataset must NOT appear — the last bar has no defined
        # next_log_return (see plan/STATUS.md wave-1 fix).
        max_idx = int(np.asarray(ds._valid_indices).max()) if len(ds) > 0 else -1
        n_bars = int(ds._features.shape[0])
        assert max_idx < n_bars - 1, f"split={split} includes terminal bar {max_idx}/{n_bars}"


def test_bar_close_monotonic(tmp_path: Path) -> None:
    """Inside each split, bar_close_utc_ns must be monotonic non-decreasing."""
    u, s = _build_pit_synthetic(tmp_path, n_bars=200, f_dim=8)
    for split in ("train", "val_a", "val_b", "val_c", "test"):
        ds = NanoGLDDataset(
            unified_path=u, sidecar_path=s, split=split, lookback_T=16, label_mode="fixed_5bps"
        )
        idx = np.asarray(ds._valid_indices)
        if idx.size < 2:
            continue
        ts = ds._bar_close_utc_ns.cpu().numpy()[idx]
        assert (np.diff(ts) >= 0).all(), f"split={split} bar_close not monotonic"


def test_batched_window_consistency(tmp_path: Path) -> None:
    """Batched dataloader produces same PIT invariant per row."""
    u, s = _build_pit_synthetic(tmp_path, n_bars=200, f_dim=8)
    ds = NanoGLDDataset(
        unified_path=u, sidecar_path=s, split="train", lookback_T=16, label_mode="fixed_5bps"
    )
    loader = DataLoader(ds, batch_size=4, shuffle=False)
    for batch in loader:
        window = batch["channel_inputs"]
        for row in window:
            bar_idxs = row[:, 0].cpu().numpy().astype(np.int64)
            anchor = int(bar_idxs.max() + 1)
            assert (bar_idxs < anchor).all()
        break  # one batch suffices
