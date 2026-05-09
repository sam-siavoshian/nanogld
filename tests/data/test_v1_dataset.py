"""Unit tests for NanoGLDDataset (synthetic minimal fixtures)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from nanogld.data.dataset import NanoGLDDataset


def _build_synthetic_unified(tmp_path: Path, n_bars: int = 256) -> Path:
    """Build a tiny unified.pt for tests."""
    rng = np.random.default_rng(0)
    feature_dim = 8
    n_articles = 16
    embed_dim = 16

    splits = np.array(
        ["train"] * (n_bars // 2)
        + ["val"] * (n_bars // 4)
        + ["test"] * (n_bars - n_bars // 2 - n_bars // 4),
        dtype=object,
    )
    base_ts = np.arange(n_bars, dtype=np.int64) * (30 * 60 * 1_000_000_000)
    base_ts += np.datetime64("2022-01-01", "ns").astype(np.int64)

    bar_news_offsets = np.zeros(n_bars + 1, dtype=np.int64)
    bar_news_values_list = []
    for i in range(n_bars):
        n_news_for_bar = int(rng.integers(0, 3))
        if n_news_for_bar > 0:
            picks = rng.integers(0, n_articles, size=n_news_for_bar)
            bar_news_values_list.extend(picks.tolist())
        bar_news_offsets[i + 1] = bar_news_offsets[i] + n_news_for_bar

    bar_news_values = np.array(bar_news_values_list, dtype=np.int64)
    payload = {
        "features": rng.standard_normal((n_bars, feature_dim)).astype(np.float32),
        "labels": rng.integers(0, 3, size=n_bars).astype(np.int8),
        "splits": splits,
        "bar_close_utc_ns": base_ts,
        "bar_news_offsets": bar_news_offsets,
        "bar_news_values": bar_news_values,
        "embeddings": rng.standard_normal((n_articles, embed_dim)).astype(np.float16),
        "feature_names": [f"f{i}" for i in range(feature_dim)],
        "article_ids": [f"a{i}" for i in range(n_articles)],
        "meta": {"schema_version": "test"},
    }
    out = tmp_path / "unified.pt"
    torch.save(payload, out)
    return out


@pytest.mark.smoke
def test_dataset_len_and_item_shapes(tmp_path: Path) -> None:
    unified = _build_synthetic_unified(tmp_path, n_bars=128)
    ds = NanoGLDDataset(
        unified_path=unified,
        sidecar_path=None,
        split="train",
        lookback_T=8,
        n_news_slots=4,
        label_mode="fixed_5bps",
    )
    assert len(ds) > 0
    item = ds[0]
    assert item["channel_inputs"].shape == (8, 8)
    assert item["news_embeddings"].shape == (4, 16)
    assert item["news_mask"].shape == (4,)
    assert item["regime_vec"].shape == (12,)
    assert item["label_3class"].dtype == torch.long
    assert item["era_label"].dtype == torch.long


@pytest.mark.smoke
def test_dataset_split_modes(tmp_path: Path) -> None:
    unified = _build_synthetic_unified(tmp_path, n_bars=400)
    sizes = {}
    for split in ("train", "val_a", "val_b", "val_c", "test"):
        ds = NanoGLDDataset(
            unified_path=unified,
            sidecar_path=None,
            split=split,
            lookback_T=4,
            label_mode="fixed_5bps",
        )
        sizes[split] = len(ds)
    assert sizes["train"] > 0
    assert sizes["val_a"] + sizes["val_b"] + sizes["val_c"] > 0


@pytest.mark.smoke
def test_dataset_lookback_truncates_early_bars(tmp_path: Path) -> None:
    unified = _build_synthetic_unified(tmp_path, n_bars=200)
    ds = NanoGLDDataset(
        unified_path=unified,
        sidecar_path=None,
        split="train",
        lookback_T=64,
        label_mode="fixed_5bps",
    )
    assert all(int(idx) >= 64 for idx in ds._valid_indices)


@pytest.mark.smoke
def test_dataset_invalid_split_raises(tmp_path: Path) -> None:
    unified = _build_synthetic_unified(tmp_path, n_bars=64)
    with pytest.raises(ValueError):
        NanoGLDDataset(
            unified_path=unified,
            sidecar_path=None,
            split="banana",
            lookback_T=4,
            label_mode="fixed_5bps",
        )


@pytest.mark.smoke
def test_dataset_missing_unified_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        NanoGLDDataset(
            unified_path=tmp_path / "missing.pt",
            split="train",
            lookback_T=4,
            label_mode="fixed_5bps",
        )
