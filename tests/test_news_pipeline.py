"""News pipeline tests — license gating, schema conformity, source-specific
edge cases. Live-data tests skip when raw parquets are missing.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from nanogld.data import news_fnspid, news_multisource, news_polygon, wayback_helpers
from nanogld.data.schema import NEWS_MANIFEST
from nanogld.data.utils import NEWS_LATENCY_MIN_GDELT, NEWS_LATENCY_SEC_ALPACA

NEEDS_RAW = pytest.mark.skipif(
    not (Path("data/raw/central_bank_news.parquet").exists()),
    reason="needs central_bank pull first",
)


# ────────────────────────────────────────────────────────────────────────────
# License gating
# ────────────────────────────────────────────────────────────────────────────


def test_fnspid_license_string_correct() -> None:
    """V4 finding: docstring originally said CC BY 4.0. Real license: CC BY-NC-4.0."""
    src = Path("src/nanogld/data/news_fnspid.py").read_text()
    assert "CC BY-NC-4.0" in src or "CC-BY-NC-4.0" in src
    # Old wrong string should be gone.
    assert "License: CC BY 4.0" not in src


def test_fnspid_gated_off_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without NANOGLD_NONCOMMERCIAL=1, FNSPID returns empty + logs warning."""
    monkeypatch.delenv("NANOGLD_NONCOMMERCIAL", raising=False)
    df = news_fnspid.fetch_filtered()
    assert df.empty


def test_fnspid_gate_passes_when_flag_set(monkeypatch: pytest.MonkeyPatch) -> None:
    """With the flag set, gate returns True so the actual fetch runs."""
    monkeypatch.setenv("NANOGLD_NONCOMMERCIAL", "1")
    assert news_fnspid._noncommercial_gate_open() is True


def test_multisource_gated_off_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("NANOGLD_NONCOMMERCIAL", raising=False)
    df = news_multisource.fetch_filtered()
    assert df.empty


def test_polygon_news_gated_off_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """Polygon free tier may not include news endpoint — gate prevents
    accidental paid-endpoint hit.
    """
    monkeypatch.delenv("NANOGLD_POLYGON_PAID", raising=False)
    df = news_polygon.fetch_news()
    assert df.empty


# ────────────────────────────────────────────────────────────────────────────
# Wayback helpers
# ────────────────────────────────────────────────────────────────────────────


def test_wayback_polite_default_2s() -> None:
    assert wayback_helpers.DEFAULT_POLITE_SEC == 2.0


def test_wayback_hard_halt_after_5() -> None:
    """5 consecutive throttle responses must halt the run cleanly."""
    assert wayback_helpers.DEFAULT_HARD_HALT_AFTER == 5


def test_wayback_cache_dir_created(
    tmp_path: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    """wayback_helpers cache dir is auto-created under data/raw/wayback_cache/."""
    monkeypatch.setenv("NANOGLD_DATA_ROOT", str(tmp_path))
    p = wayback_helpers._cache_dir("test_source")
    assert p.exists()
    assert p.name == "test_source"


# ────────────────────────────────────────────────────────────────────────────
# Schema conformity
# ────────────────────────────────────────────────────────────────────────────


def test_news_manifest_columns_canonical() -> None:
    """All news modules promise the same NEWS_MANIFEST schema."""
    cols = {c.name for c in NEWS_MANIFEST.columns}
    expected = {
        "article_id",
        "source",
        "created_at",
        "title",
        "body",
        "url",
        "symbols",
        "bias_tier",
        "release_ts",
        "t_visible",
    }
    assert cols == expected


def test_news_t_visible_buffers_per_source() -> None:
    """Spec V4 §2 + §7: per-source visibility buffers."""
    assert NEWS_LATENCY_SEC_ALPACA == 60
    assert NEWS_LATENCY_MIN_GDELT == 30


# ────────────────────────────────────────────────────────────────────────────
# Live-data sanity (skip when parquets missing)
# ────────────────────────────────────────────────────────────────────────────


@NEEDS_RAW
def test_central_bank_has_v1_window_rows() -> None:
    """ECB+FED HF dataset must yield ≥1 row in the V1 window 2021-04 → 2026-04."""
    df = pd.read_parquet("data/raw/central_bank_news.parquet")
    in_w = df[
        (df["created_at"] >= pd.Timestamp("2021-04-24", tz="UTC"))
        & (df["created_at"] <= pd.Timestamp("2026-04-24", tz="UTC"))
    ]
    assert len(in_w) > 100  # 4988 total, ~683 in window


@NEEDS_RAW
def test_no_collisions_across_news_parquets() -> None:
    """(source, article_id) must be unique across every news parquet."""
    raw = Path("data/raw")
    frames = []
    for name in (
        "alpaca_news_GLD.parquet",
        "central_bank_news.parquet",
        "kitco_news.parquet",
        "investing_gold_news.parquet",
        "bullionvault_news.parquet",
        "fnspid_gold_relevant.parquet",
        "alpha_vantage_news.parquet",
        "polygon_news_GLD.parquet",
        "multisource_news.parquet",
        "reddit_gold_filtered.parquet",
        "kaggle_gold_labeled.parquet",
    ):
        p = raw / name
        if p.exists():
            df = pd.read_parquet(p)
            if {"source", "article_id"}.issubset(df.columns):
                frames.append(df[["source", "article_id"]])
    if not frames:
        pytest.skip("no news parquets pulled yet")
    all_news = pd.concat(frames, ignore_index=True)
    dup_count = all_news.groupby(["source", "article_id"]).size().pipe(lambda s: (s > 1).sum())
    assert dup_count == 0, f"{dup_count} duplicate (source, article_id) pairs found"


@NEEDS_RAW
def test_pit_invariant_on_all_news_parquets() -> None:
    """Universal V1 hard rule: release_ts <= t_visible on every news row."""
    raw = Path("data/raw")
    for p in raw.glob("*.parquet"):
        df = pd.read_parquet(p)
        if "release_ts" not in df.columns or "t_visible" not in df.columns:
            continue
        bad = (df["release_ts"] > df["t_visible"]).sum()
        assert bad == 0, f"{p.name}: {bad} rows violate release_ts <= t_visible"
