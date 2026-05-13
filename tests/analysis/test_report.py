"""Unit tests for the analysis report aggregator (markdown formatting)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from nanogld.analysis.config import AnalysisConfig
from nanogld.analysis.report import write_report


pytestmark = pytest.mark.smoke


def test_write_report_full(tmp_path: Path) -> None:
    cfg = AnalysisConfig(
        checkpoint_path=tmp_path / "ckpt.pt",
        unified_path=tmp_path / "unified.pt",
        sidecar_path=tmp_path / "sidecar.pt",
        fold_idx=0,
        split="val_c",
        output_dir=tmp_path / "report",
        device="cpu",
    )
    feature_names = [f"f_{i:03d}" for i in range(8)]
    vsn = {
        "mean_gate": np.array([0.1] * 8, dtype=np.float32),
        "std_gate": np.array([0.01] * 8, dtype=np.float32),
        "mean_present": np.array([0.12] * 8, dtype=np.float32),
        "mean_absent": np.array([0.09] * 8, dtype=np.float32),
        "n_bars": np.array([1000], dtype=np.int64),
        "n_present": np.array([500], dtype=np.int64),
    }
    ig = {
        "mean_abs": np.array([0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.04, 0.03], dtype=np.float32),
        "mean_signed": np.array([0.5, -0.4, 0.3, -0.2, 0.1, -0.05, 0.04, -0.03], dtype=np.float32),
        "per_class_mean": np.zeros((3, 8), dtype=np.float32),
        "n_samples_seen": np.array([256], dtype=np.int64),
    }
    perm = {
        "feature_idx": np.arange(8, dtype=np.int64),
        "delta_focal_mean": np.array([0.01] * 8, dtype=np.float32),
        "delta_focal_std": np.array([0.001] * 8, dtype=np.float32),
        "delta_sharpe_mean": np.array([0.1, 0.05, 0.04, 0.03, 0.02, 0.01, 0.005, 0.001], dtype=np.float32),
        "delta_sharpe_std": np.array([0.001] * 8, dtype=np.float32),
        "baseline_focal": np.array([1.0], dtype=np.float32),
        "baseline_sharpe": np.array([1.2], dtype=np.float32),
        "n_batches": np.array([16], dtype=np.int64),
    }
    ablation = {
        "none": {"focal": 1.0, "sharpe": 1.2, "focal_present": 1.0, "focal_absent": 1.0, "sharpe_present": 1.3, "sharpe_absent": 1.1},
        "bars": {"focal": 1.5, "sharpe": 0.5, "focal_present": 1.5, "focal_absent": 1.5, "sharpe_present": 0.5, "sharpe_absent": 0.5},
        "news": {"focal": 1.1, "sharpe": 1.0, "focal_present": 1.2, "focal_absent": 1.0, "sharpe_present": 1.0, "sharpe_absent": 1.0},
        "regime": {"focal": 1.05, "sharpe": 1.15, "focal_present": 1.05, "focal_absent": 1.05, "sharpe_present": 1.2, "sharpe_absent": 1.1},
        "bars_news": {"focal": 1.6, "sharpe": 0.0, "focal_present": 1.6, "focal_absent": 1.6, "sharpe_present": 0.0, "sharpe_absent": 0.0},
    }
    attention = {
        "mean_per_slot": np.array([0.5, 0.2, 0.1, 0.05, 0.05, 0.04, 0.03, 0.03], dtype=np.float32),
        "mean_present_slot": np.array([0.6, 0.15, 0.1, 0.05, 0.04, 0.03, 0.02, 0.01], dtype=np.float32),
        "mean_absent_slot": np.array([0.95, 0.01, 0.01, 0.01, 0.01, 0.005, 0.005, 0.0], dtype=np.float32),
        "n_batches": np.array([4], dtype=np.int64),
        "n_bars": np.array([32], dtype=np.int64),
        "n_present": np.array([16], dtype=np.int64),
    }
    md_path = write_report(
        cfg=cfg,
        feature_names=feature_names,
        vsn=vsn,
        ig=ig,
        permutation=perm,
        ablation=ablation,
        attention=attention,
    )
    assert md_path.exists()
    text = md_path.read_text()
    assert "feature attribution report" in text
    assert "VSN gate" in text
    assert "Modality ablation" in text
    assert "Cross-attention rollout" in text
    assert (cfg.output_dir / "manifest.json").exists()


def test_write_report_skips_missing_methods(tmp_path: Path) -> None:
    cfg = AnalysisConfig(
        checkpoint_path=tmp_path / "ckpt.pt",
        unified_path=tmp_path / "unified.pt",
        sidecar_path=tmp_path / "sidecar.pt",
        fold_idx=0,
        output_dir=tmp_path / "rep2",
    )
    md_path = write_report(
        cfg=cfg,
        feature_names=[f"f_{i}" for i in range(4)],
        vsn=None,
        ig=None,
        permutation=None,
        ablation=None,
        attention=None,
    )
    assert md_path.exists()
    text = md_path.read_text()
    assert "skipped" in text.lower() or "no" in text.lower()
