"""Regression locks for calibrate hardening (V1-SPEC §38 / §39 / §57).

Covers:

- ``val_b`` and ``val_c`` cannot share memory (shared-tensor catches a
  caller bug where the same logits got passed for both).
- AgACI is replayed over val_c before save — its state must differ from
  a cold-init AgACI.
- ``meta.json`` carries the full reproducibility manifest.
- Atomic dir commit: a mid-write crash must leave the prior directory
  state untouched (or absent if first run).
- Optional Laplace path: ``laplace.pt`` only appears when
  ``cfg.fit_laplace=True`` and the head + loader are supplied.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import torch

from nanogld.calibration.agaci import AgACI
from nanogld.calibration.calibrate import (
    CalibrationConfig,
    calibrate,
    load_calibration,
)


def _toy_logits_labels(
    n_b: int = 64, n_c: int = 64, n_classes: int = 3, seed: int = 0
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(seed)
    b_logits = torch.randn(n_b, n_classes) * 1.5
    b_labels = torch.randint(0, n_classes, (n_b,))
    c_logits = torch.randn(n_c, n_classes) * 1.5
    c_labels = torch.randint(0, n_classes, (n_c,))
    return b_logits, b_labels, c_logits, c_labels


def test_val_b_val_c_shared_memory_rejected(tmp_path: Path) -> None:
    cfg = CalibrationConfig(fold_idx=0, output_dir=tmp_path)
    logits, labels, _, _ = _toy_logits_labels(n_b=16, n_c=16)
    with pytest.raises(ValueError, match="share memory"):
        calibrate(cfg, logits, labels, logits, labels)


def test_empty_val_set_rejected(tmp_path: Path) -> None:
    cfg = CalibrationConfig(fold_idx=0, output_dir=tmp_path)
    b_logits, b_labels, c_logits, c_labels = _toy_logits_labels()
    with pytest.raises(ValueError, match="non-empty"):
        calibrate(
            cfg,
            b_logits[:0],
            b_labels[:0],
            c_logits,
            c_labels,
        )


def test_agaci_state_changes_after_replay(tmp_path: Path) -> None:
    cfg = CalibrationConfig(fold_idx=1, output_dir=tmp_path)
    b_logits, b_labels, c_logits, c_labels = _toy_logits_labels()
    artifacts = calibrate(cfg, b_logits, b_labels, c_logits, c_labels)

    cold = AgACI(alpha_target=cfg.alpha_target)
    with open(artifacts.agaci_state_path) as f:
        saved_state = json.load(f)
    # Cold init has uniform weights and spread expert alphas; replayed
    # state should differ in at least one expert alpha or weight.
    cold_alphas = cold.expert_alphas
    saved_alphas = saved_state["expert_alphas"]
    cold_weights = cold.weights
    saved_weights = saved_state["weights"]
    delta_alpha = max(abs(a - b) for a, b in zip(cold_alphas, saved_alphas))
    delta_weight = max(abs(a - b) for a, b in zip(cold_weights, saved_weights))
    assert delta_alpha > 1e-9 or delta_weight > 1e-9


def test_meta_has_manifest(tmp_path: Path) -> None:
    cfg = CalibrationConfig(fold_idx=2, output_dir=tmp_path)
    b_logits, b_labels, c_logits, c_labels = _toy_logits_labels()
    artifacts = calibrate(cfg, b_logits, b_labels, c_logits, c_labels)
    with open(artifacts.meta_path) as f:
        meta = json.load(f)
    assert "manifest" in meta
    required = {
        "git_sha",
        "python_version",
        "torch_version",
        "platform",
        "hostname",
        "started_at_utc",
    }
    assert required <= set(meta["manifest"].keys())
    assert meta["agaci_n_replay_steps"] == c_labels.shape[0]


def test_atomic_dir_commit_rolls_back_on_crash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = CalibrationConfig(fold_idx=3, output_dir=tmp_path)
    final_dir = tmp_path / "calibration_3"
    assert not final_dir.exists()
    b_logits, b_labels, c_logits, c_labels = _toy_logits_labels()

    # The package __init__ shadows the submodule attribute with the
    # function of the same name, so reach into sys.modules directly to
    # monkey-patch the module-level `fit_raps_quantile` binding.
    import sys

    calmod = sys.modules["nanogld.calibration.calibrate"]

    def _explode(*_a: object, **_kw: object) -> None:
        raise RuntimeError("simulated mid-write crash")

    monkeypatch.setattr(calmod, "fit_raps_quantile", _explode)
    with pytest.raises(RuntimeError, match="mid-write"):
        calibrate(cfg, b_logits, b_labels, c_logits, c_labels)

    assert not final_dir.exists()
    assert not (tmp_path / "calibration_3.tmp").exists()


def test_load_calibration_returns_laplace_path_none_when_absent(
    tmp_path: Path,
) -> None:
    cfg = CalibrationConfig(fold_idx=4, output_dir=tmp_path)
    b_logits, b_labels, c_logits, c_labels = _toy_logits_labels()
    artifacts = calibrate(cfg, b_logits, b_labels, c_logits, c_labels)
    loaded = load_calibration(artifacts.t_scaler_path.parent)
    assert loaded["laplace_path"] is None
    assert loaded["laplace"] is None


def test_fit_laplace_requires_head_and_loader(tmp_path: Path) -> None:
    cfg = CalibrationConfig(fold_idx=5, output_dir=tmp_path, fit_laplace=True)
    b_logits, b_labels, c_logits, c_labels = _toy_logits_labels()
    with pytest.raises(ValueError, match="fit_laplace=True"):
        calibrate(cfg, b_logits, b_labels, c_logits, c_labels)


def test_existing_calibration_dir_replaced_cleanly(tmp_path: Path) -> None:
    cfg = CalibrationConfig(fold_idx=6, output_dir=tmp_path)
    b_logits, b_labels, c_logits, c_labels = _toy_logits_labels(seed=1)
    calibrate(cfg, b_logits, b_labels, c_logits, c_labels)
    first_run_meta = (tmp_path / "calibration_6" / "meta.json").read_text()

    b_logits2, b_labels2, c_logits2, c_labels2 = _toy_logits_labels(seed=2)
    a2 = calibrate(cfg, b_logits2, b_labels2, c_logits2, c_labels2)
    second_run_meta = a2.meta_path.read_text()

    # `started_at_utc` lives in meta.manifest and is wall-clock; two
    # consecutive runs must produce different meta files (proves the
    # second run overwrote the first rather than silently no-op'ing).
    assert first_run_meta != second_run_meta
    final_dir = tmp_path / "calibration_6"
    bak = tmp_path / "calibration_6.bak"
    tmp = tmp_path / "calibration_6.tmp"
    assert final_dir.exists()
    assert not bak.exists()
    assert not tmp.exists()
