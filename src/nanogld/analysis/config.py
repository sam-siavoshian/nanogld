"""Frozen config dataclass for the post-training feature analysis stage."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class AnalysisConfig:
    """Configuration for the analysis CLI.

    Args:
        checkpoint_path: path to llrd_final.pt (or any model state_dict).
        unified_path: path to training_v1_unified.pt for eval features.
        sidecar_path: per-fold sidecar with regime + barriers.
        fold_idx: which walk-forward fold this analysis is for.
        split: which split to attribute on. Default "val_c" (held-out
            calibration tail; doesn't double-dip the calibration set).
        output_dir: dir for analysis artifacts. One subdir per method.
        n_samples_ig: Integrated Gradients sample budget.
        n_steps_ig: number of integration steps per sample.
        n_perm_repeats: permutation importance repeats per feature.
        max_features_perm: cap on features to permute (top-N by VSN gate).
        attribution_baseline: "zero" | "mean" — IG baseline strategy.
        seed: RNG seed for sampling reproducibility.
        device: torch device.

    Spec: plan/V1-SPEC.md §11.
    """

    checkpoint_path: Path
    unified_path: Path
    sidecar_path: Path
    fold_idx: int = 0
    split: str = "val_c"
    output_dir: Path = field(default_factory=lambda: Path("reports/analysis"))
    n_samples_ig: int = 256
    n_steps_ig: int = 32
    n_perm_repeats: int = 3
    max_features_perm: int = 100
    attribution_baseline: str = "zero"
    seed: int = 42
    device: str = "cpu"

    def run_hash(self) -> str:
        """Stable 8-char hex hash over hashable fields (excluding paths).

        Mirrors the embed/config.py pattern used elsewhere in the repo.
        """
        payload = {
            "fold_idx": int(self.fold_idx),
            "split": str(self.split),
            "n_samples_ig": int(self.n_samples_ig),
            "n_steps_ig": int(self.n_steps_ig),
            "n_perm_repeats": int(self.n_perm_repeats),
            "max_features_perm": int(self.max_features_perm),
            "attribution_baseline": str(self.attribution_baseline),
            "seed": int(self.seed),
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:8]
