"""Post-training feature attribution + interpretability for nanoGLD V1.

Six methods, each producing parquet + json artifacts:
    1. VSN gate importance        (vsn_importance.py)
    2. Integrated Gradients       (integrated_gradients.py)
    3. Permutation importance     (permutation.py)
    4. Modality ablation          (modality_ablation.py)
    5. Cross-attn rollout         (attention_rollout.py)
    6. Feature group rollups      (feature_groups.py)

Aggregator: report.py renders a single markdown report per fold.

CLI: `python -m nanogld.analysis run --checkpoint <path> --config <yaml>`.

Spec: plan/V1-SPEC.md §11 (post-train interpretability — added 2026-05-08).
"""

from nanogld.analysis.config import AnalysisConfig

__all__ = ["AnalysisConfig"]
