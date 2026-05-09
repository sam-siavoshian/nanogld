# nanoGLD V1 — feature attribution suite

Six methods that together answer: **which features actually drive the model's gold-direction predictions?**

## Methods

1. **VSN gate importance** (`vsn_importance.py`) — free, native to architecture. The Variable Selection Network already produces a per-bar softmax distribution over the 681 features. Mean over an eval split = importance ranking.
2. **Integrated Gradients** (`integrated_gradients.py`) — captum-based path-integral attribution wrt the `channel_inputs` tensor. Reports per-class signed attribution + mean-abs aggregate.
3. **Permutation importance** (`permutation.py`) — model-agnostic ground truth. Shuffles each feature column across the batch axis, measures Δfocal-loss + ΔSharpe. Capped to the top-N features (by VSN gate) to fit the calibration-time budget.
4. **Modality ablation** (`modality_ablation.py`) — zeroes each input stream (bars / news / regime / bars+news) and measures the metric drop. Splits results by news-presence bucket per V1 invariant 18.
5. **Cross-attention rollout** (`attention_rollout.py`) — re-computes the softmax weights inside the NewsFuser layers to expose which news slots get attended to, split by presence bucket.
6. **Feature-group rollups** (`feature_groups.py`) — categorizes the 681 features into 9 buckets (price / volatility / macro / calendar / regime / news / flow / rates / other) and sums per-feature importance into per-category summaries.

## Why not SHAP

- **DeepSHAP / DeepLIFT**: requires layer-by-layer support; our model has SwiGLU, sLSTM, RoPE, GroupNorm, custom RMSNorm — captum's `DeepLift` falls back to gradients on unsupported layers, which converges to IG's signal. We use IG directly.
- **KernelSHAP**: O(n_features × n_perturbations × n_samples) = impractical at 681 × thousands × hundreds. Permutation importance gives the same ranking signal at much lower cost.
- **GradientShap** (also captum): viable alternative; we keep the simpler IG path for V1 and revisit if numerical signal is weak.

## CLI

```bash
python -m nanogld.analysis run \
    --checkpoint checkpoints/v1/fold_0/llrd/llrd_final.pt \
    --unified   data/processed/training_v1_unified.pt \
    --sidecar   data/processed/training_v1_sidecar.pt \
    --fold      0 \
    --split     val_c \
    --output-dir reports/analysis/fold_0 \
    --device    auto
```

Wall-clock budget per fold (val_c ≈ 4-5K bars): ~10-15 min on H100 with the default settings (256 IG samples × 32 steps + 100 features × 3 perm reps). All artifacts written atomically (tmp + os.replace) and include a manifest with git SHA, hostname, run hash for reproducibility.

## Outputs

```
reports/analysis/fold_0/
├── analysis_<run_hash>_<git_sha>.md   # the headline report
├── feature_importance.parquet         # combined per-feature row
├── vsn_gate_distribution.parquet      # raw VSN gate stats
├── integrated_gradients.parquet       # per-class IG
├── permutation_importance.parquet     # ΔSharpe per shuffle
├── modality_ablation.json             # per-stream Sharpe deltas
├── attention_rollout.json             # per-news-slot attention
└── manifest.json                      # repro manifest
```

## Spec

`plan/V1-SPEC.md §11` (post-train interpretability — added 2026-05-08).
