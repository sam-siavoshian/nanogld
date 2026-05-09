# nanoGLD

[![License](https://img.shields.io/badge/license-MIT-green.svg)](./LICENSE)
[![PyTorch](https://img.shields.io/badge/pytorch-2.11-EE4C2C.svg)](https://pytorch.org)
[![Python](https://img.shields.io/badge/python-3.11-3776AB.svg)](https://www.python.org)

A small transformer that predicts the next 30-minute direction of GLD (gold ETF) using price bars and frozen LLM news embeddings. From-scratch hybrid encoder, 24M params, trained on 10 years of intraday data.

Honest target: **1.0–1.5 out-of-sample Sharpe net of 2bp costs**, beating the simpler Gao 2014 + XGBoost ensemble by at least 0.2.

## Status

V1 implementation built and bug-hunted (101 fixes across 7 audit waves). Post-train feature attribution suite shipped. **Not yet shipped to H100**: a few blockers in `plan/STATUS.md` first (per-fold sidecar leak, RunPod script hardening, AECF mask wiring).

## Architecture

```
bars (T=64 × F=681)               news (Qwen3-4B FP16, 8 slots/bar)
       │                                     │
   RevIN + VSN                            CFA + AECF
       │                                     │
   patches P=4 ───► encoder (10 transformer + 2 sLSTM) ◄─── cross-attn at {3,7}
                                │
                                ▼
                       focal CE head (3-class)  +  Sharpe head (position weight)
                                │
                                ▼
                  T-scaling → RAPS → AgACI → Kelly + ATR exits + DD breaker
```

Why this stack: focal loss matches conformal calibration cleanly, Sharpe head turns prediction into sizing, sparse cross-attention only fuses news where it matters. See `plan/V1-SPEC.md`.

## Layout

```
src/nanogld/
  model/        # encoder, attention, sLSTM, FiLM, RoPE, RMSNorm, SwiGLU, VSN, CFA
  training/     # SSL + linear probe + LLRD, Mixout, FreeLB, EMA
  calibration/  # T-scaling, RAPS, AgACI, Laplace, ECE
  sizing/       # Kelly, vol-target, ATR exits, drawdown breaker, conformal floor
  backtest/     # engine, metrics, DSR, cost-stress, per-bucket, baselines/
  analysis/     # 6-method feature attribution (VSN, IG, perm, ablation, attn, groups)
  data/         # NanoGLDDataset
  features/     # h5, spread, ATR, regime, HMM, triple-barrier
plan/           # V1-SPEC, 00–08, HANDOFF, STATUS
tests/          # one test dir per src module
```

## Quick start

```bash
# install
uv sync --frozen

# train one fold (autodetect cuda/mps; rejects cpu silently)
uv run python -m nanogld.training run \
    --config configs/v1_main.yaml --fold 0 --device auto

# post-train feature attribution
uv run python -m nanogld.analysis run \
    --checkpoint checkpoints/v1/fold_0/llrd/llrd_final.pt \
    --unified data/processed/training_v1_unified.pt \
    --sidecar data/processed/training_v1_sidecar_fold_0.pt \
    --fold 0 --split val_c --device auto

# tests
uv run pytest -q
```

## Data

75,993 bars × 681 features + 40,032 news embeddings + per-fold sidecar (HMM regime + ATR barriers + spread + h5). 30-min NYSE RTH bars, 2016 → 2026.

Build pipeline: `scripts/build_v1_sidecar.py`. Upload: `scripts/upload_data_to_hf.py`.

## Eval gates

8 promotion gates before ship:

1. OOS Sharpe ≥ 1.0 net of 2bp (4-fold walk-forward).
2. Sharpe > 0.5 at 1.5× cost stress.
3. Beats Gao 2014 + XGBoost ensemble by ≥ 0.2 Sharpe.
4. Deflated Sharpe Ratio > 1.0.
5. Per-bucket Sharpe (news-present / news-absent / both) all positive.
6. Calibration ECE < 0.05 across all 3 buckets.
7. Max drawdown < 15% on any fold.
8. Stationary block bootstrap 95% CI excludes zero.

Per `plan/V1-SPEC.md §9` and `plan/06-BACKTEST.md`.

## Read more

- `plan/V1-SPEC.md` — canonical change list.
- `plan/STATUS.md` — what's built, what's pending.
- `plan/HANDOFF.md` — pre-H100 checklist.
- `plan/00-OVERVIEW.md` — architecture rationale.
- `plan/05-MODEL-TRAINING-CALIBRATION.md` — model + training detail.
- `plan/06-BACKTEST.md` — eval harness.
- `plan/07-SIZING-AND-EXITS.md` — F2F-style sizing layer.

## License

MIT. See `LICENSE`.
