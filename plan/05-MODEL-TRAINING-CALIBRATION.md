# 05 — Model + Training + Calibration

## V1 CHANGE SUMMARY (2026-05-08, supersedes V1 frozen 2026-05-04)

V1 redlines this doc against the V1 spec sheet at `plan-edit/V1-SPEC.md`. The V1 stack below is preserved everywhere V1 does not contradict it. All new pieces are tagged "V1:" inline. Net architecture changes here:

- **V1: Hybrid encoder backbone** (Decision 1B). 10 transformer blocks + 2 sLSTM blocks. xLSTMTime style (Alharthi & Mahmood arXiv:2407.10240). Replaces pure 12-layer transformer. Reasons: VLSTM 2.40 / xLSTM 1.79 / iTransformer 0.38 Sharpe on Saly-Kaufmann arXiv:2603.01820 benchmark.
- **V1: Channel-independent + patches** (Decision 2B). Patch length P=4, stride S=4, 16 patches per channel, 681 channels share one backbone. Replaces channel-group tokens (~14 tokens iTransformer-lite). LPatchTST 2.31 vs iTransformer 0.38 Sharpe.
- **V1: FiLM regime conditioning** every 2 layers ({2,4,6,8,10}). 12-dim regime vector (VIX-tercile, RV-tercile, FOMC-week, year-bucket, HMM-P-high-vol). ~3840 trainable params total.
- **V1: Multi-task dual head** (Decision 3B). Head A 3-class direction + Head B position weight in [-1,+1] via tanh, trained jointly with differentiable Sharpe loss. Hwang & Zohren 2025 arXiv:2510.03129; Saly-Kaufmann 2026.
- **V1: Sparse news cross-attention** at layers [3, 7, 11] only. Was: every layer / every-other. mPLUG-Owl3 arXiv:2408.04840 Table 8.
- **V1: CFA-style FiLM/orthogonal projector** before Flamingo K/V. Lee et al. arXiv:2603.22372.
- **V1: AECF entropy-gated curriculum masking** replaces 15% constant modality dropout. Per-batch p~U(0.0, 0.9). Chlon et al. arXiv:2505.15417.
- **V1: is_news_present binary embedding** concat to news token. Ma et al. CVPR 2022 arXiv:2204.05454.
- **V1: Variable per-batch modality dropout** p~U(0.1, 0.9) at training. CRITICAL FIX — V1's 15% guaranteed deployment cliff at 51% empirical absence rate.
- **V1: NEWS_NOT_PRESENT learnable token** kept (V2 candidate to swap to CMPT).
- **V1: Focal loss gamma=3** replaces vanilla CE / LS=0.05. Mukhoti 2020 arXiv:2002.09437. Critical: Xi 2024 arXiv:2402.04344 shows T-scaling on CE harms APS adaptive coverage.
- **V1: T-scaling -> RAPS -> AgACI** replaces T-scaling -> APS. Angelopoulos 2020 arXiv:2009.14193 + Zaffran 2022 ICML arXiv:2202.07282.
- **V1: Laplace last-layer (LLLA)** replaces MC dropout T=20. Daxberger 2021 arXiv:2106.14806.
- **V1: Mixout p=0.7** anchored to SSL checkpoint, Stage 3 LLRD only. Lee Cho Kang ICLR 2020 arXiv:1909.11299.
- **V1: Stochastic depth linear schedule 0.0 -> 0.2** replaces uniform 0.15. Touvron 2021.
- **V1: SimPSI + Wave-Mask augmentation** replaces naive jittering. Ryu AAAI 2024 arXiv:2312.05790, Arabi 2024 arXiv:2408.10951. Critical: naive jittering is net-negative on Sharpe (Fons 2020 arXiv:2010.15111).
- **V1: Cautious update mask** on Schedule-Free AdamW. Liang 2024 arXiv:2411.16085. 5-line patch.
- **V1: muP transfer-tune** on 2-4M tiny model first. Yang 2022 arXiv:2203.03466.
- **V1: FreeLB** on news embeddings only (not bars - PIT risk). Zhu et al. ICLR 2020 arXiv:1909.11764.
- **V1: DANN gradient reversal** on era-label = year-bucket. Feng 2019 arXiv:1810.09936.
- **V1: Mask ratio 0.20 -> 0.40** in SSL. PatchTST default.
- **V1: SimMTM** replaces plain MAE. Dong NeurIPS 2023 arXiv:2302.00861. K=3 masked views, similarity-weighted reconstruction.
- **V1: CLIP-style bars<->news contrastive head** in SSL. InfoNCE on bar-rep vs news-rep within ±5min, tau=0.07.
- **V1: Combined loss** by stage. Stage 1 SSL: L_simmtm + 0.5*L_clip + 0.05*L_DANN + L_aecf. Stage 2: L_focal_3class. Stage 3 LLRD: 0.5*L_focal + 0.5*L_sharpe_net + 0.05*L_DANN + L_aecf.

V1 explicitly NOT in scope (deferred or rejected): MoE, KAN, Mamba, TTT, online learning, Muon, Sophia, deep ensembles, CMPT (V2), bars-only distillation (V2), full xLSTMTime swap (encoder-wide), end-to-end Sharpe-only head (drops classification).

V1 NEW invariants (added to V1's 17):
- (18) Per-bucket eval (news-present / news-absent / both) is non-negotiable.
- (19) Cost-stress at {0.5x, 1.0x, 1.5x} on every reported Sharpe.
- (20) DSR > 1.0 hard gate.
- (21) SimPSI / Wave-Mask aug only; naive jittering FORBIDDEN.
- (22) Focal loss gamma=3 (NOT vanilla CE) required for clean T-scaling/APS interaction.
- (23) Triple-barrier labels with spread-adjusted neutral threshold (defined in doc 04, referenced here).
- (24) Variable per-batch modality dropout p~U(0.1, 0.9), NOT 15% constant.
- (25) Decision-aware head (multi-task with Sharpe loss) is V1 ship gate.

The full V1 redline rationale lives in `plan-edit/V1-SPEC.md`. This doc updates the parts that touch model architecture, training procedure, calibration, and regularization. Doc 04 owns triple-barrier labels and feature additions; doc 06 owns walk-forward/cost-stress/DSR; doc 07 owns sizing + F2F machinery.

---

## 🎯 START HERE — DATA IS READY

**Unified dataset shipped:** `/Users/root1/Desktop/nanogld/data/processed/training_v1_unified.pt` (Mac mini, 234 MB)

Single PyTorch file. Load:
```python
import torch
data = torch.load("/Users/root1/Desktop/nanogld/data/processed/training_v1_unified.pt")
features    = data["features"]            # (75993, 681) float32
labels      = data["labels"]              # (75993,) int8 — 0=DOWN/1=FLAT/2=UP
splits      = data["splits"]              # list[75993] — "train"/"val"/"test"
embeddings  = data["embeddings"]          # (40032, 256) float16
# Per-bar visible news: embeddings[bar_news_values[bar_news_offsets[T]:bar_news_offsets[T+1]]]
```

**READ `plan/HANDOFF.md` BEFORE WRITING ANY CODE.** Has full context on what's in the dataset, how it was built, what bugs were caught, and what to validate before training.

Architecture (V1 locked from spec): encoder-only hybrid, D=384, 6 heads. **10 transformer blocks (layers 1-10) + 2 sLSTM blocks (layers 11-12)** xLSTMTime style. Channel-independent + patches (P=4, S=4, T_bars=64 -> 16 patches per channel; 681 channels share one backbone). RMSNorm + SwiGLU + Real-form RoPE (partial 10%, QK-Norm BEFORE RoPE) on transformer layers. FiLM regime conditioning every 2 layers ({2,4,6,8,10}) with 12-dim regime vector. **Multi-task dual head**: Head A 3-class direction (focal loss gamma=3) + Head B position weight (-1 to +1 via tanh, differentiable Sharpe loss). **24-40M params** (sweet spot 30-35M; FiLM blocks add ~3840 params, near-zero).

Training: H100 RunPod $1.99/hr, ~3-5h per run, $60-150 for V1 hyperparam sweep ($5 muP tiny-model sweep added upfront).

---

## YOU ARE THE MODEL/TRAINING/CALIBRATION AGENT

You own the predictor end-to-end:
1. **Build the encoder-only hybrid transformer** (V1: 10 transformer + 2 sLSTM blocks; channel-independent + patches; FiLM regime conditioning; multi-task dual head) + baseline architectures (DLinear, TSMixer, TimeMixer, xLSTMTime, VLSTM, XGBoost, Forecast-to-Fill replica, Gao 2014 half-hour-5 single-feature rule).
2. **Train it** with V1 Cautious-Schedule-Free AdamW (Liang 2024 arXiv:2411.16085) + Friendly-SAM + DANN era-label adversarial debiasing + Mixout p=0.7 anchored to SSL checkpoint + SimMTM SSL pretrain + CLIP bars<->news contrastive head + walk-forward CV (NYSE bars_per_year=3276).
3. **Calibrate it** — V1 focal loss gamma=3 + temperature scaling + RAPS + AgACI online conformal + Laplace last-layer (LLLA) for epistemic + drift detection stack.

You start when doc 04 (features) hands off a feature DataFrame. You hand off to doc 06 (backtest) a calibrated checkpoint plus the conformal prediction layer.

**Read 00-OVERVIEW.md FIRST.** Project context is there. Read doc 04's output schema (the feature DataFrame) before coding.

### Execution Mode (short — full rules in 00-OVERVIEW.md)

**This doc IS the plan. Do not rewrite it.** Plan to *execute* what is written. If a claim is wrong, AskUserQuestion. Silent scope drift = fired.

- **Research with Nia** before guessing on libs / APIs / papers. Spawn a subagent.
- **Execution skills only:** /investigate, /review, /qa, /qa-only, /cso, /benchmark, /ship.
- **Default loop:** read this doc → Nia for unknowns → code → /review → /qa → /ship.
- **Escalate after 3 failed attempts.** AskUserQuestion.

### What You Build (top-level files)

```
src/nanogld/model/         # part 1 — architecture
src/nanogld/training/      # part 2 — training procedure
src/nanogld/calibration/   # part 3 — temperature + conformal + drift
checkpoints/v1_<hash>.pt   # output: calibrated checkpoint
```

### Acceptance Criteria

You're done when:

1. ✅ Forward-pass tests pass (V1: model returns Head A 3-class logits AND Head B tanh position weight in [-1,+1], correct shapes)
2. ✅ Walk-forward training completes 4 folds with EMA tracking (NYSE bars_per_year=3276)
3. ✅ V1: Loss is focal loss gamma=3 (not vanilla CE / not LS=0.05) on Head A + differentiable -Sharpe on Head B + 0.05 * L_DANN (era-label) + L_aecf (entropy-gated curriculum mask). NEVER MSE on returns.
4. ✅ V1: Cautious-Schedule-Free AdamW + Friendly-SAM ρ=0.05 + EMA decay=0.999 + Mixout p=0.7 anchored to SSL checkpoint (Stage 3 only). muP transfer-tune on 2-4M tiny model done before main run.
5. ✅ Temperature scaling fit on val-B fold; expect T near 1.0 with focal loss; classwise Adaptive ECE < 5% post-calibration
6. ✅ V1: T-scaling -> RAPS -> AgACI fit on val-C fold; per-class coverage matches target ±2pts; AgACI maintains coverage under arbitrary distribution shift via expert-advice over gamma grid
7. ✅ V1: Laplace last-layer (LLLA) posterior fit on val fold; posterior variance feeds Kelly multiplier (replaces MC dropout T=20)
8. ✅ Drift detection stack (3 tiers — unlabeled / labeled-daily / labeled-weekly) operational
9. ✅ All baselines (DLinear / TSMixer / TimeMixer / xLSTMTime / VLSTM / XGBoost / Forecast-to-Fill replica / Gao 2014 half-hour-5 rule) trainable from a unified config; per-bucket eval (news-present / news-absent / both) reported on each
10. ✅ Hand-off artifact: `checkpoints/v1_5_<hash>.pt` + meta JSON with all hyperparams + calibration objects (temperature, RAPS k_reg, AgACI gamma grid + expert weights, Laplace posterior, drift baselines, focal gamma, regime-vector schema)

This doc is **3 parts** combined for one agent — read all of it before starting.

---


# Part 1 — Model Architecture

_(was doc 05-MODEL-TRAINING-CALIBRATION.md before V5 merge — content unchanged)_

# 03 — Model Architecture

## YOU ARE THE ML SYSTEMS ENGINEER AGENT

You own the nanoGLD transformer code itself. You write the model from scratch in raw PyTorch (Karpathy mode — NO HuggingFace `Trainer`, NO Unsloth, NO TRL). You implement the V1 architecture spec locked below.

**Read 00-OVERVIEW.md FIRST.** Project context.
**You DO NOT train.** That's doc 05. You build the model class + verify forward pass shapes.
**Also read 00-OVERVIEW.md "Execution Mode" section before coding.**

### Execution Mode (short — full rules in 00-OVERVIEW.md)

**This doc IS the plan. Do not rewrite it.** Plan to *execute* what is written. If a claim is wrong, AskUserQuestion. Silent scope drift = fired.

- **Research with Nia** before guessing on libs / APIs / papers. Spawn a subagent: `nia papers` / `nia github` / `nia search` / `nia oracle` / `nia packages` / `nia tracer`.
- **Execution skills only:** `/investigate`, `/review`, `/qa`, `/qa-only`, `/cso`, `/benchmark`, `/ship`, `/land-and-deploy`, `/canary`, `/document-release`, `/retro`, `/learn`, `/browse`, `/setup-browser-cookies`.
- **NO planning skills:** `/office-hours`, `/plan-*`, `/autoplan`, `/design-*`, `/devex-review`. Planning is done.
- **Loop:** read doc → Nia for unknowns → code → `/review` → `/qa` → `/cso` (if security-sensitive) → `/ship`.
- **Escalate after 3 failed attempts.** AskUserQuestion. Bad work is worse than no work.

### Files You Create

```
src/nanogld/model/
├── __init__.py
├── rms_norm.py             # Llama-style RMSNorm (no mean centering, no bias)
├── rope.py                 # Real-form RoPE (NEVER torch.view_as_complex on MPS)
├── swiglu.py               # SwiGLU FFN, hidden = round(8D/3, 64)
├── attention.py            # CausalSelfAttention with QK-Norm + per-head gating + value residuals
├── tda.py                  # [A/B candidate] Threshold Differential Attention
├── sype.py                 # [A/B candidate] Symplectic Position Embedding
├── revin.py                # V1: Reversible Instance Norm per individual channel (Kim ICLR 2022; was per-group in V1)
├── patch_embed.py          # V1 NEW: Linear(P=4, D) patch projection per channel; sinusoidal pos embed
├── slstm_block.py          # V1 NEW: sLSTM block (xLSTMTime style, channel-independent), used at layers 11-12
├── film_regime.py          # V1 NEW: FiLM affine modulation per layer, conditioned on 12-dim regime vector
├── cfa_projector.py        # V1 NEW: CFA-style FiLM/orthogonal projector before Flamingo K/V (Lee arXiv:2603.22372)
├── aecf_mask.py            # V1 NEW: AECF entropy-gated curriculum masking (Chlon arXiv:2505.15417)
├── news_fuser.py           # Perceiver-Resampler-lite + Flamingo-gated cross-attn (V1: sparse layers [3,7,11] only; CFA filter; is_news_present binary embedding)
├── multi_task_head.py      # V1 NEW: Head A (3-class focal) + Head B (tanh position weight, differentiable Sharpe loss)
├── tiny_trader.py          # Main nanoGLDV1_5 class wiring everything together (10 transformer + 2 sLSTM)
├── baselines.py            # DLinear, TSMixer, TimeMixer, xLSTMTime, VLSTM, Gao 2014 half-hour-5 rule, Forecast-to-Fill (full impls in doc 06)
└── cli.py                  # `python -m nanogld.model summary` prints param count + arch table

tests/
├── test_forward_pass.py    # Synthetic input, verify shapes for both Head A (B,3) and Head B (B,1)
├── test_param_count.py     # Verify ~24-40M depending on config (V1 sweet spot 30-35M)
├── test_rope_correctness.py # Real-form RoPE matches reference impl on small example
├── test_attention_no_causal.py # Verify is_causal=False (encoder-only)
├── test_patches.py         # V1: 16 patches per channel from T_bars=64 with P=4, S=4
├── test_slstm_block.py     # V1: sLSTM block forward + backward, channel-independent
├── test_film_regime.py     # V1: FiLM conditioning produces same shape, gamma/beta apply between attn and FFN
├── test_multi_task_head.py # V1: Head A logits in (B,3), Head B tanh in [-1,+1]
└── test_aecf.py            # V1: per-batch p ~ U(0.0, 0.9) sampled correctly, entropy regularizer matches paper
```

### Files You DO NOT Touch

- Anything in `src/nanogld/data/`, `features/`, `embed/`, `training/`, `backtest/`, `sizing/`, `live/`
- Doc files other than this one
- Training scripts (you provide the model class; doc 05 writes the loop)

### Stable Interface You Publish

```python
# V1: Other docs (esp. training pipeline) instantiate:
model = nanoGLDV1_5(
    numeric_dim: int = 681,                    # V1: 681 channels (channel-independent)
    n_news_queries: int = 8,
    D: int = 384,
    num_heads: int = 6,
    num_transformer_layers: int = 10,          # V1: 10 transformer + 2 sLSTM = 12 total
    num_slstm_layers: int = 2,                 # V1 NEW
    T_bars: int = 64,
    patch_len: int = 4,                        # V1 NEW (P)
    patch_stride: int = 4,                     # V1 NEW (S)
    n_classes: int = 3,
    regime_dim: int = 12,                      # V1 NEW: VIX-tercile + RV-tercile + FOMC + year-bucket + HMM-P-high-vol
    cross_attn_layers: tuple = (3, 7, 11),     # V1 NEW: sparse news cross-attn injection points
    dropout: float = 0.2,
    drop_path_max: float = 0.20,               # V1: linear schedule 0.0 -> 0.2 (was uniform 0.15)
)

# V1 Forward signature (multi-task)
def forward(self,
            channel_inputs: torch.Tensor,      # (B, T_bars, 681) — channel-independent patched internally
            news_embeddings: torch.Tensor,     # (B, n_sources, 256) — V1: Qwen3 MRL-256
            news_mask: torch.Tensor,           # (B, n_sources)
            is_news_present: torch.Tensor,     # V1 NEW: (B,) int64 — explicit gate signal
            regime_vec: torch.Tensor,          # V1 NEW: (B, 12) — VIX/RV/FOMC/year/HMM
            ) -> dict[str, torch.Tensor]:
    return {
        "logits_3class": ...,                  # (B, 3)        Head A — focal loss target
        "position_weight": ...,                # (B,) in [-1,+1]  Head B — differentiable Sharpe target
        "regime_film_gates": ...,              # (B, num_layers, 2*D)  for diagnostic
        "aecf_mask_p": ...,                    # (B,) sampled per-batch in [0.0, 0.9] for entropy reg
    }
```

If you change this signature, update STATUS.md + AskUserQuestion before shipping.

### Acceptance Criteria

1. ✅ `pytest tests/test_forward_pass.py` passes — synthetic input produces correct output shape
2. ✅ `pytest tests/test_param_count.py` passes — model has expected param count (within 5% of spec)
3. ✅ `python -m nanogld.model summary` prints param breakdown table
4. ✅ Forward pass on M4 Pro MPS at B=32, T=64 takes <100ms (sanity benchmark)
5. ✅ `pytest tests/test_rope_correctness.py` passes — RoPE matches Karpathy nanoGPT reference impl
6. ✅ Backward pass succeeds without NaN on synthetic input (gradient flow check)
7. ✅ No use of `torch.view_as_complex` (crashes MPS — verified bug)
8. ✅ `.contiguous()` called on Q/K/V before SDPA (PyTorch #181133 workaround)

### Spawn Nia Agents When You Need To

Especially:
- **TDA implementation details** (arXiv:2601.12145) — paper has math but no reference code; spawn agent to find blog/repo implementations
- **SyPE Sp(2,R) symplectic group implementation** (arXiv:2602.08983) — verify the math before coding
- **Per-head gating + value residuals** (IMU-1 arXiv:2602.02522) — paper has spec; verify your impl matches
- **Latest PyTorch SDPA on MPS bug list** (PyTorch issue tracker) — week-to-week changes

### Critical V1 Architecture Decisions (DO NOT REVERT)

1. **ENCODER-only** (drop causal mask). Bidirectional context strictly better for next-bar classification.
2. **V1: Channel-independent + patches** (Decision 2B). Patch length P=4, stride S=4, T_bars=64 -> 16 patches per channel. 681 channels share one transformer backbone. Replaces channel-group tokens. Reason: LPatchTST 2.31 vs iTransformer 0.38 Sharpe (Saly-Kaufmann arXiv:2603.01820). Channel mixing removed from main backbone; cross-feature signal recovered via VSN feature gate at input (doc 04) + grouped cross-channel mixer at FiLM regime injection points.
3. **V1: Hybrid encoder** (Decision 1B). Layers 1-10 transformer blocks + layers 11-12 sLSTM blocks (xLSTMTime style, Alharthi & Mahmood arXiv:2407.10240). Reason: VLSTM 2.40 / xLSTM 1.79 / iTransformer 0.38 Sharpe.
4. **V1: FiLM regime conditioning** every 2 layers ({2, 4, 6, 8, 10}). 12-dim regime vector (VIX-tercile 3 + RV-tercile 3 + FOMC binary 1 + year-bucket {2016-2019, 2020-2022, 2023-2024, 2025+} 4 + HMM P(high-vol) 1). FiLM math: `gamma_l, beta_l = Linear_l(regime_vec)`, applied as `x = gamma_l * x + beta_l` between attention and FFN. ~3840 trainable params total.
5. **RMSNorm + SwiGLU + RoPE + QK-Norm + no-bias** — Llama 3 / Qwen 3 consensus stack (transformer layers only; sLSTM layers use xLSTMTime spec).
6. **Per-head gating + value residuals** — IMU-1 recipe, ~50K extra params, big sample efficiency gain.
7. **Partial RoPE** — apply RoPE to 10% of head_dim only.
8. **head_dim = 64** (D=384, num_heads=6). NOT head_dim=32 (Llama 3 / Qwen consensus).
9. **dropout = 0.2** (small data regime). NOT 0.1.
10. **V1: Param count math: ~24-40M.** FiLM blocks add ~3840 params (negligible). Sweet spot 30-35M. Don't over-engineer; verify with summary.
11. **V1: Multi-task dual head** (Decision 3B). Head A 3-class direction `nn.Linear(D, 3, bias=False)` on mean-pooled tokens, focal loss gamma=3. Head B position weight `nn.Linear(D, 1, bias=False)` then `tanh` -> [-1, +1], differentiable Sharpe loss. Combined loss documented in training section. Model outputs raw logits + raw position weight — no softmax in forward.
12. **V1: Sparse news cross-attn at layers [3, 7, 11]** of 12 (NOT every layer). Bottom 2 layers pure-bar. mPLUG-Owl3 Table 8 (arXiv:2408.04840). CFA-style FiLM/orthogonal projector before Flamingo K/V (Lee arXiv:2603.22372).

### A/B Candidates (post-baseline)

These are coded as alternative components but NOT used in the default V1 build:
- **TDA in 1 attention block** — replace `CausalSelfAttention` middle block, compare val Sharpe
- **SyPE replaces RoPE** — single hyperparameter swap
- **Muon optimizer** for 2D weights (doc 05 owns this — model code unchanged)

If your default model fails to converge, before going A/B, first verify:
- Param count within 5% of spec
- LR schedule is working (Schedule-Free should plateau around peak LR after warmup)
- No silent CPU fallback on MPS (run with `TORCH_LOGS=fallback`)

### Hand-off Protocol

1. Update STATUS.md with: model param count, forward pass benchmark, MPS dtype, any deviations
2. Notify doc 05 (training) that model class is stable
3. Document A/B candidate results IN this doc once they're tested

Now read the architecture spec below.

---

# 03 — Model Architecture

**Status:** ✅ Complete, implementation-ready, Nia-verified
**Last verified:** 2026-04-30

## ARCHITECTURE (V1) — 6-Agent 2026 Deep Research (May 2026)

After 7-agent earlier-draft research, ran 6 more agents on 2026-specific releases (Llama 4, Gemma 4, Qwen 3.5, embedding SOTA, time-series foundation models, architecture innovations Jan-May 2026, finance papers, training optimizations). Critical 2026 findings below — V1 ADDS these to earlier draft.

### CRITICAL ADDITIONS FROM 2026 PAPERS

#### A. **FORECAST COLLAPSE RULE** (arXiv:2604.00064, March 2026) — non-negotiable

Paper proves: on weak-conditional-structure data (financial returns), Transformer expressivity **increases variance WITHOUT reducing bias**, producing larger errors than linear baselines on a majority of windows. Direct empirical warning.

**Project rule: NEVER use MSE on returns. Use:**
- 3-class classification with cross-entropy (current default — keep)
- Quantile regression loss (alternative for continuous magnitude estimation)
- NEVER squared loss on raw returns

This is now a hard rule across doc 05, doc 05, doc 06. Loss-function choice is no longer up for debate.

#### B. **per-head gating + value residuals** (IMU-1 paper, arXiv:2602.02522, Jan 2026) — adopt

IMU-1 demonstrates **56× sample efficiency** on 430M model (matches teachers trained on 72B tokens). Validated each component via ablation. Recipe: QK-norm (we have) + **per-head gating** (NEW) + **value residuals** (NEW) + LayerNorm scaling + NorMuon optimizer + muP.

```python
class CausalSelfAttentionV1(nn.Module):
    """V1 adds per-head gating + value residual (IMU-1 recipe)."""
    def __init__(self, D, num_heads, max_seq, dropout=0.2):
        super().__init__()
        # ... existing qkv, proj, q_norm, k_norm, RoPE buffers ...
        # NEW: per-head gating (sigmoid scalar per head, learned)
        self.head_gate = nn.Parameter(torch.zeros(num_heads))
        # NEW: value residual (MLP that produces a residual to add to V from previous layer's V)
        self.value_residual = nn.Linear(D, D, bias=False)

    def forward(self, x, prev_v=None):
        # ... existing QK-Norm + RoPE + SDPA ...
        out_per_head = ...   # (B, H, T, head_dim)

        # Per-head gating
        gates = torch.sigmoid(self.head_gate).view(1, -1, 1, 1)  # (1, H, 1, 1)
        out_per_head = out_per_head * gates

        # Value residual (skip connection across attention layers on V)
        v_residual = self.value_residual(prev_v) if prev_v is not None else 0
        # ... combine into output ...
```

Per-head gating: ~H new params. Value residuals: 1 Linear per block. Total cost ~50K extra params on 12-block 384-dim model. Trivial. Expected gain: **significant on small data** per IMU-1 ablations.

#### C. **TDA — Threshold Differential Attention** (arXiv:2601.12145, Jan 2026) — A/B candidate

Sink-free attention with row-wise extreme-value thresholding. **>99% exact zeros, theoretical O(1) spurious survivors.** Designed precisely for low-SNR signals. Pure PyTorch (no custom kernel needed). MPS-compatible.

```python
class ThresholdDifferentialAttention(nn.Module):
    """TDA: differential attention with row-wise thresholding (arXiv:2601.12145)."""
    def __init__(self, D, num_heads, theta_init=0.5):
        super().__init__()
        # Two parallel softmax branches (subtract one from other)
        self.attn_a = MultiHeadAttention(D, num_heads)
        self.attn_b = MultiHeadAttention(D, num_heads)
        self.lambda_init = nn.Parameter(torch.full((num_heads,), 0.8))
        self.theta = nn.Parameter(torch.full((1,), theta_init))  # threshold

    def forward(self, x):
        a = self.attn_a(x)  # (B, H, T, T) softmax weights
        b = self.attn_b(x)
        diff = a - self.lambda_init.view(1, -1, 1, 1) * b
        # Row-wise thresholding (length-dependent gate)
        T = diff.size(-1)
        threshold = self.theta * (1.0 / math.sqrt(T))   # length-dependent
        diff = torch.where(diff.abs() < threshold, torch.zeros_like(diff), diff)
        # Renormalize per row
        diff = diff / diff.sum(-1, keepdim=True).clamp(min=1e-8)
        return diff @ V  # standard attention output
```

**Plan:** A/B replace ONE attention block with TDA, compare val Sharpe. If wins, replace all.

#### D. **SyPE — Symplectic Position Embedding** (arXiv:2602.08983, Feb 2026) — A/B vs RoPE

Paper **proves RoPE cannot represent non-affine temporal warping** (cyclic vs trending markets, vol clusters). Replaces SO(2) rotation with Sp(2,R) symplectic group + adaptive warp module. **Specifically motivated by financial cycles.** SOTA on multivariate forecasting per paper.

```python
class SymplecticPositionEmbedding(nn.Module):
    """SyPE: replaces RoPE for non-stationary time (arXiv:2602.08983)."""
    def __init__(self, head_dim, max_seq):
        super().__init__()
        # Sp(2,R) symplectic group has 3 parameters per pair (vs 1 for SO(2) rotation)
        # Plus adaptive warp module (small MLP)
        self.warp_mlp = nn.Sequential(
            nn.Linear(2, 16), nn.GELU(), nn.Linear(16, 3)
        )
        # ... full impl per paper ...

    def forward(self, x, position_ids):
        # Apply position-dependent symplectic transformation
        ...
```

Plan: A/B SyPE vs RoPE on val Sharpe. If financial cycles matter, SyPE wins.

#### E. **Partial RoPE** (arXiv:2603.11611, March 2026) — free win

Applying RoPE to **only ~10% of head dims** matches full RoPE convergence. ~10× KV-cache savings. Not load-bearing for us at T=64-256 but free quality at minimal cost.

```python
def apply_partial_rope(x, cos, sin, frac: float = 0.10):
    """Apply RoPE only to first frac× of head dims."""
    head_dim = x.size(-1)
    rope_dim = int(head_dim * frac) // 2 * 2   # even
    rotated_part = apply_rope(x[..., :rope_dim], cos[..., :rope_dim//2], sin[..., :rope_dim//2])
    return torch.cat([rotated_part, x[..., rope_dim:]], dim=-1)
```

#### F. **xLSTM dominates 2026 finance benchmark** (arXiv:2603.01820, March 2026)

**Most nanoGLD-relevant 2026 finding.** Large-scale benchmark on 2010-2025 daily futures across commodities, equities, bonds, FX. **xLSTM has highest breakeven transaction cost buffer and best downside-adjusted Sharpe** of all sequence models tested. Pure Transformer underperforms.

**Implication:** add xLSTM (xLSTMTime — separate paper, code released 2025) as a **mandatory baseline** alongside DLinear, TSMixer, TimeMixer. doc 06 update.

If our 24-60M Transformer can't beat xLSTM at ~10M params on val Sharpe, ship xLSTM. Same logic as TLOB's "MLP matches transformer" finding from earlier draft.

#### G. **Hybrid attention NOT recommended at our scale** (Qwen 3.5 production endorsement)

Qwen 3.5 (early 2026) ships with **75% Gated DeltaNet linear-attention + 25% softmax**. Strong production signal. But Olmo Hybrid + Long-Context Aware Upcycling ablations show gains are **marginal at T<256**. Skip for our T=64. Reconsider only if context expands to T≥1024.

#### H. **FlashAttention 4 (Blackwell-only)** — useless on MPS

FA4 (arXiv:2603.05451, March 2026) is CUDA Blackwell-targeted. PyTorch SDPA on MPS remains the right call for us.

#### I. **Mamba-3 released ICLR 2026** (arXiv:2603.15569) — skip at our scale

Trapezoidal SSM + complex state + MIMO. Half state size of Mamba-2. MPS-feasible but no CUDA selective-scan kernel = slow. At T=64-256 with 24M params, Transformer still wins on speed AND quality. Skip.

### Updated V1 Architecture Spec

```
nanoGLD V1 (May 2026 — supersedes V1)
═══════════════════════════════════════════════════════════════════
Backbone:        ENCODER-only HYBRID
                 Layers 1-10: transformer blocks (kept)
                 Layers 11-12: sLSTM blocks (V1 NEW, xLSTMTime style)
                 Source: Saly-Kaufmann/Wood/Zohren 2026 arXiv:2603.01820
                 (VLSTM 2.40 Sharpe, xLSTM 1.79 + best transaction-cost robustness)
Tokenization:    V1: Channel-independent + patches (PatchTST style)
                 Patch length P=4, stride S=4, T_bars=64 -> 16 patches per channel
                 681 channels share one transformer backbone
                 Patch projection: Linear(P, D) per patch, no bias
                 Position embedding: sinusoidal added (RoPE on Q/K)
                 Channel mixing removed from main backbone
                 Source: LPatchTST 2.31 vs iTransformer 0.38 Sharpe
Per-block:       RMSNorm + SwiGLU + RoPE + QK-Norm + no-bias (transformer layers)
                 sLSTM block (xLSTMTime: BatchNorm + sLSTM + linear + InstanceNorm)
ADDITIONS:
  • Per-head gating (IMU-1) — sigmoid scalar per head, learned
  • Value residuals (IMU-1) — Linear shortcut on V across blocks
  • Partial RoPE (apply to 10% of head_dim, leave rest unrotated)
  • V1: FiLM regime conditioning every 2 layers ({2,4,6,8,10})
    - 12-dim regime vector: VIX-tercile (3) + RV-tercile (3)
      + FOMC-week (1) + year-bucket (4) + HMM-P-high-vol (1)
    - gamma_l, beta_l = Linear_l(regime_vec); x = gamma_l*x + beta_l between attn and FFN
    - ~3840 trainable params total
  • [A/B candidate] TDA in 1+ blocks — sink-free attention
  • [A/B candidate] SyPE replaces RoPE — symplectic position for cycles
News fusion:     Perceiver-Resampler-lite + Flamingo-gated cross-attn
                 V1: SPARSE injection at layers [3, 7, 11] only (NOT every layer)
                 Source: mPLUG-Owl3 arXiv:2408.04840 Table 8
                 V1: CFA-style FiLM/orthogonal projector before Flamingo K/V
                 (Lee et al. arXiv:2603.22372; ref github.com/seunghan96/cfa)
                 V1: AECF entropy-gated curriculum masking, p ~ U(0.0, 0.9)
                 (Chlon et al. arXiv:2505.15417; ref github.com/leochlon/aecf)
                 V1: is_news_present binary embedding `nn.Embedding(2, 8)` concat to news token
                 (Ma et al. CVPR 2022 arXiv:2204.05454)
                 V1: Variable per-batch modality dropout p ~ U(0.1, 0.9) at training
                 (matches empirical 51% absence rate; CRITICAL FIX over V1's 15% constant)
                 V1: NEWS_NOT_PRESENT learnable token KEPT (V2 candidate to swap to CMPT)
News embedder:   Qwen/Qwen3-Embedding-4B 4-bit MLX (V1 — replaces Llama-3.1-8B)
Output head:     V1 MULTI-TASK DUAL HEAD (Decision 3B)
                 Head A (3-class direction): Linear(D, 3, bias=False) on mean-pooled tokens
                   Loss: focal loss gamma=3 (Mukhoti 2020 arXiv:2002.09437)
                 Head B (position weight, NEW): Linear(D, 1, bias=False) -> tanh -> [-1, +1]
                   Loss: differentiable -Sharpe over mini-batch
                   L_sharpe_net = -mean(w*r_next - cost*|w_t - w_{t-1}|) / (std(...) + eps)
                 Source: Hwang & Zohren 2025 arXiv:2510.03129; Saly-Kaufmann 2026 (-Sharpe direct)
Loss:            V1: focal loss gamma=3 (Head A) + differentiable -Sharpe (Head B)
                 Combined: L = 0.5*L_focal + 0.5*L_sharpe_net + 0.05*L_DANN + L_aecf
                 NEVER MSE on returns (forecast-collapse rule, arXiv:2604.00064)
                 V1: Vanilla CE / LS=0.05 REPLACED by focal (Xi 2024 arXiv:2402.04344
                 shows T-scaling on CE harms APS adaptive coverage; focal fixes this)
Pretrain:        V1: SimMTM SSL multi-mask reconstruction (was: plain MAE)
                 Dong NeurIPS 2023 Spotlight arXiv:2302.00861; ref thuml/SimMTM
                 K=3 masked views, similarity-weighted reconstruction
                 V1: Mask ratio 0.40 (was 0.20; PatchTST default)
                 V1: + CLIP bars<->news contrastive head
                 InfoNCE on bar-rep vs news-rep within ±5min, tau=0.07
                 -> linear-probe -> LLRD fine-tune with Mixout p=0.7 anchored to SSL ckpt
Optimizer:       V1: Cautious-Schedule-Free AdamW (Liang 2024 arXiv:2411.16085)
                 5-line patch: update *= (sign(update) == sign(grad)).float()
                 Friendly-SAM ρ=0.05 wrap kept
                 V1: muP transfer-tune on 2-4M tiny model FIRST ($5), then 30M
                 Yang 2022 arXiv:2203.03466; ref microsoft/mup
                 [A/B candidate] NorMuon — DROPPED for V1 (Muon at 30M too small per spec §11)
EMA weights:     decay=0.999
Regularization:  Dropout 0.2
                 V1: Stochastic depth LINEAR schedule 0.0 -> 0.2 (was uniform 0.15; Touvron 2021)
                 V1: Mixout p=0.7 anchored to SSL checkpoint (Stage 3 LLRD only)
                 (Lee Cho Kang ICLR 2020 arXiv:1909.11299; ref bloodwass/mixout)
                 V1: SimPSI + Wave-Mask augmentation (replaces naive jittering)
                 Ryu AAAI 2024 arXiv:2312.05790; Arabi 2024 arXiv:2408.10951
                 NAIVE JITTERING FORBIDDEN (Fons 2020 arXiv:2010.15111: net-negative on Sharpe)
                 V1: FreeLB on news embeddings ONLY (not bars - PIT risk)
                 K=2, epsilon=0.5 (Zhu et al. ICLR 2020 arXiv:1909.11764)
                 V1: DANN gradient reversal on era-label = year-bucket
                 Lambda 0->0.1 ramp (Feng 2019 arXiv:1810.09936; +3.11% on stock prediction)
                 Manifold Mixup α=0.2 at hidden states (NEVER raw input)
Calibration:     V1: focal-trained logits -> T-scaling -> RAPS -> AgACI
                 RAPS (Angelopoulos 2020 arXiv:2009.14193): APS with size penalty
                 AgACI (Zaffran 2022 ICML arXiv:2202.07282): online wrapper, expert-advice
                 over gamma grid [0.001, 0.005, 0.01, 0.05, 0.1] via BOA aggregator
                 ref ml-stat-Sustech/TorchCP, mzaffran/AdaptiveConformalPredictionsTimeSeries
Epistemic:       V1: Laplace last-layer (LLLA) (Daxberger 2021 arXiv:2106.14806)
                 (replaces MC dropout T=20; faster, better-calibrated, no 20x inference cost)
                 ref pip install laplace-torch (aleximmer/Laplace)
                 Posterior variance feeds Kelly multiplier
                 Snapshot ensemble (last 3 EMA checkpoints) and EMA 0.999: KEPT
                 Skip deep ensembles (5x compute kills budget)
Mandatory baselines (doc 06 — V1 UPDATED):
  • Buy and hold GLD net of costs
  • DLinear (~10K params)
  • TSMixer (~2M)
  • TimeMixer (~5M)
  • xLSTMTime (~10M) — strong; ship if ties
  • V1: VLSTM (~10M) — STRONG (2.40 Sharpe in Saly-Kaufmann); ship if ties
  • XGBoost on the same 681 features (committed config)
  • V1: Forecast-to-Fill replication on daily GLD bars (separate scoreboard, different problem)
  • V1: Gao 2014 half-hour-5 single-feature timing rule (THE GLD-specific bar to beat)
  • V1: 50/200 EMA crossover, Donchian breakout (kept honest)
```

### Loss Function Hard Rule (V1)

> **NEVER use squared loss (MSE) on raw returns.**
>
> V1 stack: focal loss gamma=3 on Head A (3-class direction) + differentiable -Sharpe on Head B (position weight in [-1,+1]). Vanilla CE / label smoothing REPLACED.
>
> Per arXiv:2604.00064 (March 2026): on weak-conditional-structure data, Transformer expressivity increases variance without reducing bias. Squared-loss training is provably worse than linear baselines on majority of windows in noisy financial regimes.
>
> V1: focal loss gamma=3 (Mukhoti 2020 arXiv:2002.09437) reduces ECE 30-50% pre-T-scaling and lets T post-fit converge near 1.0. CRITICAL: Xi 2024 arXiv:2402.04344 shows T-scaling on CE logits HARMS APS adaptive coverage. Focal-trained logits avoid this conflict — required for clean T-scaling/RAPS/AgACI interaction.
>
> V1: differentiable -Sharpe direct training (Hwang & Zohren 2025 arXiv:2510.03129) targets end-to-end profit not classification accuracy. Saly-Kaufmann 2026 hits 2.40 Sharpe by training directly on -Sharpe.

This rule propagates to doc 05 (here), 06 (eval per-bucket + cost-stress + DSR), 07 (sizing).

### A/B Test Hierarchy (when implementation begins)

Test in this order (cheapest→most invasive):

1. **Adopt:** per-head gating + value residuals + partial RoPE (IMU-1, ~50K params, no breaks).
2. **V1: ADOPTED (not A/B):** Hybrid 10-transformer + 2-sLSTM. xLSTMTime style. Skip the A/B; spec already commits.
3. **V1: ADOPTED (not A/B):** Channel-independent + patches. PatchTST style. Skip the A/B; spec commits.
4. **V1: ADOPTED (not A/B):** FiLM regime conditioning every 2 layers. ~3840 params, negligible cost.
5. **V1: ADOPTED (not A/B):** Multi-task dual head (Head A focal + Head B Sharpe). Decision 3B.
6. **V1: ADOPTED (not A/B):** Sparse news cross-attn at [3, 7, 11]. mPLUG-Owl3 evidence.
7. **A/B:** TDA in one block (replace `CausalSelfAttention` in middle block, compare val Sharpe over 5 seeds).
8. **A/B:** SyPE replaces RoPE (single hyperparameter swap, compare on val Sharpe).
9. **V1: SKIP at our scale:** hybrid linear+softmax attention (Qwen 3.5 pattern). Reconsider only if context expands to T>=1024.
10. **V1: SKIP — full xLSTMTime swap deferred to V2.** V1 keeps hybrid (10 transformer + 2 sLSTM).
11. **V1: A/B post-baseline:** SimMTM vs T-JEPA SSL. Spec keeps SimMTM as primary.
12. **V1: A/B post-baseline:** FreeLB ON vs OFF (some risk of distribution-shift overcorrection).

Decision criterion at each gate: ≥0.1 Sharpe improvement OOS, seed-averaged across 5 seeds. Otherwise revert.

## ARCHITECTURE — earlier-draft research (May 2026)

The 5 corrections from previous Nia rounds stand. PLUS major architecture pivots from 7 parallel research agents (modern LLM SOTA + time-series transformers + multimodal fusion + attention 2026 + small-data training + empirical SOTA + MPS optimization). Cross-agent consensus drove these decisions.

### Major Pivots

1. **DECODER-ONLY → ENCODER-ONLY** — drop causal mask. Bidirectional context is strictly better for next-bar 3-class classification (no autoregressive rollout). Free win, all time-series transformer literature converges on this. Reference: PatchTST, iTransformer, MOIRAI all encoder-only.

2. **PER-BAR TOKENS → CHANNEL-GROUP TOKENS** (iTransformer-lite). Instead of 64 time-step tokens of 384 dims, use 6-10 channel-group tokens each summarizing T=64 timesteps. Groups: price_OHLCV, macro, geopolitical, news_a/b/c (post-Resampler), multi-scale aggregates (2hr, 4hr, daily). Reference: iTransformer arXiv:2310.06625, MOIRAI, M5 results.

3. **Modern stack: RMSNorm + SwiGLU + RoPE + QK-Norm + no-bias.** Llama 3 / Qwen 3 / Mistral consensus. QK-Norm is the single biggest stability win at small scale. Reference: Raschka "The Big LLM Architecture Comparison" April 2026.

4. **Multimodal fusion upgrade**: replace `3×Linear(4096,256)` with **Perceiver-Resampler-lite + Flamingo-gated cross-attention**. 8 learnable queries cross-attend to 3 frozen news vectors → 8 fused news tokens. Tanh gate init=0 = identity at start. Multi-stage: news self-attn first, then gated cross-attn into encoder mid-depth. Reference: Flamingo arXiv:2204.14198, BLIP-2 arXiv:2301.12597.

5. **L2-norm news + learnable [NO_NEWS] token + per-source tanh gate + 15% modality dropout** — completes the multimodal fusion stack.

6. **RevIN instance norm per channel-group on input** — Kim et al. ICLR 2022. ~half of PatchTST's empirical gains came from this single trick.

7. **Two-stage training**: SSL pretrain (MLM on same 5y unlabeled) → linear-probe → LLRD fine-tune. Documented 25-50% gain on small-data classification. See doc 05.

8. **Skip Time-LLM / Chronos / Lag-Llama / TimesFM as backbone** — Tan et al. NeurIPS 2024 (arXiv:2406.16964) ablation showed ablating the LLM matches Time-LLM. None of the foundation models cleanly ingest 804-dim multivariate input with news embeddings.

9. **MANDATORY BASELINES** — ship the simpler one if it ties:
   - DLinear (1-layer linear) — sanity floor
   - TSMixer (~2M MLP-mixer) — TLOB paper showed MLP can match transformer
   - TimeMixer (~5M, multi-scale MLP)
   - XGBoost (committed config, doc 06)

   **If nanoGLD 24M doesn't beat all 4 by ≥0.2 Sharpe OOS, SHIP THE BASELINE.**

10. **MPS pin: PyTorch 2.11.0** (has SDPA fix #174945). FP32 everywhere. No autocast, no torch.compile, no quantization. `.contiguous()` Q/K/V before SDPA. DataLoader `num_workers=0`.

### Empirical Bar (from Agent 6 research)

**No published transformer beats XGBoost/MLP on single-asset 30min direction with honest walk-forward + DSR + costs.** TLOB paper: "an MLP matches our transformer." Cleanest gold result = **Forecast-to-Fill** (arXiv:2511.08571) — NOT a transformer, Sharpe 2.88, MDD 0.52%, walk-forward 5y. This is the bar to beat.

Targets:
- **Minimum viable**: 38% accuracy (3-class), beat XGBoost on same features, Sharpe > 0 after 5bps round-trip, walk-forward
- **Real claim**: Sharpe > 1.0, hit rate > 52%, DSR > 1.0, beats all 4 baselines by ≥0.2 Sharpe
- **Forecast-to-Fill tier**: Sharpe > 2.5 — almost no transformer paper here

## CRITICAL CORRECTIONS (Nia round 2 — kept from prior verification)

- ❌ Manual attention math (Q@K.T / sqrt(D) + masked_fill + softmax) → ✅ **`F.scaled_dot_product_attention(q, k, v, dropout_p=p, is_causal=True)`** — fused MPS kernel, 1.5-3× faster, fewer lines, drops causal_mask buffer. nanoGPT does this when SDPA is available.
- ❌ heads=12, head_dim=32 → ✅ **heads=6, head_dim=64** at D=384. head_dim=32 is unusually small (Llama 3 uses 128 even at 8B; sub-1B sweet spot is 64). Same total params, better-conditioned per-head subspace.
- ❌ Default dropout=0.1 → ✅ **dropout=0.2** (small data ~16K samples + transformer overfits fast). Modern LLMs use 0.0 because pretraining data is effectively infinite — opposite regime from us. Sweep 0.0-0.3 in training.
- ❌ Param count "~24M total" with "9.45M news projections" → ✅ **~18M total**, news projections are **3.15M total** (3 × 1.05M, NOT 3 × 1.05M × 3). To hit 50M target, raise to D=512, 10 layers OR D=768, 8 layers.
- ❌ Standard residual init → ✅ **scaled residual init** for `*proj.weight` layers: `std=0.02 / sqrt(2 * num_layers)`. GPT-2 trick, keeps activation variance stable as depth grows.
- ❌ NewsProjection raw output → ✅ add **`LayerNorm(projected_dim)` after each per-source Linear** (handles variable news-embedding magnitude across articles)
- ❌ `torch.compile` on MPS — try it → ✅ **DON'T**. PyTorch issue #171764: produces NaN loss on MPS where eager converges fine. Stay eager.
- ⚠️ SwiGLU vs GELU MLP: marginal gain (1-3% ppl) at this scale. Skip for v1 (Karpathy mode = type GELU, ship). future candidate if val loss plateaus.
- ⚠️ MPS BF16 autocast partial in 2026 (PyTorch #139386, #97236, #84516). Phase 1 = FP32 only. Phase 2 = autocast forward only (NEVER manual `.to(bfloat16)` on weights — explicit anti-pattern in PyTorch docs).
**Owner:** samsiavoshian
**Implementation effort:** 1 day (day 6 of week 1)

## Karpathy-Mode Discipline

You write every line. No HuggingFace `Trainer`. No Unsloth. No `transformers.AutoModel`. Raw PyTorch on MPS. The point is to understand attention, residuals, layer norm, and decoder-only causal masking by typing them.

## Specs (V1)

```
Architecture:        ENCODER-only HYBRID (NO causal mask, bidirectional)
                     V1: 10 transformer blocks + 2 sLSTM blocks (xLSTMTime style)
                     V1: Channel-independent + patches (PatchTST style)
Parameters:          ~24-40M (V1 sweet spot 30-35M; FiLM blocks add ~3840 params)
Hidden dim D:        384
Num layers:          12 total (10 transformer, 2 sLSTM at top)
Num heads:           6  (head_dim = 64 — Llama 3 / Qwen consensus, NOT 32) [transformer layers]
MLP hidden:          round(8D/3, 64) = 1024 at D=384  (SwiGLU)
Patches (V1):      P=4, S=4, T_bars=64 -> 16 patches per channel
Channels (V1):     681 (channel-independent backbone; one transformer for all)
Time context:        T=64 30min bars
Output:              V1 MULTI-TASK
                       Head A: Linear(D, 3, bias=False) on mean-pooled tokens, focal gamma=3
                       Head B: Linear(D, 1, bias=False) -> tanh -> [-1, +1], -Sharpe loss
Dropout:             0.2 (small-data regime); + V1 stochastic depth LINEAR 0.0 -> 0.2
Norm:                RMSNorm (NOT LayerNorm), pre-norm, eps=1e-6, + QK-Norm
Position encoding:   V1: Sinusoidal added to patches (per-channel) + RoPE real-form on Q/K
                     RoPE partial 10% of head_dim, theta=10000.0
                     QK-Norm BEFORE RoPE
Activation:          SwiGLU (NOT GELU) — Shazeer 2020 / Llama / Qwen consensus
Bias:                NO biases anywhere (Llama / Qwen / Mistral consensus)
RevIN:               V1: instance norm per INDIVIDUAL channel (681 instances; was per-group in V1)
                     Source: Huang & Yang ESWA 2026 — per-feature RevIN drops RMSE 50% / MAPE 54%
Series decomp:       V1: 24-bar moving-average kernel split into trend + seasonal pre-RevIN
                     (xLSTMTime requirement; Autoformer-style)
FiLM regime (V1):  Inserted at layers {2, 4, 6, 8, 10} between attn and FFN
                     12-dim regime vec: VIX-tercile (3) + RV-tercile over 60 bars (3)
                       + FOMC-week (1) + year-bucket (4) + HMM P(high-vol) (1)
                     gamma_l, beta_l = Linear_l(regime_vec) per layer
                     Total ~3840 trainable params
News fusion (V1):  Sparse cross-attn at layers [3, 7, 11] only
                     CFA-style FiLM/orthogonal projector before Flamingo K/V
                     AECF entropy-gated curriculum mask, p ~ U(0.0, 0.9)
                     is_news_present `nn.Embedding(2, 8)` concat to news token
                     Variable per-batch dropout p ~ U(0.1, 0.9) at training
```

Param count check:
- Per attention: 4 × D² = 4 × 384² = 590K
- Per MLP: 2 × D × mlp_hidden = 2 × 384 × 1536 = 1.18M
- Per block (attn + mlp): ~1.77M
- 8 blocks: ~14M
- Input projection 804 → 384: 308K
- Position embedding: 64 × 384 = 24K
- Output head 384 → 3: 1.2K
- News-source projections (3 × Linear(4096, 256)): 3.15M each = 9.45M

**Total: ~24M params** (less than the 50-100M target — we can scale up by raising D or num_layers if budget allows).

To hit 50M: try `D=512, num_layers=10, mlp_hidden=2048`. Calc: per-block ~3.1M × 10 = 31M, plus input/projections, total ~40-45M. Matches target.

To hit 100M: try `D=768, num_layers=12`. ~85-100M. Will fit on Mac mini 16GB but tight during training.

**Recommendation:** start at D=384 / 8 layers / ~24M total. Train it, see if val loss converges. Scale up only if val loss hasn't converged AND OOS Sharpe is improving with capacity (it usually doesn't on financial data — see prior warning).

## Code (~80 lines)

```python
# src/model/tiny_trader.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    def __init__(self, D: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert D % num_heads == 0
        self.D = D
        self.H = num_heads
        self.head_dim = D // num_heads
        self.qkv = nn.Linear(D, 3 * D, bias=False)
        self.proj = nn.Linear(D, D)
        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)
        # Causal mask buffer (no learnable params, registered for device-move)
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(1024, 1024)).view(1, 1, 1024, 1024),
            persistent=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.H, self.head_dim)
        q, k, v = qkv.unbind(dim=2)            # each (B, T, H, head_dim)
        q = q.transpose(1, 2)                  # (B, H, T, head_dim)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        scores = q @ k.transpose(-2, -1) / (self.head_dim ** 0.5)
        mask = self.causal_mask[:, :, :T, :T]
        scores = scores.masked_fill(mask == 0, float("-inf"))
        attn = F.softmax(scores, dim=-1)
        attn = self.attn_dropout(attn)

        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, D)
        return self.proj_dropout(self.proj(out))


class Block(nn.Module):
    def __init__(self, D: int, num_heads: int, mlp_hidden: int, dropout: float = 0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(D)
        self.attn = CausalSelfAttention(D, num_heads, dropout)
        self.ln2 = nn.LayerNorm(D)
        self.mlp = nn.Sequential(
            nn.Linear(D, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, D),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))   # pre-norm
        x = x + self.mlp(self.ln2(x))
        return x


class NewsProjection(nn.Module):
    """Projects each news source's 4096-dim raw Llama embedding down to 256."""
    def __init__(self, num_sources: int = 3, raw_dim: int = 4096, projected_dim: int = 256):
        super().__init__()
        self.projs = nn.ModuleList([
            nn.Linear(raw_dim, projected_dim) for _ in range(num_sources)
        ])

    def forward(self, news_raw: torch.Tensor) -> torch.Tensor:
        """news_raw: (B, T, num_sources * raw_dim) → (B, T, num_sources * projected_dim)"""
        B, T, total = news_raw.shape
        num_sources = len(self.projs)
        per_source = total // num_sources
        chunks = news_raw.chunk(num_sources, dim=-1)
        projected = [proj(chunk) for proj, chunk in zip(self.projs, chunks)]
        return torch.cat(projected, dim=-1)


class nanoGLD(nn.Module):
    def __init__(
        self,
        numeric_dim: int = 36,        # price + risk + macro + geo features
        news_raw_dim_per_source: int = 4096,
        num_news_sources: int = 3,
        news_projected_dim: int = 256,
        D: int = 384,
        num_heads: int = 12,
        num_layers: int = 8,
        T: int = 64,
        dropout: float = 0.1,
        num_classes: int = 3,
    ):
        super().__init__()
        self.T = T
        self.news_proj = NewsProjection(num_news_sources, news_raw_dim_per_source, news_projected_dim)
        # Total per-bar input dim after projection
        total_in = numeric_dim + num_news_sources * news_projected_dim   # e.g. 36 + 3*256 = 804
        self.input_proj = nn.Linear(total_in, D)
        self.pos_embed = nn.Parameter(torch.zeros(1, T, D))
        self.blocks = nn.ModuleList([
            Block(D, num_heads, 4 * D, dropout) for _ in range(num_layers)
        ])
        self.ln_f = nn.LayerNorm(D)
        self.head = nn.Linear(D, num_classes)
        self.dropout = nn.Dropout(dropout)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.trunc_normal_(m.weight, std=0.02)
        elif isinstance(m, nn.LayerNorm):
            nn.init.zeros_(m.bias)
            nn.init.ones_(m.weight)

    def forward(self, numeric: torch.Tensor, news_raw: torch.Tensor) -> torch.Tensor:
        """
        numeric:  (B, T, numeric_dim)            ← price + risk + macro + geo
        news_raw: (B, T, num_sources * 4096)     ← concatenated raw embeddings
        Returns logits: (B, num_classes)
        """
        news_projected = self.news_proj(news_raw)            # (B, T, 768)
        x = torch.cat([numeric, news_projected], dim=-1)     # (B, T, 804)
        x = self.input_proj(x)                                # (B, T, D)
        x = x + self.pos_embed[:, : x.size(1), :]
        x = self.dropout(x)

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        # Use final-token representation for direction prediction
        return self.head(x[:, -1, :])                         # (B, num_classes)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
```

## Forward-Pass Sanity Test

```python
def test_forward_pass():
    model = nanoGLD()
    B, T = 4, 64
    numeric = torch.randn(B, T, 36)
    news_raw = torch.randn(B, T, 3 * 4096)
    logits = model(numeric, news_raw)
    assert logits.shape == (B, 3)
    print(f"Param count: {count_params(model):,}")  # ~24M with default config
```

## Memory Estimate (Mac mini 16GB)

For batch B=32, T=64, D=384:

- Activations per layer: ~B × T × D × 4 bytes (FP32) = 32 × 64 × 384 × 4 = 3MB per tensor, with attention scores adding ~B × H × T² × 4 = 32 × 12 × 64² × 4 = 6MB per layer
- 8 layers + storage for backward: ~150-300MB activations
- Parameters: 24M × 4 bytes = 96MB (FP32) or 48MB (FP16)
- AdamW optimizer state: 2× params = 192MB FP32 or 96MB FP16
- Gradients: 1× params = 96MB FP32
- News raw embeddings batch: B × T × 3 × 4096 × 4 = 32 × 64 × 12288 × 4 = 100MB

Total: ~600MB-1GB peak during training. Fits trivially on 16GB Mac mini.

If we scale to D=768 / 12 layers / ~100M params: ~3-5GB peak. Still fits but no margin for OS.

## Mixed Precision on MPS

```python
# Apple Silicon MPS doesn't support full autocast yet, but we can use bfloat16 weights
model = model.to(device='mps', dtype=torch.float32)  # weights FP32 for stability
# OR for memory savings, FP16 weights with FP32 master copy via torch.amp.autocast (limited MPS support)
```

Current state: MPS autocast support is partial. Recommend FP32 weights for stability, bfloat16 only if OOM.

## Position Encoding Choice

**Why learned positional embedding (not sinusoidal):**
- T=64 is fixed throughout the project (we never extend sequence length)
- Learned positions have ~24K params (negligible)
- Easier to debug than sinusoidal (no math errors)
- For variable T, RoPE would be better, but we don't need that

## Output Choice (final-token vs mean-pool)

We use the **final token's hidden state** to produce logits. Alternative: mean-pool across all tokens.

**Why final token wins for autoregressive forecasting:**
- The final token has had attention over all previous tokens (causal mask)
- It's the "summary" of all 64 bars from the perspective of "predict the next bar"
- Standard pattern in GPT-style models
- Mean-pool would give equal weight to bar 1 (oldest, least relevant) and bar 64 (most recent)

## Why Not MoE / Bigger / Fancier

User mentioned MoE earlier. MoE shines when:
- Total params > 1B
- Multiple "expert" specializations are obvious (different domains, languages)
- You can amortize router overhead across many tokens

For our case (~50-100M, single domain, single asset), MoE adds complexity for no meaningful gain. Skip.

## V1 Reference Implementation (Modern Stack)

```python
# src/model/tiny_trader_v3.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class RMSNorm(nn.Module):
    """Llama-style RMSNorm. No mean centering, no bias. ~7-15% faster than LayerNorm."""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return norm * self.weight


def precompute_rope_cache(head_dim: int, max_seq: int, theta: float = 10000.0):
    """Real-form RoPE cache. NEVER use torch.view_as_complex on MPS (crashes)."""
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    t = torch.arange(max_seq).float()
    angles = torch.outer(t, freqs)  # (max_seq, head_dim/2)
    return torch.cos(angles), torch.sin(angles)


def apply_rope(x, cos, sin):
    """Real-form RoPE. x: (..., seq, head_dim)."""
    x1, x2 = x[..., 0::2], x[..., 1::2]
    cos = cos[: x.size(-2), :]
    sin = sin[: x.size(-2), :]
    rotated = torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
    return rotated.flatten(-2)


class CausalSelfAttention(nn.Module):
    """Encoder-only (NO causal mask) MHA + QK-Norm + RoPE + SDPA."""
    def __init__(self, D: int, num_heads: int, max_seq: int, dropout: float = 0.2):
        super().__init__()
        assert D % num_heads == 0
        self.D, self.H, self.head_dim = D, num_heads, D // num_heads
        self.qkv = nn.Linear(D, 3 * D, bias=False)
        self.proj = nn.Linear(D, D, bias=False)
        # QK-Norm: stabilizes attention at small scale (Qwen 3 / Gemma 3)
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)
        self.dropout = dropout
        # Pre-compute RoPE cache (real-form, MPS-safe)
        cos, sin = precompute_rope_cache(self.head_dim, max_seq)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        qkv = self.qkv(x).view(B, T, 3, self.H, self.head_dim)
        q, k, v = qkv.unbind(dim=2)              # (B, T, H, head_dim)
        # QK-Norm BEFORE RoPE
        q = self.q_norm(q)
        k = self.k_norm(k)
        # RoPE on Q and K (not V)
        q = apply_rope(q, self.rope_cos, self.rope_sin)
        k = apply_rope(k, self.rope_cos, self.rope_sin)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        # SDPA — encoder-only, NO causal mask. .contiguous() per PyTorch #181133
        out = F.scaled_dot_product_attention(
            q.contiguous(), k.contiguous(), v.contiguous(),
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,
        )
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        return self.proj(out)


class SwiGLU(nn.Module):
    """Llama-style SwiGLU FFN. hidden = round(8D/3, 64). 3 matrices, no bias."""
    def __init__(self, D: int, dropout: float = 0.2):
        super().__init__()
        hidden = ((int(8 * D / 3) + 63) // 64) * 64  # round to multiple of 64
        self.w_gate = nn.Linear(D, hidden, bias=False)
        self.w_up = nn.Linear(D, hidden, bias=False)
        self.w_down = nn.Linear(hidden, D, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.w_down(F.silu(self.w_gate(x)) * self.w_up(x)))


class Block(nn.Module):
    def __init__(self, D, num_heads, max_seq, dropout=0.2, drop_path=0.15):
        super().__init__()
        self.ln1 = RMSNorm(D)
        self.attn = CausalSelfAttention(D, num_heads, max_seq, dropout)
        self.ln2 = RMSNorm(D)
        self.mlp = SwiGLU(D, dropout)
        self.drop_path = drop_path  # stochastic depth probability

    def forward(self, x):
        if self.training and torch.rand(1).item() < self.drop_path:
            return x  # skip block (stochastic depth)
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class RevIN(nn.Module):
    """Reversible Instance Normalization per channel group (Kim ICLR 2022)."""
    def __init__(self, num_features: int, eps: float = 1e-5, affine: bool = True):
        super().__init__()
        self.eps = eps
        self.affine = affine
        if affine:
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, x, mode='norm'):
        if mode == 'norm':
            self.mean = x.mean(dim=-2, keepdim=True).detach()
            self.std = x.std(dim=-2, keepdim=True).detach() + self.eps
            x = (x - self.mean) / self.std
            if self.affine:
                x = x * self.weight + self.bias
        elif mode == 'denorm':
            if self.affine:
                x = (x - self.bias) / self.weight
            x = x * self.std + self.mean
        return x


class NewsFuser(nn.Module):
    """Perceiver-Resampler-lite + Flamingo-gated cross-attn (Agent 3 recommendation).

    V1 CHANGES (apply at integration time):
    - Insert ONLY at layers [3, 7, 11] of 12 (sparse, mPLUG-Owl3 arXiv:2408.04840 Table 8).
    - Add CFA-style FiLM/orthogonal projector before K/V projection (Lee arXiv:2603.22372):
        text_K = FiLM(gamma_bar, beta_bar) * text_proj(text)
        text_K = text_K - <text_K, bar_pool> * bar_pool / ||bar_pool||^2
        Bottleneck d_text=256 -> 64 -> d_model=384 (~50-100k extra params).
    - Replace 15% constant modality dropout with AECF (Chlon arXiv:2505.15417):
        Per-batch p ~ U(0.0, 0.9) (curriculum 0 -> 0.9, deployment matches 51% empirical absence).
        Entropy regularizer L_aecf = -lambda(x) * sum_m p_m log p_m, lambda(x) = MC-dropout entropy estimate.
    - Concat is_news_present binary embedding `nn.Embedding(2, 8)` to news token (Ma CVPR 2022).
    - Keep NEWS_NOT_PRESENT learnable token (V2 candidate to swap to CMPT).
    - Variable per-batch modality dropout p ~ U(0.1, 0.9) at training (CRITICAL FIX over V1's 15%).
    """
    def __init__(self, d_news=4096, d_model=384, n_sources=3, n_queries=8, n_heads=4):
        super().__init__()
        self.no_news = nn.Parameter(torch.randn(n_sources, d_news) * 0.02)
        self.proj = nn.ModuleList([nn.Linear(d_news, d_model, bias=False) for _ in range(n_sources)])
        self.alpha = nn.Parameter(torch.zeros(n_sources))   # init=0, opens gradually
        self.news_sa = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, batch_first=True, norm_first=True
        )
        self.queries = nn.Parameter(torch.randn(n_queries, d_model) * 0.02)
        self.cross = nn.MultiheadAttention(d_model, num_heads=n_heads, batch_first=True, bias=False)

    def forward(self, news_raw: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """news_raw: (B, n_sources, 4096), mask: (B, n_sources) 1=present 0=absent"""
        B, S, _ = news_raw.shape
        # F: substitute learned [NO_NEWS] for absent sources
        x = torch.where(mask.unsqueeze(-1).bool(), news_raw, self.no_news)
        # E: L2-normalize before projection
        x = F.normalize(x, dim=-1)
        # Project per source
        x = torch.stack([self.proj[s](x[:, s]) for s in range(S)], dim=1)  # (B, S, D)
        # C: per-source tanh gate, init=0
        x = x * torch.tanh(self.alpha).view(1, S, 1)
        # G: news self-attn (sources fuse together)
        x = self.news_sa(x)
        # B: Perceiver queries cross-attend to fused news
        q = self.queries.unsqueeze(0).expand(B, -1, -1)
        out, _ = self.cross(q, x, x)
        return out  # (B, n_queries, D)


class nanoGLDV1(nn.Module):
    """Encoder-only, channel-group tokenization, full modern stack.

    V1 NOTES (refactor on implementation):
    - Replace channel-group tokens with channel-independent + patches (P=4, S=4).
      681 channels share one transformer backbone. Patch projection Linear(P, D), no bias.
      Sinusoidal pos embedding added; RoPE on Q/K stays.
    - Replace last 2 transformer blocks (layers 11-12) with sLSTM blocks (xLSTMTime).
    - Insert FiLM regime conditioning at layers {2, 4, 6, 8, 10} between attn and FFN.
      12-dim regime vec: VIX-tercile + RV-tercile + FOMC + year-bucket + HMM-P-high-vol.
    - Sparse news cross-attn only at layers [3, 7, 11], not every layer.
    - Replace single 3-class head with multi-task dual head (Head A focal + Head B Sharpe).
    - Stochastic depth LINEAR schedule 0.0 -> 0.2 across depth (was uniform 0.15).
    """
    def __init__(
        self,
        numeric_dim: int = 36,             # price + risk + macro + geo
        news_raw_dim: int = 4096,
        n_news_sources: int = 3,
        n_news_queries: int = 8,
        D: int = 384,
        num_heads: int = 6,                # head_dim = 64
        num_layers: int = 12,
        T_bars: int = 64,
        n_classes: int = 3,
        dropout: float = 0.2,
        drop_path: float = 0.15,
    ):
        super().__init__()
        # RevIN per channel group
        self.revin_numeric = RevIN(numeric_dim)
        # Channel-group projections (each summarizes T=64 timesteps × features → 1 token of D)
        # In practice each "group" is its own time-collapsing Linear(T × group_dim, D)
        self.channel_groups = nn.ModuleDict({
            'price':       nn.Linear(T_bars * 12, D, bias=False),  # 12 price feats × 64 bars
            'macro':       nn.Linear(T_bars * 6, D, bias=False),
            'geo':         nn.Linear(T_bars * 10, D, bias=False),
            'risk':        nn.Linear(T_bars * 8, D, bias=False),
            'multi_2hr':   nn.Linear(T_bars // 4 * 12, D, bias=False),
            'multi_daily': nn.Linear(T_bars // 13 * 12, D, bias=False),
        })
        # News fuser → 8 query tokens
        self.news_fuser = NewsFuser(d_news=news_raw_dim, d_model=D,
                                     n_sources=n_news_sources, n_queries=n_news_queries)
        # Total tokens: 6 channel groups + 8 news queries = 14 tokens
        n_tokens = len(self.channel_groups) + n_news_queries
        self.pos_embed = nn.Parameter(torch.zeros(1, n_tokens, D))
        # Encoder blocks
        self.blocks = nn.ModuleList([
            Block(D, num_heads, max_seq=n_tokens, dropout=dropout, drop_path=drop_path)
            for _ in range(num_layers)
        ])
        self.ln_f = RMSNorm(D)
        self.head = nn.Linear(D, n_classes, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            # Scaled init on output projections (GPT-2 trick)
            for name, _ in self.named_parameters():
                if 'proj.weight' in name or 'w_down.weight' in name:
                    nn.init.trunc_normal_(m.weight, std=0.02 / math.sqrt(2 * len(self.blocks)))

    def forward(self, channel_inputs: dict, news_raw: torch.Tensor, news_mask: torch.Tensor):
        """
        channel_inputs: dict[group_name -> (B, T_group * features) flattened]
        news_raw: (B, 3, 4096)
        news_mask: (B, 3)
        """
        # Build channel-group tokens
        tokens = []
        for name, proj in self.channel_groups.items():
            tokens.append(proj(channel_inputs[name]).unsqueeze(1))  # (B, 1, D)
        # Add news query tokens
        news_tokens = self.news_fuser(news_raw, news_mask)  # (B, 8, D)
        x = torch.cat(tokens + [news_tokens], dim=1)  # (B, n_tokens, D)
        # Add learned positional embedding
        x = x + self.pos_embed
        x = self.dropout(x)
        # Encoder blocks
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        # Mean-pool across all tokens (vs final-token in decoder-only)
        return self.head(x.mean(dim=1))   # (B, n_classes)
```

Param count check at D=384, 12 layers, 6 heads:
- Per block: 4×D² (attn) + 3×D×(8D/3) (SwiGLU) ≈ 590K + 1.18M ≈ 1.77M
- 12 blocks: ~21.2M
- Channel projections: ~15M (depends on T)
- NewsFuser: ~4.5M
- **Total: ~40-50M params.** Bigger than earlier 24M target because we kept Channel projections fat. Tune by reducing T-collapse or D.

V1 param-count delta (vs V1 above):
- Channel-independent + patch projection: Linear(P=4, D=384) shared across 681 channels = 1536 params shared (effectively free vs V1's per-group tokens).
- 2 sLSTM blocks (layers 11-12) replace 2 transformer blocks: roughly comparable param budget; sLSTM adds gates but loses 4xD^2 attn matrices, net ~1.5-2M per layer (xLSTMTime sLSTM block size).
- FiLM regime: 5 layers x (2*D=768 params) = 3840 params total. Negligible.
- Multi-task Head B: Linear(D, 1, bias=False) = 384 params. Negligible.
- is_news_present `nn.Embedding(2, 8)`: 16 params. Negligible.
- CFA projector (text_dim=256 -> bottleneck=64 -> D=384): ~50-100k params.
- DANN domain classifier discriminator: ~50K params.
- AECF entropy regularizer params: ~3K.
- **V1 net: still 24-40M sweet spot, with above additions.** Verify with `python -m nanogld.model summary`.

## Open Questions / TODOs

- [ ] Decide final size (24M / 30M / 40M) after first training run convergence check
- [ ] V1: muP transfer-tune on 2-4M tiny model first ($5 budget); transfer LR / beta_2 / init scale / FSAM rho to 30M one-shot
- [ ] Test MPS bfloat16 weights vs FP32 — does training still converge?
- [ ] Profile attention compute — does flash-attention help on MPS? (Probably not yet supported)
- [ ] V1: A/B SimMTM vs T-JEPA after baseline lands (paper says comparable; SimMTM primary)
- [ ] V1: A/B FreeLB ON vs OFF after baseline lands (some risk of distribution-shift overcorrection)
- [ ] V1: Should regime vector include news-density bucket too? Currently no. Defer to post-baseline.

---

# Part 2 — Training Procedure

_(was doc 05-MODEL-TRAINING-CALIBRATION.md before V5 merge — content unchanged)_

# 05 — Training Procedure

## YOU ARE THE TRAINING ENGINEER AGENT

You own the training loop. You take the model class from doc 05, the feature DataFrame from doc 04, and produce trained model checkpoints + walk-forward validation results.

**Read 00-OVERVIEW.md FIRST.**
**Read 04-FEATURE-ENGINEERING.md** schema (input shape).
**Read 05-MODEL-TRAINING-CALIBRATION.md** model class signature.
**Also read 00-OVERVIEW.md "Execution Mode" section before coding.**

### Execution Mode (short — full rules in 00-OVERVIEW.md)

**This doc IS the plan. Do not rewrite it.** Plan to *execute* what is written. If a claim is wrong, AskUserQuestion. Silent scope drift = fired.

- **Research with Nia** before guessing on libs / APIs / papers. Spawn a subagent: `nia papers` / `nia github` / `nia search` / `nia oracle` / `nia packages` / `nia tracer`.
- **Execution skills only:** `/investigate`, `/review`, `/qa`, `/qa-only`, `/cso`, `/benchmark`, `/ship`, `/land-and-deploy`, `/canary`, `/document-release`, `/retro`, `/learn`, `/browse`, `/setup-browser-cookies`.
- **NO planning skills:** `/office-hours`, `/plan-*`, `/autoplan`, `/design-*`, `/devex-review`. Planning is done.
- **Loop:** read doc → Nia for unknowns → code → `/review` → `/qa` → `/cso` (if security-sensitive) → `/ship`.
- **Escalate after 3 failed attempts.** AskUserQuestion. Bad work is worse than no work.

### Files You Create

```
src/nanogld/training/
├── __init__.py
├── walk_forward.py         # Walk-forward CV splitter (4 folds at 5y, 1-week embargo)
├── dataset.py              # PyTorch Dataset wrapping feature DataFrame + cached embeddings
├── losses.py               # Class-weighted CE + label smoothing (NEVER MSE on returns)
├── optimizer.py            # Schedule-Free AdamW config + Friendly-SAM wrapper
├── ema.py                  # Exponential moving average of weights (decay=0.999)
├── augmentation.py         # Jittering + magnitude warping + Manifold Mixup
├── checkpoint.py           # Save/load with metadata (snapshot hash, fold, seed)
├── ssl_pretrain.py         # MAE-on-masked-bars (Stage 1 of 3-stage training)
├── linear_probe.py         # Stage 2: freeze encoder, train head only
├── llrd_finetune.py        # Stage 3: layer-wise LR decay full fine-tune
├── train_fold.py           # End-to-end one fold (called by walk_forward driver)
├── seeds.py                # Comprehensive RNG seeding (torch + mps + numpy + random)
└── cli.py                  # `python -m nanogld.training run --config configs/v4.yaml`

tests/
├── test_walk_forward.py    # Verify no train/val/test overlap, embargo respected
├── test_dataset.py         # Sample-by-sample shape verification
└── test_seed.py            # Same seed → same loss curve (within MPS non-determinism floor)

checkpoints/                # gitignored
├── fold_0_seed_42_ema.pt
├── fold_1_seed_42_ema.pt
└── ...
```

### Files You DO NOT Touch

- `src/nanogld/model/` — doc 05 owns the architecture; you wire it into the training loop
- `src/nanogld/embed/` — doc 03
- `src/nanogld/features/` — doc 04
- `src/nanogld/backtest/`, `sizing/`, `live/` — downstream consumers

### Stable Interface You Publish

```python
# doc 06 (backtest) loads your checkpoints:
checkpoint = torch.load("checkpoints/fold_3_seed_42_ema.pt", weights_only=True)
model.load_state_dict(checkpoint['ema_state_dict'])

# doc 08 (live) loads the SAME format
```

### Acceptance Criteria

1. ✅ `python -m nanogld.training run` completes 4 walk-forward folds
2. ✅ Loss decreases across epochs (no NaN, no plateau-at-init)
3. ✅ Val accuracy beats class-prior baseline (44% if always-flat) on at least 3/4 folds
4. ✅ Walk-forward unit test passes (no overlap, embargo respected)
5. ✅ Checkpoints saved with EMA state, metadata includes snapshot hash + fold + seed
6. ✅ Total training time < 12 hrs on M4 Mac mini for 4 folds × 1 seed
7. ✅ wandb workspace public, runs named `fold{N}_seed{N}` for X-thread material

### Spawn Nia Agents When You Need To

Especially:
- **Schedule-Free AdamW integration with PyTorch** — verify the train()/eval() calls and warmup_steps interaction with our walk-forward
- **Friendly-SAM impl details** — paper has math; spawn agent to find reference repo / usable implementation
- **MTS-JEPA implementation details** for Phase 2 — paper is from Feb 2026, code may not be public
- **MPS-specific seeding** — `torch.mps.manual_seed()` exists in PyTorch 2.11; verify it works as expected
- **wandb logging cadence on macOS** (avoid daemon issues with multiprocess)

### V1 Critical Decisions (DO NOT REVERT)

1. **V1: Cautious-Schedule-Free AdamW** — 5-line patch on top of Schedule-Free AdamW (Liang 2024 arXiv:2411.16085). `update *= (sign(update) == sign(grad)).float()`. 1.47x sample efficiency at zero hparam cost. Replaces vanilla Schedule-Free AdamW.
2. **Friendly-SAM REPLACES vanilla SAM** — filters gradient noise, drop-in. ρ=0.05 kept.
3. **V1: 3-stage training kept, with new losses per stage:**
   - Stage 1 SSL: SimMTM + CLIP-style bars<->news contrastive head + DANN era-label + AECF.
     L_pretrain = L_simmtm + 0.5*L_clip + 0.05*L_DANN + L_aecf
   - Stage 2 linear probe: focal loss gamma=3 only on Head A, encoder frozen. L = L_focal_3class.
   - Stage 3 LLRD fine-tune: multi-task. L = 0.5*L_focal + 0.5*L_sharpe_net + 0.05*L_DANN + L_aecf.
   - V1: Stage 3 anchored with Mixout p=0.7 to SSL checkpoint (Lee Cho Kang ICLR 2020 arXiv:1909.11299).
4. **EMA decay=0.999** — deployed model is EMA, NOT raw. KEPT.
5. **V1: dropout 0.2 + stochastic depth LINEAR 0.0->0.2 (was uniform 0.15) + focal loss gamma=3 (was LS=0.05)**.
6. **NEVER MSE on returns** (forecast-collapse rule arXiv:2604.00064). V1: focal gamma=3 + diff-Sharpe.
7. **Walk-forward = 4 folds.** V1 invariant: NYSE bars_per_year=3276 (NEVER 17500). Train 3y + val 6mo + test 6mo, step 3mo, 1-week embargo.
8. **PyTorch 2.11.0 pinned, FP32, num_workers=0**. V1: H100 has bf16 but stay FP32 deterministic for V1.
9. **V1: muP transfer-tune** — spend $5 on a 2-4M tiny model muP-parameterized sweep (LR, beta_2, init scale, FSAM rho). Transfer to 30M one-shot. Yang 2022 arXiv:2203.03466. Saves $30-50 of LR-guess risk on the $60-150 main run.
10. **V1: Mask ratio 0.40** in SSL (was 0.20; PatchTST default). V1's 0.20 was image-MAE inheritance, too easy.
11. **V1: SimMTM** replaces plain MAE — K=3 masked views, similarity-weighted reconstruction. Dong NeurIPS 2023 Spotlight arXiv:2302.00861. Beats MAE/TS2Vec/TF-C on UEA classification linear-probe.
12. **V1: CLIP-style bars<->news contrastive head** in SSL. InfoNCE on bar-rep vs news-rep within ±5 min, tau=0.07. Uses 40K Qwen3 embeddings as positive anchors.
13. **V1: SimPSI + Wave-Mask augmentation** REPLACES naive jittering. Ryu AAAI 2024 arXiv:2312.05790, Arabi 2024 arXiv:2408.10951. NAIVE JITTERING FORBIDDEN (Fons 2020 net-negative on Sharpe).
14. **V1: FreeLB on news embeddings ONLY** (not bars - PIT risk). K=2, epsilon=0.5. Zhu et al. ICLR 2020 arXiv:1909.11764.
15. **V1: DANN gradient reversal** on era-label = year-bucket. Lambda 0->0.1 ramp. Feng 2019 arXiv:1810.09936. ~50K extra params.
16. **V1: Variable per-batch modality dropout p ~ U(0.1, 0.9)** at training. NOT 15% constant. Training distribution must bracket inference distribution (51% empirical absence).

### A/B Candidates (after V1 baseline ships)

These are alternative components you can test if baseline plateaus:
- **V1: A/B SimMTM vs T-JEPA** (paper says comparable, SimMTM primary).
- **V1: A/B FreeLB ON vs OFF** (some risk of distribution-shift overcorrection).
- **Cross-asset transfer** SPY → GLD via LLRD (bonus experiment).
- **V2 (post-V1):** CMPT (Cross-Modal Proxy Tokens), bars-only distillation, full xLSTMTime swap, end-to-end Sharpe-only head, ACI online conformal, deep ensembles, TTT, online learning.

V1 explicit NOT (deferred / rejected per spec §11):
- MoE / KAN / Mamba / TTT / online learning / Muon / Sophia / deep ensembles.
- 75K samples too small for sparse routing. KASPER Sharpe 12.02 fraud-tier. Mamba 0.64 Sharpe on financial benchmark. Muon speedup inversely proportional to scale; 30M too small. Sophia 2x claim collapses under fair tuning (arXiv:2509.02046). Deep ensembles 5x compute kills budget.

A/B gate: ≥0.2 Sharpe improvement OOS (val), seed-averaged 5 seeds.

### Hand-off Protocol

1. Update STATUS.md with: best fold val accuracy, EMA checkpoint paths, training time, wandb workspace URL
2. Notify doc 06 (backtest) that checkpoints are ready
3. Document any deviations IN this doc

Now read the implementation specifics.

---

# 05 — Training Procedure

**Status:** ✅ Complete, implementation-ready, Nia-verified
**Last verified:** 2026-04-30

## V1 — Library addition: `accelerate==1.13.x` (May 2026)

Owner lifted "raw PyTorch only" for non-architectural infrastructure. Single library added: **`accelerate>=1.13,<2.0`** (HuggingFace).

**What it gives us (5 LOC):**
- Device-agnostic code (no `if mps else cpu` branches)
- Free checkpoint state save/load with optimizer + scheduler state
- `accelerator.set_seed()` covers torch + cuda + mps + numpy + random in one call
- Gradient accumulation handled by `accelerator.accumulate(model)`

**What it does NOT give (skip):**
- Mixed precision on MPS (near-zero gain, stay FP32)
- Distributed training (we are single-device)

**Pattern:**
```python
from accelerate import Accelerator

accelerator = Accelerator(mixed_precision="no")  # FP32 on MPS
model, optimizer, train_loader, val_loader = accelerator.prepare(
    model, optimizer, train_loader, val_loader
)
accelerator.set_seed(42)

# In loop:
for batch in train_loader:
    with accelerator.accumulate(model):
        loss = compute_loss(model, batch)
        accelerator.backward(loss)
        optimizer.step()
        optimizer.zero_grad()

# Checkpoint
accelerator.save_state("checkpoints/fold_3/")
```

**`transformers.Trainer` — optional, deferred decision.** If our Friendly-SAM `training_step` override fights Trainer assumptions (it might), stick with raw loop + accelerate. If clean override works, Trainer brings free EarlyStopping + W&B + checkpointing callbacks (~50 LOC override). Decide during implementation.

**Skipped permanently (MPS reasons):**
- Unsloth — requires Triton, no Mac support
- bitsandbytes 4-bit — CUDA-only
- `torch.compile` — still NaN on MPS in 2026
- `lightning-thunder` — NVIDIA-focused, no MPS
- torchtitan — multi-GPU LLM pretraining, overkill

**`peft==0.19.x` deferred** until/if embedder fine-tune day arrives (currently month 4+ per Agent E recommendation).

## TRAINING STACK (V1) — 2026 SOTA (May 2026, 6-agent research)

After earlier draft went live, ran another deep-research pass on training/optimization papers from Jun 2025 to May 2026. Several adopted improvements below.

### Changes (V1)

#### 1. V1: Cautious-Schedule-Free AdamW REPLACES "AdamW + cosine + warmup"

**Defazio et al., ICLR 2025 outstanding paper, arXiv:2405.15682.** Won MLCommons AlgoPerf 2024 across all workloads. Removes the LR schedule entirely via momentum-based iterate averaging. **Anytime-optimal** — checkpoint at any step is the best the model can be at that point.

V1: Apply the **Cautious update mask** patch (Liang 2024 arXiv:2411.16085) on top of Schedule-Free AdamW. 5-line patch. Mask updates where momentum disagrees with gradient. 1.47x sample efficiency at zero hparam cost. Drop-in compatible with SF-AdamW. Reference: `kyleliang919/C-Optim`.

```python
# Old draft:
# optimizer = torch.optim.AdamW(...)
# scheduler = cosine_lr_schedule(optimizer, warmup_steps, total_steps)

# V1 — Schedule-Free:
import schedulefree
optimizer = schedulefree.AdamWScheduleFree(
    model.parameters(),
    lr=1e-4,                # V1: replace with muP-transferred LR (tuned on 2-4M tiny model)
    betas=(0.9, 0.95),
    weight_decay=0.1,
    warmup_steps=300,
)

# V1 — add Cautious update mask (5-line patch inside SF-AdamW step):
# update = update * (sign(update) == sign(grad)).float()
# Mask updates where momentum disagrees with gradient.
# Reference: kyleliang919/C-Optim, arXiv:2411.16085

# In training loop:
optimizer.train()      # call at start of train phase
loss.backward()
optimizer.step()       # V1: cautious mask applied inside

# When validating, call optimizer's eval-mode method (switches to averaged weights)
# (See schedulefree docs — there is a corresponding state-switch call before validation)
```

Why for us: 16K samples means we ablate epoch count constantly. Schedule-Free removes "what if I train one more epoch" anxiety. Anytime-optimal = any saved checkpoint is the best model at that step.

Repo: https://github.com/facebookresearch/schedule_free

**Fallback if Schedule-Free underperforms:** WSD (Warmup-Stable-Decay), arXiv:2601.09000 (Jan 2026, ICML 2025). Linear warmup → flat plateau → 8-20% step cooldown. Compute-agnostic, lets you branch checkpoints.

#### 2. Friendly-SAM (F-SAM) REPLACES vanilla SAM

**arXiv:2403.12350 + revisited arXiv:2603.10048 (March 2026).** Decomposes SAM perturbation into "full gradient" + "stochastic noise"; only ascends along full-gradient component. Better generalization at same ρ.

Why for us: 16K samples = high gradient noise. Vanilla SAM amplifies noise into perturbation step. F-SAM filters it.

#### 3. Muon optimizer for 2D weight matrices — V1: SKIP

**Keller Jordan blog 2024 + arXiv:2502.16982 (Moonshot) + arXiv:2603.17970 MUD (March 2026).** Newton-Schulz orthogonalization on 2D gradients. AdamW handles embeddings/heads/norms. **DeepSeek V4 (April 2026) + Kimi-2 ship Muon in production.**

V1 SKIP: Muon speedup inversely proportional to scale. 30M is too small. V1 goes with Cautious-Schedule-Free AdamW + muP transfer-tune. (Spec §11.)

Repo: https://github.com/KellerJordan/Muon

#### 4. V1: SimMTM SSL pretrain REPLACES plain MAE

V1: plain MAE on masked bars.

V1: **SimMTM (Dong NeurIPS 2023 Spotlight, arXiv:2302.00861).** K=3 masked views per sample, encode each, reconstruct via similarity-weighted blending of neighbor representations. Beats MAE/TS2Vec/TF-C on UEA classification linear-probe.

Loss: `L_recon + lambda_sim * L_similarity` where lambda_sim=0.5.

Reference: `thuml/SimMTM`. ~30 LOC over current MAE.

Pretrain config (V1):
- Patch length 4 (consistent with main backbone). Decoder depth 2 (light decoder = better encoder, MAE finding).
- Mask ratio 0.40 (was V1's 0.20; PatchTST default). V1's 0.20 was image-MAE inheritance, too easy.
- 15-20 epochs pretrain. ~25-30% of total compute budget on SSL, 70-75% on stages 2+3.

#### 4b. V1: CLIP-style bars<->news contrastive head (NEW SSL objective)

NEW. Use the 40K Qwen3-256-d news embeddings as anchors. During SSL, add a contrastive InfoNCE loss between bar-rep (CLS) and news-rep for bars within ±5 min of a news event.

`L_clip = -log(exp(sim(z_bar, z_news+) / tau) / sum_neg exp(sim(z_bar, z_neg) / tau))`.

Tau=0.07. ~5% extra pretrain compute. Gives encoder text-semantic priors for free.

#### 4c. V1: Combined SSL loss

Stage 1 SSL: `L_pretrain = L_simmtm + 0.5 * L_clip + 0.05 * L_DANN + L_aecf`.

#### 4d. MTS-JEPA — V1: deferred A/B

**arXiv:2602.04643 (Feb 2026), Multi-variate Time Series JEPA.** Predict masked target EMBEDDINGS (not raw values) from context embeddings. V1 keeps SimMTM as primary SSL; A/B SimMTM vs T-JEPA after baseline lands (paper says comparable).

#### 5. Cross-Asset Transfer Learning (cheap experiment)

**Springer 2025 cross-asset transfer transformer.** Pretrain on broad-market index (SPY 30min, far more data), fine-tune on GLD with LLRD. Maps onto our existing LLRD pipeline. Documented gains over single-asset baselines.

Add as bonus experiment after baseline ships.

#### 6. Conformal Prediction for sizing (deployment-side, see doc 07)

Wrap softmax probabilities with split-conformal calibration on a held-out slice. Produces calibrated prediction intervals. Use **interval width** to size positions. See doc 07 for full implementation.

#### 7. Multi-Dimensional Sentiment (per arXiv:2603.11408, March 2026)

Paper found on WTI crude (gold-adjacent commodity): **intensity + uncertainty matter MORE than polarity.** Currently doc 04 conflates these — embed news, project, done.

Update: extract THREE sentiment scores per news item using LLM prompts:
- Polarity: -1 (bearish) to +1 (bullish)
- Intensity: 0 (mild) to 1 (urgent/breaking)
- Uncertainty: 0 (certain) to 1 (speculative/conditional)

Add as 3 cheap features in `geo_features` table (arXiv:2603.11408 found these dominate via SHAP).

### Training Stack Summary (V1)

```
Optimizer:      V1: Cautious-Schedule-Free AdamW (β=0.9, β2=0.95, wd=0.1, warmup_steps=300)
                LR from muP transfer-tune on 2-4M tiny model ($5 budget upfront)
                Cautious patch: update *= (sign(update) == sign(grad)).float()
                (Liang 2024 arXiv:2411.16085, ref kyleliang919/C-Optim)
                V1 SKIP: Muon (too small scale per spec §11)
LR schedule:    NONE (Schedule-Free handles this)
Sharpness:      Friendly-SAM ρ=0.05 (replace vanilla SAM, KEPT)
EMA:            decay=0.999 (KEPT)
Regularization: V1: dropout 0.2 + stochastic depth LINEAR 0.0->0.2 + focal loss gamma=3
                V1: + Mixout p=0.7 anchored to SSL ckpt (Stage 3 LLRD only)
                  Lee Cho Kang ICLR 2020 arXiv:1909.11299; ref bloodwass/mixout
                V1: + DANN gradient reversal on era-label = year-bucket
                  Lambda 0->0.1 ramp; Feng 2019 arXiv:1810.09936
                  +3.11% on stock prediction; ~50K extra params for discriminator
Augmentation:   V1: SimPSI + Wave-Mask (NAIVE JITTERING FORBIDDEN per Fons 2020)
                  Ryu AAAI 2024 arXiv:2312.05790; Arabi 2024 arXiv:2408.10951
                  SimPSI reweights aug strength so dominant freq components preserved
                  Wave-Mask masks DWT coefficients
                  Both PIT-safe (no time shift)
Adversarial:    V1: FreeLB on news embeddings ONLY (not bars - PIT risk)
                  K=2, epsilon=0.5; Zhu et al. ICLR 2020 arXiv:1909.11764
                  ~30% wall-clock overhead, addresses regime-shift defense
                  ref zhuchen03/FreeLB
Manifold Mixup: at hidden states α=0.2 (KEPT, NEVER raw input)
Modality dropout: V1: VARIABLE per-batch p ~ U(0.1, 0.9) at training (was 15% constant)
                  CRITICAL FIX: matches empirical 51% absence rate
                  V1: + AECF entropy-gated curriculum mask in news fusion
                  Chlon arXiv:2505.15417
SSL pretrain:   V1: SimMTM (was plain MAE)
                  K=3 masked views, similarity-weighted reconstruction
                  Mask ratio 0.40 (was 0.20)
                  Dong NeurIPS 2023 arXiv:2302.00861; ref thuml/SimMTM
                V1: + CLIP-style bars<->news contrastive head
                  InfoNCE on bar-rep vs news-rep within ±5min, tau=0.07
                  Uses 40K Qwen3 embeddings as positive anchors
                  ~5% extra pretrain compute
                Pretrain config: P=4, decoder depth=2, 15-20 epochs, ~25-30% budget
Loss per stage:
  Stage 1 SSL:   L = L_simmtm + 0.5*L_clip + 0.05*L_DANN + L_aecf
  Stage 2 probe: L = L_focal_3class only (encoder frozen, head only)
  Stage 3 LLRD:  L = 0.5*L_focal + 0.5*L_sharpe_net + 0.05*L_DANN + L_aecf
  V1: Mixout p=0.7 anchored to SSL checkpoint at Stage 3
  Layer-wise LR decay 0.85 at Stage 3
  V1: focal loss gamma=3 (NOT vanilla CE, NOT LS=0.05) per spec invariant 22
NEVER MSE on returns (forecast-collapse rule arXiv:2604.00064)
Linear-probe → LLRD fine-tune (KEPT, with V1 stage losses above)
Cross-asset transfer: SPY→GLD as bonus experiment after baseline
Calibration:    V1: T-scaling -> RAPS -> AgACI (was T-scaling -> APS)
                V1: Laplace last-layer (LLLA) for epistemic (was MC dropout T=20)
                Snapshot ensemble (last 3 EMA checkpoints) KEPT
Mixed precision: FP32 weights (V1: H100 has bf16 but stay FP32 deterministic for V1)
```

### What we skip (saved investigation)

- Sophia optimizer — at <150M scale ties Signum, barely beats AdamW
- FP8 mixed precision — H100-only, unstable for small models per NVIDIA + Axolotl 2025-2026 docs
- Knowledge distillation from foundation model — no useful teacher available yet
- Pure Mamba / xLSTM as backbone — covered in doc 05 (skip at our scale)

## SMALL-DATA TRAINING STACK V1 (kept for reference)

For 16K labeled samples, regularization beats capacity. Stack is non-negotiable. Each piece has documented ROI.

### Stage 1 — SSL Pretraining (MUST, single biggest win)

V1: SimMTM-style multi-mask reconstruction on same 5y unlabeled bars BEFORE classification fine-tune. Documented 25-50% gain on small-data classification (Financial Fine-tuning paper, SSL4TS literature). V1: SimMTM beats plain MAE on UEA classification linear-probe (Dong NeurIPS 2023 Spotlight).

```python
# V1 — SimMTM head: K=3 masked views, similarity-weighted reconstruction
class SimMTMHead(nn.Module):
    """V1: SimMTM (Dong NeurIPS 2023, arXiv:2302.00861).
    K=3 masked views per sample, encode each, reconstruct via similarity-weighted blending.
    Reference: thuml/SimMTM. ~30 LOC over plain MAE.
    """
    def __init__(self, D, target_dim, K=3, lambda_sim=0.5):
        super().__init__()
        self.proj = nn.Linear(D, target_dim, bias=False)
        self.K = K
        self.lambda_sim = lambda_sim
    def forward(self, hidden_views):
        # hidden_views: list of K encoded views; produce similarity-weighted reconstructions
        ...

def mask_bars_simmtm(x, mask_ratio=0.40, K=3):
    """V1: Mask 40% of input bars (was 20%; PatchTST default), generate K views."""
    B, T, F = x.shape
    n_mask = int(T * mask_ratio)
    views = []
    for _ in range(K):
        mask_idx = torch.randperm(T)[:n_mask]
        x_masked = x.clone()
        x_masked[:, mask_idx, :] = 0  # or learned [MASK] token
        views.append((x_masked, mask_idx, x[:, mask_idx, :]))
    return views
```

Train SimMTM for ~15-20 epochs (~25-30% of total compute budget). Save encoder weights. Don't train classification head yet.

V1: Also wire **CLIP-style bars<->news contrastive head** during SSL. InfoNCE on bar-rep (CLS) vs news-rep for bars within ±5 min of a news event. Tau=0.07. Uses 40K Qwen3 embeddings as positive anchors.

```python
# V1 — CLIP-style contrastive head
def clip_loss_bars_news(z_bar, z_news_pos, z_news_negs, tau=0.07):
    """InfoNCE: pull bar-rep toward news-rep within +/-5min, push away from negatives."""
    sim_pos = (z_bar * z_news_pos).sum(-1) / tau
    sim_negs = (z_bar.unsqueeze(1) * z_news_negs).sum(-1) / tau   # (B, N_neg)
    return -sim_pos + torch.logsumexp(torch.cat([sim_pos.unsqueeze(1), sim_negs], dim=1), dim=1)

# Combined SSL loss:
# L_pretrain = L_simmtm + 0.5 * L_clip + 0.05 * L_DANN + L_aecf
```

### Stage 2 — Linear-probe (5-10 epochs)

Freeze encoder. Train ONLY classification head on direction labels. Sanity check that pretrained features are useful.

```python
for p in model.encoder.parameters():
    p.requires_grad = False
# train head only, fast convergence
```

### Stage 3 — Full fine-tune with LLRD (layer-wise LR decay 0.85) + V1: Mixout p=0.7

Unfreeze encoder. Lower layers get smaller LR (preserve general features), upper layers + head get full LR.

V1: Add **Mixout p=0.7** anchored to SSL checkpoint. Bernoulli-mix between current and SSL-anchor weights. Acts as L2-toward-pretrained constraint with adaptive coefficient. Lee, Cho, Kang ICLR 2020 arXiv:1909.11299. Reference impl: `bloodwass/mixout`.

V1: Stage 3 loss is **multi-task**: `L = 0.5 * L_focal + 0.5 * L_sharpe_net + 0.05 * L_DANN + L_aecf`.

```python
def llrd_param_groups(model, base_lr=1e-4, decay=0.85):
    groups = []
    n_layers = len(model.blocks)
    for i, block in enumerate(model.blocks):
        lr = base_lr * (decay ** (n_layers - i - 1))
        groups.append({'params': block.parameters(), 'lr': lr})
    # V1: both heads get full LR (Head A 3-class focal + Head B tanh position)
    groups.append({'params': model.head_a.parameters(), 'lr': base_lr})
    groups.append({'params': model.head_b.parameters(), 'lr': base_lr})
    return groups


# V1 — Mixout p=0.7 anchor patch
class Mixout:
    """V1: Bernoulli-mix current and SSL-anchor weights. p=0.7 default.
    Acts as L2-toward-pretrained at adaptive coefficient.
    Lee Cho Kang ICLR 2020 arXiv:1909.11299; ref bloodwass/mixout.
    """
    def __init__(self, anchor_state_dict, p=0.7):
        self.anchor = anchor_state_dict
        self.p = p
    def apply(self, model):
        with torch.no_grad():
            for name, p in model.named_parameters():
                if name in self.anchor:
                    mask = torch.bernoulli(torch.full_like(p, self.p)).bool()
                    p.copy_(torch.where(mask, self.anchor[name].to(p.device), p))
```

### SAM Optimizer (MUST — 2× compute, big gain). V1: Friendly-SAM ρ=0.05.

SAMformer (ICML 2024) showed SAM is the unlock for transformers on time series — **14.33% MSE improvement over TSMixer**, 4× fewer parameters.

V1: Use **Friendly-SAM** (arXiv:2403.12350) instead of vanilla SAM, ρ=0.05. Filters gradient noise; better generalization at same ρ. Wraps the V1 Cautious-Schedule-Free AdamW base optimizer.

```python
# V1 — Friendly-SAM wrapping Cautious-Schedule-Free AdamW
import schedulefree
from friendly_sam import FriendlySAM   # ref arXiv:2403.12350

base_optim = schedulefree.AdamWScheduleFree(
    model.parameters(),
    lr=lr_from_mup_transfer,           # V1: muP-tuned on 2-4M tiny model
    betas=(0.9, 0.95),
    weight_decay=0.1,
    warmup_steps=300,
)
# Apply Cautious patch: update *= (sign(update) == sign(grad)).float()
optim = FriendlySAM(model.parameters(), base_optim, rho=0.05, adaptive=False)

# V1 multi-task loss (Stage 3 LLRD):
# L = 0.5*focal_loss_v15(logits, labels, gamma=3.0) + 0.5*sharpe_loss_v15(position_weight, r_next, w_prev)
#     + 0.05*dann_loss(z, era_label) + l_aecf
def closure():
    out = model(x_bars, x_news, news_mask, is_news_present, regime_vec)
    loss = (0.5 * focal_loss_v15(out["logits_3class"], y, gamma=3.0)
            + 0.5 * sharpe_loss_v15(out["position_weight"], r_next, w_prev)
            + 0.05 * dann_loss(out["z"], era_label)
            + l_aecf_from_aecf_module(out["aecf_mask_p"]))
    loss.backward()
    return loss

# First step
out = model(x_bars, x_news, news_mask, is_news_present, regime_vec)
loss = (0.5 * focal_loss_v15(out["logits_3class"], y, gamma=3.0)
        + 0.5 * sharpe_loss_v15(out["position_weight"], r_next, w_prev)
        + 0.05 * dann_loss(out["z"], era_label)
        + l_aecf_from_aecf_module(out["aecf_mask_p"]))
loss.backward()
optim.first_step(zero_grad=True)
# Second step
closure()  # recompute loss with perturbed weights, backward
optim.second_step(zero_grad=True)
```

### EMA Weights (MUST — cheap, robust to label noise)

```python
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn

ema_model = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(0.999))

# In training loop after each batch
ema_model.update_parameters(model)

# Deploy ema_model, NOT model
```

### Stochastic Depth + Label Smoothing (MUST — free regularization)

- Stochastic depth p=0.15 (in Block.forward)
- Label smoothing ε=0.1 in F.cross_entropy

### Manifold Mixup (recommended — at hidden states, NOT raw input)

```python
def manifold_mixup(hidden, labels, alpha=0.2):
    """Mixup at intermediate hidden state, preserves temporal structure."""
    lam = np.random.beta(alpha, alpha)
    perm = torch.randperm(hidden.size(0))
    mixed = lam * hidden + (1 - lam) * hidden[perm]
    return mixed, labels, labels[perm], lam
```

Apply ONLY at hidden states (between blocks), NEVER on raw input (destroys temporal structure).

### Augmentation: V1 SimPSI + Wave-Mask (NAIVE JITTERING FORBIDDEN)

V1 invariant 21: SimPSI / Wave-Mask only; naive jittering FORBIDDEN. Fons 2020 (arXiv:2010.15111) shows naive jittering is NET NEGATIVE on Sharpe.

V1 augmentation stack:
- **SimPSI** (Ryu et al. AAAI 2024 arXiv:2312.05790): reweights aug strength so dominant frequency components preserved (jittering kills trend signal).
- **Wave-Mask** (Arabi 2024 arXiv:2408.10951): masks DWT coefficients.
- Both PIT-safe (no time shift).

```python
# V1 — SimPSI + Wave-Mask. NAIVE JITTERING FORBIDDEN.
def augment_v15(x):
    """V1 augmentation. Spec invariant 21: SimPSI / Wave-Mask only.
    Reweights aug by spectral importance so dominant frequencies preserved.
    Both PIT-safe (no time shift).
    """
    # SimPSI spectral-preserving:
    # 1. Compute spectral importance per channel via FFT on rolling window
    # 2. Apply jitter scaled INVERSELY by spectral importance per freq band
    # Wave-Mask:
    # 1. DWT of x
    # 2. Random mask of high-freq detail coefs (preserve approximation)
    # 3. Inverse DWT
    ...

# DO NOT USE: naive jittering or magnitude warping (Fons 2020 net-negative on Sharpe)
```

Manifold Mixup α=0.2 at hidden states: KEPT.

NEVER raw-input mixup — destroys temporal structure AND can leak future across mixup pairs.

### TTA at validation (K=5)

```python
def tta_predict(model, x, K=5):
    preds = []
    for _ in range(K):
        x_aug = augment(x)
        preds.append(F.softmax(model(x_aug), dim=-1))
    return torch.stack(preds).mean(dim=0)
```

Use for validation and offline backtest. Skip for live (latency).

### V1: Modality Dropout — Variable per-batch p ~ U(0.1, 0.9) (REPLACES 15% constant)

V1 invariant 24: Variable per-batch modality dropout p~U(0.1, 0.9), NOT 15% constant. Training distribution must bracket inference distribution. 15% guarantees performance cliff at 51% empirical no-news rate.

```python
def modality_dropout_v15(news_mask, training=True):
    """V1: per-batch dropout probability sampled in [0.1, 0.9].
    Matches empirical 51% absence rate.
    CRITICAL FIX over V1's 15% constant.
    """
    if not training:
        return news_mask
    # Sample one p per batch
    p = torch.rand(1).item() * 0.8 + 0.1  # uniform in [0.1, 0.9]
    drop = torch.rand_like(news_mask.float()) < p
    return news_mask & ~drop
```

V1 also wires **AECF entropy-gated curriculum mask** in news fusion (see news_fuser.py change above; Chlon arXiv:2505.15417). Curriculum: 0.0 -> 0.9 over training. Entropy regularizer `L_aecf = -lambda(x) * sum_m p_m log p_m`, lambda(x) = MC-dropout entropy estimate. PAC-bound on calibration across 2^M-1 modality subsets.

V1 also concats **`is_news_present` binary embedding** `nn.Embedding(2, 8)` to news token. Explicit gate signal so model does not infer presence from zero values (Ma et al. CVPR 2022 arXiv:2204.05454). Trivial cost.

### Implementation Order (by ROI)

1. EMA + stochastic depth + label smoothing + jittering — cheap, immediate signal (~1 day)
2. SSL pretrain → linear-probe → LLRD fine-tune (the big one, multi-day)
3. SAM optimizer (2× compute, validate gain)
4. Manifold Mixup + TTA (polish)

Expected total stack lift: +0.3-0.6 Sharpe at val (baseline ~0.5-1.0). Compounding is where the gain lives, not any single trick.

## CRITICAL CORRECTIONS (Nia round 2 — kept)

- ❌ "~6-8 folds at 5y" → ✅ **4 folds at 5y** (math: floor((60-48)/3) = 4). To get 6 folds, shorten train window 3y→2.5y.
- ❌ Peak LR `3e-4` default → ✅ **`1e-4` default** (3e-4 is too aggressive for 24M model on 16K samples). Sweep up to 3e-4 only after LR-range test on fold 0.
- ❌ Patience 3 → ✅ **patience 5** with `min_delta=1e-4` (small val sets are noisy; 3 fires too early)
- ❌ Manual `p.data.to(torch.bfloat16)` weight cast → ✅ **autocast forward only**, keep weights FP32. PyTorch docs explicitly warn against the manual cast pattern with autocast.
- ❌ Only `torch.manual_seed(seed)` → ✅ **set_seed helper** that seeds `torch + torch.mps + numpy + random`. MPS dropout still non-deterministic — document as known limitation.
- ❌ Vague wandb auth → ✅ read `WANDB_API_KEY` env var with boot-time validation, fail loudly if missing, support `mode='offline'` for airgapped runs
- ❌ TODO "try gradient accumulation" → ✅ **delete** — arXiv:2507.07101 says gradient accumulation isn't justified for transformer LM training unless multi-device
- ⚠️ MPS bf16 transformer training has known correctness issues (PyTorch #139386, #97236, #84516). Phase 2 only after fold-0 numerical equivalence test (FP32 vs bf16 logits, abs diff < 1e-2)
**Owner:** samsiavoshian
**Implementation effort:** 1 day after model + features land

## Locked Specs (V1)

| Component | Choice | Why |
|-----------|--------|-----|
| Optimizer | **V1: Cautious-Schedule-Free AdamW** | Liang 2024 arXiv:2411.16085 + Defazio ICLR 2025; 1.47x sample efficiency at zero hparam cost; anytime-optimal |
| betas | (0.9, 0.95) | Llama / GPT convention; β2=0.95 makes second-moment more responsive |
| Peak LR | **V1: muP-transferred** | Tune on 2-4M tiny model first ($5), transfer to 30M; Yang 2022 arXiv:2203.03466 |
| Weight decay | 0.1 | Standard transformer regularization |
| Weight decay groups | bias + RMSNorm + pos_embed = no decay | Standard practice |
| LR schedule | **V1: NONE** (Schedule-Free) | Anytime-optimal; ablate epoch count without anxiety |
| Warmup | 300 steps | Schedule-Free default |
| Sharpness regularizer | Friendly-SAM ρ=0.05 | Filters gradient noise from vanilla SAM; arXiv:2403.12350 |
| Loss (Head A) | **V1: focal loss gamma=3** | Mukhoti 2020 arXiv:2002.09437; required for clean T-scaling/RAPS interaction (Xi 2024 arXiv:2402.04344) |
| Loss (Head B, V1) | **differentiable -Sharpe** | L_sharpe_net = -mean(w*r_next - cost*|w_t-w_{t-1}|) / std(...); Hwang & Zohren 2025 arXiv:2510.03129 |
| Combined loss S3 | 0.5*L_focal + 0.5*L_sharpe_net + 0.05*L_DANN + L_aecf | Multi-task decision-aware (V1 invariant 25) |
| Class weights | None | Class imbalance handled by triple-barrier neutral threshold (doc 04) + per-class reporting; class weights would break calibration |
| Label smoothing | **V1: REPLACED by focal** | LS=0.1/0.05 dropped; focal gamma=3 subsumes the calibration role |
| Gradient clipping | max_norm 1.0 | Prevents gradient explosion on outlier batches |
| Dropout | 0.2 (in attention + MLP) | Set in model |
| Stochastic depth | **V1: linear 0.0 -> 0.2** | Concentrates regularization at higher layers; Touvron 2021 |
| Mixout (V1, S3 only) | p=0.7 anchored to SSL ckpt | Lee Cho Kang ICLR 2020 arXiv:1909.11299 |
| DANN (V1) | era-label = year-bucket; lambda 0->0.1 ramp | Feng 2019 arXiv:1810.09936; +3.11% on stock prediction |
| FreeLB (V1) | News embeddings only, K=2, ε=0.5 | Zhu et al. ICLR 2020 arXiv:1909.11764; bars excluded (PIT risk) |
| Mixed precision | FP32 phase 1, bfloat16 phase 2 if needed | MPS support partial; V1 keeps FP32 deterministic |
| Batch size | 32 train, 64 val/test | Fits 16GB Mac mini comfortably; H100 use 64-128 |
| Epochs per fold | 20 max with early stopping | Usually converges in 5-12 |
| Early stopping | **Patience 5, min_delta=1e-4** | Small val sets noisy; 3 fires too early |
| Random seed | 42 default, multi-seed in week 4+ | Single-seed for week 1, 5-seed for confidence |
| Walk-forward (V1) | **NYSE bars_per_year=3276** (NEVER 17500) | V1 invariant 5; 4 folds, 1-week embargo, train 3y + val 6mo + test 6mo, step 3mo |
| Modality dropout (V1) | Variable per-batch p ~ U(0.1, 0.9) | Matches 51% empirical absence; invariant 24 |
| News cross-attn layers (V1) | [3, 7, 11] sparse | mPLUG-Owl3 arXiv:2408.04840 Table 8 |
| FiLM regime (V1) | Layers {2, 4, 6, 8, 10} | 12-dim regime vec, ~3840 params |
| Mask ratio SSL (V1) | 0.40 (was 0.20) | PatchTST default; V1's 0.20 was too easy |
| SSL objective (V1) | SimMTM + 0.5*L_clip + 0.05*L_DANN + L_aecf | SimMTM K=3 views; CLIP bars<->news InfoNCE tau=0.07 |

## Walk-Forward Cross-Validation

Anchored walk-forward with 1-week embargo:
- Train: 3 years (anchored from data start)
- Val: 6 months (after embargo)
- Test: 6 months (after embargo)
- Step: 3 months (slide forward)

For 5y of data: **4 folds exactly** (60mo - 48mo overhead) / 3mo step = 4. To get 6 folds, shorten train window from 3y to 2.5y.

V1 invariant 5: **NYSE bars_per_year = 3276** (NEVER 17500). 30-min RTH bars only. All Sharpe annualization, vol target conversion, and walk-forward arithmetic uses 3276.

V1 also adds (per spec §9 + invariants 18-20, owned by doc 06 but referenced here):
- Per-bucket eval: every metric reported separately for {news-present, news-absent, both}.
- Cost-stress: report Sharpe at {0.5x, 1.0x, 1.5x} cost levels. Hard gate Sharpe > 0.5 at 1.5x cost.
- DSR > 1.0 hard gate (Bailey & López de Prado).
- V1 promotion gates (per spec §9.4):
   1. Walk-forward Sharpe > 1.0 net of 1x cost (was 0.8).
   2. Sharpe > 0.5 net of 1.5x cost (NEW hard).
   3. Beats best baseline by >= 0.2 Sharpe on >= 3 of 4 folds.
   4. Conformal coverage within ±2% of nominal on val + per-bucket.
   5. Stage 2 sizer (decision-aware head) beats Stage 1 fallback by >= 0.2 Sharpe OOS.
   6. Drawdown circuit breaker tested on >= 2 historical regimes.
   7. Deflated Sharpe Ratio > 1.0 (NEW hard).
   8. Per-bucket Sharpe (news-present, news-absent) both positive (NEW hard).

```python
@dataclass
class Fold:
    fold_idx: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    val_start: pd.Timestamp
    val_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


def walk_forward_splits(
    timestamps: pd.Series,
    train_years: float = 3.0,
    val_months: float = 6.0,
    test_months: float = 6.0,
    step_months: float = 3.0,
    embargo_days: int = 7,
) -> Iterator[Fold]:
    """Anchored walk-forward. Train always starts at data_start, extends each step."""
    embargo = pd.Timedelta(days=embargo_days)
    train_dur = pd.Timedelta(days=train_years * 365)
    val_dur = pd.Timedelta(days=val_months * 30)
    test_dur = pd.Timedelta(days=test_months * 30)
    step = pd.Timedelta(days=step_months * 30)

    data_start = timestamps.min()
    data_end = timestamps.max()

    fold_idx = 0
    train_end = data_start + train_dur

    while True:
        val_start = train_end + embargo
        val_end = val_start + val_dur
        test_start = val_end + embargo
        test_end = test_start + test_dur

        if test_end > data_end:
            break

        yield Fold(fold_idx, data_start, train_end, val_start, val_end, test_start, test_end)
        fold_idx += 1
        train_end += step
```

## Optimizer Configuration

V1: This V1 vanilla AdamW snippet is REPLACED by Cautious-Schedule-Free AdamW (see "V1: Cautious-Schedule-Free AdamW" section above). Kept here for reference of the WD groups pattern.

```python
def configure_optimizer_v1_legacy(model: nn.Module, lr: float = 3e-4, weight_decay: float = 0.1) -> torch.optim.AdamW:
    """V1 LEGACY — superseded by V1 Cautious-Schedule-Free AdamW.
    AdamW with no decay on biases / LayerNorm / pos_embed.
    """
    decay_params = []
    no_decay_params = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.endswith('.bias') or 'norm' in name.lower() or 'pos_embed' in name:
            no_decay_params.append(p)
        else:
            decay_params.append(p)

    return torch.optim.AdamW(
        [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': no_decay_params, 'weight_decay': 0.0},
        ],
        lr=lr, betas=(0.9, 0.95), eps=1e-8,
    )


# V1 — Cautious-Schedule-Free AdamW
def configure_optimizer_v15(model: nn.Module, lr: float, weight_decay: float = 0.1):
    """V1: Cautious-Schedule-Free AdamW.
    LR comes from muP transfer-tune on 2-4M tiny model.
    Cautious patch (Liang 2024 arXiv:2411.16085): mask updates where momentum disagrees with grad.
    """
    import schedulefree
    decay_params, no_decay_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.endswith('.bias') or 'norm' in name.lower() or 'pos_embed' in name:
            no_decay_params.append(p)
        else:
            decay_params.append(p)

    optimizer = schedulefree.AdamWScheduleFree(
        [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': no_decay_params, 'weight_decay': 0.0},
        ],
        lr=lr,
        betas=(0.9, 0.95),
        warmup_steps=300,
    )
    # Wrap step() with Cautious mask: update *= (sign(update) == sign(grad)).float()
    # See kyleliang919/C-Optim for the 5-line patch.
    return optimizer
```

## LR Schedule

```python
import math

def cosine_lr_schedule(optimizer, warmup_steps: int, total_steps: int, min_lr_ratio: float = 0.1):
    """Linear warmup, then cosine decay to min_lr_ratio * peak."""
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        cosine = 0.5 * (1 + math.cos(math.pi * progress))
        return min_lr_ratio + (1 - min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
```

## V1: Loss — Focal gamma=3 (REPLACES class-weighted CE / LS)

V1 invariant 22: focal loss gamma=3 (NOT vanilla CE) required for clean T-scaling/RAPS interaction.

V1: class-weighted CE + LS=0.05.
V1: focal gamma=3, no class weights, no label smoothing. Class imbalance handled by triple-barrier neutral threshold (doc 04 — barrier_up = +1*ATR-14, barrier_down = -1*ATR-14, spread-adjusted neutral) and per-class reporting in eval. Class weights would solve recall but break calibration.

```python
def focal_loss_v15(logits, targets, gamma=3.0, reduction='mean'):
    """V1: focal loss gamma=3 (Mukhoti 2020 arXiv:2002.09437).
    Reduces ECE 30-50% pre-T-scaling. T post-fit converges near 1.0.
    CRITICAL: T-scaling on CE harms APS adaptive coverage (Xi 2024 arXiv:2402.04344).
    Focal-trained logits avoid this conflict.
    Reference: torrvision/focal_calibration.
    """
    log_p = torch.log_softmax(logits, dim=-1)
    p = log_p.exp()
    log_p_target = log_p.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    p_target = p.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    loss = -((1 - p_target) ** gamma) * log_p_target
    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    return loss


def sharpe_loss_v15(position_weight, r_next, w_prev=None, cost=2e-4, eps=1e-8):
    """V1: differentiable -Sharpe over mini-batch (Hwang & Zohren 2025 arXiv:2510.03129).
    Cost-aware: penalizes |w_t - w_{t-1}| by per-trade cost.
    L_sharpe_net = -mean(w*r_next - cost*|delta_w|) / (std(...) + eps)
    """
    pnl = position_weight * r_next
    if w_prev is not None:
        pnl = pnl - cost * (position_weight - w_prev).abs()
    return -pnl.mean() / (pnl.std() + eps)


# DEPRECATED — V1 class weights, NOT used in V1
def compute_class_weights_v1(labels: torch.Tensor, num_classes: int = 3) -> torch.Tensor:
    """V1 LEGACY (NOT V1). weight_i = N_total / (num_classes * N_i). Equivalent to sklearn 'balanced'."""
    counts = torch.bincount(labels, minlength=num_classes).float()
    N = counts.sum()
    return N / (num_classes * counts)
```

## Checkpoint Tracker (early stopping)

```python
class CheckpointTracker:
    def __init__(self, output_dir: str, patience: int = 3):
        self.output_dir = output_dir
        self.patience = patience
        self.best_val_loss = float('inf')
        self.epochs_no_improve = 0
        self.best_path = None

    def step(self, model, optimizer, val_loss: float, epoch: int) -> bool:
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.epochs_no_improve = 0
            path = f"{self.output_dir}/checkpoint_epoch{epoch}_val{val_loss:.4f}.pt"
            torch.save({
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'epoch': epoch,
                'val_loss': val_loss,
            }, path)
            self.best_path = path
            return False
        else:
            self.epochs_no_improve += 1
            return self.epochs_no_improve >= self.patience
```

## Full Training Loop (one fold)

V1 NOTE: The loop below is the V1 single-head reference. For V1 multi-task training:
- Stage 2 (linear probe) loop uses ONLY `focal_loss_v15(logits, labels, gamma=3.0)` on Head A.
- Stage 3 (LLRD fine-tune) loop uses combined: `0.5 * focal_loss_v15 + 0.5 * sharpe_loss_v15(position_weight, r_next, w_prev) + 0.05 * dann_loss + l_aecf`.
- Replace `F.cross_entropy(logits, labels, weight=class_weights, label_smoothing=label_smoothing)` with focal everywhere. NO class weights, NO label smoothing.
- Both heads forward in single pass: `out = model(...)`; `logits = out["logits_3class"]`; `position_weight = out["position_weight"]`.
- Track Mixout state at Stage 3 entry (anchor = SSL checkpoint).
- Track Schedule-Free `optimizer.train()` / averaged-weights switch; Cautious patch is internal to step.



```python
import torch
import torch.nn.functional as F
import wandb

def train_one_fold(
    model: nn.Module,
    train_loader, val_loader, test_loader,
    device: str = 'mps',
    num_epochs: int = 20,
    peak_lr: float = 3e-4,
    weight_decay: float = 0.1,
    warmup_pct: float = 0.1,
    grad_clip: float = 1.0,
    label_smoothing: float = 0.0,
    patience: int = 3,
    wandb_run=None,
) -> dict:
    model = model.to(device)

    # Class weights from training set only (no val/test peeking)
    train_labels = torch.cat([batch[2] for batch in train_loader])
    class_weights = compute_class_weights(train_labels).to(device)

    optimizer = configure_optimizer(model, peak_lr, weight_decay)
    total_steps = len(train_loader) * num_epochs
    warmup_steps = int(total_steps * warmup_pct)
    scheduler = cosine_lr_schedule(optimizer, warmup_steps, total_steps)
    tracker = CheckpointTracker('checkpoints/', patience=patience)

    for epoch in range(num_epochs):
        model.requires_grad_(True)

        epoch_train_loss = 0
        epoch_train_correct = 0
        epoch_train_total = 0

        for batch_idx, (numeric, news_raw, labels) in enumerate(train_loader):
            numeric = numeric.to(device)
            news_raw = news_raw.to(device)
            labels = labels.to(device)

            logits = model(numeric, news_raw)
            loss = F.cross_entropy(logits, labels, weight=class_weights, label_smoothing=label_smoothing)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            scheduler.step()

            with torch.no_grad():
                preds = logits.argmax(dim=-1)
                epoch_train_loss += loss.item() * len(labels)
                epoch_train_correct += (preds == labels).sum().item()
                epoch_train_total += len(labels)

            if wandb_run and batch_idx % 50 == 0:
                wandb_run.log({
                    'train/loss_step': loss.item(),
                    'train/lr': scheduler.get_last_lr()[0],
                    'step': epoch * len(train_loader) + batch_idx,
                })

        train_loss = epoch_train_loss / epoch_train_total
        train_acc = epoch_train_correct / epoch_train_total

        # Validate
        with torch.no_grad():
            val_loss = 0
            val_correct = 0
            val_total = 0
            for numeric, news_raw, labels in val_loader:
                numeric, news_raw, labels = numeric.to(device), news_raw.to(device), labels.to(device)
                logits = model(numeric, news_raw)
                loss = F.cross_entropy(logits, labels, weight=class_weights)
                val_loss += loss.item() * len(labels)
                preds = logits.argmax(dim=-1)
                val_correct += (preds == labels).sum().item()
                val_total += len(labels)
            val_loss /= val_total
            val_acc = val_correct / val_total

        if wandb_run:
            wandb_run.log({
                'train/loss_epoch': train_loss, 'train/acc_epoch': train_acc,
                'val/loss': val_loss, 'val/acc': val_acc,
                'epoch': epoch,
            })

        print(f"Epoch {epoch}: train_loss={train_loss:.4f} train_acc={train_acc:.3f} val_loss={val_loss:.4f} val_acc={val_acc:.3f}")

        if tracker.step(model, optimizer, val_loss, epoch):
            print(f"Early stopping at epoch {epoch}")
            break

    # Test on best checkpoint
    best_state = torch.load(tracker.best_path)
    model.load_state_dict(best_state['model_state'])

    with torch.no_grad():
        test_loss = 0
        test_correct = 0
        test_total = 0
        all_logits = []
        for numeric, news_raw, labels in test_loader:
            numeric, news_raw, labels = numeric.to(device), news_raw.to(device), labels.to(device)
            logits = model(numeric, news_raw)
            loss = F.cross_entropy(logits, labels, weight=class_weights)
            test_loss += loss.item() * len(labels)
            preds = logits.argmax(dim=-1)
            test_correct += (preds == labels).sum().item()
            test_total += len(labels)
            all_logits.append(logits.cpu())
        test_loss /= test_total
        test_acc = test_correct / test_total

    return {
        'best_val_loss': tracker.best_val_loss,
        'test_loss': test_loss,
        'test_acc': test_acc,
        'best_checkpoint': tracker.best_path,
        'test_logits': torch.cat(all_logits),
    }
```

## Walk-Forward Driver

```python
def run_walk_forward(snapshot_path, embeddings_path, anchors_path, num_seeds: int = 1):
    features = build_feature_table(snapshot_path, embeddings_path, anchors_path)

    fold_results = []
    for fold in walk_forward_splits(features.timestamp):
        for seed in range(num_seeds):
            torch.manual_seed(seed)

            train_data = features[fold.train_start:fold.train_end]
            val_data = features[fold.val_start:fold.val_end]
            test_data = features[fold.test_start:fold.test_end]

            train_loader = make_loader(train_data, batch_size=32, shuffle=True)
            val_loader = make_loader(val_data, batch_size=64, shuffle=False)
            test_loader = make_loader(test_data, batch_size=64, shuffle=False)

            model = nanoGLD()
            wandb_run = wandb.init(project='nanogld', name=f'fold{fold.fold_idx}_seed{seed}')

            result = train_one_fold(model, train_loader, val_loader, test_loader, wandb_run=wandb_run)
            result['fold'] = fold.fold_idx
            result['seed'] = seed
            fold_results.append(result)

            wandb_run.finish()

    return fold_results
```

## Mixed Precision Strategy

**Phase 1 (week 1, default):** all FP32 weights + activations. Slower but works. ~1.5GB model memory for 24M params, 5-10 hrs training time per fold.

**Phase 2 (if memory or speed limits hit):** bfloat16 weights + activations on MPS. ~50% memory + ~30% speedup. Keep optimizer states FP32.

```python
# Phase 2 (only if needed)
model = nanoGLD().to('mps')
for p in model.parameters():
    p.data = p.data.to(torch.bfloat16)
optimizer = configure_optimizer(model, ...)  # AdamW state stays FP32
```

If MPS-specific bugs surface (silent CPU fallback for some ops), profile with `torch.profiler` and revert affected layers to FP32.

## Implementation Day Plan (Day 7 of week 1)

| Hour | Task |
|------|------|
| 1 | Walk-forward splitter + tests |
| 2 | PyTorch Dataset + DataLoader |
| 3 | Hook up training loop with wandb |
| 4 | Run 1 fold smoke test (subsample data, 2 epochs, verify shapes + loss decrease) |
| 5-7 | Run 1 full fold |
| 8-10 | Inspect val curves, tune LR / batch / dropout if needed |
| 11-14 | Launch all folds (overnight) |
| Day 8 AM | Review fold results, save model artifacts |

## Open Questions / TODOs

- [ ] If single-seed OOS results are unstable across folds, add 5-seed averaging for reproducibility
- [ ] Decide whether to publish wandb workspace publicly (yes for X thread, double-check no secrets in run names)
- [ ] Profile MPS forward pass — does flash-attention kernel (if available) help?
- [ ] If batch 32 is too small for stable gradients on imbalanced data, try batch 64 or 128 with gradient accumulation
- [ ] Consider gradient accumulation only if memory limits force batch < 32

---

# Part 3 — Calibration

_(was doc 05-MODEL-TRAINING-CALIBRATION.md before V5 merge — content unchanged)_


---

# 07 — Confidence & Calibration (Spec)

**Status:** ✅ V1 spec, research-backed (3 parallel calibration agents 2026-05-04 — methods, conformal deep-dive, metrics + drift)
**Last verified:** 2026-05-04

## Why This Doc Exists

The plan covers calibration in fragments across 5 docs:
- **Earlier sections of this doc (model arch + training)** V1 mention focal loss gamma=3 (replacing LS=0.05) + dropout + stochastic depth linear schedule + Mixout (Stage 3 only).
- **doc 07 (sizing)** sketches temperature scaling + split conformal; V1 replaces with T-scaling -> RAPS -> AgACI + LLLA Kelly multiplier.
- **doc 07 (exits Part 2)** V1 uses smooth p-value shrinkage on top of set-size gate.
- **doc 08 (live trading)** has a single-signal drift detector (entropy z-score > 2 sigma + KL on argmax). V1 keeps the 3-tier stack.

This doc consolidates and upgrades. Three parallel research agents on 2026-05-04 produced ~9000 words of analysis on:
- Which scalar best measures confidence for a 3-class weak-signal financial model (signed for sizing, MSP for gate, entropy for drift, set size for abstention, MC dropout decomposition for aleatoric/epistemic split).
- Which post-hoc calibration method (temperature primary, Dirichlet-ODIR fallback; skip Platt / isotonic / BBQ / matrix / vector / spline / Mix-n-Match).
- Which conformal variant. **V1 UPDATE: T-scaling -> RAPS -> AgACI replaces V1's T-scaling -> APS** (Angelopoulos 2020 arXiv:2009.14193 + Zaffran 2022 ICML arXiv:2202.07282; AgACI online wrapper provides coverage under arbitrary distribution shift via expert-advice over gamma grid).
- Which metrics survive class imbalance (classwise AdaECE + macro Brier + NLL, not top-label ECE).
- How to monitor drift before realized PnL collapses (3-tier stack with named thresholds).
- **V1 UPDATE: Laplace last-layer (LLLA) replaces MC dropout T=20** (Daxberger 2021 arXiv:2106.14806 + ref `aleximmer/Laplace`; faster, better-calibrated epistemic, no 20x inference cost; posterior variance feeds Kelly multiplier).

## Glossary

- **MSP:** max softmax probability = max(P_down, P_flat, P_up). Used as confidence gate post-temperature.
- **Signed score:** s = P_up - P_down ∈ [-1, +1]. Used as sizing magnitude × direction.
- **APS:** Adaptive Prediction Sets (Romano-Sesia-Candès 2020). Cumulative-sum non-conformity score.
- **V1 RAPS:** Regularized APS (Angelopoulos 2020 arXiv:2009.14193). APS + size penalty `lambda * max(0, rank - kreg)`.
- **V1 AgACI:** Aggregated Adaptive Conformal Inference (Zaffran 2022 ICML arXiv:2202.07282). Online wrapper aggregating over a gamma grid via expert advice (BOA aggregator). Maintains target coverage under arbitrary distribution shift.
- **Mondrian CP:** class-conditional conformal prediction. Per-class quantile, per-class coverage guarantee.
- **AdaECE:** Adaptive Expected Calibration Error. Equal-mass binning instead of equal-width. Roelofs 2020.
- **Classwise ECE:** ECE evaluated per class (one-vs-rest), then macro-averaged. Kull 2019.
- **V1 LLLA:** Laplace last-layer approximation (Daxberger 2021 arXiv:2106.14806). Posterior `N(MAP_weights, H^{-1})` over last-layer weights. Single-pass posterior; replaces MC dropout T=20.
- **Aleatoric uncertainty:** irreducible noise. E[H[p]] over MC samples (V1) / entropy of MAP softmax (V1).
- **Epistemic uncertainty:** model uncertainty. H[E[p]] - E[H[p]] (mutual information / BALD) (V1) / posterior variance (V1 via LLLA).
- **Snapshot ensemble:** average over the last K EMA checkpoints from a single training run. Huang 2017.
- **V1 Focal loss:** `(1 - p_t)^gamma * log(p_t)` with gamma=3 (Mukhoti 2020 arXiv:2002.09437). Replaces vanilla CE / label smoothing on Head A.
- **V1 Diff-Sharpe loss:** `L_sharpe_net = -mean(w * r_next - cost * |w_t - w_{t-1}|) / (std(...) + eps)` on Head B (Hwang & Zohren 2025 arXiv:2510.03129).
- **V1 Multi-task head:** Head A (3-class direction, focal) + Head B (position weight in [-1, +1], diff-Sharpe), trained jointly on the same backbone.
- **V1 FiLM regime:** affine modulation `gamma * x + beta` between attn and FFN, conditioned on a 12-dim regime vector (VIX-tercile + RV-tercile + FOMC + year-bucket + HMM-P-high-vol).
- **V1 sLSTM block:** xLSTMTime sLSTM block (Alharthi & Mahmood arXiv:2407.10240). Layers 11-12 of V1 hybrid encoder.
- **V1 SimMTM:** SSL multi-mask reconstruction with similarity-weighted blending (Dong NeurIPS 2023 arXiv:2302.00861).
- **V1 AECF:** entropy-gated curriculum masking for multimodal fusion (Chlon et al. arXiv:2505.15417).
- **V1 Mixout:** Bernoulli-mix between current and SSL-anchor weights, p=0.7 at Stage 3 LLRD (Lee Cho Kang ICLR 2020 arXiv:1909.11299).
- **V1 Cautious update mask:** `update *= (sign(update) == sign(grad)).float()` patch on Schedule-Free AdamW (Liang 2024 arXiv:2411.16085).
- **V1 muP transfer-tune:** parameterize at small scale, transfer hparams to large (Yang 2022 arXiv:2203.03466).
- **V1 FreeLB:** adversarial perturbation on news embeddings only, K=2 inner ascent steps, epsilon=0.5 (Zhu et al. ICLR 2020 arXiv:1909.11764).
- **V1 DANN:** domain-adversarial gradient reversal on era-label = year-bucket (Feng et al. IJCAI 2019 arXiv:1810.09936).
- **V1 SimPSI / Wave-Mask:** spectral-preserving augmentation (Ryu AAAI 2024 arXiv:2312.05790; Arabi 2024 arXiv:2408.10951). Replaces forbidden naive jittering.
- **V1 DSR:** Deflated Sharpe Ratio (Bailey & López de Prado). Hard gate > 1.0.

---

## PART A — The Confidence Scalar Layer

There is no single "confidence scalar" because different downstream consumers want different things. We publish a structured object:

```python
@dataclass(frozen=True)
class Confidence:
    signed_score: float         # P_up - P_down in [-1, +1]    — for sizing direction × magnitude
    msp: float                  # max(p) in [0, 1]              — for re-entry gate, abstention gate
    entropy: float              # -sum(p log p)                  — for drift monitoring
    set_size: int               # |C(x)| in {1, 2, 3}            — for abstention with coverage guarantee
    top_pvalue: float           # APS p-value of top class      — for continuous shrinkage
    aleatoric: float            # E[H[p]] from MC dropout       — for "halve size when ambiguous"
    epistemic: float            # mutual info from MC dropout   — for "abstain when out-of-distribution"
```

| Use case | Scalar | Threshold (V1) | Action |
|---|---|---|---|
| Sizing direction | signed_score | min_signed_signal = 0.05 | flat below |
| Sizing magnitude | abs(signed_score) × sigmoid(5 × (top_pvalue - alpha)) | continuous | scale Kelly |
| Re-entry gate | msp | >= 0.55 (post-T) | allow same-side re-entry |
| Abstention (hard gate) | set_size | >= 2 | flat |
| Aleatoric ambiguity | aleatoric z-score | > 1.5 (vs train baseline) | halve size |
| Epistemic OOD | epistemic z-score | > 2.0 (vs train baseline) | flat for this bar, alert |
| Drift monitoring | entropy z-score | > 2.5 warn / > 3.5 page | reduce / halt |

**The hard rule:** if any of set_size >= 2, epistemic_z > 2.0, or (aleatoric_z > 1.5 AND set_size = 2) fires, the bar is treated as abstain (size = 0). Failures are logged with the triggering scalar.

---

## PART B — Training-Time Calibration

### V1: Loss Function — Focal gamma=3 (REPLACES LS=0.05)

V1 invariant 22: focal loss gamma=3 (NOT vanilla CE).

```python
# V1 — focal loss on Head A (3-class)
def focal_loss_v15(logits, labels, gamma=3.0):
    """V1: Mukhoti 2020 arXiv:2002.09437.
    Reduces ECE 30-50% pre-T-scaling. T post-fit converges near 1.0.
    CRITICAL: T-scaling on CE harms APS adaptive coverage (Xi 2024 arXiv:2402.04344).
    Focal-trained logits avoid this conflict.
    Reference: torrvision/focal_calibration.
    """
    log_p = torch.log_softmax(logits, dim=-1)
    log_p_t = log_p.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    p_t = log_p_t.exp()
    return -((1 - p_t) ** gamma) * log_p_t

loss_focal = focal_loss_v15(logits, labels, gamma=3.0).mean()
# NO label smoothing (subsumed by focal)
# NO class weights (would break calibration; class imbalance handled by triple-barrier neutral threshold + per-class reporting)
```

**V1 Rationale:**
- V1's LS=0.05 + Friendly-SAM + EMA 0.999 + dropout 0.2 + stoch depth 0.15 was already over-regularized for a weak-edge target.
- LS uniformly squashes the logit distribution (Müller-Kornblith-Hinton arXiv:1906.02629). T-scaling does the same job post-hoc, but tunable.
- V1: focal gamma=3 absorbs LS's calibration role AND directly addresses the T-scaling/APS-RAPS conflict (Xi 2024 arXiv:2402.04344). Mukhoti shows ECE drops 30-50% pre-T-scaling, T post-fit lands near 1.0.
- Class weights would solve a different problem (recall on minority classes) at the cost of probability calibration. Class imbalance is handled by **triple-barrier neutral threshold** (doc 04) + **stratified sampling in val-B** (T-fitter sees balanced classes) + **per-class reporting** (classwise AdaECE).
- V1: T-scaling bounds [0.7, 3.0] kept; expect T to land near 1.0 with focal-trained logits — existing guard rarely activates (a feature, not a bug).

### Snapshot Ensemble (modifies the training section above in this doc)

After training the final model, save the last 3 EMA checkpoints from epochs N-2, N-1, N. At inference, average their softmax outputs. Free calibration improvement (Huang arXiv:1704.00109). No extra training compute.

### Mixup — DROP for V1

The training section above mentions Manifold Mixup. Mixup IS a calibration improver (Thulasidasan arXiv:1905.11001) — input-space label smoothing. **But:** mixup on time-series financial features can leak future information across mixup pairs (mixing bars from different timestamps). Drop for V1 unless time-safe implementation is verified. Defer to V2 with explicit "no cross-time mixing" constraint.

---

## PART C — Post-Hoc Temperature Scaling

### Algorithm

```python
class TemperatureScaler(nn.Module):
    def __init__(self, init_T: float = 1.0):
        super().__init__()
        self.log_T = nn.Parameter(torch.tensor(float(np.log(init_T))))

    @property
    def T(self) -> torch.Tensor:
        return self.log_T.exp()

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.T

    def calibrated_probs(self, logits: torch.Tensor) -> torch.Tensor:
        return F.softmax(logits / self.T, dim=-1)

    def fit(self, val_logits, val_labels, max_iter: int = 50) -> float:
        opt = torch.optim.LBFGS(
            [self.log_T], lr=0.1, max_iter=max_iter,
            line_search_fn="strong_wolfe",
        )
        def closure():
            opt.zero_grad()
            loss = F.cross_entropy(val_logits / self.T, val_labels)
            loss.backward()
            return loss
        opt.step(closure)
        return float(self.T.item())
```

**Properties:**
- 1 parameter. LBFGS converges in < 50 iterations, < 1 second.
- Preserves argmax (monotone in logits).
- Empirical T values for calibrated transformers cluster in [1.0, 2.5]; T < 1 is rare and indicates underconfidence.
- Acceptance: max_T_per_fold <= 3.0 and min_T_per_fold >= 0.7. Outside that range = degenerate fit, escalate.

### Fitting On val-B (Stratified)

The val-B slice (25% of val month, ~750 bars) is **stratified by class** before T fits. Stratification guarantees the LBFGS sees DOWN, FLAT, UP in roughly equal measure, so T isn't tuned by the FLAT-dominated class.

### Fallback: Dirichlet-ODIR

Trigger only if post-T classwise-AdaECE > 0.10 on UP or DOWN. Kull et al. arXiv:1910.12656.

12 params (W 9 + b 3) with off-diagonal+intercept L2 regularization. Designed by the Dirichlet authors specifically for small calibration sets (~750 stratified bars). Don't use vanilla Dirichlet (overfits) or matrix scaling (worse: argmax flips under tiny perturbations).

---

## PART D — Conformal Prediction Recipe (V1: T-scaling -> RAPS -> AgACI)

V1 REPLACES V1's "T-scaling -> APS Mondrian" with **T-scaling -> RAPS -> AgACI**. Two upgrades:
1. **RAPS** (Angelopoulos 2020 arXiv:2009.14193): APS with size penalty `lambda * max(0, rank - kreg)`. Sharper sets, prevents pathological full-set predictions on 3-class.
2. **AgACI** (Zaffran 2022 ICML arXiv:2202.07282): online wrapper updates `alpha_t` based on miscoverage history, aggregates over gamma grid `[0.001, 0.005, 0.01, 0.05, 0.1]` via expert advice (BOA aggregator). Provably maintains target coverage under arbitrary distribution shift.

References: `mzaffran/AdaptiveConformalPredictionsTimeSeries` (R, port to PyTorch ~50 LOC). `aangelopoulos/conformal_classification`. Or use `ml-stat-Sustech/TorchCP` (production library, has RAPS/SAPS/ConfTS).

### V1 invariant 5.1 (per spec): focal-trained logits required

CRITICAL: Xi 2024 arXiv:2402.04344 shows **T-scaling on CE logits HARMS APS adaptive coverage**. Focal-trained logits (gamma=3) fix this. Without focal, the T-scaling -> RAPS -> AgACI stack collapses. This is why V1 invariant 22 (focal gamma=3) is non-negotiable.

### APS Score Function (still the basis for RAPS)

Cumulative softmax mass up to and including the true class, in descending-probability order.

For each test point, sort classes by probability descending. The non-conformity score for class y is the cumulative sum from the top down to y. Calibration takes the (1-alpha)(1+1/n) quantile of calibration scores. At inference, include classes (in descending prob order) until cumulative mass exceeds q_hat.

### V1: RAPS Adds Size Penalty

RAPS = APS + size-penalty term `lambda * max(0, rank - kreg)` per included class. Prevents the K=3 case from pathologically hitting full-set on weak-confidence bars (which would size to 0 by our shrinkage rule, wasting trades).

```python
# V1 — RAPS (sketch). Use TorchCP for production impl.
def raps_score_v15(probs, label, lambda_reg=0.01, k_reg=1):
    """V1: RAPS non-conformity score (Angelopoulos 2020 arXiv:2009.14193)."""
    sorted_p, sorted_idx = torch.sort(probs, descending=True)
    rank_of_label = (sorted_idx == label).nonzero(as_tuple=True)[0].item()
    cum_p = sorted_p[:rank_of_label + 1].sum()
    size_penalty = lambda_reg * max(0, rank_of_label + 1 - k_reg)
    return cum_p + size_penalty
```

### V1: AgACI Online Wrapper

AgACI maintains a grid of gamma candidates and aggregates predictions via expert-advice (BOA aggregator). Fast adaptation to distribution shift without manual gamma tuning.

```python
# V1 — AgACI (sketch). Use mzaffran/AdaptiveConformalPredictionsTimeSeries port.
class AgACI:
    """V1: Aggregated ACI (Zaffran 2022 ICML arXiv:2202.07282).
    Online wrapper over RAPS quantiles. Expert-advice aggregator over gamma grid.
    Provably maintains target coverage under arbitrary distribution shift.
    """
    def __init__(self, gamma_grid=(0.001, 0.005, 0.01, 0.05, 0.1), alpha=0.1):
        self.gammas = gamma_grid
        self.alphas = [alpha for _ in gamma_grid]
        self.weights = [1.0 / len(gamma_grid) for _ in gamma_grid]
        self.alpha_target = alpha
    def update(self, miscoverage_t):
        # Per-expert ACI update: alpha_i <- alpha_i + gamma_i * (alpha_target - 1{miscovered})
        for i, gamma in enumerate(self.gammas):
            self.alphas[i] += gamma * (self.alpha_target - miscoverage_t)
        # BOA expert-advice weight update on observed coverage loss
        ...
    def predict_set(self, calibrated_probs):
        alpha_t = sum(w * a for w, a in zip(self.weights, self.alphas))
        return raps_set(calibrated_probs, alpha_t)
```

### Why Mondrian Is Mandatory (KEPT from V1)

With 70/15/15 imbalance, marginal CP at alpha=0.10 typically achieves:
- Marginal coverage: ~90%
- FLAT coverage: ~95% (over-covered)
- UP coverage: ~75% (under-covered)
- DOWN coverage: ~75% (under-covered)

UP and DOWN are the directions we trade. Mondrian uses 3 separate quantiles, each calibrated on its own class, and guarantees >= 1-alpha coverage per class. ~10 LOC for 3 quantiles. ~750 cal points per class on a 3000-bar val fold — well above the 50-point minimum (Ding et al. 2023). V1: apply Mondrian on top of RAPS+AgACI (per-class quantiles, online-adapted).

### V1 promotes RAPS from "skip" to "primary"

V1 (this doc previously) said: "RAPS = APS when k_reg >= 1, K=3 is small, save the tuning." V1 OVERRULES that. The size-penalty term in RAPS becomes meaningful because focal gamma=3 changes the probability mass distribution: sets that were trivial-singletons under CE become smooth-confidence-weighted under focal. RAPS k_reg=1, lambda=0.01 default.

### V1 promotes AgACI from "V2 only" to "primary"

V1 deferred ACI to V2 (gamma misconfiguration risk). V1 sidesteps the gamma-tuning failure mode by using **AgACI** (aggregated over a gamma grid via expert advice). No manual gamma to tune. Gibbs-Candès arXiv:2106.00170 stays the foundation; AgACI is the production-safe wrapper.

### Continuous Confidence Scalar

Set size on K=3 is discrete {1, 2, 3}. For a smoother sizing signal, use the **top-class APS p-value**: the fraction of calibration scores that are MORE non-conforming than the test score for the top class. Maps continuously from 0 (just barely included) to ~(1-alpha) (very confident).

Combined sizing replaces the doc 07 discrete shrinkage {1: 1.0, 2: 0.5, 3: 0.0} with:

```
if set_size == 0 or set_size == 3:
    shrinkage = 0.0
elif set_size == 1:
    shrinkage = 1.0 * sigmoid(5 * (p_top - alpha))
elif set_size == 2:
    shrinkage = 0.5 * sigmoid(5 * (p_top - alpha))
```

The discrete set-size acts as a hard gate; the p-value adds smooth magnitude inside each bucket.

---

## PART E — V1: Laplace Last-Layer (LLLA) (REPLACES MC Dropout T=20)

V1 invariant 5.3: LLLA replaces MC dropout. Rationale: faster (1 forward pass, not 20), better-calibrated epistemic, no 20x inference cost. Posterior variance feeds Kelly multiplier.

Source: Daxberger 2021 arXiv:2106.14806. Reference impl: `pip install laplace-torch` (`aleximmer/Laplace`).

### LLLA Algorithm Sketch

Fit a Laplace approximation around the trained MAP weights of the **last layer** (Head A 3-class). Posterior is `N(MAP_weights, H^{-1})` where H is the Hessian of the log-posterior at MAP.

```python
# V1 — LLLA (sketch). Use laplace-torch for production.
from laplace import Laplace

la = Laplace(model, 'classification', subset_of_weights='last_layer', hessian_structure='kron')
la.fit(train_loader)
la.optimize_prior_precision()

# At inference:
pred_dist = la(x)   # posterior predictive over (B, K)
mean_p = pred_dist.mean
epistemic = pred_dist.variance.sum(-1)  # posterior variance feeds Kelly multiplier
# Kelly multiplier: min(1, sigma_target / sigma_posterior)
```

### V1 LEGACY: MC Dropout (kept for reference; not used in V1)

V1 ran T=20 forward passes with dropout enabled at inference. Stack T probability vectors (T, B, K). Compute:
- mean_p = mean over T → calibrated point estimate
- H_mean = entropy of mean_p → total predictive uncertainty
- E_H = mean over T of entropy(p_t) → expected entropy = aleatoric
- epistemic = H_mean - E_H → mutual information / BALD

V1 SKIPS this. LLLA gives a single-pass posterior with comparable epistemic decomposition at 1x cost.

### Cost on Macbook M4 Pro

- Single forward: ~80-150 ms (24-40M params, FP32, MPS).
- V1: LLLA single forward + posterior sample = ~100 ms per cycle (was 1.6-3 s for MC T=20).

### Decomposition

V1: H[E[p]] = aleatoric + epistemic = E[H[p]] + (H[E[p]] - E[H[p]]) (MC dropout decomposition).

V1: with LLLA, epistemic = posterior variance over logits (Hessian-based); aleatoric = entropy of MAP softmax. Same downstream actions (halve / abstain).

- **Aleatoric high → mean prediction is high-entropy → "data is genuinely ambiguous."** Action: halve size via the conformal shrinkage path.
- **Epistemic high → MC samples disagree → "model is out of training distribution."** Action: abstain (size = 0). Different members predicting different classes is the classical OOD signal.

V1 baselines (calibrated on val of fold 0, replaced after 30 days of paper trading):
- aleatoric_z = (aleatoric - mu_train) / sigma_train > 1.5 → halve
- epistemic_z = (epistemic - mu_train) / sigma_train > 2.0 → abstain
- both fire → abstain + alert

### Combine With Snapshot Ensemble

V1: snapshot ensemble (3 EMA checkpoints) + MC dropout (T=20) combined multiplicatively. Cost: 23 forward passes ≈ 2-3.5 s.

V1: snapshot ensemble (3 EMA checkpoints) + LLLA posterior on each. Cost: 3 forward passes + 3 LLLA samples ≈ 0.5 s. Budget reclaimed for other inference work.

---

## PART F — Calibration Metrics

### Adaptive ECE (Equal-Mass Bins)

Instead of equal-width bins (which empty out where the softmax piles up), use 15 equal-mass quantile bins. Each bin holds ~N/15 samples. Roelofs 2020.

### Classwise AdaECE

Per-class AdaECE computed one-vs-rest, then macro-averaged. Worst-class also reported as a guardrail. Kull 2019.

### Macro Brier

Brier score = sum over classes of (p_k - 1{y=k})^2, averaged over samples. Random uniform (1/3, 1/3, 1/3) = 0.667 for K=3. Strictly proper scoring rule — cannot be gamed by uninformative outputs. Target macro-Brier < 0.62 (resolution must beat random by > 7%).

### Conformal Coverage

Empirical fraction of bars where the prediction set contains the true label. Target marginal >= 0.90 at alpha=0.10. Track per-class via Mondrian.

### Targets (V1)

| Metric | Offline target (per fold) | Live target (rolling) | Source |
|---|---|---|---|
| Classwise AdaECE | macro < 0.04, worst < 0.10 | — | Roelofs 2020; Kull 2019 |
| Macro Brier | < 0.62 (vs random 0.667) | < 0.66 (50-bar) | Murphy 1973 |
| NLL | per-fold logged, no fixed threshold | — | proper scoring rule |
| Marginal CP coverage | >= 0.88 (target 0.90) | >= 0.85 (200-bar) | Vovk |
| Per-class CP coverage | >= 0.83 each | >= 0.80 each | Mondrian |
| Macro-F1 | > 0.40 (vs random 0.27) | > 0.40 (5-day rolling) | imbalance |
| Set-size mean | < 1.6 | < 1.8 (200-bar) | informative model |
| Singleton rate | 30%-90% | 20%-95% | not collapsed / exploded |

---

## PART G — Drift Detection (3-Tier Stack)

Replaces doc 08's single-signal entropy z. Tiered by label-availability latency.

### Tier 1 — Unlabeled (Fires Same Cycle, No Label Needed)

| Signal | Window | Threshold | Action |
|---|---|---|---|
| Predictive entropy z | 60d baseline | abs(z) > 2.5 warn / > 3.5 page | warn → log; page → halve size |
| Argmax-mix KL | 1 day vs train prior | KL > 0.20 warn / > 0.50 halve | warn / halve size |
| MC-dropout variance z | 60d baseline | z > 3 | abstain (size=0) on this bar |
| Feature KL (post-RevIN) | 1 day vs 60d | KL > 0.30 | warn — covariate shift |
| Singleton rate (rolling 200-bar) | live | > 0.95 or < 0.20 | auto-pause for review |
| MSP distribution KS | 1 day vs train | p < 0.001 | warn |

### Tier 2 — Labeled (Fires Next-Cycle When Prior Bar Realizes)

| Signal | Window | Threshold | Action |
|---|---|---|---|
| Rolling Brier | 50 bars | > 0.66 (random) for 100 bars | recalibrate |
| Classwise reliability gap | 200 bars | max-class gap > 0.10 | recalibrate |
| Conformal marginal coverage | 200 bars | < 0.85 for 2 consecutive days | recalibrate q_hat; persists → halt new entries |
| Conformal per-class coverage | 200 bars | any class < 0.80 | recalibrate Mondrian; persists → halt |
| Mean set size | 200 bars | > 1.8 | model uninformative → halve |

### Tier 3 — Labeled (Fires Weekly Friday EOD)

| Signal | Window | Threshold | Action |
|---|---|---|---|
| Per-class precision | 5 trading days | UP-precision < 0.40 OR DOWN-precision < 0.40 | retrain trigger |
| Macro-F1 | 5 days | < 0.40 | retrain trigger after consecutive |
| AvUC | 5 days | < 0.55 | retrain trigger |
| Signed-score Sharpe | 5 days | < 0.05 (was > 0.10 in val) | retrain trigger |

### False-Positive Control

- Tier-1: require **2 consecutive days** of the same warn-level signal OR **2 different signals same day** before any size reduction. A single-cycle Tier-1 spike is logged-only.
- Tier-2: always reduces size 50% on first trip. Two consecutive trips → halt new entries.
- Tier-3: triggers full retrain queue.

### Lead-Time Ranking (Per Ovadia et al. arXiv:1906.02530)

1. Entropy z, MC-variance z (lead PnL drift by 1-3 days)
2. Argmax KL (leads by 0-1 day)
3. Coverage, rolling Brier (~coincident with PnL)
4. AvUC, realized accuracy (lagging)

Fastest signals get the most-aggressive thresholds because they cost the least to false-fire.

---

## PART H — Recalibration & Retrain Cadence

### Recalibration (Weekly Friday EOD)

Cost: < 10 seconds per fold. Scheduled task:

1. Fetch last 390 RTH bars of (logits, label) pairs from SQLite state.
2. Split 60% / 40% temporally: trailing 234 bars for T fit, leading 156 bars for q_hat fit.
3. Refit T_star via LBFGS on val-B-equivalent slice.
4. Refit q_marginal and q_per_class via APS on val-C-equivalent slice.
5. Persist new T and q values; old values archived for rollback.
6. Run a self-test: classwise AdaECE on the trailing 200 bars must be < 0.10. If fail, alert and KEEP old T/q.

### Reactive Recalibration (Triggered By Tier-2)

Same procedure but on detection of a Tier-2 trigger between scheduled refits.

### Retrain Triggers (Full Transformer Fit, ~12 h on Mac mini)

- **Scheduled:** every 6 months (walk-forward fold rotation).
- **Performance:** macro-F1 < 0.40 for 10 consecutive trading days.
- **Calibration:** 3 consecutive weekly recals fail to bring classwise AdaECE under 0.10 OR Brier under 0.66.
- **Distribution:** feature KL > 0.50 nats sustained 5d; OR argmax KL > 0.80 sustained 3d.
- **Edge:** signed-score correlation with realized return < 0.05 for 15d (was > 0.10 in val).
- **Risk:** drawdown circuit-breaker fires at -8% paper or -5% live (per doc 07).

### Don't Confuse The Two

- **Recalibration handles "model is right but probabilities shifted."** Refit T + q. Cheap. Weekly.
- **Retrain handles "model is wrong."** Refit weights. Expensive. Triggered.

Empirically Guo / Roelofs / Wright all show T-scaling fixes ECE drift up to ~0.15. Beyond that you need new weights.

---

## PART I — Val Fold Split Protocol

Currently val is used for: early stopping + temperature scaling + conformal q_hat + sizing hyperparameter scan. That's 4 things on one slice — overuse risk. The split:

```
Walk-forward window (3y train + 6mo val + 6mo test):
   train    : 3 years    — model weights
   val-A    : first 50%  — early stopping ONLY
   val-B    : next 25%   — temperature scalar T (stratified across classes)
   val-C    : last 25%   — conformal q_hat (Mondrian per-class quantiles)
   test     : 6 months   — reported metrics only, no tuning
```

Rationale:
- **Early stopping is the heaviest user of val** — it gets the largest slice.
- **T-scaling + conformal q_hat** can't share the same slice without inflating each other (Vovk's CP guarantee assumes calibration set is exchangeable with future test, distinct from any earlier tuning).
- **Sizing hyperparameters** (Kelly fraction, target_vol) tune on val-A + val-B *combined*, never test. Two scalars, leakage risk small.

For live trailing-window recalibration: use trailing 390 bars but split 60/40 (T then q_hat) preserving temporal ordering.

---


## PART J — Hyperparameter Table (Consolidated, V1)

| Pillar | Param | V1 default | Source |
|---|---|---|---|
| **V1: Loss (Head A)** | **focal gamma** | **3.0** | **Mukhoti 2020 arXiv:2002.09437; required by Xi 2024 arXiv:2402.04344** |
| **V1: Loss (Head B)** | **diff -Sharpe coef** | **0.5** | **Hwang & Zohren 2025 arXiv:2510.03129** |
| **V1: Loss combined** | **lambda_DANN** | **0.05** | **Feng 2019 arXiv:1810.09936** |
| **V1: Loss combined** | **lambda_aecf** | **MC-dropout entropy estimate** | **Chlon arXiv:2505.15417** |
| Loss | label_smoothing | **V1: 0.0 (REPLACED by focal)** | Müller arXiv:1906.02629 (V1 used 0.05) |
| Loss | class_weight | None | preserve calibration |
| Snap-ensemble | K | 3 (epochs N-2, N-1, N) | Huang arXiv:1704.00109 |
| **V1: Epistemic** | **method** | **Laplace last-layer (LLLA)** | **Daxberger 2021 arXiv:2106.14806; ref aleximmer/Laplace** |
| **V1: Epistemic** | **kelly multiplier** | **min(1, sigma_target / sigma_posterior)** | this doc |
| MC dropout | T (V1 LEGACY) | **V1: REPLACED by LLLA** | Gal arXiv:1506.02142 §4.3 (V1 used T=20) |
| MC dropout | aleatoric_z halve | 1.5 (KEPT — applied to LLLA aleatoric) | this doc |
| MC dropout | epistemic_z abstain | 2.0 (KEPT — applied to LLLA epistemic) | this doc |
| Temperature | optimizer | LBFGS, lr=0.1, max_iter=50, strong-wolfe | Guo arXiv:1706.04599 |
| Temperature | sanity bounds | T in [0.7, 3.0] (V1: expect T near 1.0 with focal) | this doc |
| Conformal | alpha | 0.10 (90% coverage) | this doc |
| **V1: Conformal** | **score** | **RAPS (was APS)** | **Angelopoulos 2020 arXiv:2009.14193; ref TorchCP** |
| **V1: Conformal** | **RAPS k_reg** | **1** | this doc |
| **V1: Conformal** | **RAPS lambda** | **0.01** | this doc |
| **V1: Conformal** | **online wrapper** | **AgACI** | **Zaffran 2022 ICML arXiv:2202.07282; ref mzaffran/AdaptiveConformalPredictionsTimeSeries** |
| **V1: Conformal** | **AgACI gamma grid** | **(0.001, 0.005, 0.01, 0.05, 0.1)** | spec §5.2 |
| **V1: Conformal** | **AgACI aggregator** | **BOA expert advice** | spec §5.2 |
| Conformal | mondrian | True (per-class, applied on top of RAPS+AgACI) | Vovk 2012; Ding 2023 |
| Conformal | min cal points / class | 50 | Ding 2023 |
| Conformal | set_size = 1 shrinkage | sigmoid(5 * (p_top - alpha)) | this doc |
| Conformal | set_size = 2 shrinkage | 0.5 * sigmoid(5 * (p_top - alpha)) | this doc |
| Conformal | set_size = 3 or 0 shrinkage | 0.0 (abstain) | this doc |
| Metrics offline | classwise-AdaECE bins | 15 (equal-mass) | Roelofs 2020 |
| Metrics offline | classwise-AdaECE macro target | < 0.04 | this doc |
| Metrics offline | classwise-AdaECE worst-class | < 0.10 | this doc |
| Metrics offline | macro Brier target | < 0.62 (vs random 0.667) | Murphy 1973 |
| Metrics offline | macro-F1 floor | > 0.40 | imbalance |
| Metrics live | rolling Brier window | 50 bars | this doc |
| Metrics live | rolling Brier alarm | > 0.66 for 100 bars | random baseline |
| Metrics live | coverage window | 200 bars (~2 weeks) | Vovk |
| Metrics live | coverage floor | 0.85 (target 0.90) | margin |
| Drift T1 | entropy z warn / page | 2.5 / 3.5 | gaussian tail |
| Drift T1 | argmax-KL warn / halve | 0.20 / 0.50 nats | this doc |
| Drift T1 | MC-var z abstain | 3.0 | Krishnan 2020 FP budget |
| Drift T1 | feat-KL warn | 0.30 nats | this doc |
| Drift T2 | per-class coverage | < 0.80 | this doc |
| Drift T3 | per-class precision | < 0.40 | random=0.33 |
| Drift T3 | macro-F1 retrain | < 0.40 for 10d | imbalance |
| Drift T3 | feat-KL retrain | > 0.50 sustained 5d | this doc |
| Drift T3 | argmax-KL retrain | > 0.80 sustained 3d | this doc |
| Drift T3 | signed-Sharpe retrain | < 0 for 15d | this doc |
| Recalibration | cadence | weekly Friday EOD | this doc |
| Recalibration | window | trailing 390 bars (~30d) | Bandi 2008 |
| Recalibration | T/q split within window | 60/40 temporal | leakage budget |
| Retrain | scheduled | 6 months (fold rotation) | walk-forward CV |
| Retrain | recal-fail consecutive | 3 | this doc |
| Val split | val-A early-stop | 50% | this doc |
| Val split | val-B T scaling | 25% | this doc |
| Val split | val-C conformal | 25% | this doc |
| Val split | val-B stratification | balanced 250 / class | imbalance |
| **V1: ACI** | **promoted to V1 via AgACI wrapper** | gamma grid above | spec §5.2 |
| (V2) Deep ensembles | members | DROPPED — 5x compute kills budget; snapshot+EMA matches 80-90% (Abe 2022 arXiv:2202.06985) | Lakshminarayanan 2017 |
| **V1: Mixout** | **p (Stage 3 LLRD only)** | **0.7** | **Lee Cho Kang ICLR 2020 arXiv:1909.11299** |
| **V1: Stoch depth** | **schedule** | **linear 0.0 -> 0.2** | **Touvron 2021** |
| **V1: SSL** | **method** | **SimMTM** | **Dong NeurIPS 2023 arXiv:2302.00861; ref thuml/SimMTM** |
| **V1: SSL** | **mask ratio** | **0.40 (was 0.20)** | **PatchTST default** |
| **V1: SSL** | **K masked views** | **3** | spec §7.2 |
| **V1: SSL** | **CLIP head tau** | **0.07** | spec §7.3 |
| **V1: SSL** | **CLIP coefficient** | **0.5** | spec §7.5 |
| **V1: Optimizer** | **method** | **Cautious-Schedule-Free AdamW** | **Liang 2024 arXiv:2411.16085 + Defazio ICLR 2025** |
| **V1: muP** | **transfer-tune budget** | **$5 (2-4M tiny model)** | **Yang 2022 arXiv:2203.03466; ref microsoft/mup** |
| **V1: FreeLB** | **scope** | **news embeddings only (NOT bars)** | **Zhu et al. ICLR 2020 arXiv:1909.11764; PIT risk on bars** |
| **V1: FreeLB** | **K, epsilon** | **2, 0.5** | spec §6.6 |
| **V1: DANN** | **era-label** | **year-bucket {2016-2019, 2020-2022, 2023-2024, 2025+}** | **Feng 2019 arXiv:1810.09936** |
| **V1: DANN** | **lambda ramp** | **0 -> 0.1 over training** | spec §6.7 |
| **V1: News fusion** | **cross-attn layers** | **[3, 7, 11] sparse** | **mPLUG-Owl3 arXiv:2408.04840 Table 8** |
| **V1: News fusion** | **CFA projector** | **bottleneck d_text=256 -> 64 -> D=384** | **Lee arXiv:2603.22372; ref github.com/seunghan96/cfa** |
| **V1: News fusion** | **AECF curriculum p** | **U(0.0, 0.9)** | **Chlon arXiv:2505.15417; ref github.com/leochlon/aecf** |
| **V1: News fusion** | **is_news_present** | **`nn.Embedding(2, 8)`** | **Ma CVPR 2022 arXiv:2204.05454** |
| **V1: Modality dropout** | **per-batch p** | **U(0.1, 0.9)** | spec invariant 24; CRITICAL FIX over V1's 15% |
| **V1: FiLM regime** | **layers** | **{2, 4, 6, 8, 10}** | spec §1.5 |
| **V1: FiLM regime** | **vec dim** | **12** (VIX-tercile 3 + RV-tercile 3 + FOMC 1 + year-bucket 4 + HMM-P-high-vol 1) | spec §1.5 |
| **V1: Patches** | **P, S** | **4, 4 -> 16 patches** | spec §2B |
| **V1: Patches** | **channels** | **681 (channel-independent)** | spec §2B |
| **V1: Hybrid** | **transformer layers** | **10** | spec §1B |
| **V1: Hybrid** | **sLSTM layers** | **2 (top of stack)** | spec §1B; xLSTMTime arXiv:2407.10240 |
| **V1: Aug** | **method** | **SimPSI + Wave-Mask** | **Ryu AAAI 2024 arXiv:2312.05790; Arabi 2024 arXiv:2408.10951** |
| **V1: Aug** | **naive jittering** | **FORBIDDEN (invariant 21)** | **Fons 2020 arXiv:2010.15111** |
| **V1: Multi-task** | **head A loss coef** | **0.5** | spec §3B |
| **V1: Multi-task** | **head B loss coef** | **0.5** | spec §3B |
| **V1: Walk-forward** | **bars_per_year** | **3276 (NEVER 17500)** | invariant 5 |

---

## PART K — Ablation Plan (V1)

Run on the same val fold with bootstrap CIs. Decision rules below.

| # | Variant | Loss | Post-hoc | Conformal | Uncertainty | Live drift |
|---|---|---|---|---|---|---|
| 1 | Baseline V1 reference | CE+LS=0.05 | none | naive split-CP | none | entropy z only |
| 2 | + Focal gamma=3 (V1) | **focal gamma=3** | none | naive split-CP | none | entropy z only |
| 3 | + Temperature | focal gamma=3 | T-scaling | naive split-CP | none | entropy z only |
| 4 | + RAPS (V1) | focal gamma=3 | T-scaling | **RAPS marginal** | none | entropy z only |
| 5 | + Mondrian | focal gamma=3 | T-scaling | RAPS Mondrian | none | entropy z only |
| 6 | + AgACI online (V1) | focal gamma=3 | T-scaling | **RAPS Mondrian + AgACI** | none | entropy z only |
| 7 | + LLLA (V1) | focal gamma=3 | T-scaling | RAPS+AgACI Mondrian | **LLLA** | entropy z only |
| 8 | + Snap ensemble | focal gamma=3 | T-scaling | RAPS+AgACI Mondrian | LLLA + snap | entropy z only |
| 9 | + 3-tier drift | focal gamma=3 | T-scaling | RAPS+AgACI Mondrian | LLLA + snap | 3-tier |
| 10 | + Dirichlet-ODIR fallback | focal gamma=3 | T+Dirichlet | RAPS+AgACI Mondrian | LLLA + snap | 3-tier |
| 11 | + Multi-task Sharpe head (V1) | **focal + diff-Sharpe** | T-scaling | RAPS+AgACI Mondrian | LLLA + snap | 3-tier |

Decision (V1):
- Ship variant 11 (full V1 stack with multi-task head) if classwise-AdaECE macro < 0.04 AND worst < 0.10 on val AND Stage 3 multi-task beats Stage 2 single-task by >= 0.2 Sharpe OOS (V1 promotion gate 5).
- Escalate to variant 10 (Dirichlet fallback) only if 9 fails the worst-class threshold on UP or DOWN.
- If variant 5 alone matches variant 9 on all calibration metrics AND on val Sharpe (via doc 07 sizing path), drop LLLA + snap ensemble + AgACI from V1. Simplicity wins.
- If Stage 3 multi-task fails to beat Stage 2 single-task by >= 0.2 Sharpe OOS, ship the simpler (Head A only) per V1 invariant 25 + V1 invariant 3 (ship simpler if ties).

---

## PART L — Acceptance Gate For V1 Calibration Pipeline

To ship V1 calibration, ALL must pass on every walk-forward fold:

1. ✅ T fits within sanity bounds (T in [0.7, 3.0]). V1: expect T near 1.0 with focal gamma=3.
2. ✅ classwise-AdaECE macro < 0.04, worst-class < 0.10.
3. ✅ V1: Marginal CP coverage on test fold >= 0.88 with **RAPS+AgACI** (target 0.90, 2pp margin).
4. ✅ V1: Per-class CP coverage >= 0.83 each via Mondrian on top of RAPS+AgACI.
5. ✅ Macro Brier < 0.62.
6. ✅ Macro-F1 > 0.40.
7. ✅ V1: LLLA aleatoric + epistemic decomposition checks (posterior variance + MAP entropy match within 1e-5; replaces V1's MC dropout decomposition test).
8. ✅ V1: End-to-end calibration latency < 100 ms per bar (LLLA single forward replaces MC T=20).
9. ✅ V1: LLLA + snapshot ensemble combined latency < 0.5 s per cycle (was MC dropout 5 s budget in V1).
10. ✅ Snapshot ensemble adds < 1 s per cycle.
11. ✅ Drift monitor unit tests all pass.
12. ✅ Friday recalibration job runs end-to-end on a synthetic week.
13. ✅ V1: focal loss gamma=3 trained logits stored alongside checkpoint (required for AgACI clean coverage; CE-trained logits would silently break coverage per Xi 2024 arXiv:2402.04344).
14. ✅ V1: per-bucket eval metrics (news-present / news-absent / both) reported on every Sharpe / accuracy / coverage figure (invariant 18).
15. ✅ V1: cost-stress at {0.5x, 1.0x, 1.5x} reported on every Sharpe (invariant 19).
16. ✅ V1: Deflated Sharpe Ratio > 1.0 (invariant 20).
17. ✅ V1: Stage 3 multi-task head beats Stage 2 single-task fallback by >= 0.2 Sharpe OOS (invariant 25 / promotion gate 5). If fail, ship simpler.

If any fail on any fold, document under `## DEVIATION FROM SPEC: <date> - <issue>` at the top of this doc and AskUserQuestion.

---

## PART M — Open Questions For Owner Decision (V1 status updated)

1. ~~**LS=0.05 or LS=0.0?**~~ **V1 RESOLVED**: focal loss gamma=3 replaces label smoothing entirely (invariant 22).
2. ~~**MC dropout T=20 or T=30?**~~ **V1 RESOLVED**: Laplace last-layer (LLLA) replaces MC dropout (spec §5.3).
3. **Snapshot ensemble size: 3 or 5 EMA checkpoints?** 3 captures most of the gain (Huang 2017). 5 = more storage, more inference cost. Recommend 3 (KEPT).
4. **Mondrian or marginal CP for V1?** Recommend Mondrian on top of RAPS+AgACI — minority class coverage matters for trading. ~10 LOC cost (KEPT).
5. **Recalibration window: 390 bars (30d) or 195 bars (15d) or 780 bars (60d)?** Recommend 390. 30d is the empirical sweet spot for intraday financial models per Bandi/Mykland (KEPT).
6. ~~**ACI in V1 or V2?**~~ **V1 RESOLVED**: AgACI (online wrapper around RAPS) is now in V1 (spec §5.2). Removes V1's gamma misconfiguration risk by aggregating over a gamma grid via expert advice.
7. **Dirichlet-ODIR auto-trigger or manual?** Recommend auto-trigger if classwise-AdaECE worst > 0.10 on UP/DOWN after T-scaling (KEPT, applies to focal-trained logits).
8. **Retrain trigger thresholds — calibrate now or after first 30 days of paper trading?** Recommend lock the structure now, recalibrate the exact threshold values on day 31 from observed baselines (KEPT).
9. **V1 NEW: muP transfer-tune budget — $5 confirmed?** Spec says $5 on 2-4M tiny model before 30M one-shot. Owner approved.
10. **V1 NEW: SimMTM vs T-JEPA — A/B after baseline?** Paper says comparable. Spec primary = SimMTM. Defer A/B to post-baseline.
11. **V1 NEW: FreeLB ON vs OFF?** Some risk of distribution-shift overcorrection. Defer A/B to post-baseline.
12. **V1 NEW: Should regime vector include news-density bucket?** Currently no (12-dim: VIX-tercile + RV-tercile + FOMC + year-bucket + HMM-P-high-vol). Defer to post-baseline.
13. **V1 NEW: Hyperparameter freeze date** — post-muP sweep, before main H100 run.

---

## Citations

### V1 NEW Citations (architecture, training, calibration)

- Saly-Kaufmann, Wood, Zohren — DL futures benchmark (VLSTM 2.40 / xLSTM 1.79 Sharpe) — arXiv:2603.01820
- Alharthi, Mahmood — xLSTMTime — arXiv:2407.10240
- Hwang, Zohren — multi-task Sharpe loss — arXiv:2510.03129
- Lee et al. — CFA Constrained Fusion Adapter — arXiv:2603.22372 (ref github.com/seunghan96/cfa)
- Chlon et al. — AECF entropy-gated curriculum masking — arXiv:2505.15417 (ref github.com/leochlon/aecf)
- Ma et al. — is_present binary embedding — CVPR 2022 arXiv:2204.05454
- mPLUG-Owl3 — sparse cross-attn ablation — arXiv:2408.04840
- Mukhoti et al. — focal calibration gamma=3 — arXiv:2002.09437 (ref torrvision/focal_calibration)
- Xi et al. — T-scaling on CE harms APS — arXiv:2402.04344
- Angelopoulos et al. — RAPS — arXiv:2009.14193 (ref aangelopoulos/conformal_classification)
- Zaffran et al. — AgACI online conformal — ICML 2022 arXiv:2202.07282 (ref mzaffran/AdaptiveConformalPredictionsTimeSeries)
- TorchCP — production conformal library — github.com/ml-stat-Sustech/TorchCP
- Daxberger et al. — Laplace last-layer — arXiv:2106.14806 (ref aleximmer/Laplace, pip install laplace-torch)
- Lee, Cho, Kang — Mixout — ICLR 2020 arXiv:1909.11299 (ref bloodwass/mixout)
- Touvron et al. — stochastic depth linear schedule — DeiT 2021
- Ryu et al. — SimPSI spectral-preserving aug — AAAI 2024 arXiv:2312.05790
- Arabi et al. — Wave-Mask DWT aug — arXiv:2408.10951
- Fons et al. — naive jittering net-negative on Sharpe — arXiv:2010.15111
- Liang — Cautious update mask — arXiv:2411.16085 (ref kyleliang919/C-Optim)
- Yang — muP transfer-tune — arXiv:2203.03466 (ref microsoft/mup)
- Zhu et al. — FreeLB adversarial training — ICLR 2020 arXiv:1909.11764 (ref zhuchen03/FreeLB)
- Feng et al. — DANN gradient reversal on era-label — IJCAI 2019 arXiv:1810.09936
- Dong et al. — SimMTM SSL — NeurIPS 2023 Spotlight arXiv:2302.00861 (ref thuml/SimMTM)
- Defazio et al. — Schedule-Free AdamW — ICLR 2025 arXiv:2405.15682 (ref github.com/facebookresearch/schedule_free)
- Abe et al. — snapshot+EMA matches deep ensembles — arXiv:2202.06985
- Bailey, López de Prado — Deflated Sharpe Ratio — V1 invariant 20

### Calibration (training-time + post-hoc)
- Müller, Kornblith, Hinton — "When Does Label Smoothing Help?" — arXiv:1906.02629
- Mukhoti et al. — "Calibrating Deep Neural Networks using Focal Loss" — arXiv:2002.09437
- Guo, Pleiss, Sun, Weinberger — "On Calibration of Modern Neural Networks" — arXiv:1706.04599
- Kull et al. — "Beyond temperature scaling: Obtaining well-calibrated multi-class probabilities with Dirichlet calibration" — arXiv:1910.12656
- Hendrycks, Gimpel — "A Baseline for Detecting Misclassified and Out-of-Distribution Examples" — arXiv:1610.02136
- Thulasidasan et al. — "On Mixup Training: Improved Calibration and Predictive Uncertainty" — arXiv:1905.11001

### Conformal Prediction
- Romano, Sesia, Candès — "Classification with Valid and Adaptive Coverage" (APS) — arXiv:2006.02544
- Angelopoulos, Bates, Malik, Jordan — "Uncertainty Sets for Image Classifiers using Conformal Prediction" (RAPS) — arXiv:2009.14193
- Gibbs, Candès — "Adaptive Conformal Inference Under Distribution Shift" (ACI) — arXiv:2106.00170
- Bhatnagar, Wang, Xiong, Bai — "Improved Online Conformal Prediction via Strongly Adaptive Online Learning" (SAOCP) — arXiv:2302.07869
- Bhatnagar, Wang — auto-tuned ACI — arXiv:2208.08401
- Angelopoulos, Bates — "A Gentle Introduction to Conformal Prediction" (tutorial) — arXiv:2107.07511
- Vovk — "Conditional validity of inductive conformal predictors" — ACML 2012
- Ding, Angelopoulos, Bates, Jordan, Tibshirani — "Class-Conditional Conformal Prediction with Many Classes"
- Kato — "Conformal Predictive Portfolio Selection" — arXiv:2410.16333
- "A Gentle Introduction to Conformal Time Series Forecasting" — arXiv:2511.13608
- **Wright (2026), arXiv:2601.07852 — confirmed misattributed in plan doc 07. Has zero conformal content.**

### Uncertainty Estimation
- Gal, Ghahramani — "Dropout as a Bayesian Approximation" — arXiv:1506.02142
- Lakshminarayanan, Pritzel, Blundell — "Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles" — arXiv:1612.01474
- Kendall, Gal — "What Uncertainties Do We Need in Bayesian Deep Learning?" — arXiv:1703.04977
- Depeweg et al. — "Decomposition of Uncertainty in Bayesian Deep Learning" — arXiv:1710.07283
- Huang et al. — "Snapshot Ensembles: Train 1, Get M for Free" — arXiv:1704.00109
- Folgoc et al. — "Is MC Dropout Bayesian?" — arXiv:2102.08501

### Metrics + Drift
- Nixon et al. — "Measuring Calibration in Deep Learning" — arXiv:1904.10683
- Roelofs et al. — "Mitigating Bias in Calibration Error Estimation" — arXiv:1909.10155
- Vaicenavicius et al. — "Evaluating Model Calibration in Classification"
- Kumar et al. — "Verified Uncertainty Calibration"
- Krishnan, Tickoo — "Improving model calibration with accuracy versus uncertainty optimization" (AvUC) — arXiv:2007.10546
- Ovadia et al. — "Can You Trust Your Model's Uncertainty?" — arXiv:1906.02530
- Murphy 1973 — "A New Vector Partition of the Probability Score" — Brier decomposition

---

## Implementation Day Plan (V1)

| Hour | Task |
|---|---|
| 1 | V1: focal loss gamma=3 + sharpe loss + test_focal_argmax + test_focal_ECE_improvement |
| 2 | TemperatureScaler class + test_temperature_argmax_preserved (expect T near 1.0 with focal) |
| 3 | metrics.py: AdaECE + classwise-AdaECE + macro Brier + NLL + DSR + per-bucket eval + tests |
| 4 | V1: RAPS (Angelopoulos arXiv:2009.14193) + test_raps_coverage; integrate TorchCP if available |
| 5 | V1: AgACI online wrapper (Zaffran arXiv:2202.07282) + test_agaci_distribution_shift |
| 6 | RAPS+AgACI Mondrian per-class + test_mondrian_per_class_coverage_under_shift |
| 7 | V1: LLLA via laplace-torch + test_llla_decomposition (replaces mc_dropout_predict) |
| 8 | snapshot_ensemble + test_snapshot_ensemble_avg |
| 9 | DriftMonitor Tier 1 + test_drift_tier1_thresholds |
| 10 | DriftMonitor Tier 2 + Tier 3 + tests |
| 11 | val split protocol + test_val_split_no_leakage |
| 12 | V1: Mixout p=0.7 + test_mixout_anchor_invariant |
| 13 | V1: SimPSI + Wave-Mask aug + test_aug_pit_safe + test_no_naive_jitter |
| 14 | V1: Cautious patch on Schedule-Free AdamW + test_cautious_mask |
| 15 | V1: muP transfer-tune script (2-4M tiny model) + test_mup_lr_transfer |
| 16 | V1: FreeLB on news embeddings only + test_freelb_news_only |
| 17 | V1: DANN gradient reversal on era-label + test_dann_lambda_ramp |
| 18 | V1: SimMTM SSL pretrain + CLIP bars<->news head + test_simmtm_K_views |
| 19 | V1: AECF entropy-gated curriculum mask + test_aecf_p_uniform_curriculum |
| 20 | V1: is_news_present `nn.Embedding(2, 8)` + test_is_news_present |
| 21 | V1: variable per-batch modality dropout + test_modality_dropout_uniform |
| 22 | V1: FiLM regime conditioning every 2 layers + test_film_gamma_beta |
| 23 | V1: sLSTM block (xLSTMTime style) + test_slstm_block + test_hybrid_10_2 |
| 24 | V1: channel-independent + patches + test_patches_16_per_channel |
| 25 | V1: multi-task dual head + test_head_a_focal + test_head_b_tanh |
| 26 | V1: sparse cross-attn at [3,7,11] + CFA projector + test_sparse_layers |
| 27 | Friday recalibration job + integration test |
| 28 | Live cycle integration patch (cycle.py) |
| 29 | Ablation runner (V1 variants 1-11) + bootstrap CIs |

Total: ~3-4 days for a competent agent (V1 added significant scope). Concurrent with doc 07 changes; sequential with doc 08 changes. muP tiny-model sweep ($5) goes first to lock LR / beta_2 / init scale / FSAM rho before main run.

---

## Hand-off Protocol (V1)

1. Update STATUS.md with: V1 multi-task head outputs (Head A logits + Head B position weight), T per fold (expect near 1.0 with focal), classwise-AdaECE per fold, marginal + per-class coverage per fold, **RAPS+AgACI** quantiles per fold, **LLLA** posterior precision per fold, snapshot ensemble checkpoint hashes, Friday recal job status, **per-bucket eval** results (news-present / news-absent / both), **cost-stress** results at {0.5x, 1.0x, 1.5x}, **DSR** per fold.
2. Within this doc (Parts 1+2): use **focal loss gamma=3** (NOT LS=0.05) + **multi-task dual head** (Head A focal + Head B diff-Sharpe).
3. doc 07 (sizing+exits): use the calibrated probs + **RAPS+AgACI** p-values from Part 3 of this doc, not raw softmax. Use **LLLA posterior variance** for Kelly multiplier. Position weight from Head B is the primary; Head A conformal interval scales position down when uncertainty high. Sizing math (V1): set_size_gate × sigmoid(5 × (p_top - alpha)) × Head_B_position × min(1, sigma_target / sigma_LLLA).
4. doc 08 (live): wire the 3-tier DriftMonitor; add Friday EOD launchd plist for recalibration; add per-bucket eval reporting; add cost-stress reporting.

Now go build.
