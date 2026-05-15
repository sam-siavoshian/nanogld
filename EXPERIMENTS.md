# nanoGLD Experiment Ledger

One row per code change. Each change is its own git commit so it can be reverted in isolation. Every retrain run records its starting commit and the resulting backtest report path. When a change is found to be neutral or harmful, the **Verdict** column is updated and the commit is reverted (or kept behind a config flag default-off).

The convention:

| Field | Meaning |
|---|---|
| **ID** | `V<n><letter>` — `V3a`, `V3b`, ... Each letter is one isolated change. A bundled run lists all letters in its name. |
| **Commit** | git sha of the change. Empty if not yet committed. |
| **Hypothesis** | One sentence — what the agent believed and what the change does. |
| **Predicted outcome** | What metric must move and by how much for the change to be considered a win. Falsifiable. |
| **Actual outcome** | Filled in after the run lands. |
| **Verdict** | `WIN` (kept), `LOSE` (reverted), `NEUTRAL` (kept behind flag), `PENDING`. |
| **Rollback** | git command to undo. Empty if no commit. |
| **Source agent** | Which of the 5 ML eng agents proposed it (1-5). |

---

## Baseline runs

### V1 — baseline (`barrier_mult=1.0`)

- **Commit:** `10f514b`
- **Report:** `reports/backtest/v1_backtest_f5801965_10f514bce0.md`
- **Result:** Sharpe -0.780 @ 1.0× cost, -1.011 @ 1.5×. Class-collapse to NEUTRAL on 100% of bars. Folds 2/3 zero trades.
- **Documented in:** `PAPER.md` §5.

### V2 — label fix (`barrier_mult=0.25`)

- **Commit:** `9a56136`
- **Report:** `reports/backtest/v1_backtest_aa79ba47_9a56136bfb.md`
- **Result:** Sharpe -0.850 @ 1.0× cost, -1.069 @ 1.5×. Label balance fixed (30/40/30), no edge emerged. News-present Sharpe -6.24 (V1 -0.95) — news pathway now actively anti-edge.
- **Documented in:** `PAPER.md` §10.

---

## Five ML-eng agent diagnostic reports (2026-05-15)

Full prompts + responses archived in `~/.claude/projects/<...>/sessions` via conversation transcript. Summary table here. Each agent was given the full V1+V2 paper and asked for one top hypothesis + falsifiable prediction + code change spec + 5 supporting citations + risks.

### Agent 1 — Target / Label design

- **Hypothesis:** Triple-barrier discards magnitude (a +1.2σ move and a +0.0σ move both land in NEUTRAL). Replace 3-class focal CE Head A with **5-quantile pinball regression on `next_log_return / ATR_14`**. Demote triple-barrier to **meta-label** (predict-the-bet-will-hit per López de Prado AFML Ch. 3).
- **Falsifiable prediction:** Fold-0 val_b Sharpe -2.19 → ≥-0.3 (sign change in worst fold) within 4h retrain. Specifically `corr(q_0.5_pred, realized_vol_scaled_ret) > 0.04` on val_b.
- **Cost:** HIGH. Rewrite heads + losses + sizer + calibration stack (RAPS → CQR).
- **Risk:** quantile crossing + most predicted magnitudes sub-spread → sizer outputs zero everywhere (same failure mode as V1 reincarnated).
- **Citations:** López de Prado AFML Ch. 3; arXiv:2602.03395 (label-horizon paradox 2026); arXiv:2408.07497 (quantile NN stock distributions 2025); arXiv:2504.02249 (triple-barrier sensitivity 2025); MDPI Appl. Sci. 15:13204 (vol-scaled adaptive labels 2025).

### Agent 2 — News embeddings (cheapest test)

- **Hypothesis:** CLIP InfoNCE in SSL aligned `bar_pool ↔ contemporaneous news_pool`, so the encoder maps news context onto **backward-looking** price drift. At ±5 min alignment with GLD short-term reversal patterns, the model bets the priced-in direction and loses. Mean-pooling 8 news slots into one vector destroys article-level dispersion.
- **Falsifiable prediction:** **10-min inference-only test** — set `news_embeddings=0`, `news_mask=0`, `is_news_present=0`, rerun V2 backtest. If news-present-bucket Sharpe rises from **-6.24 → within ±0.5 of news-absent -0.996**, news is the dominant negative contributor.
- **Cost:** TINY. Config flag in inference path.
- **Risk:** if news ablation drops to -1.0 but overall Sharpe stays negative, news is one of multiple problems.
- **Citations:** FinMTEB Feb 2025 (arXiv:2502.10990); FinBERT2 Jun 2025 (arXiv:2506.06335); FinGPT dissemination Dec 2024 (arXiv:2412.10823); LLM sentiment S&P 500 Jul 2025 (arXiv:2507.09739); Cross-the-Gap CLIP misalignment ICLR 2025.

### Agent 3 — Architecture

- **Hypothesis:** Channel-independent encoding (B×F=1302 channel sequences) + post-encoder **channel-mean pool** (`pooled.mean(dim=1)` over F=651) over heterogeneous features (OHLCV + macro + COT + ...) drowns high-SNR channels in noise. PatchTST-CI assumes homogeneous channels; we have heterogeneous. Single shared transformer cannot specialize per-channel. Two-stream decomposition + sLSTM tail wasted at T=32.
- **Falsifiable prediction:** Replace CI + mean-pool with **iTransformer variate-token encoder** (each feature → one token; attention is over features, not time). Fold-0 macro-F1 0.33 → ≥0.42, per-bucket Sharpe ≥0. Compute: 6.5h on MPS (SSL must redo).
- **Cost:** HIGH. Rewrite encoder, redo SSL (cached anchors incompatible).
- **Risk:** seq_len=651 attention is O(N²·d); fits in 16GB at bs=8 but ~1.4× wall-clock. Pure-feature attention may wipe temporal structure.
- **Citations:** iTransformer (arXiv:2310.06625 ICLR 2024); TimeMixer (arXiv:2405.14616 ICLR 2024); Cross-Variate Patch Embedding (arXiv:2505.12761 May 2025); Beyond xLSTM short-term financial (MDPI Math 14:1282 2026); Channel Dependence under Limited Lookback (arXiv:2502.09683 Feb 2025).

### Agent 4 — Features

- **Bug found:** `src/nanogld/model/vsn.py:91-93` applies `softmax(raw) * num_features` → mean gate = 1.0 by construction. **VSN cannot prune any feature**, only redistribute mass.
- **Hypothesis:** ~60% of 651 features are weekly-monthly-cadence macro (FRED, COT, WGC, GPR, macro_bundle) forward-filled onto 30-min bars. After RevIN per-instance z-scoring of 32-bar windows these slow features collapse to near-constants — pure noise the broken VSN cannot suppress.
- **Falsifiable prediction:** Lean-200 channels (price + microstructure + regime + cross-asset DXY/TLT/SLV) → fold-0 Sharpe ≥+0.4 vs V2. Train 2 probes (Lean-200 vs full-651) on fold 0 only.
- **Cost:** MED-HIGH. Rebuild `unified.pt` (30h compute), fix VSN to sigmoid+L1, add microstructure features (OFI proxy from OHLCV, signed volume, microprice, range-position).
- **Risk:** macro features may matter via regime interaction; pruning may lose signal. Mitigation: PCA-compress macro block to 8 dims as FiLM side-channel.
- **Citations:** arXiv:2508.06788 (OFI intraday dynamics 2025); arXiv:2506.05764v2 (better inputs > deeper layers 2025); arXiv:2510.04667 (RevIN noise-or-signal 2025); arXiv:2511.08571 (forecast-to-fill gold 2025); ESwA S0957417425032099 (gold indicators 2025).

### Agent 5 — Training recipe

- **Hypothesis:** Head-B differentiable -Sharpe at bs=2 is unidentifiable (1-DoF sample → gradients of `O(1/√eps)` and arbitrary sign). Cautious masks-on-sign-agreement which on noisy 1-DoF acts as **high-pass filter on noise**. FSAM 2nd forward pass re-amplifies that noise. Three optimizer wrappers solving different problems on a starved-batch regime.
- **Falsifiable prediction:** Three-knob ablation on fold 0 (~3.5h): (a) `sharpe_weight=0`, (b) disable FSAM + Cautious, (c) grad accumulation bs_eff=16. Fold-0 OOS Sharpe ≥-0.2, macro-F1 ≥0.38, per-batch Sharpe loss std drops 5×.
- **Cost:** LOW. Config flags + accum loop.
- **Risk:** confounded if all 3 disabled at once. If macro-F1 still pins to ~0.31 the recipe is not the dominant drag (then features/horizon/news take over).
- **Citations:** Cautious Optimizers (arXiv:2411.16085 Nov 2024); Schedule-Free (Defazio NeurIPS 2024); Small-batch training (arXiv:2507.07101 Jul 2025); Friendly SAM (arXiv:2403.12350 2024); novel daily-stock loss (arXiv:2502.17493 Feb 2025).

---

## V3 plan (sequential, not parallel — attribution > coverage)

Each phase is a separate retrain with its own git tag + report file so we can identify which knob produced the effect.

### V3a — news ablation (Phase 1, ~10 min, INFERENCE ONLY)

| | |
|---|---|
| **ID** | V3a |
| **Commit** | (config-flag commit) |
| **Hypothesis** | Agent 2 — news pathway is anti-edge under V2 balanced labels |
| **Change** | Add `--ablate-news` flag to `nanogld.backtest run` that zeros `news_embeddings`, `news_mask`, `is_news_present` in the inference path. No retrain. |
| **Predicted outcome** | news-present Sharpe -6.24 → within ±0.5 of news-absent -0.996. Overall Sharpe should rise above V2 -0.85 toward the news-absent average. |
| **Actual outcome** | PENDING |
| **Verdict** | PENDING |
| **Rollback** | `git revert <sha>` |
| **Source agent** | 2 |

### V3b — strip recipe (Phase 2, ~12h, RETRAIN probe+LLRD)

Three independent code changes, each in its own commit so we can `git revert` any of them in isolation.

| ID | Change | Commit | Source agent |
|---|---|---|---|
| **V3b-1** | `llrd.use_fsam: false` — drop FriendlySAM wrapper from LLRD optimizer build. | | 5 + 3 |
| **V3b-2** | `llrd.use_cautious: false` — drop Cautious wrapper from LLRD optimizer build. | | 5 |
| **V3b-3** | `llrd.sharpe_weight: 0.0` — disable Head-B differentiable Sharpe loss (Head A focal CE only). | | 5 |
| **V3b-4** | `llrd.accum_steps: 8` — gradient accumulation, micro-bs=2, effective bs=16. | | 5 + 3 |

Hypothesis: stripping the over-engineered recipe lets the model actually learn from the (already-balanced V2) labels. Falsifiable: fold-0 LLRD val_b macro-F1 ≥0.38 (V1 was ~0.31, the majority prior on V1 labels).

SSL anchors are label-agnostic and stay frozen (sentinels intact). Only probe + LLRD + calibration retrain. ~3h/fold × 4 = ~12h on Mac mini MPS.

### V3c — backtest + paper

Run `nanogld.backtest` against the 4 new LLRD checkpoints + recalibrated dirs. Report path goes into the row of `EXPERIMENTS.md` corresponding to the change set. Compare V1 → V2 → V3a → V3b in `PAPER.md` §13 (new section).

---

## V4 candidates (only if V3 is also a fail)

In priority order:

1. **Agent 4 — VSN fix + feature pruning.** Replace `softmax×F` with `sigmoid + L1`. Rebuild `unified.pt` with `feature_set=lean` (≤200 cols) + add OFI/signed-volume/microprice/cross-asset features. 30h compute, needs end-to-end retrain.
2. **Agent 1 — Quantile target.** Replace 3-class focal CE with 5-quantile pinball regression on vol-scaled returns. Rewrite heads + losses + sizer + calibration. 20h compute.
3. **Agent 3 — iTransformer.** Replace channel-independent + mean-pool with variate-token encoder. Rewrite encoder, redo SSL. 26h compute (~6.5h/fold incl. SSL).

Pick whichever V3 result most strongly fails to falsify, OR the cheapest remaining hypothesis.

---

## Compute budget (cumulative)

| Run | Compute | Cumulative |
|---|---|---|
| V1 (initial) | 20h | 20h |
| V2 (label fix, SSL reused) | 22h | 42h |
| V3a (news ablation, inference only) | ~10 min | 42.2h |
| V3b (strip recipe, probe+LLRD only) | ~12h | 54.2h |
| Each V4 candidate | 20-30h | 74-84h |

---

## Rollback playbook

```bash
# Find commit
git log --oneline | grep "<ID>"
# Revert
git revert <sha>
git push origin main
# Mac mini
ssh root1@100.83.86.5 'cd ~/Desktop/nanogld && git pull --ff-only origin main'
# If retraining is needed:
ssh root1@100.83.86.5 'cd ~/Desktop/nanogld && rm -rf checkpoints/v1/fold_*/fold_*/{probe,llrd,calibration_*}'
# SSL anchors stay — they are label-agnostic and architecture-agnostic provided VSN/encoder shape didn't change.
```

---

*Last updated: 2026-05-15 (V3 plan, pre-execution).*
