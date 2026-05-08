# nanoGLD

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](./LICENSE)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.11-EE4C2C.svg)](https://pytorch.org)
[![Python](https://img.shields.io/badge/Python-3.13-3776AB.svg)](https://www.python.org)
[![Plan](https://img.shields.io/badge/Plan-V1%20locked-blueviolet.svg)](./plan)
[![Status](https://img.shields.io/badge/Data-Complete-brightgreen.svg)](./plan/HANDOFF.md)
[![Status](https://img.shields.io/badge/Model-Next-orange.svg)](./plan/05-MODEL-TRAINING-CALIBRATION.md)

> **Data phase complete 2026-05-08. V1 spec locked 2026-05-08.** Single unified dataset shipped on Mac mini at `/Users/root1/Desktop/nanogld/data/processed/training_v1_unified.pt` (234 MB). 75,993 bars × 681 features + 40,032 Qwen3-4B news embeddings. Triple-barrier label rebuild may be required pre-train (see HANDOFF). Read `plan/V1-SPEC.md` then `plan/HANDOFF.md` before starting model training.

A from-scratch hybrid encoder (transformer + sLSTM head) that predicts the next 30-minute direction of **GLD** (gold ETF) as a 3-class problem (UP / FLAT / DOWN) jointly with a continuous position-weight head, fused with frozen LLM news embeddings via Flamingo-gated cross-attention, trained on **10 years** (2016-2026) of multi-asset 30-minute bars + 40 FRED macro series + 40K embedded news articles.

> **TL;DR (V1):** ~24-60M-parameter hybrid encoder (10 transformer + 2 sLSTM at head, xLSTMTime style), channel-independent + patches (P=4, S=4 -> 16 patches), FiLM regime conditioning every 2 layers, sparse cross-attention to news at layers {3, 7, 11}, dual head (focal CE gamma=3 + Sharpe loss), CFA projector + AECF entropy gate before Flamingo cross-attn, frozen Qwen3-Embedding-4B for news, Cautious-Schedule-Free-AdamW + Friendly-SAM + muP transfer-tune, 3-stage SimMTM + CLIP SSL -> linear-probe -> LLRD fine-tune (Mixout p=0.7), walk-forward CV with 1-week embargo, focal -> RAPS -> AgACI conformal + Laplace last-layer, Deflated Sharpe Ratio + cost-stress + per-bucket eval as hard gates, full baseline ladder (xLSTMTime / VLSTM / Gao 2014 / XGBoost / F2F).

```
   bars (T=64, 30min RTH)             news (sparse, multi-source)
   series-decomp + per-channel RevIN  Qwen3-Embedding-4B (frozen, MRL -> 256)
          |                                      |
          v                                      v
   VSN feature gate (681 channels)       is_news_present embed (8-dim)
          |                                      |
          v                                      v
   Channel-independent patches                CFA projector
   (P=4, S=4, 16 patches/channel)        (FiLM + orthogonal residual)
          |                                      |
          v                                      v
   +------------------------------------------------------+
   | nanoGLDV1 hybrid encoder, ~24-60M params           |
   |                                                      |
   |   Layers 1-10: transformer blocks                    |
   |     RMSNorm + SwiGLU + real-form RoPE + QK-Norm      |
   |     + per-head gating + value residuals + no bias    |
   |   Layers 11-12: sLSTM blocks (xLSTMTime style)       |
   |   FiLM regime modulation @ layers {2,4,6,8,10}       |
   |     (12-dim regime vector)                           |
   |   Sparse Flamingo cross-attn @ layers {3,7,11}       |
   |     (AECF entropy-gated curriculum mask)             |
   |   Stochastic depth schedule linear 0.0 -> 0.2        |
   |   FP32 weights, no autocast, no compile              |
   +-------------------------+----------------------------+
                             |
            +----------------+----------------+
            |                                 |
            v                                 v
   3-class logits (UP/FLAT/DOWN)    position weight in [-1, +1]
   focal CE gamma=3                 Sharpe loss (cost-aware)
            |                                 |
            v                                 v
   T-scaling -> RAPS -> AgACI       friction-adjusted Kelly + ATR
   Laplace last-layer epistemic     vol-target 15% ann + 30d timeout
            |                                 |
            +----------------+----------------+
                             |
                             v
                conformal-floored position size
```

## Why this exists

Most ML-trading repos do one of two things. Print a 4.0 Sharpe on training data and quietly never validate out-of-sample. Or wire an LLM into a strategy with no ablations, no baselines, no honest comparison, and call it research.

This is the third thing. A real transformer (with sLSTM head), written from scratch in raw PyTorch the way Karpathy writes nanoGPT, with a multimodal news-fusion head wired to a frozen Qwen3-Embedding-4B. Walk-forward CV with embargo. Bootstrap confidence intervals. Deflated Sharpe Ratio. A full baseline ladder, including the GLD-specific Gao 2014 half-hour-5 rule that is the actual bar to beat. If a 2M-param TSMixer wins, ship TSMixer. If buy-and-hold wins after costs, ship buy-and-hold. The story is the rigor.

**Honest target (V1 reframe):** 1.0 to 1.5 OOS Sharpe net of 2 bps round-trip costs over 4-fold walk-forward, beating Gao 2014 + XGBoost ensemble by >= 0.2 Sharpe. The 2.88 Sharpe number from Wright et al Forecast-to-Fill is **daily gold futures EOD-to-EOD with ~30-day holding**, not directly comparable to 30-min intraday GLD direction. Apples-to-apples published intraday GLD record is Gao-Han-Li-Zhou 2014 (5.43 Sharpe single-feature half-hour-5 timing); daily futures DL frontier per Saly-Kaufmann/Wood/Zohren 2026 is VLSTM 2.40 Sharpe. We report against both scoreboards but only commit to the intraday one.

---

## Status

```
Plan:           ██████████████ V1 locked 2026-05-08, 9-agent Nia synthesis on top of V1's 27 agents
Data:           ██████████████ unified.pt shipped (75,993 bars × 681 features + 40,032 news embeddings)
Training:       ░░░░░░░░░░░░░░ NEXT — V1 model + multi-task head + multi-stage pipeline
Backtest:       ░░░░░░░░░░░░░░ pending checkpoints
```

**This README is a sketch until proven.** Numbers, claims, and architecture details get rewritten only when an implementation agent ships verified results, replaces a hypothesis with a measured truth, and cites the file path that produced it. The journey from sketch to real artifact is the journey from `plan/` -> trained checkpoints -> reported metrics. Until then, treat everything below as the design we are testing, not what we are claiming.

V1 was frozen 2026-05-04. V1 redlines drafted 2026-05-08 from a 9-agent Nia research synthesis (recurrent gated state on noisy financial benchmarks, Sharpe-loss heads, focal calibration, RAPS / AgACI online conformal, AECF entropy gating, CFA cross-modal fusion, SimMTM + CLIP pretraining, triple-barrier labels, F2F-style sizing). Owner approved Decisions 1B + 2B + 3B + 4A and all small wins. `plan/V1-SPEC.md` is the canonical change list; the 8 implementation docs in [`plan/`](./plan) have been updated against it. Each doc is owned by a single agent with Nia-verified citations, an interface contract, and acceptance criteria. The plan is the spec. Read [`plan/V1-SPEC.md`](./plan/V1-SPEC.md) and [`plan/00-OVERVIEW.md`](./plan/00-OVERVIEW.md) first.

---

## What's interesting about it

### 1. Multimodal time-series transformer with hybrid xLSTM head, from scratch

Numerical bars and natural-language headlines fuse inside a single hybrid encoder. Bars get tokenized as **channel-independent patches** (PatchTST-style, P=4, S=4 -> 16 patches per channel) so 681 channels share a single backbone without cross-channel-attn overfit on 75K samples. The first 10 layers are pre-norm transformer blocks; the last 2 layers are **sLSTM blocks** (xLSTMTime recipe — Alharthi & Mahmood 2024) where regime-conditional classification mixing matters most. Recurrent gated state is what beat pure attention on the 2026 Saly-Kaufmann finance benchmark (VLSTM 2.40 Sharpe vs iTransformer 0.38). News headlines pass through a frozen **Qwen3-Embedding-4B** (Apache 2.0, MTEB-en 74.6, MRL-truncated 2560 -> 256), get filtered by a **CFA projector** (FiLM + orthogonal residual against bar pool — Lee 2025), and merge into the bar stream via **sparse Flamingo-gated cross-attention** at layers {3, 7, 11} only with an **AECF entropy-gated curriculum mask** (Chlon 2025). The fuser is the trainable bridge; the LLM stays frozen.

### 2. Modern transformer block, not the 2017 original

```
RMSNorm                Llama / Qwen consensus, no mean centering
SwiGLU FFN             hidden = round(8 * D / 3, 64)
Real-form RoPE         (NEVER torch.view_as_complex on MPS — silently broken)
QK-Norm                stabilizes long-context attention
Per-head gating        IMU-1 recipe (~50K extra params, big sample-efficiency win)
Value residuals        IMU-1
Partial RoPE           applied to 10% of head_dim
no bias                everywhere
```

### 3. 3-stage training, not "fit one loop"

```
Stage 1   SSL: masked-bar reconstruction (MAE-style)
Stage 2   linear probe: freeze encoder, train classification head
Stage 3   LLRD fine-tune: layer-wise learning rate decay full pass
```

### 4. Cautious-Schedule-Free-AdamW + Friendly-SAM + muP transfer

No warmup. No cosine. No tuning the LR schedule. **Schedule-Free AdamW** ([Defazio 2024](https://arxiv.org/abs/2405.15682)) self-anneals. **Cautious update mask** ([Liang 2024](https://arxiv.org/abs/2411.16085), 5-line patch on top of SF-AdamW) zeroes out updates where momentum disagrees with gradient — 1.47x sample efficiency at zero hparam cost. **Friendly-SAM** (filtered-gradient SAM, rho=0.05) finds flatter minima for better generalization on non-stationary regimes. **muP transfer-tune** ([Yang 2022](https://arxiv.org/abs/2203.03466)) — spend $5 on a 2-4M-param tiny model muP-parameterized sweep (LR, beta_2, init scale, F-SAM rho), transfer to 30M for the one-shot H100 run. Saves $30-50 of LR-guess risk on the $60-150 main run. EMA decay 0.999 on top. Stochastic depth schedule linear 0.0 -> 0.2 across depth (Touvron 2021). **Mixout p=0.7** ([Lee 2020 ICLR](https://arxiv.org/abs/1909.11299)) at Stage 3 LLRD anchored to SSL checkpoint.

### 5. Multi-task head: focal classification + Sharpe loss

3-class **focal cross-entropy** (gamma=3, [Mukhoti 2020](https://arxiv.org/abs/2002.09437)) co-trained with a continuous **position-weight head** that minimizes negative cost-aware Sharpe directly. Vanilla CE conflicts with T-scaling + APS adaptive coverage ([Xi 2024](https://arxiv.org/abs/2402.04344)); focal-trained logits land cleanly with T near 1.0. Never MSE on returns ([arXiv:2604.00064](https://arxiv.org/abs/2604.00064)) — on weak-conditional-structure data MSE collapses into predicting the conditional mean (~0). **End-to-end Sharpe head** because MSE-optimal forecasts produce non-optimal allocations ([Hwang & Zohren 2025](https://arxiv.org/abs/2510.03129)) and Saly-Kaufmann 2026 trains directly on -Sharpe to hit 2.40. Combined Stage 3 loss: `0.5 * L_focal + 0.5 * L_sharpe_net + 0.05 * L_DANN + L_aecf`. Head A feeds calibration + conformal floor; Head B is the primary position weight.

### 6. Honest evaluation harness with hard gates

Walk-forward CV, 4 folds across 5 years, **1-week embargo** between train and val to kill label leakage from overlapping 30-min lookahead windows. **3,276 bars/year** for Sharpe annualization (NYSE regular trading hours, NOT the 17,500 you get from a 24/7 calendar — that's a 2.31x silent inflation). **Per-bucket eval** for {news-present, news-absent, both} as a non-negotiable diagnostic — 51% of bars have no news, training that on the wrong distribution is a silent kill. **Cost-stress** as a hard gate: report Sharpe at {0.5x, 1.0x, 1.5x} cost levels (half-spread 0.7bps + sqrt-impact gamma=0.02), must show Sharpe > 0.5 at 1.5x. **Deflated Sharpe Ratio > 1.0** as a hard gate ([Bailey & Lopez de Prado](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2460551)). **Stationary block bootstrap** for CIs. **Regime breakdowns** by vol tercile, FOMC week, news density.

### 7. The baseline ladder

Every result gets reported alongside:

```
Naive       Buy-and-hold                              baseline floor
            50/200 EMA crossover                      classical TA
            20-period Donchian breakout               classical TA

Linear      DLinear (Zeng AAAI 2023)                  "transformers are not effective"
            TSMixer (Chen 2023)                       MLP-Mixer for time series
            TimeMixer (ICLR 2024)                     decomposition mixing

RNN         xLSTMTime (Korkmaz 2024)                  extended LSTM, finance benchmark winner
            VLSTM (Saly-Kaufmann/Wood/Zohren 2026)    2.40 Sharpe daily futures DL frontier
            xLSTM (Saly-Kaufmann 2026)                1.79 Sharpe with best transaction-cost robustness

GLD-spec    Gao-Han-Li-Zhou 2014 half-hour-5 rule     5.43 Sharpe single-feature (the GLD-specific bar)

Tree        XGBoost on the same 681 features          gradient boosted truth-stick

Replica     Forecast-to-Fill (Wright et al, daily)    rules-only — 2.88 Sharpe daily futures, separate scoreboard

Ours        nanoGLDV1                               the thing being trained
```

If V1 fails to beat the simpler ensemble (Gao 2014 + XGBoost) by >= 0.2 Sharpe OOS net of costs, the ship recommendation is the simpler ensemble, full stop. Same rule for xLSTMTime / VLSTM ties — ship the simpler one.

### 8. Sizing as a separate, ablated stage (F2F-style machinery)

Sizing is not jammed into the loss. It is a separate stage with its own ablation table:

```
Stage 1     Head B output * vol target                              (basic)
Stage 2     Head B + friction-adjusted Kelly (lambda=0.4)
            + ATR-14 exits (2x stop, 1.5x trail)
            + vol target 15% annualized
            + 30-day timeout (per F2F)
            + sqrt-impact cost model (gamma=0.02, k=0.7bps)
            + conformal floor (RAPS lower-bound < 0.40 -> position 0)
```

Stage 2 must beat Stage 1 by >= 0.2 Sharpe out-of-sample to ship the full pipeline. Otherwise ship Stage 1. Conformal floor wraps the position before Kelly, because miscalibrated probabilities x Kelly is how you blow up. Head B (continuous position-weight, trained jointly on cost-aware Sharpe loss) is the V1 ship gate; Head A (focal CE) feeds the calibration / conformal floor.

---

## Architecture

```python
# Forward signature (V1, from plan/05-MODEL-TRAINING-CALIBRATION.md)
class nanoGLDV1_5(nn.Module):
    def forward(
        self,
        bars: Tensor,                        # (B, T=64, C=681), per-channel RevIN'd, decomposed
        regime_vec: Tensor,                  # (B, 12), VIX/RV/FOMC/year/HMM
        news_embeddings: Tensor,             # (B, K, 256), Qwen3 truncated
        news_mask: Tensor,                   # (B, K)
        is_news_present: Tensor,             # (B,) int, 0/1
    ) -> tuple[Tensor, Tensor]:
        return logits_3class, position_weight  # (B, 3), (B,) in [-1, +1]
```

Default config: `D=384`, `num_heads=6` (head_dim=64), `num_layers=12` (10 transformer + 2 sLSTM), `T_bars=64`, `patch_len=4`, `patch_stride=4`, `drop_path` linear `0.0 -> 0.2`. ~24M params at this size; spec scales to 60M. FiLM regime conditioning at layers {2, 4, 6, 8, 10}. Cross-attn to news at layers {3, 7, 11}. Full math + V1 redlines in [`plan/V1-SPEC.md`](./plan/V1-SPEC.md), implementation contract in [`plan/05-MODEL-TRAINING-CALIBRATION.md`](./plan/05-MODEL-TRAINING-CALIBRATION.md).

---

## Data

| Source | Window | Cost |
|---|---|---|
| GLD 30min bars (Alpaca) | 5 years | free |
| GDELT GKG (geopolitical) | 5 years | free, BigQuery free tier |
| FRED macro (DXY, DGS10, DGS2, oil) | 5 years | free |
| Alpaca News API headlines | rolling | free |
| Public RSS (Reuters, Bloomberg, MarketWatch, SEC EDGAR) | rolling | free |

Snapshots are immutable, content-addressed by SHA256, and reproducible bit-for-bit from [`plan/02-DATA-PIPELINE.md`](./plan/02-DATA-PIPELINE.md).

---

## Hard rules (the bugs that look fine in code review)

These get caught in `tests/`. They cost a week if missed.

V1 invariants (1-17, kept):

- **`bars_per_year = 3276`**, not 17,500. NYSE regular trading hours only. Wrong annualization -> entire backtest wrong, ~2.31x inflated Sharpe.
- **No MSE on returns. Ever.** Focal CE + Sharpe loss only. See forecast-collapse rule above.
- **Stay from-scratch.** Encoder + sLSTM head + heads all trainable. Only Qwen3 frozen.
- **Ship the simpler model if it ties.** TLOB lesson.
- **No `torch.view_as_complex` for RoPE on MPS.** Real-form rotation only. Same math, runs everywhere.
- **No autocast, no `torch.compile`, no quantization on the trainer.** FP32 weights, deterministic seeds. PyTorch 2.11 + MPS still has sharp edges; H100 has bf16 but V1 stays FP32 deterministic.
- **`.contiguous()` Q/K/V before SDPA** (PyTorch #181133).
- **1-week embargo between train and val folds.** Otherwise label leakage from overlapping 30-min lookahead windows.
- **Pin `torch>=2.11.0,<2.12`.** SDPA fix [#174945](https://github.com/pytorch/pytorch/pull/174945) lands here.
- **Sortino with the canonical formula** (`sqrt(mean(min(0, r)^2))`), not `std()` of the negative-only subset.
- **Every feature row carries `t_visible`.** CI gate: `test_release_ts_lte_t_visible_all_rows`.
- **ALFRED `get_series_all_releases` for ALL FRED series** (CPI/PCE annual revisions silently rewrite 5y of history).
- **pandas-ta KAMA / Ichimoku / KST / DPO / TRIX / Vortex FORBIDDEN** (look-ahead bugs).
- **Calendar features = binary windows ONLY** (no `minutes_until_event`).
- **Anchor templates = hand-crafted with NO event provenance.**
- **Alpaca News field = `created_at`** (NOT `published_at`, NEVER `updated_at`).
- **DFF for daily Fed Funds** (NOT FEDFUNDS, monthly).
- **Gitleaks runs BEFORE the first commit.** Pre-commit hook installs in bootstrap.

V1 invariants (18-25, NEW):

- **Per-bucket eval (news-present / news-absent / both) is non-negotiable.** 51% of bars are news-absent; reporting only the average flies blind.
- **Cost-stress at {0.5x, 1.0x, 1.5x} on every reported Sharpe.** Hard gate: Sharpe > 0.5 at 1.5x.
- **Deflated Sharpe Ratio > 1.0 hard gate.** No cherry-picking across configs.
- **SimPSI / Wave-Mask aug only.** Naive jittering FORBIDDEN — Fons 2020 net-negative on Sharpe.
- **Focal loss gamma=3 (NOT vanilla CE).** Required for clean T-scaling / APS interaction (Xi 2024).
- **Triple-barrier labels with spread-adjusted neutral threshold.** Replaces fixed 5-bps cutoff.
- **Variable per-batch modality dropout p ~ U(0.1, 0.9).** NOT 15% constant — training distribution must bracket inference.
- **Decision-aware head (multi-task with Sharpe loss) is V1 ship gate.** End-to-end profit metric, not just classification accuracy.

Every doc in `plan/` lists its own rules at the top.

---

## Promotion gates (V1)

A model promotes from "trained" to "reportable" only after passing:

```
Gate 1   Walk-forward Sharpe > 1.0 net of 1x cost across 4 folds       (was 0.8)
Gate 2   Sharpe > 0.5 net of 1.5x cost                                 (NEW hard)
Gate 3   Beats best baseline by >= 0.2 Sharpe on >= 3 of 4 folds
Gate 4   Conformal coverage within +/- 2% of nominal on val + per-bucket
Gate 5   Stage 2 sizer (decision-aware head) beats Stage 1 by >= 0.2 Sharpe OOS
Gate 6   Drawdown circuit breaker tested on >= 2 historical regimes
Gate 7   Deflated Sharpe Ratio > 1.0                                   (NEW hard)
Gate 8   Per-bucket Sharpe (news-present, news-absent) both positive   (NEW hard)
```

Fail any gate, the negative result gets reported. Cherry-picking is a fireable offense.

---

## Repo layout

```
plan/                              the spec, V1 locked (8 docs + V1-SPEC + HANDOFF + STATUS)
  V1-SPEC.md                     canonical V1 redlines (read first)
  HANDOFF.md                       data -> model phase hand-off
  00-OVERVIEW.md                   project context, hard rules, execution mode
  01-INFRA-AND-SECURITY.md         uv, pre-commit, gitleaks, CI, secrets
  02-DATA-PIPELINE.md              GLD bars + ETF basket + GDELT + FRED + COT/WGC + news -> parquet
  03-NEWS-EMBEDDING.md             Qwen3-Embedding-4B per-article + LAFTR + anchor cosines
  04-FEATURE-ENGINEERING.md        681 features + VSN + series decomp + per-channel RevIN + triple-barrier
  05-MODEL-TRAINING-CALIBRATION.md nanoGLDV1 hybrid encoder + dual head + SimMTM SSL -> probe -> LLRD + focal -> RAPS -> AgACI + Laplace
  06-BACKTEST.md                   walk-forward CV, baseline ladder, DSR + bootstrap CI + per-bucket + cost-stress
  07-SIZING-AND-EXITS.md           F2F-style sizing: friction-adjusted Kelly + ATR exits + vol target + conformal floor
  08-LIVE-TRADING.md               launchd cron + Alpaca live + drift detection
  STATUS.md                        doc tracker + execution-mode rules

src/nanogld/                       the code, mirrored 1:1 with plan/
  data/, embed/, features/         shipped 2026-05-08 (data phase complete)
  model/, training/, calibration/, sizing/, backtest/  not-yet-built
tests/                             pytest suite (data + features today; model + calib + sizing as docs ship)
data/                              (gitignored) parquet snapshots + unified .pt
checkpoints/                       (gitignored) trained models
docs/                              SETUP.md (secrets) + REPRODUCE.md (clone-to-running)
```

Every `src/nanogld/<module>/` directory maps 1:1 to a `plan/0X-*.md` doc. Read the doc, build the module, ship.

---

## Built with

- **PyTorch 2.11** — pinned, FP32
- **Qwen3-Embedding-4B** — frozen news embedder, MRL-truncated 2560 -> 256
- **Schedule-Free AdamW** — [Defazio 2024](https://arxiv.org/abs/2405.15682), with Cautious update mask ([Liang 2024](https://arxiv.org/abs/2411.16085)) on top
- **Friendly-SAM** — flatter minima, filtered-gradient SAM
- **muP** — [Yang 2022](https://arxiv.org/abs/2203.03466), tiny-model sweep transferred to 30M
- **xLSTMTime sLSTM blocks** — [Alharthi & Mahmood 2024](https://arxiv.org/abs/2407.10240)
- **PatchTST channel-independent patches** — small-data overfit defense
- **CFA cross-modal projector** — Lee 2025, FiLM + orthogonal residual before Flamingo K/V
- **AECF entropy-gated curriculum mask** — [Chlon 2025](https://arxiv.org/abs/2505.15417)
- **SimMTM SSL** — [Dong NeurIPS 2023](https://arxiv.org/abs/2302.00861), multi-mask + similarity reconstruction
- **RAPS + AgACI conformal** — [Angelopoulos 2020](https://arxiv.org/abs/2009.14193) + [Zaffran ICML 2022](https://arxiv.org/abs/2202.07282)
- **Laplace last-layer** — [Daxberger 2021](https://arxiv.org/abs/2106.14806), epistemic uncertainty without 20x MC dropout cost
- **HuggingFace `transformers`, `accelerate`, `datasets`** — plumbing only; the model is from scratch
- **GDELT GKG** via Google BigQuery — geopolitical signal
- **FRED API** — macro time series
- **`pandas-market-calendars`** — NYSE RTH calendar (the 3,276 number)
- **`arch.bootstrap.StationaryBootstrap`** — honest CIs
- **Nia** — research agents that verified every paper claim and library version above (27 V1 + 9 V1 = 36 specialized agents)

---

## License

[MIT](./LICENSE) © Saam Siavoshian

## Author

**Saam Siavoshian**

- X: [@samsiavoshian](https://x.com/samsiavoshian)
- Email: [samsiavoshian2009@gmail.com](mailto:samsiavoshian2009@gmail.com)

---

Built at the speed of ⚡ and with ❤️.
