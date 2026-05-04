# nanoGLD

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](./LICENSE)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.11-EE4C2C.svg)](https://pytorch.org)
[![Python](https://img.shields.io/badge/Python-3.13-3776AB.svg)](https://www.python.org)
[![Plan](https://img.shields.io/badge/Plan-V1%20frozen-blueviolet.svg)](./plan)
[![Status](https://img.shields.io/badge/Status-Training-orange.svg)](#status)

A from-scratch encoder-only transformer that predicts the next 30-minute direction of **GLD** (gold ETF) as a 3-class problem (UP / FLAT / DOWN), fused with frozen LLM news embeddings via Flamingo-gated cross-attention, trained on 5 years of NYSE 30-minute bars.

> **TL;DR:** ~24-60M-parameter encoder-only transformer with channel-group tokens, RMSNorm + SwiGLU + real-form RoPE + QK-Norm + per-head gating + value residuals, frozen Qwen3-Embedding-4B for news, Schedule-Free AdamW + Friendly-SAM, 3-stage SSL → linear-probe → LLRD fine-tune, walk-forward CV with 1-week embargo, Deflated Sharpe Ratio, full baseline ladder.

```
   bars (T=128, 30min RTH)            news (T=128, multi-source)
          │                                      │
          ▼                                      ▼
      RevIN per                           Qwen3-Embedding-4B
      channel group                       (frozen, MRL → 256)
          │                                      │
          ▼                                      ▼
   ┌───────────────────────────────────────────────────┐
   │  nanoGLDV1 encoder, ~24-60M params                │
   │                                                   │
   │   RMSNorm + SwiGLU + RoPE (real-form, MPS-safe)   │
   │   + QK-Norm                                       │
   │   + per-head gating                               │
   │   + value residuals (IMU-1)                       │
   │   + Perceiver-Resampler-lite + Flamingo-gated     │
   │     cross-attention to news                       │
   │   no bias, FP32                                   │
   └─────────────────────────┬─────────────────────────┘
                             │
                             ▼
                  3-class logits (UP / FLAT / DOWN)
                             │
                             ▼
            temperature scaling + conformal calibration
```

## Why this exists

Most ML-trading repos do one of two things. Print a 4.0 Sharpe on training data and quietly never validate out-of-sample. Or wire an LLM into a strategy with no ablations, no baselines, no honest comparison, and call it research.

This is the third thing. A real transformer, written from scratch in raw PyTorch the way Karpathy writes nanoGPT, with a multimodal news-fusion head wired to a frozen Qwen3-Embedding-4B. Walk-forward CV with embargo. Bootstrap confidence intervals. Deflated Sharpe Ratio. A full baseline ladder, including a 2026 paper replication that is the actual bar to beat. If a 2M-param TSMixer wins, ship TSMixer. If buy-and-hold wins after costs, ship buy-and-hold. The story is the rigor.

---

## Status

```
Plan:           ██████████████ V1 frozen, 3 verification rounds, 19 Nia agents
Training:       ░░░░░░░░░░░░░░ in progress
Backtest:       ░░░░░░░░░░░░░░ pending checkpoints
```

**This README is a sketch until proven.** Numbers, claims, and architecture details get rewritten only when an implementation agent ships verified results. The journey from sketch to real artifact is the journey from `plan/` → trained checkpoints → reported metrics. Until then, treat everything below as the design we are testing, not what we are claiming.

11 implementation docs live in [`plan/`](./plan). Each is owned by a single agent with Nia-verified citations, an interface contract, and acceptance criteria. The plan is the spec. Read [`plan/00-OVERVIEW.md`](./plan/00-OVERVIEW.md) first.

---

## What's interesting about it

### 1. Multimodal time-series transformer, end-to-end from scratch

Numerical bars and natural-language headlines fuse inside a single encoder. Bars get tokenized as **channel groups** (iTransformer-lite, ~14 tokens) so attention learns cross-group structure cleanly. News headlines pass through a frozen **Qwen3-Embedding-4B** (Apache 2.0, MTEB-en 74.6, MRL-truncated 2560 → 256), then condense through a **Perceiver-Resampler-lite** into 8 query slots, then merge into the bar stream via **Flamingo-gated cross-attention**. The fuser is the trainable bridge; the LLM stays frozen.

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

### 4. Schedule-Free AdamW + Friendly-SAM

No warmup. No cosine. No tuning the LR schedule. **Schedule-Free AdamW** ([Defazio 2024](https://arxiv.org/abs/2405.15682)) self-anneals. Friendly-SAM (filtered-gradient SAM, ρ=0.05) finds flatter minima — measurably better generalization on non-stationary regimes. EMA decay 0.999 on top. Dropout 0.2, stochastic depth 0.15, label smoothing 0.1, modality dropout 15% on news.

### 5. Classification, not regression

3-class cross-entropy, never MSE on returns. [arXiv:2604.00064](https://arxiv.org/abs/2604.00064) (March 2026) proved that on weak-conditional-structure data like financial returns, transformer expressivity *increases variance without reducing bias*, and MSE collapses into predicting the conditional mean (~0). The model "wins" by always saying flat. Classification forces signal.

### 6. Honest evaluation harness

Walk-forward CV, 4 folds across 5 years, **1-week embargo** between train and val to kill label leakage from overlapping 30-min lookahead windows. **3,276 bars/year** for Sharpe annualization (NYSE regular trading hours, NOT the 17,500 you get from a 24/7 calendar — that's a 2.31× silent inflation). **Deflated Sharpe Ratio** ([Bailey & Lopez de Prado](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2460551)) on top of raw Sharpe. **Stationary block bootstrap** for confidence intervals. **Regime breakdowns** by vol tercile, FOMC week, and news density.

### 7. The baseline ladder

Every result gets reported alongside:

```
Naive       Buy-and-hold                              baseline floor
            50/200 EMA crossover                      classical TA
            20-period Donchian breakout               classical TA

Linear      DLinear (Zeng AAAI 2023)                  "transformers are not effective"
            TSMixer (Chen 2023)                       MLP-Mixer for time series
            TimeMixer (ICLR 2024)                     decomposition mixing

RNN         xLSTMTime (Korkmaz 2024)                  extended LSTM, won 2026 finance bench

Tree        XGBoost on the same features              gradient boosted truth-stick

Replica     Forecast-to-Fill (Wright et al)           no ML, just rules — 2.88 Sharpe to beat

Ours        nanoGLDV1                                 the thing being trained
```

If nanoGLD does not beat all baselines by ≥0.2 Sharpe out-of-sample, the report explicitly recommends shipping the simpler model.

### 8. Sizing as a separate, ablated stage

Sizing is not jammed into the loss. It is a separate stage with its own ablation table:

```
Stage 1     argmax → 1 share when class != flat
Stage 1.5a  Kelly-lite only
Stage 1.5b  vol-target only
Stage 2     vol-target × Kelly-lite × conformal-confidence
```

Stage 2 must beat Stage 1 by ≥0.2 Sharpe out-of-sample to ship. Conformal calibration ([Wright 2026](https://arxiv.org/abs/2601.07852)) wraps the logits before Kelly-lite, because miscalibrated probabilities × Kelly-lite is how you blow up.

---

## Architecture

```python
# Forward signature (real, from plan/03-MODEL-ARCHITECTURE.md)
class nanoGLDV1(nn.Module):
    def forward(
        self,
        channel_inputs: dict[str, Tensor],   # channel-group dict, RevIN-normed
        news_embeddings: Tensor,             # (B, n_sources, 256), Qwen3 truncated
        news_mask: Tensor,                   # (B, n_sources)
    ) -> Tensor:
        return logits  # (B, 3)
```

Default config: `D=384`, `num_heads=6` (head_dim=64), `num_layers=12`, `T_bars=64`, `dropout=0.2`, `drop_path=0.15`. ~24M params at this size; spec scales to 60M. Full math + ablation candidates (TDA, SyPE, Muon for 2D weights) in [`plan/03-MODEL-ARCHITECTURE.md`](./plan/03-MODEL-ARCHITECTURE.md).

---

## Data

| Source | Window | Cost |
|---|---|---|
| GLD 30min bars (Alpaca) | 5 years | free |
| GDELT GKG (geopolitical) | 5 years | free, BigQuery free tier |
| FRED macro (DXY, DGS10, DGS2, oil) | 5 years | free |
| Alpaca News API headlines | rolling | free |
| Public RSS (Reuters, Bloomberg, MarketWatch, SEC EDGAR) | rolling | free |

Snapshots are immutable, content-addressed by SHA256, and reproducible bit-for-bit from [`plan/01-DATA-PIPELINE.md`](./plan/01-DATA-PIPELINE.md).

---

## Hard rules (the bugs that look fine in code review)

These get caught in `tests/`. They cost a week if missed.

- **`bars_per_year = 3276`**, not 17,500. NYSE regular trading hours only. Wrong annualization → entire backtest wrong, ~2.31× inflated Sharpe.
- **No MSE on returns. Ever.** 3-class CE only. See forecast-collapse rule above.
- **No `torch.view_as_complex` for RoPE on MPS.** Real-form rotation only. Same math, runs everywhere.
- **No autocast, no `torch.compile`, no quantization on the trainer.** PyTorch 2.11 + MPS still has sharp edges. FP32 weights, deterministic seeds.
- **1-week embargo between train and val folds.** Otherwise label leakage from overlapping 30-min lookahead windows.
- **Pin `torch>=2.11.0,<2.12`.** SDPA fix [#174945](https://github.com/pytorch/pytorch/pull/174945) lands here.
- **Sortino with the canonical formula** (`sqrt(mean(min(0, r)^2))`), not `std()` of the negative-only subset.
- **Gitleaks runs BEFORE the first commit.** Pre-commit hook installs in bootstrap.

Every doc in `plan/` lists its own rules at the top.

---

## Promotion gates

A model promotes from "trained" to "reportable" only after passing:

```
Gate 1   Walk-forward Sharpe > 0.8 net of costs across 4 folds
Gate 2   Beats best baseline on at least 3 of 4 folds
Gate 3   Conformal coverage within ±2% of nominal on val
Gate 4   Stage 2 sizer beats Stage 1 by ≥0.2 Sharpe OOS
Gate 5   Drawdown circuit breaker tested on ≥2 historical regimes
Gate 6   Deflated Sharpe Ratio > 1.0
```

Fail any gate, the negative result gets reported. Cherry-picking is a fireable offense.

---

## Repo layout

```
plan/                         the spec, V1 frozen
  00-OVERVIEW.md              project context, hard rules, execution mode
  01-DATA-PIPELINE.md         GLD bars + GDELT + macro + news → parquet
  02-FEATURE-ENGINEERING.md   42 features + RevIN + 3-class labels
  03-MODEL-ARCHITECTURE.md    nanoGLDV1 spec, raw PyTorch
  04-NEWS-EMBEDDING.md        Qwen3-Embedding-4B, anchor cosines
  05-TRAINING-PROCEDURE.md    SSL → probe → LLRD, Schedule-Free + F-SAM
  06-BACKTEST.md              walk-forward CV, baseline ladder, DSR
  07-SIZING-STAGE2.md         Kelly-lite × vol-target × conformal
  09-LIVE-TRADING.md          deployment cycle (downstream)
  10-INFRA-AND-SECURITY.md    uv, pre-commit, gitleaks, CI, secrets
  STATUS.md                   doc tracker + execution-mode rules

src/nanogld/                  (coming) the code, mirrored 1:1 with plan/
tests/                        (coming) pytest suite per doc
data/                         (gitignored) parquet snapshots
checkpoints/                  (gitignored) trained models
```

Every `src/nanogld/<module>/` directory maps 1:1 to a `plan/0X-*.md` doc. Read the doc, build the module, ship.

---

## Built with

- **PyTorch 2.11** — pinned, FP32
- **Qwen3-Embedding-4B** — frozen news embedder, MRL-truncated 2560 → 256
- **Schedule-Free AdamW** — [Defazio 2024](https://arxiv.org/abs/2405.15682)
- **Friendly-SAM** — flatter minima, filtered-gradient SAM
- **HuggingFace `transformers`, `accelerate`, `datasets`** — plumbing only; the model is from scratch
- **GDELT GKG** via Google BigQuery — geopolitical signal
- **FRED API** — macro time series
- **`pandas-market-calendars`** — NYSE RTH calendar (the 3,276 number)
- **`arch.bootstrap.StationaryBootstrap`** — honest CIs
- **Nia** — research agents that verified every paper claim and library version above

---

## License

[MIT](./LICENSE) © Saam Siavoshian

## Author

**Saam Siavoshian**

- X: [@samsiavoshian](https://x.com/samsiavoshian)
- Email: [samsiavoshian2009@gmail.com](mailto:samsiavoshian2009@gmail.com)

---

Built at the speed of ⚡ and with ❤️.
