# nanoGLD

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](./LICENSE)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.11-EE4C2C.svg)](https://pytorch.org)
[![Python](https://img.shields.io/badge/Python-3.13-3776AB.svg)](https://www.python.org)
[![Apple Silicon](https://img.shields.io/badge/Hardware-M4%20Mac%20mini%2016GB-000000.svg)](https://www.apple.com/mac-mini/)
[![Capital](https://img.shields.io/badge/Live%20capital-%24100-FFD700.svg)](https://alpaca.markets)
[![Status](https://img.shields.io/badge/Status-Building%20in%20public-blue.svg)](#status)

A from-scratch encoder-only transformer that predicts the next 30-minute direction of **GLD** (gold ETF), fused with frozen LLM news embeddings, trained on 5 years of 30-minute bars on a single **16GB M4 Mac mini**, then deployed live on **$100 of real Alpaca capital** from a **16GB M4 Pro Macbook**.

> **TL;DR:** `nanoGPT` for gold trading. ~24-60M params, raw PyTorch on MPS, no cloud GPU, no API training tricks, no quantization, no autocast. Real money. Honest results, including the bad ones.

```
   ┌──────────────┐      ┌──────────────────┐      ┌──────────────┐
   │  M4 Mac mini │      │   GLD 30min bars │      │  Live cycle  │
   │   (trainer)  │◄────►│  + GDELT + Alpaca│◄────►│   M4 Pro     │
   │   16 GB MPS  │      │  + FRED macro    │      │  $100 Alpaca │
   └──────────────┘      └──────────────────┘      └──────────────┘
          │                       │                        │
          ▼                       ▼                        ▼
     PyTorch 2.11           parquet snapshots        launchd cron
     Schedule-Free          (immutable, hashed)       every 30min
     + Friendly-SAM                                   during RTH
          │                                                │
          └──────────────────► nanoGLDV1 ◄─────────────────┘
                              ~24-60M params
                              channel-group tokens
                              + Qwen3-Embedding-4B (frozen)
```

## Why?

Most ML-trading repos do one of two things. Either they print a 4.0 Sharpe on training data and quietly never trade live, or they wire LLMs into a strategy with no ablations, no baselines, no honest comparison, and call it research.

This is the third thing. Build the model from scratch in raw PyTorch the way Karpathy builds nanoGPT. Use the best frozen LLM for news (Qwen3-Embedding-4B, 4-bit MLX, `<200ms` per news bar on the M4 Pro). Run a real walk-forward CV with embargo. Compare against DLinear, TSMixer, TimeMixer, xLSTMTime, XGBoost, and Forecast-to-Fill. If a 2M-param TSMixer beats us, ship TSMixer. If buy-and-hold beats all of them after costs, ship buy-and-hold. The story is the rigor, not the win.

And then we put **real $100** behind it. Paper trading is dev. Real money is the only honest test.

---

## Status

```
Planning:        ██████████████ 100%   3 verification rounds, 19 Nia research agents
Implementation:  ░░░░░░░░░░░░░░   0%   agents dispatched, day 1 in progress
Live capital:    ░░░░░░░░░░░░░░   $0   deploys after walk-forward Sharpe gates pass
```

**11 implementation docs** live in [`plan/`](./plan). Each one is owned by a single Opus 4.7 agent. Each has Nia-verified citations, an agent-isolation header, an interface contract, and acceptance criteria. The plan IS the code spec. Read [`plan/00-OVERVIEW.md`](./plan/00-OVERVIEW.md) first.

Follow the build live: **[@samsiavoshian](https://x.com/samsiavoshian)** on X.

---

## What we're building (one paragraph)

A from-scratch encoder-only transformer (~24-60M params) that predicts next-30min direction of GLD as a **3-class classification** (UP / FLAT / DOWN, never MSE on returns, see [arXiv:2604.00064](https://arxiv.org/abs/2604.00064) on forecast collapse). Inputs are **channel-group tokens** combining price + volatility + macro + geopolitical features (RevIN-normalized) fused via Flamingo-gated cross-attention with semantic news embeddings from **Qwen3-Embedding-4B** (frozen, 4-bit MLX, MRL-truncated 2560→256). Trained with **3-stage SSL → linear-probe → LLRD fine-tune** using **Schedule-Free AdamW + Friendly-SAM (ρ=0.05)** on **3,276 bars/year** of NYSE RTH data. Sized at inference time by **Kelly-lite × vol-target × conformal confidence** (Wright 2026). Deployed live by a launchd cron every 30 minutes during market hours.

---

## Architecture

```
Bars (T=128, 30min RTH)            News (T=128, multi-source)
       │                                   │
       ▼                                   ▼
   RevIN per                       Qwen3-Embedding-4B
   channel group                   (frozen, 4-bit MLX)
       │                                   │
       ▼                                   │
   Channel-group                           │
   tokenizer                               │
       │                                   │
       ▼                                   ▼
   ┌──────────────────────────────────────────┐
   │  nanoGLDV1 encoder                       │
   │  RMSNorm + SwiGLU + RoPE (real-form)     │
   │  + QK-Norm + per-head gating             │
   │  + value residuals                       │
   │  + Perceiver-Resampler-lite news fuser   │
   │  + Flamingo-gated cross-attn             │
   │  no bias, FP32, MPS-safe                 │
   └────────────────────┬─────────────────────┘
                        │
                        ▼
                3-class logits (UP/FLAT/DOWN)
                        │
                        ▼
            Temperature scaling + Conformal
                        │
                        ▼
        Kelly-lite × vol-target × confidence
                        │
                        ▼
                  Alpaca order
```

Why these choices, briefly:

- **Encoder-only, not decoder.** Predicting one token (the next-30min class), not generating text. Causal masking is overkill.
- **Channel-group tokens, not patch tokens.** Bars carry heterogeneous signals (price, vol, macro, geo). Group them so attention can learn cross-group structure cleanly.
- **RoPE in real form.** `torch.view_as_complex` silently breaks on MPS. Real-form rotation is identical math, runs everywhere.
- **RMSNorm, SwiGLU, no bias.** Same recipe as Llama / Qwen / nanoGPT. Cleaner gradients, fewer params, faster.
- **Schedule-Free AdamW + Friendly-SAM.** Schedule-Free skips the warmup/decay tuning. Friendly-SAM finds flatter minima, which generalize better in non-stationary regimes (markets).
- **Frozen LLM for news.** Fine-tuning an 8B LLM on M4 hardware is wishful thinking. Use the frozen embeddings, fuse via cross-attention, train the fuser. Same approach as Flamingo / BLIP-2.
- **3-class CE, never MSE.** MSE on log-returns degenerates into predicting the conditional mean (~0). The model "wins" by always saying flat. Classification forces signal.

Full spec lives in [`plan/03-MODEL-ARCHITECTURE.md`](./plan/03-MODEL-ARCHITECTURE.md).

---

## Hardware

| Role | Machine | RAM | Notes |
|---|---|---|---|
| Trainer | M4 Mac mini | 16 GB | MPS backend, FP32, no autocast, batch=8 grad-accum=8 |
| Live + dev | M4 Pro Macbook | 16 GB | Runs Qwen3 4-bit MLX in `<200ms`, launchd cron every 30min |

No cloud GPU. No Modal / Runpod / Lambda. No CUDA. The constraint is the feature.

---

## Data

| Source | Window | Free? |
|---|---|---|
| GLD 30min bars (Alpaca) | 5 years | ✅ |
| GDELT GKG (geopolitical) | 5 years | ✅ via BigQuery free tier |
| FRED macro (DXY, DGS10, DGS2, oil) | 5 years | ✅ |
| Alpaca News API headlines | rolling | ✅ |
| Public RSS (Reuters / Bloomberg / etc) | rolling | ✅ |

**$0 monthly cost target.** BigQuery free tier (1 TB query/mo, 10 GB storage) caps GDELT cost. `maximum_bytes_billed` set per query. Custom 1024 GiB/day project quota installed before first query. Everything else fits in `~8GB` local parquet snapshots, immutable, content-addressed by SHA256.

Pipeline lives in [`plan/01-DATA-PIPELINE.md`](./plan/01-DATA-PIPELINE.md). Storage decision rule (60% free disk threshold) is in there too.

---

## Honest comparison ladder

Every result gets benchmarked against:

```
Naive    Buy-and-hold                       baseline floor
         50/200 EMA crossover               classical TA
         20-period Donchian breakout        classical TA

Linear   DLinear (Zeng AAAI 2023)           "transformers are not effective"
         TSMixer (Chen 2023)                MLP-Mixer for time series
         TimeMixer (ICLR 2024)              decomposition mixing

RNN      xLSTMTime (Korkmaz 2024)           extended LSTM, time-series flavor

Tree     XGBoost on the same features       gradient boosted truth-stick

Replica  Forecast-to-Fill (Wright et al)    no ML at all, just rules

Ours     nanoGLDV1                          the thing we're building
```

The X thread shows every row, with stationary block bootstrap CIs, regime breakdowns (vol terciles, FOMC weeks, news density), and Deflated Sharpe Ratio. Cherry-picking is a fireable offense.

Full discipline in [`plan/06-BACKTEST.md`](./plan/06-BACKTEST.md).

---

## Repo layout

```
plan/                         the spec (read first)
  00-OVERVIEW.md              project context, hard rules, execution mode
  01-DATA-PIPELINE.md         GLD bars + GDELT + macro + news → parquet
  02-FEATURE-ENGINEERING.md   42 features + RevIN + 3-class labels
  03-MODEL-ARCHITECTURE.md    nanoGLDV1 spec, raw PyTorch
  04-NEWS-EMBEDDING.md        Qwen3-Embedding-4B, 4-bit MLX, anchor cosines
  05-TRAINING-PROCEDURE.md    SSL → probe → LLRD, Schedule-Free + F-SAM
  06-BACKTEST.md              walk-forward CV, baseline ladder, DSR
  07-SIZING-STAGE2.md         Kelly-lite × vol-target × conformal
  09-LIVE-TRADING.md          launchd cron, Alpaca, drift detection
  10-INFRA-AND-SECURITY.md    uv, pre-commit, gitleaks, CI, secrets
  STATUS.md                   doc tracker + execution-mode rules

src/nanogld/                  (coming) the code, mirrored 1:1 with plan/
tests/                        (coming) pytest suite per doc
data/                         (gitignored) parquet snapshots
checkpoints/                  (gitignored) trained models
```

Every `src/nanogld/<module>/` directory maps 1:1 to a `plan/0X-*.md` doc. Read the doc, build the module, ship.

---

## Hard rules (the ones that bite)

These are the bugs that look fine in code review and silently waste a week.

- **`bars_per_year = 3276`**, not 17,500. NYSE regular trading hours only. Sharpe annualization wrong = entire backtest wrong.
- **No MSE on returns.** Ever. 3-class CE only. See [arXiv:2604.00064](https://arxiv.org/abs/2604.00064).
- **No `torch.view_as_complex` for RoPE on MPS.** Real-form only.
- **No autocast, no `torch.compile`, no quantization on the trainer.** PyTorch 2.11 + MPS is brittle. FP32 weights, deterministic seeds.
- **1-week embargo between train and val folds.** Otherwise label leakage from overlapping 30-min lookahead windows.
- **Pin `torch>=2.11.0,<2.12`.** SDPA fix [#174945](https://github.com/pytorch/pytorch/pull/174945) lands here.
- **Gitleaks runs BEFORE first commit.** Pre-commit hook installs in the bootstrap step.

Every doc in `plan/` lists its own rules at the top. Read them.

---

## Ship gates

The model goes live on real money only after passing:

```
Gate 1   Walk-forward Sharpe > 0.8 net of costs across 4 folds
Gate 2   Beats best baseline on at least 3 of 4 folds
Gate 3   Conformal coverage within ±2% of nominal on val
Gate 4   60-day paper-trade matches backtest within 1σ
Gate 5   Drawdown circuit breaker tested in 2 historical regimes
Gate 6   /cso pass on the live cycle (secrets, network, order flow)
```

Fail any gate, no live deploy. Write the post-mortem instead. Either result is content for the thread.

---

## Built with

- **PyTorch 2.11** — pinned, MPS backend, FP32
- **Qwen3-Embedding-4B** — frozen news embedder, 4-bit MLX
- **Schedule-Free AdamW** — [Defazio 2024](https://arxiv.org/abs/2405.15682)
- **Friendly-SAM** — flatter minima for non-stationary regimes
- **HuggingFace `transformers`, `accelerate`, `datasets`** — plumbing only
- **Alpaca Markets** — broker, news, real-time bars
- **GDELT GKG** via Google BigQuery free tier — geopolitical signal
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

Built at the speed of ⚡ and with ❤️ on a 16GB Mac mini.
