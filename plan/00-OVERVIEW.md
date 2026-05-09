# 00 — nanoGLD Project Overview

**Read this doc first. Every implementation agent reads this before touching code.**

**Project:** nanoGLD — Karpathy-mode LLM-augmented gold trader on local hardware
**Owner:** samsiavoshian
**Status:** ⭐ **DATA PHASE COMPLETE 2026-05-08. V1 ARCHITECTURE FROZEN 2026-05-08.** Single unified dataset on remote host ready for model training. Read `plan/V1-SPEC.md` first, then `plan/HANDOFF.md` before starting model phase.
**Version:** **V1 — frozen.** Replaces V1 (frozen 2026-05-04). Plan does not get a version bump until the owner explicitly says so. Agents must not introduce new version stamps. V1 spec sheet at `plan/V1-SPEC.md` is the authoritative change list; anything not on that list stays as V1.
**Last verified:** 2026-05-08 (V1 = 9-agent Nia research synthesis on top of V1's 3 verification rounds + 19 Nia agents + ~30 critical findings + 17 leakage hard rules)
**Data phase complete:** 2026-05-08 (3 paranoid audit iterations × 8 agents = 19 bugs caught + fixed; cron loop continuing)

---

## ⭐ DATA PHASE COMPLETE — READ THIS FIRST

**Single unified dataset:** `$NANOGLD_REMOTE/data/processed/training_v1_unified.pt`
- remote host at `$NANOGLD_HOST` (Tailscale)
- 234 MB PyTorch native file
- 75,993 30-min bars × 681 features (float32) + 40,032 news embeddings (256-dim float16)
- Train/Val/Test splits: 57,697 / 7,540 / 10,756 (chronological, 2016 → 2026)
- Zero leakage, zero inf, zero 100%-NaN cols, max |corr| with next_log_return = 0.0117

**For full data-phase context:** read `plan/HANDOFF.md`

**For status of all docs:** read `plan/STATUS.md`

---

## YOU ARE AN OPUS 4.7 IMPLEMENTATION AGENT

You were assigned ONE doc in this folder. You will:

1. **Read this overview FIRST.** Understand what we're building, why, and where your piece fits.
2. **Read your assigned doc thoroughly.** Every claim has a citation; every architectural decision has been Nia-verified.
3. **Build ONLY what your doc covers.** Do not touch other docs' source files. Other agents are working on those in parallel; collisions break everything.
4. **Spawn Nia agents whenever you have questions.** Don't guess on library versions, paper claims, API behavior, or implementation patterns. The field moves weekly. Use `nia papers`, `nia github`, `nia search web`, `nia packages` aggressively.
5. **Move forward factually.** If you find a better approach that aligns with the locked architecture (this doc + your doc), take it. If you find the locked architecture is wrong, document it in your doc, propose alternative, ask user via AskUserQuestion before changing.
6. **Don't expect micromanagement.** You are Opus 4.7 — same model writing this. The doc gives direction, not line-by-line orders. Read code, write code, debug, ship. The acceptance criteria in your doc tell you when you're done.

---

## README Update Rule (READ THIS — owner explicitly demanded)

The repo `README.md` is **AI slop until proven otherwise.** It was written before any code shipped. Treat every claim, number, architecture detail, and bullet inside it as a hypothesis, not a fact.

**You MAY update README.md if and ONLY if ALL of these are true:**

1. You shipped a real, runnable artifact (code, checkpoint, report, benchmark) that the README references or should reference.
2. You have a finding, number, or fact that is **10000% verified** — reproducible from the codebase, with a test or report producing it, with a citation if it depends on external sources.
3. The current README claim about that thing is **wrong, missing, or outdated** in a way that misleads a reader.
4. Your update **replaces a hypothesis with a measured truth**, not a hypothesis with a different hypothesis.

**You MAY NOT update README.md to:**

- Reword for vibes, tone, or "polish"
- Add features you have not built yet
- Add metrics you have not measured (no Sharpe numbers, no accuracy numbers, no hit rates until they exist in a committed report file)
- Add decorative diagrams, headings, or sections
- Change the architecture description because the spec evolved on paper — the spec lives in `plan/`, not the README
- Insert your own marketing voice

**The README's journey from slop to real artifact = the journey from `plan/` → trained checkpoints → reported, reproducible numbers.** Each verified finding earns its line in the README. No findings, no edits.

**Format rule when you do update:** every numerical claim cites the file path or report that produced it (e.g. `Sharpe 1.23 — see reports/v1_<sha>_backtest.md`). Every architectural claim cites the file path that implements it (e.g. `RMSNorm — src/nanogld/model/rms_norm.py`). If you cannot point at a file, the claim is not ready for the README.

**One commit per real finding.** Commit message format: `docs(readme): replace <hypothesis> with verified <finding> — see <evidence path>`.

If unsure whether your finding clears the bar, AskUserQuestion before editing the README. The owner would rather see five short verified lines than fifty unverified ones.

---

## Execution Mode (Every Agent — Read Before Coding)

**This doc is V1-frozen as of 2026-05-08.** Agents do NOT replan, redesign, or regenerate scope without owner approval. The V1 spec sheet (`plan/V1-SPEC.md`) is the authoritative change list against V1. Read it before reading any individual sub-doc.

**These docs ARE the plan. Do not rewrite them.** Plan how to *execute* what is written. Do not replan, do not redesign, do not regenerate scope. If a doc claim is wrong, write the issue + proposed fix at the bottom of your assigned doc and AskUserQuestion before changing direction. Silent scope drift is a fireable offense.

**Research with Nia, not blind web search.** Before guessing on a library version, paper claim, or API behavior, use Nia (CLI at `/Users/samsiavoshian/.bun/bin/nia`):

- `nia papers <query>` — arXiv (RoPE on MPS, Schedule-Free, Friendly-SAM, Conformal Prediction)
- `nia github <query>` — live GitHub code search (no indexing needed)
- `nia search <query>` — indexed docs (run `nia deps` first to subscribe project deps to their docs)
- `nia packages <name>` — npm / PyPI / crates / Go registry
- `nia oracle <question>` — autonomous multi-step research for hard questions
- `nia tracer <query>` — autonomous code search across many repos

Spawn a subagent (general-purpose or Explore) to run Nia so raw output stays out of your main context. Run `nia auth` once if any command errors on auth.

**Use gstack skills during execution. Listed in priority order:**

| Skill | When |
|---|---|
| `/investigate` | bug, weird error, "why is this broken". Iron Law: no fix without root cause |
| `/review` | staff-engineer review on your diff before commit. Catches prod bugs CI does not |
| `/qa` | exercise the code path. Auto-generates regression tests for every fix |
| `/qa-only` | report bugs without auto-fix |
| `/cso` | OWASP + STRIDE security audit. Run before any code touching secrets, network, or live $ |
| `/benchmark` | baseline before perf-sensitive change, compare after |
| `/ship` | sync main, test, push, open PR |
| `/land-and-deploy` | merge → CI → deploy → verify production |
| `/canary` | post-deploy monitoring loop |
| `/document-release` | update README + docs to match what shipped |
| `/retro` | weekly retro on shipping streaks, test health |
| `/learn` | review/prune what gstack remembered across sessions |
| `/browse` | headless Chromium for any web check (NEVER `mcp__claude-in-chrome__*`) |
| `/setup-browser-cookies` | import real-browser cookies for authenticated browsing |
| `/pair-agent` | share your browser with another AI agent if you need a second opinion |

**You may NOT use these (planning skills — planning is done):**

`/office-hours`, `/plan-ceo-review`, `/plan-eng-review`, `/plan-design-review`, `/plan-devex-review`, `/autoplan`, `/design-consultation`, `/design-shotgun`, `/design-html`, `/design-review`, `/devex-review`.

These replan from scratch. The user said explicitly: *plan to execute the docs, do not replan*. Calling them wastes tokens and risks silent scope drift.

**Default execution loop per file you build:**

```
1. Read your doc + 00-OVERVIEW.md
2. Spawn Nia subagent for any unknown (lib version, paper claim, API)
3. Write code per doc spec
4. /review your diff
5. /qa the code path (generates regression test)
6. /cso if security-sensitive
7. /ship
```

**One escalation path.** If you have attempted a task 3 times without success, STOP and AskUserQuestion. If unsure on a security-sensitive change, STOP and AskUserQuestion. Bad work is worse than no work.

---

## What We're Building (one paragraph)

A from-scratch hybrid encoder (10 transformer blocks + 2 sLSTM blocks at the head, ~30M params) that predicts next-30min direction of GLD (gold ETF) using channel-independent patches (P=4, S=4, T=64 -> 16 patches per channel) over 681 features fused with semantic news embeddings from Qwen3-Embedding-4B (frozen, 4-bit MLX). FiLM regime conditioning every 2 transformer layers on a 12-dim regime vector (VIX terciles, RV terciles, FOMC week, year buckets, HMM P(high-vol)). Sparse Flamingo-gated cross-attention at layers {3, 7, 11} only. Dual head: Head A (3-class focal CE gamma=3) for calibration, Head B (tanh -> position weight, trained on cost-aware Sharpe loss) for sizing. Trained via SimMTM SSL pretrain + linear probe + LLRD fine-tune on Cautious-Schedule-Free-AdamW + Friendly-SAM. Sizing layer = Head B output through friction-adjusted Kelly + vol-target + conformal floor. Deployed live via Alpaca with $100 of real money on the Macbook M4 Pro. Artifact is a build-in-public X thread + blog post documenting the journey honestly.

## North Stars (read these — they decide every tradeoff)

1. **SOTA quality + WOW factor.** This is going on the internet to be read by ML researchers, quants, and tech Twitter. Everything must be impressive AND honest. Negative results with rigorous methodology beat cherry-picked positives.
2. **Use the best tool for the job.** No artificial restrictions. **HuggingFace Trainer, accelerate, peft, Unsloth, axolotl, datasets, lightning — all on the table.** If a library makes the project better, USE it. The quality of the artifact and the user's learning compound when professional tooling is in the stack.
3. **Owner learns deeply.** Owner is samsiavoshian. They want to understand transformers at Karpathy depth. So: **build the core model FROM SCRATCH in raw PyTorch** (this is the learning artifact). Use HF/Unsloth/peft for boilerplate (training loop infrastructure, data loading, mixed precision, LoRA, distributed). Read every library you import; understand what it does before depending on it.
4. **Real $100 skin-in-the-game.** Paper trade is dev. Live Alpaca with real money once paper validates.
5. **Local hardware only.** M4 remote host 16GB (trainer) + M4 Pro Macbook 16GB (live + dev). NO cloud GPU. NO Modal/Runpod/Lambda.
6. **Honest beats clever.** If a 2M-param TSMixer beats our 50M transformer on val Sharpe, ship TSMixer. If a Forecast-to-Fill replication (no ML at all) beats both, ship that. Story matters more than complexity.

### Library Policy (V1)

**Permitted now:**

- `transformers` library (HF) — for the frozen embedder (Qwen3-Embedding-4B), tokenizers, pretrained backbones if they help
- `accelerate` — for mixed precision, distributed (future-proof, single-device today)
- `datasets` — efficient data loading, streaming
- `peft` — LoRA / DoRA fine-tunes when fine-tuning a foundation model is justified
- `Unsloth` — 4× faster LoRA training if we ever fine-tune the embedder or a foundation backbone
- `TRL` — only if we add Stage 3 RL (gated, deferred)
- `lightning` (PyTorch Lightning) — if its callbacks/checkpointing help
- `axolotl` — if we need declarative fine-tune configs
- `vllm` / `mlx-lm` — for fast inference of frozen LLMs

**Still build from scratch** (the learning core):

- The nanoGLD transformer architecture itself (RMSNorm, SwiGLU, attention, RoPE, channel-group tokens, news fuser) — doc 05
- The vectorized backtest engine — doc 06
- The sizing math (Kelly-lite × vol-target × conformal) — doc 07
- The Live trading cycle (Alpaca SDK is fine; cycle orchestration is yours) — doc 08

**Why this split:** the model architecture is what people will read on the X thread. They want to see transformer math written cleanly. Training infrastructure is plumbing — use the best library, document the choice.

## Hard Constraints (do not violate without user override)

- $0 cloud spend. $100 trading capital. ~$10 misc (domain).
- Single asset: GLD only.
- Free data sources only (Alpaca historical bars, Alpaca News API, GDELT BigQuery free tier, FRED + ALFRED, yfinance for daily oil/macro, GPR Index monthly).
- 3-class cross-entropy or quantile loss only — **NEVER MSE on returns** (forecast-collapse rule, arXiv:2604.00064).
- ~~No fine-tuning of the embedder LLM~~ — **LIFTED in V1.** If LoRA fine-tune of Qwen3-Embedding-4B on financial news improves quality, do it.
- ~~No HuggingFace `Trainer` / Unsloth / TRL~~ — **LIFTED in V1.** Use HF infra freely. Build the MODEL from scratch (learning core), use libraries for everything else.
- PyTorch 2.11.0 pinned (has SDPA fix #174945 for MPS).
- FP32 weights everywhere (no autocast, no torch.compile, no quantization).

## Architecture Spec (V1, locked 2026-05-08; replaces V1 frozen 2026-05-04)

```
DATA PIPELINE (doc 02) — V1 expanded 2026-05-04
├── Alpaca historical 30min GLD × 5y (free Basic tier, IEX feed since 2016)
├── Alpaca historical 30min ETF basket × 5y: SPY/QQQ/IWM/GDX/SLV/XLF/XLE/XLK/XLU
├── Alpaca News API (Benzinga only — Reuters paywalled 2024)
├── GDELT 2.0 GKG_partitioned via BigQuery (themes — events table has no themes)
├── FRED + ALFRED — 34 series total (vintage-correct):
│   ├── Treasury curve (6): DGS3MO/6MO/2/5/10/30
│   ├── TIPS + breakevens (5): DFII5/10, T5YIE/T10YIE/T5YIFR
│   ├── FX + vol (2): DTWEXBGS, VIXCLS
│   ├── Oil (2): DCOILBRENTEU, DCOILWTICO
│   ├── Labor (5): UNRATE, PAYEMS, ICSA, CCSA, JTSJOL
│   ├── Inflation (4): CPIAUCSL, CPILFESL, PCEPI, PCEPILFE
│   ├── Growth (5): GDPC1, INDPRO, RSAFS, HOUST, UMCSENT
│   └── Money/Fed (5): M2SL, WALCL, RRPONTSYD, FEDFUNDS, SOFR
├── yfinance for Brent/WTI daily (NOT 30min — 60d cap is real)
├── GPR Index monthly (matteoiacoviello.com)
├── CFTC COT weekly disaggregated for COMEX gold (free CSV)
├── WGC quarterly central bank net purchases (free CSV)
├── Calendar event schedule (deterministic FOMC/CPI/NFP/GDP/JOLTS/PCE)
└── Joined parquet with point-in-time discipline + 15min news latency

FEATURE ENGINEERING (doc 04) — V1 expanded 2026-05-04, V1 redlines 2026-05-08
├── Existing: Price (12) + Risk/Vol (8) + Macro short (12) + Geo (10) = 42
├── NEW: Equity ETF features (~72) + Equity ratios incl. gold/silver, GDX/GLD (~9) = 81
├── NEW: Treasury curve features (~30) — levels + spreads + butterfly + real rates
├── NEW: Macro bundle (~60) — labor/inflation/growth/Fed YoY+MoM
├── NEW: COT positioning (~6) — managed money net long, OI z-score
├── NEW: WGC central bank (~3) — quarterly net purchases
├── NEW: Calendar events (~10) — event proximity windows + sin/cos cyclicals
├── + Multi-dim sentiment: polarity + intensity + uncertainty (per arXiv:2603.11408)
├── V1 NEW: Half-hour-5 intraday momentum r_h5 = log(close_t/close_{t-30min}) at RTH-5 (Gao 2014, Sharpe 5.43 single-feature)
├── V1 NEW: gld_h5_x_vol_high interaction with high-vol-tercile binary
├── V1 NEW: gld_spread_bps_t (5-min trailing avg of (ask-bid)/mid * 10000) for triple-barrier neutral and sizing
├── V1 NEW: VSN feature gate at input (Lim 2021 TFT, ~2M params, +0.92 Sharpe vs plain LSTM in Saly-Kaufmann)
├── V1 NEW: Series decomposition pre-VSN — trend = MA(x, kernel=24), seasonal = x - trend (xLSTMTime requirement)
├── V1 update: RevIN per individual channel (681 instances), REPLACES per-group RevIN — Huang & Yang ESWA 2026 dropped RMSE 50%
├── V1 update: Labels = triple-barrier method (López de Prado), REPLACES fixed 5-bps threshold
│   ├── Up barrier = +1.0 * ATR-14, down barrier = -1.0 * ATR-14, timeout = 1 bar
│   ├── Spread-adjusted neutral: even if barrier touched but |ret| < spread_t, label = 0
│   └── Three columns: label_triple_barrier {-1, 0, +1}, barrier_up, barrier_down (ATR-scaled)
├── pandas-ta-classic for indicators (NOT stale `ta`)
├── Garman-Klass volatility (NOT Parkinson — same OHLC, more efficient)
└── Z-score with rolling 1000-bar lookback + clip(-10, 10) — extended to 3276 for YoY-bearing macro features

NEWS EMBEDDING (doc 03) — V4 expanded news pipeline + bias-aware aggregator
├── Qwen3-Embedding-4B 4-bit MLX (Apache 2.0, ~18K tok/s on M4 mini)
│   ├── 45× faster than Llama-3.1-8B mean-pool (earlier draft)
│   ├── MTEB-en avg 74.6 (+10pts vs LLM2Vec)
│   └── MRL truncation: 2560-dim → 256-dim, 99% quality retention
├── 12+ news sources (was 3): Alpaca News + GDELT + Kitco + Investing.com + BullionVault + CNBC + FNSPID + central bank speeches (Fed/ECB/BIS) + government press (Treasury/CFTC) + Reddit Arctic Shift + WGC + Kaggle gold-labeled
├── FORBIDDEN: FT (robots.txt bans ML), DEFERRED: Reuters/FXStreet/Trading Economics (paid), SKIPPED: Metals Daily (syndication dup)
├── Source registry with 12 bias tiers — every article tagged at ingestion (industry_bullish, dealer_bullish, mainstream_neutral, central_bank_official, retail_social, etc.)
├── LAFTR adversarial debiasing head (arXiv:1802.06309) + gradient reversal (DANN, arXiv:1505.07818) + inverse-frequency reweighting
├── Per-article embedding (NOT per-source mean-pool) — preserves per-article meaning
├── Aggregator V4 = Per-source PMA pre-pool (Set Transformer) → bar-conditioned FiLM Q-Former (K=8 latents, was 16) → Flamingo tanh-gated cross-attn (gate init=0)
│   ├── Per-source PMA: 2 seeds/source, handles 50+ articles/bar (FinGPT dissemination, arXiv:2412.10823)
│   ├── Bar-conditioned FiLM: latent queries adapt to current price/vol regime (CMTF arXiv:2504.13522, FiCoTS arXiv:2512.00293)
│   └── Flamingo gate: init=0 ensures stable training when news is sparse
├── Anchor-cosine semantic features — V4 anchors are HAND-CRAFTED TEMPLATES (no event provenance, fixes leakage)
└── Per-article parquet storage (~500MB - 2GB depending on volume)

MODEL ARCHITECTURE (doc 05) — V1 hybrid encoder
├── ENCODER-only (no causal mask — bidirectional context for classification)
├── V1: channel-independent + patches (PatchTST style, REPLACES iTransformer-lite)
│   ├── Patch length P=4 bars, stride S=4, lookback T_bars=64 → 16 patches per channel
│   ├── 681 channels share one transformer backbone (channel-independent)
│   ├── Patch projection Linear(P, D), no bias; sinusoidal pos-emb on patches
│   └── Cross-channel mixing removed from main backbone (recovered via VSN gate at input)
├── V1 HYBRID BACKBONE: 10 transformer blocks + 2 sLSTM blocks at the head
│   ├── Layers 1-10: pre-norm transformer (RMSNorm + SwiGLU + partial RoPE + QK-Norm + no-bias)
│   ├── Layers 11-12: sLSTM blocks (xLSTMTime style, channel-independent)
│   ├── D=384, num_heads=6, head_dim=64
│   ├── Per-head gating + value residuals (IMU-1 recipe, arXiv:2602.02522)
│   ├── Partial RoPE (10% of head_dim — arXiv:2603.11611)
│   ├── Stochastic depth: linear schedule p_l = 0.2 * l / num_layers (Touvron 2021)
│   └── F.scaled_dot_product_attention(is_causal=False) with .contiguous() Q/K/V
├── V1 FiLM regime blocks at layers {2, 4, 6, 8, 10}
│   ├── 12-dim regime vector: VIX-tercile (3) + RV-tercile (3) + FOMC-week (1) + year-bucket (4) + HMM P(high-vol) (1)
│   └── gamma_l, beta_l = Linear_l(regime_vec); applied between attn and FFN
├── V1 SPARSE cross-attn for news at layers {3, 7, 11} only (mPLUG-Owl3 finding)
│   ├── CFA-style FiLM/orthogonal projector before Flamingo K/V (arXiv:2603.22372)
│   ├── AECF entropy-gated curriculum masking p ~ U(0.0, 0.9) (arXiv:2505.15417)
│   ├── is_news_present binary embedding concatenated to news token
│   └── NEWS_NOT_PRESENT learnable token kept as fallback
├── V1 DUAL HEAD (multi-task, REPLACES single classification head)
│   ├── Head A: Linear(D, 3, bias=False), focal loss gamma=3 (arXiv:2002.09437)
│   ├── Head B: Linear(D, 1, bias=False) → tanh → position weight in [-1, +1]
│   ├── L_sharpe_net = -mean(w * r_next - cost * |delta_w|) / std(...)
│   └── Combined: L = 0.5 * L_focal + 0.5 * L_sharpe_net + 0.05 * L_DANN + L_aecf
└── Sizing pipeline: Head B = primary position weight; Head A conformal interval as defensive shutoff

TRAINING (doc 05) — V1 update 2026-05-08
├── Stage 1: SSL pretrain — V1 SimMTM (arXiv:2302.00861, REPLACES plain MAE)
│   ├── K=3 masked views per sample, similarity-weighted blending of neighbor reps
│   ├── Mask ratio 0.40 (PatchTST default; V1's 0.20 was image-MAE inheritance)
│   ├── Patch length 4, decoder depth 2, 15-20 epochs (~25-30% of compute budget)
│   ├── + CLIP-style bars↔news contrastive head, tau=0.07, ±5 min news window
│   └── L_pretrain = L_simmtm + 0.5 * L_clip + 0.05 * L_DANN + L_aecf
├── Stage 2: Linear-probe (frozen encoder, focal CE head only)
├── Stage 3: LLRD fine-tune (layer-wise LR decay 0.85, Mixout p=0.7 anchored to SSL ckpt)
│   └── L = 0.5 * L_focal + 0.5 * L_sharpe_net + 0.05 * L_DANN + L_aecf
├── Optimizer: V1 Cautious-Schedule-Free-AdamW (5-line patch, arXiv:2411.16085, +1.47x sample efficiency)
│   ├── lr=1e-4 transferred from $5 muP sweep on 2-4M tiny model (Yang 2022)
│   └── betas=(0.9, 0.95), wd=0.1, warmup=300
├── Sharpness: Friendly-SAM ρ=0.05 kept (NOT vanilla SAM)
├── EMA: decay=0.999 on weights (deployed = EMA, not raw)
├── V1 NEW: FreeLB adversarial perturbation on news embeddings only (K=2, eps=0.5, +30% wall-clock)
├── V1 NEW: DANN gradient reversal on era-label = year-bucket {2016-2019, 2020-2022, 2023-2024, 2025+}, lambda 0→0.1
├── Walk-forward: 4 folds (3y train + 6mo val + 6mo test, 3mo step, 1-week embargo)
├── Loss: focal CE gamma=3 + Sharpe loss + DANN + AECF (NEVER MSE on returns)
├── V1 augmentation: SimPSI (arXiv:2312.05790) + Wave-Mask (arXiv:2408.10951), REPLACES naive jittering
│   └── Manifold Mixup alpha=0.2 at hidden states kept (NEVER raw input)
├── Cross-asset transfer (bonus): SPY → GLD via LLRD
└── PyTorch 2.11.0, FP32, num_workers=0 (macOS fork issues)

BACKTEST (doc 06) — V1 update 2026-05-08
├── V1 cost-stress HARD GATE: report Sharpe at {0.5x, 1.0x, 1.5x} cost levels
│   ├── Half-spread = 0.7 bps base; sqrt-impact gamma=0.02; round-trip ~2 bps base
│   └── Must show Sharpe > 0.5 at 1.5x cost (Wright F2F died at 1.5x)
├── V1 PER-BUCKET EVAL HARD: every metric reported separately for {news-present, news-absent, both}
├── V1 DSR > 1.0 HARD across all reported configs (no cherry-picking)
├── bars_per_year = 3276 (NYSE RTH only — NOT 17500)
├── Baselines: buy-hold, MA crossover, Donchian, DLinear, TSMixer, TimeMixer,
│              xLSTMTime (won 2026 finance benchmark per arXiv:2603.01820), VLSTM (Saly-Kaufmann 2026, 2.40 Sharpe),
│              XGBoost (committed config), Forecast-to-Fill replication, Gao 2014 half-hour-5 single-feature rule
├── Stationary block bootstrap CI on Sharpe (arch.bootstrap)
├── Regime stratification: vol terciles + FOMC weeks + news density
├── Sortino: target downside dev (NOT std of negative subset)
└── V1 hard rule: if V1 fails to beat simpler ensemble (Gao 2014 + XGBoost) by ≥ 0.2 Sharpe OOS net of costs,
    SHIP THE SIMPLER ENSEMBLE (TLOB lesson + Gao 2014 GLD-specific bar)

SIZING + EXITS (doc 07 V1 update 2026-05-08; locks F2F machinery on top of dual-head model)
├── V1 primary signal = Head B output (tanh → position weight in [-1, +1])
├── Stage 1 (basic): Head B output × vol target
├── Stage 2 (full): Head B + friction-adjusted Kelly + ATR exits + vol target + 30-day timeout + conformal floor
│   ├── kelly_size = lambda * edge / variance, lambda=0.4 (half-Kelly-ish)
│   ├── edge from Head B output; variance from Laplace last-layer posterior + rolling 60-bar realized var
│   ├── effective_size = kelly_size * max(0, 1 - cost / |edge|) (zero out if cost > edge)
│   ├── target_vol = 15% annualized (NYSE RTH bars_per_year=3276)
│   ├── sqrt-impact cost: cost_t = gamma * sqrt(|delta_w|) + k_bps * |delta_w|, gamma=0.02, k=0.7bps
│   └── conformal floor: if AgACI/RAPS lower bound on top-class prob < 0.40, force position to 0
│   GATE: Stage 2 must beat Stage 1 by ≥ 0.2 Sharpe OOS to ship the full pipeline (else ship Stage 1)
├── Per-trade stop-loss (NEW, doc 07)
│   ├── Hard ATR stop: 2.0 × ATR(14)_at_entry (frozen at entry)
│   ├── Trailing ATR stop: 1.5 × ATR(14)_live, ratchet only
│   ├── Time stop: 390 bars (30 RTH days × 13 30-min bars)
│   ├── Re-entry gate: cooldown 1 bar + max_prob ≥ 0.55 OR argmax flipped
│   ├── Session-flat at 15:55 ET, re-eligible 09:35 ET
│   └── News blackout ±15 min around FOMC / CPI / NFP / GDP / JOLTS / PCE
├── Profit-taking (NEW, doc 07)
│   ├── NO fixed take-profit (F2F + Baur-Dimpfl + 5bp cost gate)
│   ├── Model re-decision IS the TP (continuous re-rebalance)
│   └── OPTIONAL signal-decay exit (gated: ship only if val A/B ≥30bp lift)
├── Live execution: client-side stop polling (Alpaca rejects bracket on fractional, error 42210000)
├── Drawdown circuit-breaker (portfolio-level): -5% halve / -10% quarter / -15% halt
└── Stage 3 (RL) deferred — gated, only built if Stage 2 leaves Sharpe on table

LIVE TRADING (doc 08)
├── Macbook M4 Pro runs cron via launchd every 30min
├── StartCalendarInterval (NOT StartInterval=1800) — fires only RTH M-F
├── pmset -c sleep 0 disablesleep 1 (#1 prod risk if skipped)
├── Alpaca SDK: alpaca-py >=0.43, paper for dev, live for prod
├── Pre-cycle check for open orders (avoid double-submit on partial fills)
├── Two-key separation: paper keys in dev, live keys ONLY in launchd env
├── 1Password CLI for live keys (NOT chmod 600 in prod)
├── Idempotent reconciliation via get_all_positions
└── Drift detection (entropy z-score + KL on argmax distribution)

INFRA + SECURITY (doc 01)
├── PyTorch 2.11.0 pinned, uv.lock committed
├── gitleaks v8.24.2 pre-commit hook (BEFORE first commit)
├── ruff v0.11+ (replaces black entirely)
├── pre-commit-hooks v5.0.0
├── astral-sh/setup-uv@v8.1.0 in CI
├── GCP ADC for dev, WIF for CI (NEVER service-account JSON keys in repo)
├── Snapshot hashing: hash_pandas_object + columns/dtypes (10-100× faster than to_csv)
├── Log rotation: in-process RotatingFileHandler (NOT newsyslog — root trap)
├── HuggingFace Hub for checkpoint backup (LoRA adapters ~30MB)
└── Quarterly GitHub Actions smoke test (catches API drift)
```

## Doc Index


| #   | Doc                          | Owner agent role           | Implementation effort            |
| --- | ---------------------------- | -------------------------- | -------------------------------- |
| 00  | OVERVIEW (this)              | n/a (read-first reference) | n/a                              |
| 01  | INFRA-AND-SECURITY           | DevOps                     | 0.5 day (START HERE)             |
| 02  | DATA-PIPELINE                | Data engineer              | 4-5 days                         |
| 03  | NEWS-EMBEDDING               | ML engineer                | 1.5 day setup + ~120min precompute |
| 04  | FEATURE-ENGINEERING          | Feature engineer           | 1.5 days                         |
| 05  | MODEL-TRAINING-CALIBRATION   | ML systems engineer        | 3 days (model + train + calib in one) |
| 06  | BACKTEST                     | Quant engineer             | 1 day                            |
| 07  | SIZING-AND-EXITS             | Quant risk engineer        | 2 days (sizing + SL + profit-take in one) |
| 08  | LIVE-TRADING                 | Production engineer        | 1.5 days                         |
| --  | STATUS                       | n/a (anyone can update)    | n/a                              |

**V5 Merge (2026-05-04):** 11 docs → 8 docs. Old 05+06+07 (model+training+calibration) merged into new 05. Old 09+10 (sizing+exits) merged into new 07. Each remaining doc depends only on the immediately-preceding one — pure linear chain, single agent per doc, no blocking. Old `08-RL-STAGE3.md` was deleted May 1 (RL deferred to V2). Old `11-X-THREAD-AND-BLOG.md` deleted (owner writes himself).

## Implementation Order — Sequential, One Agent Per Doc

Each agent depends ONLY on the previous agent's output. No overlap. No multi-doc blocking.

```
Agent 01 → doc 01 INFRA-AND-SECURITY               (0.5 day)
Agent 02 → doc 02 DATA-PIPELINE                    (4-5 days)
Agent 03 → doc 03 NEWS-EMBEDDING                   (1.5 days + ~120min precompute)
Agent 04 → doc 04 FEATURE-ENGINEERING              (1.5 days)
Agent 05 → doc 05 MODEL-TRAINING-CALIBRATION       (3 days, all in one file)
Agent 06 → doc 06 BACKTEST                         (1 day)
Agent 07 → doc 07 SIZING-AND-EXITS                 (2 days, all in one file)
Agent 08 → doc 08 LIVE-TRADING                     (1.5 days)

Total: ~14-16 days end-to-end (sequential, 8 self-contained agents).
Per-agent details in STATUS.md.
```

## Verification History

- **Round 1 (data pipeline):** 4 Nia agents verified Alpaca / GDELT / yfinance / FRED claims against live APIs. Found ~10 critical bugs (TimeFrame.Minute_30 doesn't exist, GDELT theme codes wrong, get_series_as_of_date returns DataFrame).
- **Round 2 (architecture v3):** 7 Nia agents on transformer SOTA / time-series / multimodal / attention / small-data / empirical SOTA / MPS. Found 12 critical bugs (bars_per_year=17500 → 3276, decoder→encoder pivot, head_dim=32 too small, etc).
- **Round 3 (2026 SOTA):** 6 Nia agents on May 2026 releases (Llama 4, Gemma 4, Qwen 3.5, embedding leaderboard, TS foundation models, architecture innovations, finance papers, training optimization). Major pivot: Llama-3.1-8B → Qwen3-Embedding-4B (45× faster). Schedule-Free AdamW replaces cosine + warmup. Forecast-collapse hard rule (NEVER MSE).
- **Round 4 (V1 dataset expansion + leakage audit, 2026-05-04):** 5 Nia agents verified all sources for the V1 expansion AND audited every existing source for leakage. Found **17 high-severity issues**: bar timestamp = START not END, Alpaca News field = `created_at` not `published_at`, FEDFUNDS is monthly (need DFF), 6 GDELT theme codes refuted, GDELT buffer 30min not 15min, WGC URL was wrong (correct: gold.org/download/8052), WGC is monthly not quarterly, AI-GPR not real-time (30-day lag), GPR no vintage archive, pandas-ta look-ahead bugs (KAMA/Ichimoku/KST/DPO/TRIX/Vortex forbidden), CFTC 2025 shutdown gap, multi-symbol pagination interleaves, `adjustment="all"` is retroactive, WALCL Thursday 4:30pm release-time gating, ICSA Thursday 8:30am release-time gating, anchor-cosine source must precede train period, no `minutes_until_event` features. All fixes encoded as 17 hard rules + 28 mandatory tests in `tests/test_no_leakage.py`.
- **Round 5 (news pipeline expansion + ML aggregation refactor, 2026-05-04):** 5 Nia agents verified the user's 10-source list (Kitco / Metals Daily / BullionVault / Investing.com / CNBC / Reuters / FT / Trading Economics / FXStreet / WGC) against live URLs + ToS + 10y archive depth, audited bias profiles, surveyed free 10y datasets, and researched multi-document aggregation SOTA from 2024-2026. **Key findings:** FT robots.txt explicitly bans ML training (legal blocker — skip). Reuters paywall + Reuters Connect enterprise-only (defer paid). FNSPID dataset (15.7M articles, 1999-2023, CC BY 4.0, on HF) is the biggest free win for filling pre-2021 gap. Kitco/BullionVault/Investing.com are all free-scrape with 10y depth. Central bank speeches (Fed/ECB/BIS) + government press (Treasury/CFTC) are public-domain (US 17 USC §105). Reddit Arctic Shift dumps free through 2026-04. Aggregation SOTA: per-source PMA + bar-conditioned FiLM Q-Former (K=8) + Flamingo gate is 2025-26 sweet spot (CMTF arXiv:2504.13522, FiCoTS arXiv:2512.00293). Bias debiasing recipe: LAFTR adversarial head (arXiv:1802.06309) + gradient reversal (arXiv:1505.07818) + inverse-frequency reweighting. Plan: 12+ news sources + 12 bias tiers + LAFTR head + new aggregator. doc 03 effort 0.5d → 1.5d.
- **Round 6 (V1 architecture synthesis, 2026-05-08):** 9-agent Nia synthesis on backbone choice, multimodal fusion, training/SSL recipe, calibration stack, sizing/F2F machinery, eval discipline, and 2025-26 SOTA delta. Major redlines: V1 pure transformer → V1 hybrid (10 transformer + 2 sLSTM head) per Saly-Kaufmann/Wood/Zohren 2026 (VLSTM 2.40 Sharpe vs iTransformer 0.38) + xLSTMTime small-data recipe; iTransformer-lite group tokens → channel-independent + patches (PatchTST P=4, S=4); single classification head → dual head (focal CE gamma=3 + Sharpe loss); MAE → SimMTM + CLIP bars↔news; CE → focal loss (clean T-scaling/APS); MC-dropout → Laplace last-layer; jittering → SimPSI/Wave-Mask; fixed-bp labels → triple-barrier; 15% news dropout → variable U(0.1, 0.9); add cost-stress + DSR + per-bucket as HARD gates; target reframe 2.88 → 1.0-1.5 OOS Sharpe net of 2bp. Full change list at `plan/V1-SPEC.md`.

## Empirical Bar (V1, what success actually looks like)

V1 chased "beat Wright F2F 2.88 Sharpe." V1 reframes that target. F2F 2.88 is daily gold futures EOD-to-EOD with ~30-day holding. Not comparable to 30-min intraday GLD direction. Apples-to-apples GLD intraday published record is Gao-Han-Li-Zhou 2014 (5.43 Sharpe single-feature half-hour-5 timing). Daily futures DL frontier is Saly-Kaufmann/Wood/Zohren 2026 (arXiv:2603.01820): VLSTM 2.40 Sharpe.

**V1 honest target: 1.0 to 1.5 OOS Sharpe net of 2 bps round-trip costs over 4-fold walk-forward.**

| Tier                  | Threshold                                                                                              | Status                   |
| --------------------- | ------------------------------------------------------------------------------------------------------ | ------------------------ |
| Minimum viable        | 38% direction accuracy + Sharpe > 0 net of 2bp + beat XGBoost on same 681 features                     | Mandatory to ship        |
| V1 honest           | OOS Sharpe 1.0-1.5 net of 2bp, DSR > 1.0, beat baselines by ≥ 0.2 Sharpe on ≥ 3 of 4 folds, per-bucket positive | Mandatory for X thread   |
| Stretch               | OOS Sharpe > 2.0 net of 2bp + survives 1.5x cost stress                                                | Publishable contribution |

**Bars to beat (kept honest, V1):** buy-and-hold GLD net of costs; 50/200 EMA crossover; Donchian breakout; DLinear, TSMixer, TimeMixer; xLSTMTime + VLSTM (these are STRONG; if either ties or wins, ship the simpler one per hard rule 3); XGBoost on same 681 features; F2F replication on daily GLD bars (different problem, separate scoreboard); Gao 2014 half-hour-5 single-feature rule (the GLD-specific bar).

**V1 ship rule:** if V1 fails to beat the simpler ensemble (Gao 2014 + XGBoost) by ≥ 0.2 Sharpe OOS net of costs, the ship recommendation is the simpler ensemble, full stop.

## Hard Project Rules (apply across all docs)

1. **NEVER use MSE on returns.** Use 3-class CE or quantile loss. (forecast-collapse, arXiv:2604.00064)
2. **STAY FROM-SCRATCH.** No TS foundation model fine-tune. (arXiv:2511.18578 — TSFMs underperform)
3. **MANDATORY MLP/xLSTM baselines.** Ship simpler model if it ties. (TLOB lesson)
4. **Apply peer-benchmark discount.** Backtests capture launch-period regime, not skill. (arXiv:2604.18821)
5. **bars_per_year = 3276** (NYSE RTH only). NEVER 17500.
6. **Point-in-time discipline.** Every feature uses `.shift(1).rolling(...)`. **News buffer = 30min for GDELT, 60s for Alpaca News.** **Bar visibility = `bar.timestamp + bar_duration` (Alpaca bar `t` = bar START).**
7. **gitleaks before first commit.** Verify by trying to commit a fake key — must fail.
8. **PyTorch 2.11.0 pinned.** Has SDPA fix #174945 for MPS.
9. **FP32 weights everywhere.** No autocast, no torch.compile, no quantization at our scale.
10. **`.contiguous()` Q/K/V before SDPA.** PyTorch #181133.
11. **Every feature row carries `t_visible: pd.Timestamp` column.** Every join uses `t_visible <= prediction_time` with strict `<`. CI gate: `test_release_ts_lte_t_visible_all_rows`.
12. **Use ALFRED `get_series_all_releases` for ALL FRED series.** Never current snapshot. CPI/PCE annual revisions silently rewrite 5y of history.
13. **`pandas-ta` KAMA / Ichimoku / KST / DPO / TRIX / Vortex are FORBIDDEN** (look-ahead bugs, bukosabino/ta#181). Every indicator passes growing-window stability test.
14. **Calendar features = binary windows ONLY.** No `minutes_until_event` raw features (calendar memorization risk).
15. **Anchor-cosine anchors must be hand-crafted templates OR pre-train-period samples.** Otherwise anchor set encodes future events.
16. **News field is `created_at` (Alpaca News).** `published_at` does NOT exist. Never join on `updated_at`.
17. **Use `DFF` for daily Fed Funds.** `FEDFUNDS` is monthly — using it as daily silently leaks values that don't exist until next month.

### V1 invariants (18-25, added 2026-05-08)

18. **Per-bucket eval is non-negotiable.** Every metric reported separately for {news-present, news-absent, both}. Without this we fly blind on the 51% no-news bars.
19. **Cost-stress at {0.5x, 1.0x, 1.5x} on every reported Sharpe.** Hard gate: must show Sharpe > 0.5 at 1.5x cost. Wright F2F died at 1.5x; ours likely worse.
20. **DSR > 1.0 hard gate** across all reported configs (Bailey & López de Prado). Multi-config selection penalty enforced. No cherry-picking.
21. **SimPSI / Wave-Mask aug only.** Naive jittering FORBIDDEN (Fons 2020 arXiv:2010.15111: net-negative on Sharpe).
22. **Focal loss gamma=3 (NOT vanilla CE).** Required for clean T-scaling/APS interaction (Mukhoti 2020 + Xi 2024 — T-scaling on CE logits HARMS APS adaptive coverage).
23. **Triple-barrier labels with spread-adjusted neutral threshold.** ATR-14 up/down barriers, 1-bar timeout, label = 0 if `|return| < spread_t` even when barrier touched.
24. **Variable per-batch modality dropout `p ~ U(0.1, 0.9)`.** NOT 15% constant. Empirical news-absence rate is 51%; 15% guarantees performance cliff at inference.
25. **Decision-aware head (multi-task with Sharpe loss) is V1 ship gate.** End-to-end profit metric, not just classification accuracy. Stage 2 sizer must beat Stage 1 by ≥ 0.2 Sharpe OOS to ship the full pipeline.

## V1 Promotion Gates (replaces V1 gates)

Fail any gate, the negative result gets reported. Cherry-picking is fireable.

```
Gate 1   Walk-forward Sharpe > 1.0 net of 1x cost
Gate 2   Sharpe > 0.5 net of 1.5x cost (NEW hard)
Gate 3   Beats best baseline by >= 0.2 Sharpe on >= 3 of 4 folds
Gate 4   Conformal coverage within ±2% of nominal on val + per-bucket
Gate 5   Stage 2 sizer (decision-aware head) beats Stage 1 fallback by >= 0.2 Sharpe OOS
Gate 6   Drawdown circuit breaker tested on >= 2 historical regimes
Gate 7   Deflated Sharpe Ratio > 1.0 (NEW hard)
Gate 8   Per-bucket Sharpe (news-present, news-absent) both positive (NEW hard)
```

## Things explicitly NOT in V1

These were considered and rejected. Revisit only after V1 baseline lands.

- MoE / Switch / Mixtral / OLMoE / Soft MoE. 75K samples too small for sparse routing.
- KAN / KASPER. KASPER's Sharpe 12.02 is fraud-tier; Renaissance Medallion is ~2.5 net.
- Mamba / S-Mamba / S5 / Hyena. 0.64 Sharpe on financial benchmark (Saly-Kaufmann 2026).
- Jamba. 50B+ scale required.
- Test-Time Training (TTT). V2 candidate; inference cost too high for one-shot.
- Online learning / EWC. Walk-forward CV already simulates fresh data.
- Muon optimizer. Speedup inversely proportional to scale; 30M too small.
- Sophia. Stanford benchmark (arXiv:2509.02046) shows 2x claim collapses under fair tuning.
- Deep ensembles. 5x compute kills budget; snapshot+EMA matches 80-90% of DE accuracy at 1x.
- LLaVA-Mini one-token compression. Wrong scale.
- Cross-Modal Proxy Tokens (CMPT). Deferred to V2 (Decision 4A).
- Bars-only distillation. Deferred to V2.
- Set Transformer over present articles. V2 alternative to anchor pooling.
- Full xLSTMTime swap (encoder-wide). V1 keeps hybrid (10 transformer + 2 sLSTM).
- End-to-end Sharpe-loss-only head (drops classification). V1 keeps multi-task.

## Key Citations Driving the Design

Architecture:

- Llama 3 Herd (arXiv:2407.21783) — RMSNorm + SwiGLU + RoPE convention
- Qwen 3 (arXiv:2505.09388) — QK-Norm
- iTransformer (arXiv:2310.06625) — channel-as-token tokenization
- IMU-1 (arXiv:2602.02522, Jan 2026) — per-head gating + value residuals + 56× sample efficiency
- TDA (arXiv:2601.12145, Jan 2026) — sink-free attention for low-SNR
- SyPE (arXiv:2602.08983, Feb 2026) — symplectic position embedding for financial cycles
- Forecast collapse (arXiv:2604.00064, March 2026) — NEVER MSE on returns

Time-series:

- PatchTST (arXiv:2211.14730), DLinear (arXiv:2205.13504) — encoder-only baselines
- xLSTM finance benchmark (arXiv:2603.01820, March 2026) — xLSTM wins on 2010-2025 daily futures
- Tan et al. NeurIPS 2024 (arXiv:2406.16964) — LLMs don't help time-series
- Re(Visiting) TSFMs in Finance (arXiv:2511.18578, Nov 2025) — TSFMs underperform

News fusion:

- Flamingo (arXiv:2204.14198) — gated cross-attention init=0
- BLIP-2 / Q-Former (arXiv:2301.12597)
- Multi-dim sentiment WTI (arXiv:2603.11408, March 2026) — intensity + uncertainty > polarity
- Embedding: Qwen3-Embedding-4B (qwenlm.github.io/blog/qwen3-embedding/, June 2025)

Training:

- Schedule-Free AdamW (Defazio arXiv:2405.15682, ICLR 2025 outstanding)
- F-SAM (arXiv:2403.12350) + revisited (arXiv:2603.10048, March 2026)
- MTS-JEPA (arXiv:2602.04643, Feb 2026) — Phase 2 SSL alternative
- SAMformer (arXiv:2402.10198, ICML 2024) — 14% MSE improvement
- Conformal prediction for trading (Wright arXiv:2601.07852, Jan 2026; Kato arXiv:2410.16333)

Empirical bar:

- Forecast-to-Fill (arXiv:2511.08571, Nov 2025) — Sharpe 2.88 on gold, unreplicated

## How to Spawn Nia Research Agents (encouraged)

If your doc says X but you suspect Y is better, or if you have ANY question, spawn a Nia subagent:

```python
# Pseudocode for Agent tool
Agent(
    description="Verify [specific claim]",
    prompt="""Research [topic]. Use nia search web, nia papers, nia github,
    nia packages. Today is 2026-05-01. Find post-2025 papers/releases.

    Specific questions:
    1. ...
    2. ...

    Output: VERIFIED/REFUTED/NEEDS_REVISION per claim with citations.
    Mark unverified claims clearly.""",
)
```

If you spawn an agent and it disagrees with the doc, document the disagreement IN your doc, propose alternative, ask user via AskUserQuestion before changing locked architecture. Don't ship contradicting code silently.

## Decision Log (cross-cutting)


| Date       | Decision                                                | Rationale                                                           |
| ---------- | ------------------------------------------------------- | ------------------------------------------------------------------- |
| 2026-04-25 | initial admissions-framed → V1 (Karpathy + viral X + $100) | User pivot                                                          |
| 2026-04-30 | Decoder-only → Encoder-only                             | Time-series agent: bidirectional better for next-bar classification |
| 2026-04-30 | Per-bar tokens → Channel-group tokens                   | iTransformer-lite, empirically dominant                             |
| 2026-04-30 | Add SAM ρ=0.05                                          | SAMformer ICML 2024: 14% MSE improvement                            |
| 2026-04-30 | Mandatory MLP baselines                                 | TLOB: "MLP matches transformer"                                     |
| 2026-05-01 | Llama-3.1-8B mean-pool → Qwen3-Embedding-4B             | 45× faster, Apache 2.0, +10pts MTEB                                 |
| 2026-05-01 | AdamW + cosine + warmup → Schedule-Free AdamW           | Won AlgoPerf 2024                                                   |
| 2026-05-01 | Vanilla SAM → Friendly-SAM                              | Filters gradient noise, drop-in upgrade                             |
| 2026-05-01 | Hard rule: NEVER MSE on returns                         | Forecast collapse, arXiv:2604.00064                                 |
| 2026-05-01 | Add conformal prediction sizing                         | 30% lower decision loss, Wright 2026                                |
| 2026-05-01 | Add xLSTMTime baseline                                  | Won 2026 finance benchmark                                          |
| 2026-05-01 | Delete doc 07 (RL) + 11 (content)                      | Speculative + non-implementation                                    |
| 2026-05-04 | Dataset expansion — 9 equity ETFs (SPY/QQQ/IWM/GDX/SLV/XLF/XLE/XLK/XLU), full Treasury curve + TIPS + breakevens (11 FRED), full macro bundle (19 FRED), CFTC COT weekly, WGC quarterly, calendar events | Owner directive — capture more market drivers (real rates, risk-on/off, sector rotation, gold-silver ratio, positioning extremes). Per-bar feature dim grows ~804 → ~1000. doc 02 effort 2-3d → 4-5d. No model arch change. |
| 2026-05-04 | Wrote `07-SIZING-AND-EXITS.md` (4 parallel research agents — sizing / SL / TP / Forecast-to-Fill + Alpaca constraints). Supersedes doc 07 sizing math and doc 08 line 198 stop-loss. | Owner flagged 3 missing pieces. Findings: (1) Sizing formula in doc 07 was magnitude-only (`max_prob-0.33`), discards signed info, fragile `confidence_scale=3` guess, fabricated "30% lower decision loss" Wright 2026 citation. Replaced with signed score `s = P_up - P_down`, quarter-Kelly default, `vol_mult` capped at 3.0, EWMA+20d-floor σ_t, continuous conformal shrinkage. (2) Per-trade SL absent; literature (Kaminski-Lo, Han-Zhou-Zhu, F2F arXiv:2511.08571) and 5%-of-bars-with-news-tail-risk argue for wide ATR stop. Match F2F: 2.0×ATR14 hard + 1.5×ATR14 trail + 390-bar time-stop + re-entry gate + 15:55 ET session-flat + news blackout. (3) NO fixed take-profit (F2F + Baur-Dimpfl + 5bp cost gate); optional signal-decay exit gated on val A/B ≥30bp lift. (4) Live: Alpaca rejects bracket orders for fractional positions (error 42210000); at $100 + GLD ~$200 every position is fractional, so stops enforced client-side via `cycle.py` polling. |
| 2026-05-04 | Verification Round 4 — 17 leakage findings encoded as hard rules + 28 mandatory tests | 5 Nia agents audited every source. Bar timestamp=START leakage, FEDFUNDS→DFF, 6 GDELT codes refuted, GDELT 30min buffer, WGC URL fix (monthly not quarterly), AI-GPR not real-time, anchor leakage rule, pandas-ta forbidden indicators, CFTC release-time gate, calendar-binary-only, etc. CI gate via `test_release_ts_lte_t_visible_all_rows`. |
| 2026-05-04 | News pipeline expansion — 3 sources → 12+ sources + bias-aware LAFTR debiasing + V4 aggregator (per-source PMA + bar-conditioned FiLM Q-Former K=8 + Flamingo gate) | 5 Nia agents verified user's 10-source list + free-news datasets + multi-doc aggregation SOTA. Add Kitco/Investing.com/BullionVault/CNBC/FNSPID/central bank speeches/government press/Reddit/Kaggle. Forbid FT (robots.txt bans ML — legal). Defer Reuters/FXStreet/TE (paid). Skip Metals Daily (syndication dup). Source registry with 12 bias tiers + LAFTR adversarial head fights per-source prior. Aggregator upgrade (CMTF + FiCoTS 2025-26 papers): K=16→8, add bar-conditioned FiLM, add per-source PMA pre-pool. Per-article embedding (was per-source mean-pool). doc 03 effort 0.5d → 1.5d. |
| 2026-05-08 | V1 → V1 freeze. 9-agent Nia synthesis. Decisions 1B (hybrid xLSTM head: 10 transformer + 2 sLSTM), 2B (channel-independent + patches P=4 S=4), 3B (multi-task: focal CE gamma=3 + Sharpe loss head), 4A (V1 missing-modality fixes only, defer CMPT to V2). Target reframe: 2.88 → 1.0-1.5 OOS Sharpe net of 2bp. | Owner-approved spec sheet at `plan/V1-SPEC.md`. Saly-Kaufmann/Wood/Zohren 2026 (arXiv:2603.01820): VLSTM 2.40 > iTransformer 0.38; xLSTM 1.79 best transaction-cost robustness. xLSTMTime (arXiv:2407.10240): sLSTM small-data winner. PatchTST channel-independent + LPatchTST 2.31 vs iTransformer 0.38. Hwang & Zohren 2025 (arXiv:2510.03129): MSE-optimal forecasts ≠ optimal allocations. Mukhoti 2020 focal loss + Xi 2024 (T-scaling on CE HARMS APS). mPLUG-Owl3 sparse cross-attn at {3, 7, 11}. CFA filter (arXiv:2603.22372) + AECF (arXiv:2505.15417). SimMTM (arXiv:2302.00861) + CLIP bars↔news pretrain. Cautious-SF-AdamW (arXiv:2411.16085) + muP transfer-tune (arXiv:2203.03466) + FreeLB on news only + DANN era-label. SimPSI (arXiv:2312.05790) + Wave-Mask (arXiv:2408.10951) REPLACE jittering (Fons 2020 net-negative). Triple-barrier labels + spread-adjusted neutral. RAPS + AgACI (arXiv:2009.14193 + arXiv:2202.07282) + Laplace last-layer REPLACES MC-dropout. F2F sizing locks Head B → friction-Kelly → vol-target → ATR exits → conformal floor. New invariants 18-25; new 8 promotion gates with cost-stress + DSR + per-bucket as HARD. |


## File Layout (when implementation begins)

```
ml-trading/  (gh repo create nanogld --public)
├── pyproject.toml          (uv-managed, locked deps per doc 01)
├── uv.lock
├── .gitignore              (per doc 01 template)
├── .pre-commit-config.yaml (gitleaks, ruff, hooks per doc 01)
├── .github/workflows/      (CI smoke test per doc 01)
├── README.md               (project overview + reproduce instructions)
├── data/
│   ├── raw/                (.gitignored, per doc 02)
│   ├── snapshots/          (immutable hashed parquets)
│   ├── embeddings/         (memmap fp16 .npy, per doc 03)
│   └── anchors/            (.npz, per doc 04)
├── src/nanogld/
│   ├── data/               (doc 02: pipeline + joiner + golden fixture test)
│   ├── features/           (doc 04: feature engineering + RevIN)
│   ├── embed/              (doc 03: Qwen3 embedder + anchor cosines)
│   ├── model/              (doc 05: tiny_trader_v4.py + baselines)
│   ├── training/           (doc 05: walk-forward + Schedule-Free + F-SAM)
│   ├── backtest/           (doc 06: vectorized engine + bootstrap)
│   ├── sizing/             (doc 07: stage2 + conformal)
│   ├── live/               (doc 08: Alpaca cron + drift detection)
│   └── utils/              (shared: snapshot hashing, point-in-time helpers)
├── tests/
│   ├── test_pit.py         (golden fixture for joiner — doc 02)
│   ├── test_features.py    (no-leakage tests — doc 04)
│   ├── test_model.py       (forward pass shapes — doc 05)
│   ├── test_backtest.py    (cost model arithmetic — doc 06)
│   └── test_sizing.py      (Kelly-lite formula — doc 07)
├── notebooks/
│   ├── 01_explore_data.ipynb
│   ├── 02_baseline_xgboost.ipynb
│   └── 03_results_writeup.ipynb
└── checkpoints/            (.gitignored)
```

Each doc tells the agent EXACTLY which files to create. Other agents must not write to those.

## Cross-Doc Coordination

If two docs depend on each other:

- Agent X publishes a STABLE INTERFACE (function signature, file format, schema) in their doc
- Agent Y reads the interface from doc X, codes against it
- If interface changes mid-implementation, agent who changes notifies via STATUS.md update + AskUserQuestion to user

Example: doc 04 (features) consumes parquet from doc 02. Schema documented in doc 02 — doc 04 codes against it. If doc 02 changes schema, doc 04 must be notified.

## When Implementation Diverges from Doc

If during implementation you discover:

- A library version is broken on MPS that the doc said works
- A paper's claim doesn't reproduce
- An API has changed since Nia verification
- A simpler approach achieves the same goal

DOCUMENT IT IN YOUR DOC. Add a section at the top: `## DEVIATION FROM SPEC: [date] - [issue]`. Don't silently work around — future agents need to see what changed.

If your deviation breaks another doc's interface, ask user via AskUserQuestion before shipping.

## Glossary (for new agents)

- **nanoGLD:** the project name + the V1 hybrid encoder model (10 transformer + 2 sLSTM blocks)
- **GLD:** SPDR Gold Shares ETF (what Alpaca trades — not GC=F futures)
- **30min bar:** 30-minute OHLCV candle on NYSE RTH (09:30-16:00 ET, 13 bars/day)
- **Walk-forward CV:** time-ordered cross-validation with embargo, never random splits
- **Point-in-time correct:** features at time T use only data with timestamp < T (no future leakage)
- **MRL:** Matryoshka Representation Learning — embedding dim truncatable
- **Channel-group token:** iTransformer-lite — each token represents one feature group across all 64 bars (not per-bar)
- **DSR:** Deflated Sharpe Ratio (Bailey-Lopez de Prado) — corrects for selection bias
- **Forecast collapse:** Transformer overfit pattern on weak-signal data (arXiv:2604.00064)
- **F-SAM:** Friendly Sharpness-Aware Minimization — filters gradient noise vs vanilla SAM
- **TDA:** Threshold Differential Attention — sink-free attention for low-SNR
- **SyPE:** Symplectic Position Embedding — replaces RoPE for non-stationary cycles
- **Forecast-to-Fill:** arXiv:2511.08571 — clean published gold result, Sharpe 2.88, unreplicated 2026

## Last Note

This plan has been verified across **6 verification rounds (V1: 5 rounds + 19 Nia agents + 30+ findings + 17 leakage hard rules; V1: 9-agent architecture synthesis 2026-05-08)**. Every claim has a citation. Every architectural decision has been challenged. The plan is exhaustive enough to implement without ambiguity.

But the field moves weekly. **Always run a fresh** `nia search` **on your topic before starting your implementation.** A paper from last week could change your decision. The plan is a snapshot, not a contract. **V1 is frozen as of 2026-05-08; agents do NOT replan further without owner approval.** If you find something the V1 spec sheet missed, raise it via AskUserQuestion, do not silently change.

**Now go read your assigned doc. Spawn Nia agents freely. Build the thing.**
