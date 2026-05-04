# 00 — nanoGLD Project Overview

**Read this doc first. Every implementation agent reads this before touching code.**

**Project:** nanoGLD — Karpathy-mode LLM-augmented gold trader on local hardware
**Owner:** samsiavoshian
**Status:** Planning complete. Implementation phase.
**Version:** **V1 — frozen.** Plan does not get a version bump until the owner explicitly says so. All earlier draft labels (V2/V3/V4/V5) collapsed into V1. Agents must not introduce new version stamps.
**Last verified:** 2026-05-01 (3 verification rounds, 19 specialized Nia research agents, ~30 critical findings absorbed)

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

A from-scratch encoder-only transformer (~24-60M params) that predicts next-30min direction (UP / FLAT / DOWN) of GLD (gold ETF) using channel-group tokens combining price + volatility + macro + geopolitical features fused with semantic news embeddings from Qwen3-Embedding-4B (frozen, 4-bit MLX). Trained supervised with 3-class cross-entropy via Schedule-Free AdamW + Friendly-SAM on 5y of 30min bars. Sizing layer combines Kelly-lite × vol-target × conformal-confidence. Deployed live via Alpaca with $100 of real money on the Macbook M4 Pro. Artifact is a build-in-public X thread + blog post documenting the journey honestly.

## North Stars (read these — they decide every tradeoff)

1. **SOTA quality + WOW factor.** This is going on the internet to be read by ML researchers, quants, and tech Twitter. Everything must be impressive AND honest. Negative results with rigorous methodology beat cherry-picked positives.
2. **Use the best tool for the job.** No artificial restrictions. **HuggingFace Trainer, accelerate, peft, Unsloth, axolotl, datasets, lightning — all on the table.** If a library makes the project better, USE it. The quality of the artifact and the user's learning compound when professional tooling is in the stack.
3. **Owner learns deeply.** Owner is samsiavoshian. They want to understand transformers at Karpathy depth. So: **build the core model FROM SCRATCH in raw PyTorch** (this is the learning artifact). Use HF/Unsloth/peft for boilerplate (training loop infrastructure, data loading, mixed precision, LoRA, distributed). Read every library you import; understand what it does before depending on it.
4. **Real $100 skin-in-the-game.** Paper trade is dev. Live Alpaca with real money once paper validates.
5. **Local hardware only.** M4 Mac mini 16GB (trainer) + M4 Pro Macbook 16GB (live + dev). NO cloud GPU. NO Modal/Runpod/Lambda.
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

- The nanoGLD transformer architecture itself (RMSNorm, SwiGLU, attention, RoPE, channel-group tokens, news fuser) — doc 03
- The vectorized backtest engine — doc 06
- The sizing math (Kelly-lite × vol-target × conformal) — doc 07
- The Live trading cycle (Alpaca SDK is fine; cycle orchestration is yours) — doc 09

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

## Architecture Spec (V1, locked May 2026)

```
DATA PIPELINE (doc 01)
├── Alpaca historical 30min GLD × 5y (free Basic tier, IEX feed since 2016)
├── Alpaca News API (Benzinga only — Reuters paywalled 2024)
├── GDELT 2.0 GKG_partitioned via BigQuery (themes — events table has no themes)
├── FRED + ALFRED for vintage-correct DXY/DGS10/DGS2/T5YIE/VIXCLS/oil
├── yfinance for Brent/WTI daily (NOT 30min — 60d cap is real)
├── GPR Index monthly (matteoiacoviello.com)
└── Joined parquet with point-in-time discipline + 15min news latency

FEATURE ENGINEERING (doc 02)
├── Price (12) + Risk/Vol (8) + Macro (6) + Geopolitical (10) = 36 numeric
├── + Multi-dim sentiment: polarity + intensity + uncertainty (per arXiv:2603.11408)
├── Channel-group tokenization (iTransformer-lite, ~14 group tokens)
├── RevIN per channel-group (Kim ICLR 2022)
├── pandas-ta-classic for indicators (NOT stale `ta`)
├── Garman-Klass volatility (NOT Parkinson — same OHLC, more efficient)
├── Labels: 3-class via 5bps threshold
└── Z-score with rolling 1000-bar lookback + clip(-10, 10)

NEWS EMBEDDING (doc 04)
├── Qwen3-Embedding-4B 4-bit MLX (Apache 2.0, ~18K tok/s on M4 mini)
│   ├── 45× faster than Llama-3.1-8B mean-pool (earlier draft)
│   ├── MTEB-en avg 74.6 (+10pts vs LLM2Vec)
│   └── MRL truncation: 2560-dim → 256-dim, 99% quality retention
├── Per-source instruction-aware prompts ("Represent this financial news...")
├── Anchor-cosine semantic features (4 anchors: conflict, monetary, dollar, recession)
├── Learned [NO_NEWS] token + 15% modality dropout
└── memmap fp16 .npy storage (~133MB total — was 4GB in earlier draft)

MODEL ARCHITECTURE (doc 03)
├── ENCODER-only (no causal mask — bidirectional context for classification)
├── ~14 channel-group tokens (NOT 64 per-bar tokens — iTransformer pattern)
├── 12 layers, D=384, num_heads=6, head_dim=64 (Llama 3 / Qwen 3 sweet spot)
├── RMSNorm + SwiGLU + RoPE (real-form, MPS-safe) + QK-Norm + no-bias
├── Per-head gating + value residuals (IMU-1 recipe, arXiv:2602.02522)
├── Partial RoPE (10% of head_dim — arXiv:2603.11611)
├── F.scaled_dot_product_attention(is_causal=False) with .contiguous() Q/K/V
├── Perceiver-Resampler-lite + Flamingo-gated cross-attn for news fusion
├── A/B candidates: TDA (arXiv:2601.12145) + SyPE (arXiv:2602.08983)
└── Mean-pool over tokens → Linear(D, 3) → softmax

TRAINING (doc 05)
├── Stage 1: SSL pretrain (MAE on masked bars, 10 epochs)
│   [A/B Phase 2: MTS-JEPA arXiv:2602.04643 — replaces MAE]
├── Stage 2: Linear-probe (frozen encoder, 5-10 epochs head-only)
├── Stage 3: LLRD fine-tune (decay 0.85, unfreezes encoder)
├── Optimizer: Schedule-Free AdamW (Defazio ICLR 2025, won AlgoPerf 2024)
│   [A/B: Muon for 2D weights — DeepSeek V4 / Kimi-2 production]
├── Sharpness: Friendly-SAM ρ=0.05 (NOT vanilla SAM — F-SAM filters noise)
├── EMA: decay=0.999 on weights (deployed = EMA, not raw)
├── Walk-forward: 4 folds at 5y (3y train + 6mo val + 6mo test, 3mo step)
├── Loss: 3-class CE + class weights + label smoothing 0.1 (NEVER MSE on returns)
├── Regularization: dropout 0.2 + stoch depth 0.15 + jittering + Manifold Mixup
├── Cross-asset transfer (bonus): SPY → GLD via LLRD
└── PyTorch 2.11.0, FP32, num_workers=0 (macOS fork issues)

BACKTEST (doc 06)
├── Cost model: 5bps round-trip (sensitivity test 3/5/7/10 bps)
├── bars_per_year = 3276 (NYSE RTH only — NOT 17500)
├── Baselines: buy-hold, MA crossover, Donchian, DLinear, TSMixer, TimeMixer,
│              xLSTMTime (won 2026 finance benchmark per arXiv:2603.01820),
│              XGBoost (committed config), Forecast-to-Fill replication
├── Stationary block bootstrap CI on Sharpe (arch.bootstrap)
├── Deflated Sharpe Ratio (Bailey-Lopez de Prado)
├── Regime stratification: vol terciles + FOMC weeks + news density
├── Sortino: target downside dev (NOT std of negative subset)
└── Hard rule: if 24-60M Transformer can't beat all baselines by ≥0.2 Sharpe,
    SHIP THE BASELINE (TLOB lesson: "MLP can match transformer")

SIZING (doc 07)
├── Stage 1: fixed (1 share when argmax ≠ flat)
├── Stage 2: vol-target × Kelly-lite (0.5 Kelly, target 10% annual vol)
│   GATE: must beat Stage 1 by ≥0.2 Sharpe OOS
├── + Conformal Prediction (Wright 2026, arXiv:2601.07852)
│   30% lower realized decision loss vs uncalibrated
├── Drawdown circuit-breaker with explicit re-entry rule
└── Stage 3 (RL) deferred — gated, only built if Stage 2 leaves Sharpe on table

LIVE TRADING (doc 09)
├── Macbook M4 Pro runs cron via launchd every 30min
├── StartCalendarInterval (NOT StartInterval=1800) — fires only RTH M-F
├── pmset -c sleep 0 disablesleep 1 (#1 prod risk if skipped)
├── Alpaca SDK: alpaca-py >=0.43, paper for dev, live for prod
├── Pre-cycle check for open orders (avoid double-submit on partial fills)
├── Two-key separation: paper keys in dev, live keys ONLY in launchd env
├── 1Password CLI for live keys (NOT chmod 600 in prod)
├── Idempotent reconciliation via get_all_positions
└── Drift detection (entropy z-score + KL on argmax distribution)

INFRA + SECURITY (doc 10)
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


| #   | Doc                 | Status                           | Owner agent role           | Implementation effort            |
| --- | ------------------- | -------------------------------- | -------------------------- | -------------------------------- |
| 00  | OVERVIEW (this)     | ✅                                | n/a (read-first reference) | n/a                              |
| 01  | DATA-PIPELINE       | ✅ Complete + Nia-verified        | Data engineer              | 2-3 days                         |
| 02  | FEATURE-ENGINEERING | ✅ Complete + Nia-verified        | Feature engineer           | 1 day                            |
| 03  | MODEL-ARCHITECTURE  | ✅ V1                  | ML systems engineer        | 1 day                            |
| 04  | NEWS-EMBEDDING      | ✅ V1 (Qwen3-Embedding-4B) | ML engineer                | 0.5 day setup + 30min precompute |
| 05  | TRAINING-PROCEDURE  | ✅ V1 (Schedule-Free + F-SAM)     | ML engineer                | 1 day                            |
| 06  | BACKTEST            | ✅ Complete + Nia-verified        | Quant engineer             | 1 day                            |
| 07  | SIZING-STAGE2       | ✅ Complete + conformal           | Quant engineer             | 0.5 day                          |
| 09  | LIVE-TRADING        | ✅ Complete + Nia-verified        | Production engineer        | 1.5 days                         |
| 10  | INFRA-AND-SECURITY  | ✅ Complete + Nia-verified        | DevOps                     | 0.5 day (day 1 setup)            |
| --  | STATUS              | ✅ Live tracker                   | n/a (anyone can update)    | n/a                              |


**Deleted from earlier draft:** doc 08 (RL Stage 3 — speculative, gated, only build if Stage 2 leaves Sharpe), doc 11 (X-thread + blog content — user writes himself).

## Implementation Order (concurrent-friendly)

```
Day 1 (single agent — DevOps):
└── doc 10: repo setup + gitleaks + uv + pre-commit + .env structure
    → unblocks everything

Day 2-4 (concurrent if 4 agents):
├── doc 01: data pipeline (data eng — Alpaca + GDELT + FRED setup, 2-3 days)
├── doc 03: model architecture skeleton (ML systems eng — code only, no training)
├── doc 04: news embedder setup + first 1K bar A/B test (ML eng)
└── doc 06: backtest engine on synthetic data (quant eng — verify math)

Day 4-5 (depends on 01 + 04):
├── doc 02: feature engineering on real cached data
└── Continue 04: full 87K-bar embedding precompute (~30 min on M4 mini)

Day 5-6 (depends on 02 + 03):
└── doc 05: training procedure end-to-end on cached features

Day 7-8 (depends on 03, 05, 06):
├── Run all baselines (DLinear, TSMixer, TimeMixer, xLSTMTime, XGBoost)
├── Run nanoGLD Stage 1 + Stage 2 sizing
└── doc 06: full backtest report + bootstrap CIs + DSR

Day 9-10 (depends on 03, 05, 07):
├── doc 07: Stage 2 sizing + conformal calibration
└── doc 09: live trading loop on Macbook (paper trading)

Day 11-12: paper-trade soak, debug, polish

Day 13-14: fund $100, switch to live, build-in-public X thread
```

## Verification History

- **Round 1 (data pipeline):** 4 Nia agents verified Alpaca / GDELT / yfinance / FRED claims against live APIs. Found ~10 critical bugs (TimeFrame.Minute_30 doesn't exist, GDELT theme codes wrong, get_series_as_of_date returns DataFrame).
- **Round 2 (architecture v3):** 7 Nia agents on transformer SOTA / time-series / multimodal / attention / small-data / empirical SOTA / MPS. Found 12 critical bugs (bars_per_year=17500 → 3276, decoder→encoder pivot, head_dim=32 too small, etc).
- **Round 3 (2026 SOTA):** 6 Nia agents on May 2026 releases (Llama 4, Gemma 4, Qwen 3.5, embedding leaderboard, TS foundation models, architecture innovations, finance papers, training optimization). Major pivot: Llama-3.1-8B → Qwen3-Embedding-4B (45× faster). Schedule-Free AdamW replaces cosine + warmup. Forecast-collapse hard rule (NEVER MSE).

## Empirical Bar (what success looks like)

Per Agent 6 (empirical SOTA research) + 2026 finance papers (Agent E):


| Tier                  | Threshold                                                                  | Status                   |
| --------------------- | -------------------------------------------------------------------------- | ------------------------ |
| Minimum viable        | 38% direction accuracy + Sharpe > 0 + beat XGBoost on same features        | Mandatory to ship        |
| Real claim            | Sharpe > 1.0, hit rate > 52%, DSR > 1.0, beat ALL baselines by ≥0.2 Sharpe | Mandatory for X thread   |
| Forecast-to-Fill tier | Sharpe > 2.5, MDD < 5% on 5y walk-forward                                  | Publishable contribution |


**Forecast-to-Fill (arXiv:2511.08571, Sharpe 2.88) is unreplicated in 2026.** Building our own honest 30min gold benchmark with cost+DSR is genuinely publishable.

## Hard Project Rules (apply across all docs)

1. **NEVER use MSE on returns.** Use 3-class CE or quantile loss. (forecast-collapse, arXiv:2604.00064)
2. **STAY FROM-SCRATCH.** No TS foundation model fine-tune. (arXiv:2511.18578 — TSFMs underperform)
3. **MANDATORY MLP/xLSTM baselines.** Ship simpler model if it ties. (TLOB lesson)
4. **Apply peer-benchmark discount.** Backtests capture launch-period regime, not skill. (arXiv:2604.18821)
5. **bars_per_year = 3276** (NYSE RTH only). NEVER 17500.
6. **Point-in-time discipline.** Every feature uses .shift(1).rolling(...). News timestamps shifted forward by 15min latency.
7. **gitleaks before first commit.** Verify by trying to commit a fake key — must fail.
8. **PyTorch 2.11.0 pinned.** Has SDPA fix #174945 for MPS.
9. **FP32 weights everywhere.** No autocast, no torch.compile, no quantization at our scale.
10. `**.contiguous()` Q/K/V before SDPA.** PyTorch #181133.

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
| 2026-05-01 | Delete docs 08 (RL) + 11 (content)                      | Speculative + non-implementation                                    |


## File Layout (when implementation begins)

```
ml-trading/  (gh repo create nanogld --public)
├── pyproject.toml          (uv-managed, locked deps per doc 10)
├── uv.lock
├── .gitignore              (per doc 10 template)
├── .pre-commit-config.yaml (gitleaks, ruff, hooks per doc 10)
├── .github/workflows/      (CI smoke test per doc 10)
├── README.md               (project overview + reproduce instructions)
├── data/
│   ├── raw/                (.gitignored, per doc 01)
│   ├── snapshots/          (immutable hashed parquets)
│   ├── embeddings/         (memmap fp16 .npy, per doc 04)
│   └── anchors/            (.npz, per doc 02)
├── src/nanogld/
│   ├── data/               (doc 01: pipeline + joiner + golden fixture test)
│   ├── features/           (doc 02: feature engineering + RevIN)
│   ├── embed/              (doc 04: Qwen3 embedder + anchor cosines)
│   ├── model/              (doc 03: tiny_trader_v4.py + baselines)
│   ├── training/           (doc 05: walk-forward + Schedule-Free + F-SAM)
│   ├── backtest/           (doc 06: vectorized engine + bootstrap)
│   ├── sizing/             (doc 07: stage2 + conformal)
│   ├── live/               (doc 09: Alpaca cron + drift detection)
│   └── utils/              (shared: snapshot hashing, point-in-time helpers)
├── tests/
│   ├── test_pit.py         (golden fixture for joiner — doc 01)
│   ├── test_features.py    (no-leakage tests — doc 02)
│   ├── test_model.py       (forward pass shapes — doc 03)
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

Example: doc 02 (features) consumes parquet from doc 01. Schema documented in doc 01 — doc 02 codes against it. If doc 01 changes schema, doc 02 must be notified.

## When Implementation Diverges from Doc

If during implementation you discover:

- A library version is broken on MPS that the doc said works
- A paper's claim doesn't reproduce
- An API has changed since Nia verification
- A simpler approach achieves the same goal

DOCUMENT IT IN YOUR DOC. Add a section at the top: `## DEVIATION FROM SPEC: [date] - [issue]`. Don't silently work around — future agents need to see what changed.

If your deviation breaks another doc's interface, ask user via AskUserQuestion before shipping.

## Glossary (for new agents)

- **nanoGLD:** the project name + the encoder-only transformer model
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

This plan has been verified across **3 verification rounds, 19 specialized Nia agents, ~30+ critical findings**. Every claim has a citation. Every architectural decision has been challenged. The plan is comprehensive enough to implement without ambiguity.

But the field moves weekly. **Always run a fresh** `nia search` **on your topic before starting your implementation.** A paper from last week could change your decision. The plan is a snapshot, not a contract. But don't try to change things, we are at a stable point and we need to ship ASAP.

**Now go read your assigned doc. Spawn Nia agents freely. Build the thing.**