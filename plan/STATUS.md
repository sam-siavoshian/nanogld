# STATUS — nanoGLD Implementation Tracker

**Project:** nanoGLD
**Owner:** samsiavoshian
**Date last updated:** 2026-05-08
**Phase:** ⭐ **DATA PHASE COMPLETE. V1 SPEC LOCKED.** Single unified dataset shipped on Mac mini. Ready for V1 model phase.
**Version:** **V1 — locked 2026-05-08.** No further replanning without owner approval. Agents do NOT introduce V2/V3/etc references or further redline V1. Past planning iterations collapsed into V1; V1 is a single-pass redline on top.

---

## V1 transition complete — 2026-05-08

9-agent Nia research synthesis on top of V1's 27 verifying agents produced V1 redlines. Owner approved Decisions 1B (hybrid encoder), 2B (channel-independent + patches), 3B (multi-task focal CE + Sharpe head), 4A (CFA + AECF + sparse cross-attn) + all small wins (Cautious update mask, muP transfer, Mixout, SimPSI/Wave-Mask, FreeLB on news, DANN era, SimMTM + CLIP SSL, triple-barrier labels, half-hour-5 feature, VSN gate, series decomposition, per-channel RevIN, focal loss gamma=3, T-scaling -> RAPS -> AgACI, Laplace last-layer, F2F-style sizing, per-bucket eval, cost-stress {0.5x, 1.0x, 1.5x}, DSR > 1.0).

`plan/V1-SPEC.md` is the canonical change list. All 8 plan docs touched + this STATUS + HANDOFF.

| Doc | V1 touched | Date |
|---|---|---|
| V1-SPEC.md | NEW | 2026-05-08 |
| 00-OVERVIEW.md | yes | 2026-05-08 |
| 01-INFRA-AND-SECURITY.md | minimal (deps unchanged) | 2026-05-08 |
| 02-DATA-PIPELINE.md | minimal (spread + ATR pre-features added) | 2026-05-08 |
| 03-NEWS-EMBEDDING.md | minimal (CFA + AECF call-out) | 2026-05-08 |
| 04-FEATURE-ENGINEERING.md | yes (VSN + series decomp + per-channel RevIN + h5 + spread + triple-barrier) | 2026-05-08 |
| 05-MODEL-TRAINING-CALIBRATION.md | yes (hybrid encoder + dual head + focal + RAPS/AgACI + Laplace + Cautious + muP + Mixout + SimMTM + CLIP) | 2026-05-08 |
| 06-BACKTEST.md | yes (per-bucket + cost-stress + DSR hard gate + 8 promotion gates) | 2026-05-08 |
| 07-SIZING-AND-EXITS.md | yes (F2F-style: friction-Kelly + ATR + vol-target + 30d timeout + sqrt-impact + conformal floor) | 2026-05-08 |
| 08-LIVE-TRADING.md | minimal (sizing API contract follows new doc 07) | 2026-05-08 |
| HANDOFF.md | yes (V1 transition section + pre-training-run checklist + label rebuild caveat) | 2026-05-08 |
| STATUS.md | this entry | 2026-05-08 |
| README.md | yes (target reframe + arch diagram + hard rules 18-25 + 8 gates) | 2026-05-08 |

**Execution rule update:** V1 is the new lock. Agents do NOT replan further without owner approval. Same Iron Law as V1: do not silently rewrite, redesign, or extend. If a V1 spec line is wrong, document the issue and AskUserQuestion before changing anything. README/STATUS slot reserved for V1 backtest report once the model is trained.

---

## Quick Status

```
Planning:        ██████████████ V1 locked 2026-05-08, 27 V1 + 9 V1 = 36 specialized Nia agents
Data phase:      ██████████████ 100% — unified dataset shipped 2026-05-08
                                  /Users/root1/Desktop/nanogld/data/processed/training_v1_unified.pt
                                  234 MB, 75,993 bars x 681 features + 40,032 news embeddings
                                  CAVEAT: V1 fixed-5bps labels; V1 wants triple-barrier (rebuild in dataloader or sidecar)
Model phase:     ░░░░░░░░░░░░░░   0% — read plan/V1-SPEC.md then plan/HANDOFF.md before starting
```

**🎯 NEXT STEP:** Model agent reads `plan/V1-SPEC.md` + `plan/HANDOFF.md` then implements per `plan/05-MODEL-TRAINING-CALIBRATION.md`. Run pre-training-run checklist (HANDOFF.md) before any H100 spend.

---

## Implementation phase tracker (V1)

| Module | Path | Status | Notes |
|---|---|---|---|
| Model | `src/nanogld/model/` | not-yet-built | hybrid encoder (10 transformer + 2 sLSTM) + FiLM + sparse cross-attn + CFA + AECF + dual head |
| Training | `src/nanogld/training/` | not-yet-built | Stage 1 SimMTM + CLIP SSL, Stage 2 linear probe, Stage 3 LLRD + Mixout p=0.7, Cautious-SF-AdamW + F-SAM + muP transfer |
| Calibration | `src/nanogld/calibration/` | not-yet-built | T-scaling (focal-trained logits) -> RAPS -> AgACI online wrapper, Laplace last-layer epistemic |
| Sizing | `src/nanogld/sizing/` | not-yet-built | F2F machinery: friction-adjusted Kelly + ATR-14 exits + vol-target 15% ann + 30d timeout + sqrt-impact + conformal floor |
| Backtest | `src/nanogld/backtest/` | not-yet-built | walk-forward CV + per-bucket eval + cost-stress {0.5x, 1.0x, 1.5x} + DSR + bootstrap CIs + 8 promotion gates |

Estimated implementation: **~14-16 days** end-to-end. **Sequential** — one agent per doc, hand off when done. **8 docs after V5 merge** (was 11; merged model+train+calib into doc 05, sizing+exits into doc 07).

---

## Doc Status (each doc owned by one Opus 4.7 agent)

| Doc | Owner role | Status | Effort | Blocked by |
|-----|-----------|--------|--------|-----------|
| 00 OVERVIEW | n/a | ✅ Read-first reference (V1-touched 2026-05-08) | n/a | n/a |
| 01 INFRA-AND-SECURITY | DevOps | ✅ **Implemented 2026-05-04** (see hand-off below; deps unchanged for V1) | 0.5 day | n/a |
| 02 DATA-PIPELINE | Data engineer | ✅ **DONE 2026-05-08.** All 27 ETFs/crypto + 40 FRED + GDELT + GPR + COT + WGC + calendar + DXY ingested. Unified dataset shipped. (V1: spread + ATR pre-features added to spec.) | **4 days actual** | doc 01 |
| 03 NEWS-EMBEDDING | ML engineer | ✅ **DONE 2026-05-08.** 40,032 articles embedded with Qwen3-Embedding-4B (256-dim FP16) on Mac mini MPS at bs=2. 99.7% coverage. (V1: CFA + AECF call-outs in spec, no re-embedding required.) | **2 days actual** | doc 02 |
| 04 FEATURE-ENGINEERING | Feature engineer | ✅ **DONE 2026-05-08** for v1 base + v2 engineered (681 features). **V1 ADDS** half-hour-5, spread bps, VSN gate, series decomposition, per-channel RevIN, triple-barrier labels — added in dataloader (see HANDOFF.md). | **0.5 day actual + V1 ~0.5 day** | doc 03 |
| 05 MODEL-TRAINING-CALIBRATION | ML systems engineer | 🔜 **NEXT (V1).** Read `plan/V1-SPEC.md` + `plan/HANDOFF.md` first. Hybrid encoder + dual head + SimMTM + CLIP SSL + focal -> RAPS -> AgACI + Laplace + Cautious + muP + Mixout. | **4-5 days (V1)** | doc 04 |
| 06 BACKTEST | Quant engineer | ✅ Spec ready (V1: per-bucket + cost-stress + DSR hard gate + 8 promotion gates) | 1 day | doc 05 |
| 07 SIZING-AND-EXITS | Quant risk engineer | ✅ Spec ready (V1: F2F-style — friction-adjusted Kelly + ATR exits + vol-target 15% ann + 30d timeout + sqrt-impact + conformal floor) | **2 days** | doc 06 |
| 08 LIVE-TRADING | Production engineer | ✅ Spec ready (V1: sizing API contract follows new doc 07) | 1.5 days | doc 07 |

**V5 Merge (2026-05-04):** old docs 05+06+07 merged into the new doc 05 (MODEL-TRAINING-CALIBRATION). Old docs 09+10 merged into the new doc 07 (SIZING-AND-EXITS). One agent owns each merged doc end-to-end. Each doc's `Blocked by` is now exactly the immediately-preceding doc — pure linear chain.

**Deleted docs (V1 cleanup, May 1):**
- ~~old `08-RL-STAGE3.md`~~ — RL deferred to V2.
- ~~old `11-X-THREAD-AND-BLOG.md`~~ — content strategy, owner writes himself.

---

## Execution Mode (every agent reads 00-OVERVIEW.md "Execution Mode" section)

**These docs ARE the plan. Do not replan, do not rewrite, do not redesign.** The user said explicitly: *plan to execute the docs, do not replan*. Silent scope drift = fired. If a doc claim is wrong, document the issue and AskUserQuestion before changing anything.

**V1 is the new lock as of 2026-05-08.** Same rule applies: agents do NOT redline V1 further. Build what `plan/V1-SPEC.md` says, test it, ship it. If a V1 spec line is wrong or under-specified, document and AskUserQuestion. Do not silently rewrite.

**Research with Nia.** Before guessing on a library version, paper claim, or API, spawn a subagent (general-purpose or Explore) to run `nia papers` / `nia github` / `nia search` / `nia oracle` / `nia packages` / `nia tracer`. Run `nia auth` once if any command errors on auth. Nia CLI: `/Users/samsiavoshian/.bun/bin/nia`.

**Use gstack execution skills only:**

| Skill | When |
|---|---|
| `/investigate` | bug / weird error / "why is this broken". Iron Law: no fix without root cause. |
| `/review` | staff-engineer review on diff before commit. Catches prod bugs CI does not. |
| `/qa` | exercise the code path. Auto-generates regression tests. |
| `/qa-only` | bug report without auto-fix |
| `/cso` | OWASP + STRIDE security audit. Mandatory before any code touching secrets, network, live $. |
| `/benchmark` | baseline before perf-sensitive change, compare after |
| `/ship` | sync, test, push, open PR |
| `/land-and-deploy` | merge → CI → deploy → verify |
| `/canary` | post-deploy monitoring loop |
| `/document-release` | update README + docs to match shipped state |
| `/retro` | weekly retro on shipping streaks, test health |
| `/learn` | review/prune what gstack remembered |
| `/browse` | headless Chromium for any web check (NEVER `mcp__claude-in-chrome__*`) |
| `/setup-browser-cookies` | import real-browser cookies for authenticated browsing |

**Forbidden (planning skills — done):** `/office-hours`, `/plan-ceo-review`, `/plan-eng-review`, `/plan-design-review`, `/plan-devex-review`, `/autoplan`, `/design-consultation`, `/design-shotgun`, `/design-html`, `/design-review`, `/devex-review`. Calling these wastes tokens and risks scope drift.

**Default loop per file you build:**

```
read doc → Nia subagent for unknowns → code → /review → /qa → /cso (if security) → /ship
```

**Escalate after 3 failed attempts.** AskUserQuestion. Bad work is worse than no work.

---

## Implementation Order — Sequential, One Agent Per Doc (V5 — 8 docs after merge)

Each agent starts only when the previous agent has handed off. **No blocking on multiple upstream docs — pure linear chain.**

```
Agent 01 → doc 01 INFRA-AND-SECURITY               (0.5 day)
  └── Repo + .gitignore + pre-commit + uv + .env. Unblocks everything.

Agent 02 → doc 02 DATA-PIPELINE                    (4-5 days)
  └── Pull all sources (Alpaca bars+ETFs+News, GDELT, FRED 35 series, CFTC COT,
      WGC, GPR, Brent/WTI, FNSPID, Kitco, Investing.com, BullionVault, CNBC,
      central bank speeches, government press, Reddit, Kaggle gold-labeled,
      calendar events). Hash snapshot. Output: data/snapshots/v1_<hash>.parquet

Agent 03 → doc 03 NEWS-EMBEDDING                   (1.5 days + ~120min precompute)
  └── Qwen3-Embedding-4B per-article embeddings. Source registry (12 bias tiers).
      Aggregator (per-source PMA + bar-conditioned FiLM Q-Former K=8 + Flamingo gate).
      LAFTR adversarial head. Anchors (V4 hand-crafted templates).
      Output: data/embeddings/v1_<hash>_articles.parquet

Agent 04 → doc 04 FEATURE-ENGINEERING              (1.5 days)
  └── 12 feature categories (price, risk, macro short, geo, equities, treasury curve,
      macro bundle, COT, WGC, calendar). Z-score + clip(-10, 10) + RevIN.
      Output: feature DataFrame.

Agent 05 → doc 05 MODEL-TRAINING-CALIBRATION       (3 days, all in one)
  └── Part 1 — Build encoder-only transformer + baselines (DLinear, TSMixer, TimeMixer,
      xLSTMTime, XGBoost, Forecast-to-Fill replica).
  └── Part 2 — Train it: 3-stage (SSL-MAE → linear-probe → LLRD fine-tune),
      Schedule-Free AdamW + Friendly-SAM + EMA + walk-forward CV + LAFTR adversary.
  └── Part 3 — Calibrate it: temperature scaling (val-B) + APS Mondrian conformal
      (val-C) + classwise Adaptive ECE + drift stack (3 tiers).
      Output: checkpoints/v1_<hash>.pt with calibration objects baked in.

Agent 06 → doc 06 BACKTEST                         (1 day)
  └── Vectorized engine + 5bps cost model + bars_per_year=3276 + DSR + bootstrap CIs
      + regime stratification. Run all baselines + nanoGLD.
      Output: reports/v1_<hash>_backtest.md

Agent 07 → doc 07 SIZING-AND-EXITS                 (2 days, all in one)
  └── Part 1 — Vol-target × Kelly-lite × calibrated conformal shrinkage.
  └── Part 2 — Per-trade stop-loss + profit-take ladders + drawdown circuit-breaker.
      Output: position-mgmt module.

Agent 08 → doc 08 LIVE-TRADING                     (1.5 days)
  └── launchd cron + Alpaca SDK + drift detection + 1Password live keys.
      Paper-trade soak (5+ days), debug. Then fund $100, switch to live,
      build-in-public X thread.

────────────────────────────────────────────────────
Total: ~14-16 days end-to-end (sequential, 8 self-contained agents).
```

---

## V1 — Library Policy Update (2026-05-01)

Owner lifted "raw PyTorch only / no HuggingFace" restriction. Researched modern training libraries.

**Adopted:** `accelerate>=1.13,<2.0` only (5 LOC for device-agnostic boilerplate + free checkpoint state + seed handling). Added to doc 01 deps + doc 05 stack.

**Deferred:** `transformers.Trainer` (optional, decide during impl based on Friendly-SAM override fit), `peft` (until/if embedder fine-tune day).

**Skipped permanently (MPS-incompatible regardless of restriction):** Unsloth (no Triton on Mac), bitsandbytes 4-bit (CUDA-only), torch.compile on MPS (NaN bug), lightning-thunder (NVIDIA-focused), torchtitan (multi-GPU overkill).

**No architectural changes.** V1 from-scratch nanoGLD stays locked. Foundation model untouched. Embedder still frozen Qwen3-Embedding-4B.

## Verification History

| Round | Date | Agents | Findings |
|-------|------|--------|----------|
| 1 | 2026-04-30 | 4 (Alpaca, GDELT, yfinance, FRED) | ~10 critical bugs in data layer (TimeFrame.Minute_30 fake, GDELT theme codes wrong, yfinance 30m cap real) |
| 2 | 2026-04-30 | 7 (LLM SOTA, TS, multimodal, attention, small-data, empirical SOTA, MPS) | 12 critical bugs (bars_per_year=17500 → 3276 fatal, decoder→encoder pivot, head_dim=32 too small, fold count off) |
| 3 | 2026-05-01 | 6 (2026 LLMs, embedders, TS foundation, architecture, finance, training) | 8+ critical (Qwen3-Embedding-4B 45× faster, Schedule-Free AdamW, F-SAM, NEVER MSE on returns hard rule) |
| 4 | 2026-05-04 | 5 (Alpaca + bars/news/IEX leakage, FRED+ALFRED full audit, CFTC+WGC verify, GDELT+yfinance+GPR verify, cross-source leakage taxonomy) | **17 silent killers**: bar `t`=START leak, `created_at` field correction, FEDFUNDS→DFF, 6 GDELT codes refuted, GDELT 30min buffer, WGC URL was wrong + monthly, AI-GPR has 30-day lag, GPR no vintage archive, pandas-ta KAMA/Ichimoku/KST/DPO/TRIX/Vortex forbidden, CFTC 2025 shutdown gap, multi-symbol pagination interleaves, `adjustment="all"` retroactive, WALCL/ICSA release-time gating, anchor-cosine pre-train-period rule, no minutes-until-event features. All encoded as 17 hard rules + 28 mandatory tests. |
| 5 | 2026-05-04 | 5 (gold-news sources Kitco/MetalsDaily/BullionVault/Investing.com, mainstream CNBC/Reuters/FT/TE/FXStreet, multi-source bias debiasing, multi-doc embedding aggregation SOTA, free 10y news datasets) | **News pipeline 3→12+ sources**. FT REFUTED (robots.txt bans ML). Reuters/FXStreet/TE DEFER (paid). Kitco/Investing.com/BullionVault/CNBC ADD (free scrape). FNSPID (15.7M articles, 1999-2023, CC BY 4.0) ADD as biggest free win. Central bank speeches + government press releases (Fed/ECB/BIS/Treasury/CFTC) ADD (US gov public domain). Reddit Arctic Shift ADD. Bias-aware: 12 source-bias tiers + LAFTR adversarial head + inverse-frequency reweighting. Aggregator V4: K=16→8, +bar-conditioned FiLM, +per-source PMA pre-pool. Per-article embedding (was per-source mean-pool). doc 03 effort 0.5d → 1.5d. |

**Total: 27 specialized Nia agents, ~70 critical findings absorbed.**

---

## V1 Architecture Locked (May 1, 2026) -> V1 Locked (May 8, 2026)

**Backbone (V1):** hybrid encoder, ~24-60M params. 10 transformer blocks (layers 1-10) + 2 sLSTM blocks (layers 11-12, xLSTMTime style). Channel-independent + patches (P=4, S=4, T=64 -> 16 patches/channel) replaces V1's channel-group tokens.
**Per-block:** RMSNorm + SwiGLU + RoPE (real-form) + QK-Norm + per-head gating + value residuals + no-bias. FiLM regime modulation @ {2,4,6,8,10} on a 12-dim regime vector. Stochastic depth schedule linear 0.0 -> 0.2.
**News fusion (V1):** Qwen3-Embedding-4B (frozen) -> CFA projector (FiLM + orthogonal residual) -> sparse Flamingo-gated cross-attn at layers {3, 7, 11} with AECF entropy-gated curriculum mask. is_news_present binary embedding concatenated.
**Input (V1):** VSN feature gate (Lim 2021 TFT) -> series decomposition (24-bar MA kernel) -> per-channel RevIN (681 instances) -> patches.
**Training (V1):** Stage 1 SSL = SimMTM (mask 0.40, K=3 multi-mask) + CLIP-style bars<->news contrastive. Stage 2 linear probe. Stage 3 LLRD fine-tune with Mixout p=0.7 anchored to SSL checkpoint. Cautious-Schedule-Free-AdamW + Friendly-SAM rho=0.05 + EMA 0.999 + muP transfer-tune from 2-4M tiny model. SimPSI + Wave-Mask aug only (NO naive jittering). FreeLB on news embeddings, DANN era-label gradient reversal.
**Loss (V1):** dual head jointly trained — focal CE gamma=3 (Head A, 3-class) + cost-aware Sharpe loss (Head B, position weight in [-1, +1]). NEVER MSE on returns. Combined Stage 3: `0.5 * L_focal + 0.5 * L_sharpe_net + 0.05 * L_DANN + L_aecf`.
**Calibration (V1):** focal-trained logits -> T-scaling [0.7, 3.0] guard -> RAPS (size penalty) -> AgACI online wrapper. Laplace last-layer epistemic (replaces MC dropout T=20). Snapshot ensemble (last 3 EMA checkpoints) kept.
**Sizing (V1, F2F-style):** Head B output -> friction-adjusted Kelly (lambda=0.4, edge from Head B, variance from Laplace + rolling 60-bar realized) -> ATR-14 hard stop (2x) + trailing (1.5x) -> 30-day timeout -> sqrt-impact cost (gamma=0.02, k=0.7bps) -> vol target 15% annualized -> conformal floor (RAPS lower-bound < 0.40 -> position 0).
**Backtest (V1):** vectorized engine, bars_per_year=3276, baselines = DLinear/TSMixer/TimeMixer/xLSTMTime/VLSTM/Gao 2014 half-hour-5/XGBoost/Forecast-to-Fill replica, DSR > 1.0 hard gate, per-bucket {news-present, news-absent, both} hard requirement, cost-stress {0.5x, 1.0x, 1.5x} hard gate, bootstrap CI.
**Honest target (V1):** 1.0 to 1.5 OOS Sharpe net of 2bps round-trip, beating Gao 2014 + XGBoost ensemble by >= 0.2 Sharpe. The 2.88 number is daily gold futures EOD-to-EOD, not 30-min intraday.
**Live:** launchd StartCalendarInterval cron on Macbook M4 Pro, alpaca-py >=0.43, pmset sleep prevention.
**Infra:** PyTorch 2.11.0 pinned, FP32 weights, gitleaks before first commit, ADC for GCP, 1Password for live keys.

---

## A/B Candidates (post-baseline)

These are coded as alternative components. Test if baseline plateaus. Adopt only if win ≥0.1 Sharpe OOS, seed-averaged 5 seeds.

| # | Candidate | Doc | Trigger |
|---|-----------|-----|---------|
| 1 | TDA in 1 attention block | 03 | Baseline noise-rejection insufficient |
| 2 | SyPE replaces RoPE | 03 | Cyclic regime patterns missed |
| 3 | Muon optimizer for 2D weights | 05 | Schedule-Free plateaus |
| 4 | MTS-JEPA replaces MAE pretrain | 05 | MAE pretrain converges but val gap stays large |
| 5 | Cross-asset transfer SPY → GLD | 05 | Bonus experiment, free |
| 6 | RL Stage 3 (deferred — was doc 07) | n/a | Stage 2 leaves Sharpe; user decision |

---

## Empirical Bar (success criteria — V1 reframe)

| Tier | Threshold | Status to ship |
|------|-----------|---------------|
| Minimum viable | 38% accuracy + Sharpe > 0 net of 1x cost + beat XGBoost | Mandatory |
| V1 ship | Sharpe > 1.0 net of 1x + Sharpe > 0.5 net of 1.5x + DSR > 1.0 + per-bucket Sharpe both positive + beat Gao 2014 + XGBoost ensemble by >= 0.2 | Mandatory for X thread |
| Stretch | Sharpe > 1.5 net of 2bps + MDD < 5% + beats VLSTM 2.40 (daily futures, separate scoreboard) | Publishable contribution |

**The actual bars (V1):**
- GLD 5y buy-and-hold Sharpe ~ 2.4 (2020-2025 was a great gold run). Beating this gross is easy; net of costs and on a 30-min timing strategy is the real test.
- **Gao-Han-Li-Zhou 2014 half-hour-5 single-feature rule (5.43 Sharpe)** is the apples-to-apples GLD intraday published bar. If V1 loses to this, we shipped a worse model than 2014 and the recommendation is the simpler ensemble.
- **Forecast-to-Fill 2.88 Sharpe** is daily gold futures EOD-to-EOD, ~30-day holding. Different problem, separate scoreboard.
- **Saly-Kaufmann/Wood/Zohren 2026 VLSTM 2.40 Sharpe** is the daily futures DL frontier, separate scoreboard.

---

## Hard Project Rules (apply across all docs)

V1 invariants (1-17, kept):

1. **NEVER use MSE on returns** (forecast-collapse, arXiv:2604.00064)
2. **STAY FROM-SCRATCH** (encoder + sLSTM head + heads all trainable; only Qwen3 frozen, arXiv:2511.18578)
3. **SHIP THE SIMPLER MODEL IF IT TIES** (TLOB lesson)
4. **APPLY PEER-BENCHMARK DISCOUNT** to backtests (arXiv:2604.18821)
5. **bars_per_year = 3276** (NYSE RTH only)
6. **Point-in-time correctness on every feature** (`.shift(1).rolling(...)`, news buffer = 30min for GDELT, 60s for Alpaca News, bar visibility = `bar.timestamp + bar_duration`)
7. **gitleaks BEFORE first commit** (verify with fake key)
8. **PyTorch 2.11.0 pinned** (SDPA fix #174945 for MPS)
9. **FP32 weights everywhere** (no autocast, no torch.compile, no quantization). H100 has bf16 but V1 stays FP32 deterministic.
10. **`.contiguous()` Q/K/V before SDPA** (PyTorch #181133)
11. **Every feature row carries `t_visible`** column. CI gate: `test_release_ts_lte_t_visible_all_rows`.
12. **ALFRED `get_series_all_releases` for ALL FRED series** (CPI/PCE annual revisions silently rewrite 5y of history)
13. **pandas-ta KAMA/Ichimoku/KST/DPO/TRIX/Vortex FORBIDDEN** (look-ahead bugs, bukosabino/ta#181)
14. **Calendar features = binary windows ONLY** (no `minutes_until_event`)
15. **Anchor-cosine anchors = hand-crafted templates OR pre-train-period samples**
16. **Alpaca News field = `created_at`** (`published_at` does NOT exist; never join on `updated_at`)
17. **`DFF` for daily Fed Funds, NOT `FEDFUNDS`** (FEDFUNDS is monthly)

V1 invariants (18-25, NEW):

18. **Per-bucket eval (news-present / news-absent / both) is non-negotiable.** 51% of bars are news-absent.
19. **Cost-stress at {0.5x, 1.0x, 1.5x} on every reported Sharpe.** Hard gate: Sharpe > 0.5 at 1.5x.
20. **DSR > 1.0 hard gate.** No cherry-picking across configs.
21. **SimPSI / Wave-Mask aug only.** Naive jittering FORBIDDEN (Fons 2020 net-negative on Sharpe).
22. **Focal loss gamma=3 (NOT vanilla CE).** Required for clean T-scaling / APS interaction (Xi 2024 arXiv:2402.04344).
23. **Triple-barrier labels with spread-adjusted neutral threshold.** Replaces fixed 5-bps cutoff.
24. **Variable per-batch modality dropout p ~ U(0.1, 0.9).** NOT 15% constant — training distribution must bracket inference (51% news-absent).
25. **Decision-aware head (multi-task with Sharpe loss) is V1 ship gate.** End-to-end profit metric, not just classification accuracy.

---

## Decision Log (cross-cutting decisions)

| Date | Decision | Rationale |
|------|----------|-----------|
| 2026-04-25 | initial admissions-framed → V1 | User pivot |
| 2026-04-30 | Decoder-only → Encoder-only | Bidirectional better for next-bar classification |
| 2026-04-30 | Per-bar tokens → Channel-group tokens | iTransformer-lite, empirically dominant |
| 2026-04-30 | Add SAM ρ=0.05 | SAMformer ICML 2024 |
| 2026-04-30 | Mandatory MLP baselines | TLOB |
| 2026-05-01 | Llama-3.1-8B → Qwen3-Embedding-4B | 45× faster, Apache 2.0 |
| 2026-05-01 | AdamW + cosine + warmup → Schedule-Free AdamW | AlgoPerf 2024 winner |
| 2026-05-01 | Vanilla SAM → Friendly-SAM | Filters gradient noise |
| 2026-05-01 | Hard rule: NEVER MSE on returns | Forecast collapse |
| 2026-05-01 | Add conformal prediction sizing | Calibrated set-size shrinkage; doc 07 supersedes. (Note: original 30% claim from Wright 2026 was fabricated — citation removed in doc 07.) |
| 2026-05-01 | Add xLSTMTime baseline | 2026 finance benchmark winner |
| 2026-05-01 | Delete doc 07 + 11 | Speculative + non-implementation |
| 2026-05-01 | Add agent-isolation headers to all docs | Concurrent multi-agent implementation phase |
| 2026-05-04 | Dataset expansion (V1, owner directive): 9-ETF basket + full Treasury curve + 19-series macro bundle + CFTC COT + WGC + calendar | Capture real-rate / risk-on-off / sector rotation / positioning drivers. Per-bar dim ~804 → ~1000. doc 02 effort 2-3d → 4-5d, doc 04 1d → 1.5d. No model arch change. |
| 2026-05-04 | Owner flagged 3 missing pieces: confidence sizing, per-trade SL, profit-taking. Spawned 4 parallel research agents (sizing / SL / TP / Forecast-to-Fill replication + Alpaca constraints). Wrote `plan/07-SIZING-AND-EXITS.md` superseding doc 07 sizing math and doc 08 line 198 stop-loss. | (1) Sizing: replace `(max_prob-0.33)` with signed score `P_up-P_down`; quarter-Kelly default; vol-target with `min(σ*/σ_t, 3.0)` cap; continuous conformal shrinkage; EWMA λ=0.94 + 20d floor. (2) Stop-loss: 2.0×ATR(14) hard + 1.5×ATR(14) trailing + 390-bar time-stop + re-entry gate + 15:55 ET session-flat + ±15min FOMC/CPI/NFP blackout. Match Forecast-to-Fill exactly. (3) Profit-take: NO fixed TP (F2F + Baur-Dimpfl + 5bp cost gate); optional signal-decay exit `max_prob < entry_max_prob × 0.7` gated on val A/B ≥30bp lift. (4) Live: client-side stop polling — Alpaca rejects bracket orders for fractional positions (error 42210000), and at $100 with GLD ~$200 every position is fractional. (5) Deleted fabricated "30% lower decision loss" Wright 2026 citation. |
| 2026-05-08 | **V1 lock.** 9-agent Nia synthesis approved by owner: Decisions 1B (hybrid 10 transformer + 2 sLSTM head) + 2B (channel-independent + patches replaces channel-group tokens) + 3B (multi-task focal CE + Sharpe loss head) + 4A (CFA + AECF + sparse cross-attn at {3,7,11}) + small wins (Cautious update mask, muP transfer, Mixout, SimPSI/Wave-Mask, FreeLB news-only, DANN era, SimMTM + CLIP SSL, triple-barrier labels, half-hour-5 feature, VSN gate, series decomposition, per-channel RevIN, focal loss gamma=3, T-scaling -> RAPS -> AgACI, Laplace last-layer, F2F-style sizing, per-bucket eval hard gate, cost-stress {0.5x, 1.0x, 1.5x} hard gate, DSR > 1.0 hard gate). Honest target reframe: 1.0-1.5 OOS Sharpe net of 2bps, beating Gao 2014 + XGBoost ensemble by >= 0.2. The 2.88 number is daily futures, not 30-min intraday. | Recurrent gated state (VLSTM 2.40 / xLSTM 1.79 / iTransformer 0.38 on Saly-Kaufmann/Wood/Zohren 2026), small-data PatchTST channel-independent dominance (75K samples), end-to-end Sharpe loss > MSE forecasts (Hwang & Zohren 2025), focal cleans T-scaling/APS conflict (Xi 2024), AECF entropy gate PAC bound (Chlon 2025), sparse cross-attn beats dense (mPLUG-Owl3), SimMTM > MAE (Dong NeurIPS 2023), AgACI provable coverage under shift (Zaffran ICML 2022), Laplace last-layer faster + better-calibrated than 20x MC dropout (Daxberger 2021). |

---

## Open Questions (live)

These need user decisions OR Nia verification BEFORE implementation begins:

- [ ] Verify Alpaca paper account actually returns 5y of 30min GLD on YOUR account (small chance of region restriction)
- [ ] Verify Alpaca returns 5y of 30min for ALL 9 ETFs in V1 basket (SPY/QQQ/IWM/GDX/SLV/XLF/XLE/XLK/XLU) — most likely SPY/QQQ are fine; SLV/GDX low-volume IEX may have gaps
- [ ] Verify GCP billing approval if non-US card (may have day-delay)
- [ ] Confirm GDELT BigQuery dry-run estimate matches reality on YOUR query phrasing (±10% expected)
- [x] T5YIE vs T5YIFR — V1 expansion settles: pull both + T10YIE + use as separate features. doc 04 builds derived features from all three.
- [ ] Do we A/B test TDA + SyPE , before or after baseline?
- [ ] CFTC disaggregated COT historical zip URL verify (path rotated in past — confirm 2021-2026 still live with `/browse`)
- [ ] WGC central-bank-quarterly direct download URL — needs `/browse` to extract (form-based)
- [ ] Owner re-decision (after V1 baseline lands): include the deferred specialty bundles (GVZ + credit spreads + bond vol) and (USD cross-rates + crypto + industrial metals)?

---

## Implementation Pre-Flight Checklist

Before starting Day 1, owner verifies:

- [ ] Mac mini 16GB available + macOS 14+ (or 26 with caveats)
- [ ] Macbook M4 Pro 16GB available + macOS 15.x preferred
- [ ] $100 ready to fund Alpaca live account (3 days bank-link lag)
- [ ] Alpaca paper account approved
- [ ] Alpaca live account approved (or in-progress)
- [ ] GitHub account
- [ ] HuggingFace account (for Qwen3-Embedding-4B download)
- [ ] GCP account + billing (for BigQuery — biggest first-day friction)
- [ ] FRED API key (free, instant)
- [ ] wandb account (free)
- [ ] Personal domain registered (for blog, ~$10)

Total wall-time before any code can run: **~3-5 days** (mostly waiting for Alpaca + bank link).
Total active setup time: **~3-4 hours** (when not waiting).

---

## Risk Register

| Risk | Severity | Mitigation |
|------|----------|------------|
| Llama-3.1-8B-4bit too slow on Mac mini | LOW (V1 uses Qwen3) | Already mitigated |
| Qwen3-Embedding-4B doesn't fit 16GB | LOW | Falls back to Qwen3-Embedding-0.6B (44K tok/s, 900MB RAM) |
| yfinance breaks during implementation | MEDIUM | Pin v1.3.0; Alpaca historical for 30m bars (NOT yfinance) |
| GDELT BigQuery free tier exceeded | MEDIUM | maximum_bytes_billed cap + 1024 GiB/day quota cap, materialize once |
| Model overfits, OOS Sharpe < 0 | HIGH | Honest negative result thread is still a great thread |
| Real $100 Alpaca account issues | MEDIUM | Stay in paper longer, document friction |
| 16GB Mac mini OOMs during training | MEDIUM | Reduce batch size, gradient accumulation, or model size |
| Leakage bug not caught by golden fixture | HIGH | Multiple golden fixtures + visual sanity check |
| Macbook sleeps during market hours | HIGH | pmset -c sleep 0 disablesleep 1 + caffeinate (doc 08) |
| Concurrent doc updates corrupt plan | MEDIUM | Each agent owns ONE doc, AskUserQuestion on cross-doc changes |

---

## How Agents Communicate

- **Across docs:** via STATUS.md (this file). Agents append to relevant section.
- **User decisions:** via AskUserQuestion when crossing locked boundaries.
- **Cross-doc deps:** via stable interfaces documented IN each doc's "Stable Interface You Publish" section.
- **Bug reports:** via DEVIATION sections appended to top of relevant doc.

**No Slack, no async, no wait time.** Each agent works from spec, ships, updates STATUS, hands off.

---

## Implementation Hand-offs

Each agent appends a record here when their doc is shipped. Subsequent agents read this to understand the in-flight system state.

### Doc 01 INFRA-AND-SECURITY — shipped 2026-05-04

**Repo:** `https://github.com/sam-siavoshian/nanogld` (public, MIT, branch `main`)
**Local dir:** `/Users/samsiavoshian/Desktop/Coding Stuff/Side Projects/nanogld/`
**Python:** 3.11.14 (pinned via `.python-version` + `requires-python = ">=3.11,<3.13"`)

**What's in:**
- `pyproject.toml` + `uv.lock` with V1 pinned deps (torch 2.11.0 / transformers 5.7.0 / sentence-transformers 5.4.1 / mlx-lm 0.31.3 (darwin-only) / schedulefree 1.4.1 / accelerate 1.13 / alpaca-py / yfinance 1.3.0 / fredapi 0.5.2 / pandas-ta-classic 0.5.44 / arch / xgboost / etc.). Cold-cache `uv sync`: 41s on M4.
- `.pre-commit-config.yaml`: gitleaks v8.24.2 + pre-commit-hooks v5.0.0 + ruff v0.11.0 (lint --fix + format). Verified against fake `ALPACA_API_KEY=PKTEST...` commit — exit code 1, blocked. ✅
- `.github/workflows/test.yml` — runs `ruff check` + `ruff format --check` + `pytest -q` + `gitleaks-action@v2` on push/PR. setup-uv@v8, ubuntu-latest, 15min timeout.
- `.github/workflows/smoke-test.yml` — monthly cron `0 0 1 * *` + manual dispatch. Imports every critical dep (catches API drift between active dev cycles). 20min timeout.
- `tests/test_smoke.py` — 3 trivial tests (Python version range, critical-imports, `nanogld.__version__`). Keeps CI green from day 1.
- `Makefile` with `help / install / lock / sync / upgrade / lint / format / test / pre-commit / clean` + placeholder targets `data / train / backtest / live`.
- `docs/SETUP.md` — per-key signup walkthrough, `~/.config/nanogld/` layout, two-key principle, ADC-not-JSON for GCP, rotation policy, optional 1Password CLI for `.env.live`.
- `docs/REPRODUCE.md` — fresh-clone walkthrough, prereqs, per-doc execution order with effort + output, doc 01 acceptance checklist.
- `src/nanogld/__init__.py` — empty package root with `__version__ = "0.1.0"`. Hatchling auto-detects via `[tool.hatch.build.targets.wheel] packages = ["src/nanogld"]`.
- `~/.config/nanogld/.env.paper` + `.env.live` (chmod 600, dir chmod 700, NOT in repo) — populated with `<FILL_ME>` placeholders. Owner fills before doc 02.
- `README.md` — mechanical link fixes only (8 stale `plan/0X-*.md` paths corrected to V5 numbering). No architecture/metric edits.
- `.gitignore` — augmented (added `*.h5`, `*.ckpt`, `alpaca-*`, `.neptune/`, `.comet/`, `outputs/`, `.hydra/`, `.coverage`, `htmlcov/`; removed `.python-version` from ignore so uv pin tracks).

**Acceptance criteria status:**
1. ✅ Public repo from commit 1.
2. ✅ Pre-commit blocks fake-key commits (verified).
3. ✅ `uv sync --frozen` produces working env (41s cold cache).
4. ✅ `uv.lock` committed.
5. CI green on first push — verified at push time (step 13 of phase 1 plan).
6. ✅ Monthly smoke-test cron scheduled.
7. ✅ `~/.config/nanogld/.env.{paper,live}` chmod 600.
8. ✅ Fresh-clone-to-`make test`-pass: ~2min on M4.

**Verification commands the next agent can run:**
```bash
uv sync --frozen      # ~40s cold, instant warm
make test             # 3 tests pass
make lint             # ruff clean
make pre-commit       # gitleaks + ruff + whitespace + EOF — all pass
gitleaks detect --no-git --verbose   # clean
```

**Open items for owner before doc 02:**
- Fill `~/.config/nanogld/.env.paper` with real Alpaca paper + FRED + HF + wandb keys.
- `gcloud init` + `gcloud auth application-default login` (BigQuery / GDELT — see `docs/SETUP.md`).
- Optional: bump pinned versions if Nia spots a 5+ day update; otherwise spec versions resolved cleanly.

**Doc 02 unblocked.** Owner: data engineer agent. Effort: 4-5 days. Spec: `plan/02-DATA-PIPELINE.md`.

### Doc 02 DATA-PIPELINE — code shipped 2026-05-04

**18 source modules** under `src/nanogld/data/`, all schema-validated, all
emit `(release_ts, t_visible)` PIT-correct frames.

**Foundation:**
- `utils.py` — FRED_RELEASE_TOD_ET (35-series ET tod table), `cot_release_ts_utc`,
  `assert_t_visible_invariant`, `merge_asof_pit` (strict-< asof), retry HTTP,
  rotating logger.
- `schema.py` — 9 source manifests + `validate()`, no pandera dep.
- `snapshot.py` — `pd.util.hash_pandas_object` + cols + dtypes (per doc 01
  Critical Corrections; 10-100× faster than `to_csv`). meta.json sidecar
  carries git_commit + full source manifest + time_range.

**Sources:**
| # | Module | Status |
|---|--------|--------|
| 1 | alpaca_bars.py (GLD 30m) | code complete; owner runs after `.env.paper` |
| 8 | alpaca_etfs.py (9 ETF basket) | code complete; same |
| 2 | alpaca_news.py (Benzinga) | code complete; same — `created_at` (NOT `published_at`) |
| 3 | gdelt.py | code complete; owner runs `gdelt_materialize` after `gcloud auth ADC` |
| 4 | fred.py (35 ALFRED series) | code complete; owner runs after FRED_API_KEY filled |
| 5 | yfinance_helpers.py (Brent/WTI) | ✅ pulled — 1258 BZ + 1257 CL daily rows |
| 6 | gpr.py | ✅ pulled — 213,540 rows, 116 series |
| 9 | cot.py | ✅ pulled — 278 weekly rows (5y) |
| 10 | wgc.py | ⚠️ HTML form-wall — owner extracts real URL via /browse |
| 11 | calendar_events.py | ✅ pulled — 408 events |
| 12-18 | news_{fnspid,kitco,investing,bullionvault,central_bank,reddit,kaggle}.py | code complete; HF datasets need HF_TOKEN; scrapers run on demand |

**Joiner + CLI:**
- `join.py` — strict-< asof on `bar_close_utc`, allow_exact_matches=False
  everywhere. Lag-1 bars, FRED forward-fill, news counts, calendar binary
  proximity.
- `cli.py` + `__main__.py` — `python -m nanogld.data {list,pull,join,build}`
  with `--skip-keyed`, `--force`. Idempotent.

**Tests (49 pass, 6 skip):**
- `tests/test_pit.py` — golden fixture from spec (NON-NEGOTIABLE).
- `tests/test_join_schema.py` — manifest + dtype + null + PIT invariant.
- `tests/test_snapshot_hash.py` — determinism + content-addressing.
- `tests/test_no_leakage.py` — 28 mandatory tests, V4 leakage findings.
  Live-data tests (`@NEEDS_DATA`) auto-skip until owner runs `build`.

**Pulled artifacts (data/raw/, 13 MB):**
```
brent_daily.parquet            1258 rows
calendar_events_v1.parquet      408 rows
cftc_cot_gold_weekly.parquet    278 rows
gpr_combined.parquet         213540 rows
wti_daily.parquet              1257 rows
```

**Dataset deviations from V1 spec (logged for /browse follow-up):**
- WGC `gold.org/download/{8052,7739}` returns HTML form-wall on direct GET.
  Spec line 162 acknowledges this.
- CPI / JOLTS / PCE / GDP calendar dates use deterministic approximations
  (CPI 12th, JOLTS 9th, PCE last BD). Spec line 1054 acknowledges this.

**Owner-action checklist before doc 03:**
1. Fill `~/.config/nanogld/.env.paper` with real Alpaca paper + FRED + HF +
   wandb keys (see `docs/SETUP.md`).
2. `gcloud init && gcloud auth application-default login` for BigQuery.
3. (Optional) /browse-extract real WGC URL + /browse-verify CPI/JOLTS dates
   against BLS calendar.
4. (Optional) Download Reddit Arctic Shift `.jsonl.zst` dumps to
   `data/raw/reddit/`.
5. Run `python -m nanogld.data build` end-to-end. Verify 16K-bar snapshot
   under `data/snapshots/v1_<sha>.parquet` + sidecar meta.json.
6. Re-run `pytest tests/` — `@NEEDS_DATA` tests should now exercise.

**Doc 03 unblocked.** Owner: ML engineer agent. Effort: 1.5 days setup +
~120 min precompute. Spec: `plan/03-NEWS-EMBEDDING.md`.

### News pipeline rewrite — code shipped 2026-05-04 (Phase News-1 to News-10)

Rewrite triggered by user dropping Alpaca (KYC-gated paper signup confusion)
and the verification pass exposing 4 broken scrapers + 1 license bug + 0
geopolitical / 0 retail rows on disk.

**Decisions (owner-confirmed):**
- All-free wire news (Wayback CDX); `polygon_news.py` ships gated behind
  `NANOGLD_POLYGON_PAID=1` for owner to flip if they pay $29/mo Polygon Stocks
  Starter later.
- `NANOGLD_NONCOMMERCIAL=1` default ON for V1 personal/research training.
  Disables cleanly for any future commercial deployment.
- GDELT — owner authed gcloud already (`~/.config/gcloud/application_default_credentials.json`,
  quota project `nanogld`). Owner needs to enable BigQuery API in the project
  (one-click) before `gdelt_materialize` works.
- Push policy: local commits only, owner pushes when ready.

**Code complete:**
- `wayback_helpers.py` — CDX search + polite fetch (300s timeout, exp backoff
  on 429/503, halt-after-5, raw-byte cache under `data/raw/wayback_cache/`)
- `news_kitco.py` — Wayback CDX backfill (RSS endpoint serves HTML, broken)
- `news_investing.py` — Wayback CDX (Cloudflare-walled live scrape)
- `news_bullionvault.py` — Wayback CDX (selectors don't match modern layout)
- `news_polygon.py` — drop-in for dropped `alpaca_news.py`, gated behind
  `NANOGLD_POLYGON_PAID=1`. published_utc -> created_at; never `updated_at`.
- `news_alpha_vantage.py` — free 25 req/day NEWS_SENTIMENT, 4y back to 2022-03,
  journaled cursor at `data/raw/alpha_vantage_state.json` for daily-cron resume
- `news_reddit.py` — DuckDB query against HF `open-index/arctic` (mirror covers
  2005-2017 only — owner downloads post-2017 Arctic Shift torrents for V1)
- `news_multisource.py` — HF `Brianferrell787/financial-news-multisource`
  (57M rows 1990-2025, NON-COMMERCIAL gated, HF-token gated dataset)
- `news_central_bank.py` — extended with 5 regional Fed scrapers
  (Cleveland/Chicago/NY/SF/Atlanta — selectors per-Fed needed, 0 rows so far)
- `news_fnspid.py` — license bug fixed (CC BY 4.0 -> CC BY-NC-4.0),
  `NANOGLD_NONCOMMERCIAL=1` gate, parallel `Dataset.filter(num_proc=4)` perf fix
- `cli.py` — SOURCES dict gains polygon_news/alpha_vantage/multisource;
  KEYLESS_SOURCES expanded for HF-backed sources

**Pulls run + landed:**
| source | rows on disk | notes |
|---|---:|---|
| central_bank_news.parquet | 4988 | refreshed; regional Feds returned 0 (selector work needed per-Fed) |
| (other news parquets) | not yet | running pulls hit timeouts / API gates — see owner actions below |

**Test count:** 49 (data) + 14 (features) + 13 (news pipeline) = **76 pass**, 6 skip.

**Owner-action checklist (continue news work):**

1. Sign up at https://www.alphavantage.co/support/#api-key — email-only,
   instant. Paste `ALPHA_VANTAGE_API_KEY=...` into `~/.config/nanogld/.env.paper`.
2. Add `HF_TOKEN=...` to `.env.paper` (huggingface.co/settings/tokens, read
   scope, free). Unblocks gated `Brianferrell787/financial-news-multisource`.
3. Enable BigQuery API in GCP `nanogld` project (one-click):
   https://console.developers.google.com/apis/api/bigquery.googleapis.com/overview?project=nanogld
   Then re-run `NANOGLD_GCP_PROJECT=nanogld python -m nanogld.data pull gdelt_materialize`.
4. (Optional) Decide on $29/mo Polygon News Stocks Starter — set
   `NANOGLD_POLYGON_PAID=1` + `POLYGON_API_KEY=...` if yes.
5. Wayback CDX backfills (Kitco/Investing/BullionVault) — each is a multi-hour
   soak at 2 s/req polite rate. Run overnight: `python -m nanogld.data pull
   kitco --force` etc. Cache resumes cleanly across runs. Free.
6. FNSPID HF download is ~50 GB before filter applies — owner runs
   `NANOGLD_NONCOMMERCIAL=1 python -m nanogld.data pull fnspid` overnight
   when bandwidth permits.
7. Reddit — **SKIPPED for V1** (owner decision 2026-05-04). Full firehose torrents
   too big for 67 GB free disk. Selective per-subreddit torrent only covers
   2021-04 → 2023-12 of V1 window. Wire news + central bank + Kaggle + multisource
   provide enough density without retail social. Revisit after backtest results.

---

## V1 Data Gap-Fill (2026-05-05) — owner directive: "fill any data gap"

Audit revealed two NaN-warm-up bugs + missing GDELT tone + missing dims +
incomplete doc 03 prep. All filled. Snapshot hash advances 79678e1d → 11b24f78.

### Bugs fixed

| Issue | Was | Now |
|---|---|---|
| **DXY warm-up** (DTWEXBGS post-2020 only) | 24,387 NaN bars 2016-2019 (32% of data) | yfinance DX-Y.NYB, 0.2% NaN, full 10y. Spec deviation: V1 substitutes ICE futures DXY for FRED DTWEXBGS. |
| **SOFR sparseness** (only post-2018-04) | fred_sofr 30% NaN | dropped from FRED_SERIES_V1 (was 35→33 series). DFF covers daily Fed Funds role per spec line 21. |
| **GDELT tone missing** | spec mandated tone_mean / tone_std (most validated GDELT signal); per-bar parquet had counts only | gdelt_tone_mean / gdelt_tone_std / gdelt_polarity_mean / gdelt_tone_mean_gold / gdelt_tone_mean_conflict (5 cols, <2% NaN per year) |
| **GDELT 5y window** (only 2021-2026) | 50% snapshot bars had 0 article_count | materialized `gkg_5y_pre` table covering 2016-01-04 → 2021-04-23 (119M rows, 1664 GB scan). Re-aggregated full 10y per-bar parquet (177,867 bars). |
| **Missing feature dims** | price 11/12, macro short 6/12, equity ratios 7/9 | added macd_hist, bb_width, dxy_level, vix_log_change_1d, brent_log_change_1d, wti_log_change_1d, gold_silver_log_ret_30d, slv_gld_corr_30d. New total: 367 cols (was 356). |
| **Late-window news** (2024-2026 thin) | 2024 = 398 articles, 2025 = 92, 2026 = 1 | AV multi-topic backfill: 5,107 articles 2024 + 1,534 articles 2025. 2026 awaiting AV daily quota reset. Free tier 25 calls/day, journaled cursor. |

### Doc 03 (news embedding) inputs ready

- `data/processed/news_corpus_v1.parquet` — **31,840 unique articles**, 68% with body, 8 sources unified, schema-conforming, deduped (URL-aware, body-preferred), V1 window filtered.
- `data/processed/news_source_registry.json` — 30 sources × 12 bias_tiers per doc 03 V4 spec. Used by Flamingo gate as source-conditional bias.
- `data/anchors/v1_templates.json` — 4 anchor categories × 20 templates each (V4 leakage-fixed: hand-crafted, no event provenance).
- `data/embeddings/` — empty dir created for doc 03 outputs.

### Final snapshot v1_11b24f78a3ad6ea9

- 75,673 bars × **367 cols**, 2016-01-04 → 2026-04-24, 2,747 trading days
- splits: train 57,696 (76%) / val 7,540 (10%) / test 10,437 (14%)
- class balance: DOWN 29.1% / FLAT 39.9% / UP 31.0% (target 28/44/28)
- **first all-clean row: idx 2,676 = 2016-05-23** (vs prior 24,387 = 2019-06-24 — 9× warm-up reduction)
- tests: 88 pass / 4 skip / 0 fail
- News bar coverage: 9% of bars have ≥1 article; 13.9% in 2016, 16.7% in 2024 (best); 2.9% in 2025 (will improve with daily AV pulls)

### Costs incurred

- BigQuery scan (GDELT pre-2021 materialize + tone re-agg + per-bar full 10y): ~2,200 GB, ~$5-8 over free tier (one-time backfill).
- AV API: 25 calls/day free quota — currently exhausted. Resumes 2026-05-06.
- Wayback IA: hit 504 + connection errors during attempted re-pulls. IA infra issues today; kitco/investing/bullionvault remain at 24/11/26 articles. Multisource (39K) + FNSPID (19K) + AV (6K) + central_bank (5K) + polygon (863) compensate. Re-pull via Wayback can be retried later when IA stabilizes.

### Spec deviations (logged here for traceability)

1. DXY = yfinance DX-Y.NYB (ICE futures), not FRED DTWEXBGS. Pre-2020 DTWEXBGS had no real PIT data.
2. SOFR dropped (FRED_SERIES_V1 size 35 → 33). DFF covers daily-rate role.
3. AV news pulled via topic filter (finance/monetary_policy/economy_macro/...) instead of ticker filter, since gold-ticker filter returned 0 articles for 2024+ on the free tier.

### What's queued

- AV daily pull (25 calls × 7 topics @ 3 cursors) — resumes when quota refreshes; will further fill 2025-2026 gap. Persisted via journaled cursor in `data/raw/alpha_vantage_state.json`.
- Wayback re-pull (kitco/investing) — IA infra was 504-ing; retry overnight.

---

## Phase 3 — Embedding (2026-05-06)

**Doc 03 precompute pipeline shipped + running on Mac mini.**

### Code (committed cc3ad02 + this session)

`src/nanogld/embed/` — 12 modules:
- `config.py` — frozen EmbedConfig + content-addressed run_hash
- `registry.py` — SOURCE_REGISTRY loader + alias resolver (colon, underscore, UNK fallback)
- `text_builder.py` — title+body construction + NFC-normalized text_hash
- `checkpoint.py` — atomic JSON Progress + errors.jsonl
- `qwen3_embedder.py` — sentence-transformers wrapper (FP16 MPS, MRL, L2-norm, fallback model)
- `anchors.py` — 80 templates → 4 vectors → v1.npz
- `precompute.py` — main loop: sharded, resumable, OOM-tolerant
- `cli.py` + `__main__.py` — `python -m nanogld.embed {precompute|anchors|verify}`
- `bar_index.py` — bar→article_ids window index (NEW for doc 04 hand-off)
- `anchor_cosines.py` — per-bar cosine features (NEW pre-build for doc 04)

Tests: 17 new (test_embed_bar_index 9, test_embed_anchor_cosines 8). All green.

### Mac mini run (in progress)

| Step | Status |
|---|---|
| rsync corpus + code → Mac mini | ✅ |
| uv sync on Mac mini | ✅ |
| Qwen3-Embedding-4B downloaded (8 GB) | ✅ |
| anchors → data/anchors/v1.npz | ✅ |
| smoke 64 articles (anchors cohesion 0.469-0.526, gate 0.40 PASS) | ✅ |
| probe 500 articles bs=16 (10.5 min, 0 OOM, 1.26 sec/article) | ✅ |
| **full 31,840 article precompute** (nohup+caffeinate) | 🟡 running |
| ETA | ~13 hours total wall, started 12:58 PDT |
| Heartbeat | data/embeddings/.heartbeat (touched every 60s) |
| Sentinel | data/embeddings/.done OR .failed |

### Hardware decision

**Mac mini stays as data + embedding box.** Training goes to **rented 1× H100 ($1.99/hr RunPod Community)**.

Reasoning:
- Mac mini 16 GB unified RAM caps practical training at ~30-40M FP32 (60M = OOM with SAM + EMA + AdamW state)
- MPS gradient reliability for transformer training unverified at 30M+ scale (PyTorch issues #103343 silent grad divergence, #157345 RMSNorm slow)
- Wall-clock estimate on Mac mini: 4-10 days with restart risk vs 3-5 hours per H100 run

### V1 model size — locked

**24-40M params (D=384, 12 layers, 6 heads, channel-token T=64).**

**Why not bigger:**
- Industry: Lag-Llama 2.5M, TLOB 10M, Chronos-Bolt-base 205M, TimesFM 2.5 200M (halved from 500M, got better)
- Forecast-to-Fill (our Sharpe 2.88 target) uses ZERO neural net — just EWMA + Kelly
- Single-asset 30min direction: ~33K bars × 50+ features → Chinchilla-optimal compute caps at 30-80M
- TLOB lesson: 10M model on RTX 3090 beat SOTA F1 by 3.7
- Hard rule still applies: if 30M transformer doesn't beat XGBoost + DLinear baselines by ≥0.2 Sharpe → ship the baseline

### Cost-of-fund (V1 + V2 if scale)

| Item | Cost |
|---|---|
| Embedding (Mac mini, free) | $0 |
| V1 training (1× H100, ~10 hyperparam runs × 3-5h × $1.99) | $60-150 |
| V2 if scale to 200M (5 runs × 30h × $1.99) | $200-400 |
| Backtest (Mac mini, free) | $0 |
| **Total V1+V2** | **$300-600** |

Rounding error. Don't optimize.

### Timeline to first ship

- **2 days** to first trained model on H100 after embedding completes
- **14-21 days** to ship-ready calibrated model (V1 baseline + hyperparam sweep + walk-forward backtest + isotonic/Platt calibration + drift audit)

### Story (publishable)

**"30M from-scratch encoder + Qwen3-Embedding-4B news fuser beats hand-tuned Forecast-to-Fill EMA on 10y of 30min GLD bars (Sharpe ≥ 2.88, DSR ≥ 1.0). Sub-$500 compute budget."**

Lab-level. Beats industry-standard hand-tuned alpha. Engineering > scale.

---

## Data Phase Achievements (2026-05-04 → 2026-05-08)

### What got built

**Asset universe (27 instruments × 30-min × 10y where possible):**
- Metals: GLD, SLV, GDX
- Indices: SPY, QQQ, IWM, DIA, VTI, EEM, EFA
- Sectors: XLE, XLF, XLK, XLU
- Treasury: TLT, IEF
- Real estate: VNQ, IYR
- Oils: USO (WTI), BNO (Brent), UNG (nat gas)
- Volatility: VXX (2018+)
- Crypto: BTC, ETH (2016+, Bitfinex), XRP (2017+), ADA, SOL, DOGE (2021+)

**Macro (40 FRED series):** full Treasury curve + TIPS + breakevens, inflation (CPI, PCE), employment (UNRATE, PAYEMS, ICSA, CCSA, JOLTS), Fed (DFF, M2, WALCL, RRP), growth (GDP, INDPRO, RSAFS, HOUST, UMCSENT), retirement (PSAVERT, DSPI, DSPIC96), real estate (CSUSHPISA, MORTGAGE30US, HSN1F), oil daily, VIX daily.

**Other:** DXY dollar index daily, COT positioning weekly, WGC central bank quarterly, GPR geopolitical risk daily, GDELT 30-min tone aggregates (66M raw → 177K), 752 calendar events.

**News (40,032 articles embedded):**
- FNSPID 12,301 + Multisource HF 6,683 + ECB+Fed speeches 1,881 + Multisource Benzinga 1,756
- Polygon (Benzinga, Zacks, Seeking Alpha, MarketWatch) ~750
- Alpha Vantage ~1,000 across multiple wires
- Fox News + Fox Business 8,192 (Common Crawl WARC range-fetch)
- Kitco, BullionVault, central_bank_news ~5,000

All embedded with **Qwen3-Embedding-4B** (frozen, FP16 MPS) + MRL truncation 2560→256 + L2-norm.

**v2 engineered features (46 added on top of v1 base):**
- Cross-asset interactions (4): flight-to-safety, digital-gold-rotation, real-rate × dollar, pm-cohesion
- Volatility regime (5): VRP, vol-of-vol, VIX z-scores, RV breakout
- Calendar (7): NFP/CPI windows, London Fix, FOMC time decay, weekend, quarter-end
- Regime + microstructure (8): bull/bear MA-cross, drawdown, volume z-score, tick imbalance, VWAP deviation, variance ratio 8/48
- Macro term structure (6): 2y10y inversion + persistence, term premium, mortgage spread, recession flag
- Momentum (10): 6 extended log-ret horizons + acceleration ratios + cross-horizon agreement + trend efficiency + mean-reversion z + silver lead-lag
- News × price (5): velocity ratio, news-price interaction, sentiment momentum, recession-monetary spread
- Anchor cosines (4 anchors × 3 stats + 2 visibility = 14)

### Bug-hunt iterations (paranoid audit loop, 8 agents per iter)

**Iteration 1:** 9 bugs found + fixed
- 6 leakage bugs (bull_bear_regime, drawdown, volume_z, tick_imbalance, vwap_dev, mean_rev_z used current bar — added .shift(1))
- VRP annualization off by 630× (fixed to ×252)
- INDPRO recession fallback wrong window
- 11 features missing (col name mismatch — `gld_log_ret_X` vs `log_return_X`)
- timestamp_x/timestamp_y merge collisions
- spy_rs_spy_24 zero-variance constant
- 3,309 inf values from rrpontsyd_*_change (fixed: replace inf → NaN)
- label dtype float64 → int8

**Iteration 2:** 3 bugs found + fixed
- rv_breakout_flag warmup all 0 (boolean cast lost NaN)
- cross_horizon_agreement warmup all 0 (sign(NaN)=0 issue)
- label dtype Int8 nullable → int8 non-nullable (after dropping 1 NaN row at very end)

**Iteration 3:** 5 bugs found + fixed
- All v2 cross-asset features used `fillna(0)` then arithmetic, masking warmup as fake zeros (flight_to_safety, digital_gold_rotation, real_rate_dollar, pm_cohesion, vrp). Dropped fillna(0) so NaN propagates.

**Total bugs fixed: 19. All v2 features now propagate NaN correctly during warmup.**

### Final unified dataset (single file)

```
/Users/root1/Desktop/nanogld/data/processed/training_v1_unified.pt
```

234 MB PyTorch native. Loadable in 1 line: `data = torch.load(path)`.

Contains: features (75993, 681) float32 + labels (75993,) int8 + splits + bar_close_utc_ns + 40032 article embeddings (256-dim FP16) + bar→article CSR index + meta dict.

Read `plan/HANDOFF.md` before training.

### Verification (last paranoid loop iteration)

| Check | Result |
|---|---|
| Future leakage | **0 violations** (strict-< t_visible) |
| FRED PIT correctness | 0 violations across sampled series |
| gld_lag1_close == prior bar close | 0 mismatch |
| Rolling/shift operations PIT-correct | 48/48 safe |
| Max \|corr\| feature × next_log_return | 0.0117 (research-typical, no leakage) |
| 100% NaN cols | 0 |
| Inf values | 0 |
| Constant cols | 0 |
| Duplicate bars | 0 |
| Chronological order | True |
| Calendar/event window timing | All PASS |
| Train/Val/Test chronological non-overlap | True |
| Dangling article references | 0 |

### Cron audit loop active

Job ID `2dc09613`. Recurring `*/10 * * * *` (every 10 min). 8 verification agents per iteration. Auto-expires 7d. Cancel with `CronDelete 2dc09613` once model training starts.

### Tooling for next agent

- Mac mini SSH: `ssh -o StrictHostKeyChecking=no root1@100.83.86.5` (Tailscale 100.83.86.5)
- All code under `~/Desktop/nanogld/` synced from local laptop
- `uv` at `~/.local/bin/uv`, Python 3.11, PyTorch 2.11 with MPS available + built
- 16 GB unified RAM (training >30M params requires H100)

---

## Final Notes

The plan has been verified across **5 rounds + V1 redline = 36 specialized Nia agents (27 V1 + 9 V1), ~80 critical findings absorbed**. Every load-bearing claim has citation. Every architectural decision has been challenged. Every doc is now a self-contained spec for one Opus 4.7 agent.

**The field moves weekly.** Always run a fresh Nia search before starting your section. Plan is a snapshot, not a contract.

**V1 is the new lock.** Agents do NOT replan further without owner approval. If a V1 spec line is wrong, document the issue and AskUserQuestion before changing anything. Same Iron Law as V1.

**Agents: read `plan/V1-SPEC.md` first, then 00-OVERVIEW.md, then your assigned doc. Spawn Nia agents freely. Build the thing.**
