# STATUS — nanoGLD Implementation Tracker

**Project:** nanoGLD
**Owner:** samsiavoshian
**Date last updated:** 2026-05-01
**Phase:** Planning complete. Implementation phase begins.
**Version:** **V1 — frozen.** No version bump until owner explicitly says so. Agents do NOT introduce V2/V3/etc references. Past planning iterations collapsed into V1.

---

## Quick Status

```
Planning:        ██████████████ 100% locked, Nia-verified across 3 rounds (19 agents)
Implementation:  ░░░░░░░░░░░░░░   0% — agents ready to be dispatched
```

Estimated implementation: **~14-16 days** end-to-end. **Sequential** — one agent per doc, hand off when done. **8 docs after V5 merge** (was 11; merged model+train+calib into doc 05, sizing+exits into doc 07).

---

## Doc Status (each doc owned by one Opus 4.7 agent)

| Doc | Owner role | Status | Effort | Blocked by |
|-----|-----------|--------|--------|-----------|
| 00 OVERVIEW | n/a | ✅ Read-first reference | n/a | n/a |
| 01 INFRA-AND-SECURITY | DevOps | ✅ Spec ready | 0.5 day | nothing — START HERE |
| 02 DATA-PIPELINE | Data engineer | ✅ Spec ready (V1 expanded 2026-05-04) | **4-5 days** | doc 01 |
| 03 NEWS-EMBEDDING | ML engineer | ✅ Spec ready (V1 Qwen3 + V4 expanded pipeline + LAFTR + new aggregator) | **1.5 day** setup + ~120min precompute | doc 02 |
| 04 FEATURE-ENGINEERING | Feature engineer | ✅ Spec ready (V1 expanded 2026-05-04) | **1.5 days** | doc 03 |
| 05 MODEL-TRAINING-CALIBRATION | ML systems engineer | ✅ Spec ready (V5 merge: model + training + calibration in one file) | **3 days** | doc 04 |
| 06 BACKTEST | Quant engineer | ✅ Spec ready | 1 day | doc 05 |
| 07 SIZING-AND-EXITS | Quant risk engineer | ✅ Spec ready (V5 merge: sizing + per-trade SL + profit-take + drawdown circuit-breaker) | **2 days** | doc 06 |
| 08 LIVE-TRADING | Production engineer | ✅ Spec ready | 1.5 days | doc 07 |

**V5 Merge (2026-05-04):** old docs 05+06+07 merged into the new doc 05 (MODEL-TRAINING-CALIBRATION). Old docs 09+10 merged into the new doc 07 (SIZING-AND-EXITS). One agent owns each merged doc end-to-end. Each doc's `Blocked by` is now exactly the immediately-preceding doc — pure linear chain.

**Deleted docs (V1 cleanup, May 1):**
- ~~old `08-RL-STAGE3.md`~~ — RL deferred to V2.
- ~~old `11-X-THREAD-AND-BLOG.md`~~ — content strategy, owner writes himself.

---

## Execution Mode (every agent reads 00-OVERVIEW.md "Execution Mode" section)

**These docs ARE the plan. Do not replan, do not rewrite, do not redesign.** The user said explicitly: *plan to execute the docs, do not replan*. Silent scope drift = fired. If a doc claim is wrong, document the issue and AskUserQuestion before changing anything.

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

## V1 Architecture Locked (May 1, 2026)

**Backbone:** encoder-only transformer, ~24-60M params, channel-group tokens (~14)
**Per-block:** RMSNorm + SwiGLU + RoPE (real-form) + QK-Norm + per-head gating + value residuals + no-bias
**News fusion:** Qwen3-Embedding-4B → Perceiver-Resampler-lite → Flamingo-gated cross-attn
**Training:** SSL-MAE → linear-probe → LLRD fine-tune, Schedule-Free AdamW + Friendly-SAM ρ=0.05 + EMA 0.999
**Loss:** 3-class cross-entropy + label smoothing 0.1 (NEVER MSE on returns — forecast-collapse rule)
**Sizing:** Stage 2 = vol-target × Kelly-lite × conformal-confidence
**Backtest:** vectorized engine, bars_per_year=3276, baselines = DLinear/TSMixer/TimeMixer/xLSTMTime/XGBoost/Forecast-to-Fill replica, DSR + bootstrap CI
**Live:** launchd StartCalendarInterval cron on Macbook M4 Pro, alpaca-py >=0.43, pmset sleep prevention
**Infra:** PyTorch 2.11.0 pinned, FP32 weights, gitleaks before first commit, ADC for GCP, 1Password for live keys

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

## Empirical Bar (success criteria)

| Tier | Threshold | Status to ship |
|------|-----------|---------------|
| Minimum viable | 38% accuracy + Sharpe > 0 + beat XGBoost | Mandatory |
| Real claim | Sharpe > 1.0 + DSR > 1.0 + beat ALL baselines by ≥0.2 | Mandatory for X thread |
| Forecast-to-Fill tier | Sharpe > 2.5 + MDD < 5% | Publishable contribution |

**The actual bar:** GLD 5y buy-and-hold Sharpe ≈ 2.4 (2020-2025 was a great gold run). nanoGLD must beat this honestly to claim alpha. Forecast-to-Fill (Sharpe 2.88) on gold remains UNREPLICATED in 2026 — building this honestly is publishable.

---

## Hard Project Rules (apply across all docs)

1. **NEVER use MSE on returns** (forecast-collapse, arXiv:2604.00064)
2. **STAY FROM-SCRATCH** (no TS foundation model fine-tune, arXiv:2511.18578)
3. **SHIP THE SIMPLER MODEL IF IT TIES** (TLOB lesson)
4. **APPLY PEER-BENCHMARK DISCOUNT** to backtests (arXiv:2604.18821)
5. **bars_per_year = 3276** (NYSE RTH only)
6. **Point-in-time correctness on every feature** (`.shift(1).rolling(...)`, news buffer = 30min for GDELT, 60s for Alpaca News, bar visibility = `bar.timestamp + bar_duration`)
7. **gitleaks BEFORE first commit** (verify with fake key)
8. **PyTorch 2.11.0 pinned** (SDPA fix #174945 for MPS)
9. **FP32 weights everywhere** (no autocast, no torch.compile, no quantization)
10. **`.contiguous()` Q/K/V before SDPA** (PyTorch #181133)
11. **Every feature row carries `t_visible`** column. CI gate: `test_release_ts_lte_t_visible_all_rows`.
12. **ALFRED `get_series_all_releases` for ALL FRED series** (CPI/PCE annual revisions silently rewrite 5y of history)
13. **pandas-ta KAMA/Ichimoku/KST/DPO/TRIX/Vortex FORBIDDEN** (look-ahead bugs, bukosabino/ta#181)
14. **Calendar features = binary windows ONLY** (no `minutes_until_event`)
15. **Anchor-cosine anchors = hand-crafted templates OR pre-train-period samples**
16. **Alpaca News field = `created_at`** (`published_at` does NOT exist; never join on `updated_at`)
17. **`DFF` for daily Fed Funds, NOT `FEDFUNDS`** (FEDFUNDS is monthly)

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

## Final Notes

The plan has been verified across **3 rounds, 19 agents, ~30+ critical findings**. Every load-bearing claim has citation. Every architectural decision has been challenged. Every doc is now a self-contained spec for one Opus 4.7 agent.

**The field moves weekly.** Always run a fresh Nia search before starting your section. Plan is a snapshot, not a contract.

**Agents: read 00-OVERVIEW.md first. Then your assigned doc. Spawn Nia agents freely. Build the thing.**
