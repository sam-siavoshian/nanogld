# 01 — Data Pipeline

## YOU ARE THE DATA ENGINEER AGENT

You own data ingestion, joining, and snapshot artifacts. You will build the pipeline that produces immutable hashed parquets every other doc consumes.

**Read 00-OVERVIEW.md FIRST.** That doc has full project context — nanoGLD, hardware constraints, locked architecture, hard rules. This doc assumes you've read it. **Also read the "Execution Mode" section in 00-OVERVIEW.md before coding.** Summary repeated below.

### Execution Mode (short version — full rules in 00-OVERVIEW.md)

**This doc IS the plan. Do not rewrite it.** Plan to *execute* what is written. If a claim is wrong, document the issue and AskUserQuestion. Silent scope drift is a fireable offense.

- **Research with Nia** before guessing on any library, API, or paper claim. Spawn a subagent to run `nia papers` / `nia github` / `nia search` / `nia oracle` / `nia packages` / `nia tracer`. Run `nia auth` once if any command errors on auth.
- **Use gstack execution skills only:** `/investigate`, `/review`, `/qa`, `/qa-only`, `/cso`, `/benchmark`, `/ship`, `/land-and-deploy`, `/canary`, `/document-release`, `/retro`, `/learn`, `/browse`, `/setup-browser-cookies`.
- **Do NOT use planning skills:** `/office-hours`, `/plan-*`, `/autoplan`, `/design-*`, `/devex-review`. Planning is done. Calling them wastes tokens and risks silent scope drift.
- **Loop:** read doc → Nia for unknowns → code → `/review` → `/qa` → `/cso` (if security-sensitive) → `/ship`.
- **Escalate after 3 failed attempts.** AskUserQuestion. Bad work is worse than no work.

### Storage Plan (V1 — checked May 2026, owner's machine)

**Owner disk check at planning time:**
- Total: 494 GB Mac internal
- Free: ~67-72 GB
- 60% of free (storage budget): **~40 GB**
- Estimated data footprint: **~25-30 GB** (post-2026-05-04 V4 dataset + news-pipeline expansion)
- Utilization: 28 / 40 = **70% of budget** → KEEP LOCAL but watch carefully

V4 storage breakdown:
- Original snapshot + GLD bars + ETF basket + FRED ALFRED cubes + GDELT + GPR + COT + WGC: ~10-12 GB
- FNSPID gold-relevant subset: ~5 GB
- Kitco scrape (10y): ~1 GB
- Investing.com scrape (10y): ~2 GB
- BullionVault scrape: ~0.5 GB
- CNBC scrape + Wayback backfill: ~2 GB
- Central bank speeches + government press releases: ~0.5 GB
- Reddit Arctic Shift filtered: ~5 GB
- Kaggle gold-labeled: ~0.05 GB
- Per-article embeddings parquet: ~2 GB

### Dataset Expansion (2026-05-04 — owner directive)

Owner directed expanding dataset to capture more market drivers. Approved scope:

| Bundle | Added | Reason |
|---|---|---|
| **Equity ETFs (Source 8)** | SPY, QQQ, IWM, GDX, SLV, XLF, XLE, XLK, XLU — 30m bars via Alpaca | Risk-on/off, sector rotation, GDX (gold miners) + SLV (silver) = direct gold cross-correlation |
| **Treasury curve (Source 4 expansion)** | DGS3MO, DGS6MO, DGS5, DGS30, DFII5, DFII10, T10YIE, T5YIFR | Real rates = #1 gold driver. Full curve + TIPS direct + 5Y forward inflation. |
| **Macro economic (Source 4 expansion)** | UNRATE, PAYEMS, ICSA, CCSA, JTSJOL, CPIAUCSL, CPILFESL, PCEPI, PCEPILFE, GDPC1, INDPRO, RSAFS, HOUST, UMCSENT, M2SL, WALCL, RRPONTSYD, FEDFUNDS, SOFR | Each macro release moves gold. Vintage-correct via ALFRED. |
| **CFTC COT (Source 9)** | Weekly speculator positioning for COMEX gold futures | Predictive at extremes; non-commercial net long |
| **WGC central bank flows (Source 10)** | Quarterly net central bank gold purchases | Slow signal, structurally bullish gold when positive |
| **Calendar event schedule (Source 11)** | Pre/post-release windows for FOMC, CPI, NFP, GDP, JOLTS + cyclicals | Captures known information-flow patterns; deterministic schedule, no API |

**Deferred (owner did NOT select — leave in TODOs):**
- GVZ (CBOE Gold VIX), HY/IG OAS credit spreads, MOVE bond vol
- USD direct cross-rates (DEXUSEU/DEXJPUS/DEXCHUS/DEXUSUK)
- Crypto (BTC/ETH)
- Industrial metals (HG=F copper, PL=F platinum, SI=F silver futures)

These are documented in "Open Questions / TODOs" at the bottom. Re-ask after V1 baseline lands.

**New per-bar feature dim:** ~1000 (was ~804). Channel-group count grows from ~14 → ~25. Model architecture (doc 05) unchanged — input projection layer just gets wider (~75K extra params, trivial).

---

## Verification Round 4 (2026-05-04) — Source Corrections + Leakage Audit

**5 Nia subagents** verified every source against live APIs/docs. Found **17 high-severity issues** that would silently leak future information into the model. **Every issue below MUST be implemented.** Hard rules.

### CRITICAL CORRECTIONS (silent killers)

**1. Alpaca bar `t` is bar START not bar END.** Bar with `t=09:30:00` covers half-open interval `[09:30, 10:00)`. OHLCV reflects all trades in that window. **A 30m bar at `t=09:30` is NOT safe to use as a feature for any decision before `t + 30min = 10:00`.**

```python
# WRONG — leaks 30 min of future
features.iloc[T] = bars.iloc[T].close   # bar `t=09:30` close knows trades through 10:00

# RIGHT — define visibility
bars["t_visible"] = bars["timestamp"] + pd.Timedelta(minutes=30)
features = features.merge_asof(bars, left_on="t_visible", direction="backward",
                               allow_exact_matches=False)
```

Apply identically to GLD AND all 9 ETFs.

**2. Alpaca News field is `created_at` (NOT `published_at`).** The `published_at` field does not exist. Use `created_at` only. Never join on `updated_at` (drifts forward on edits — guaranteed leak).

```python
NEWS_LATENCY_MIN = 1   # 60s safety buffer
news["t_visible"] = news["created_at"] + pd.Timedelta(minutes=NEWS_LATENCY_MIN)
# NEVER join on news["updated_at"]
```

**3. FEDFUNDS is MONTHLY, not daily.** Replace with `DFF` (Effective Federal Funds Rate, daily) for any daily PIT signal. Keep FEDFUNDS only for monthly aggregates.

**4. FRED `realtime_start` is DATE-PRECISE, not timestamp-precise.** Need static `release_tod_et` lookup per series. Bar at 07:00 ET on date D MUST NOT use a series that publishes at 08:30 ET on date D, even though both have `realtime_start = D`.

```python
# Static release-time table — built from BLS/BEA/Fed schedules
FRED_RELEASE_TOD_ET = {
    "DGS3MO": time(16, 15),  # H.15 daily ~4:15 PM ET
    "DGS6MO": time(16, 15),
    "DGS2":   time(16, 15),
    "DGS5":   time(16, 15),
    "DGS10":  time(16, 15),
    "DGS30":  time(16, 15),
    "DFII5":  time(16, 15),
    "DFII10": time(16, 15),
    "T5YIE":  time(16, 15),
    "T10YIE": time(16, 15),
    "T5YIFR": time(16, 15),
    "DTWEXBGS": time(16, 15),  # H.10
    "VIXCLS": time(8, 37),     # CBOE close + FRED next-morning ingest (defensive)
    "DCOILBRENTEU": time(16, 0),  # EIA spot
    "DCOILWTICO": time(16, 0),
    "UNRATE": time(8, 30),     # BLS Employment Situation, 1st Friday
    "PAYEMS": time(8, 30),
    "ICSA":   time(8, 30),     # DOL Thursday
    "CCSA":   time(8, 30),
    "JTSJOL": time(10, 0),     # BLS JOLTS
    "CPIAUCSL": time(8, 30),   # BLS CPI mid-month
    "CPILFESL": time(8, 30),
    "PCEPI":  time(8, 30),     # BEA Personal Income
    "PCEPILFE": time(8, 30),
    "GDPC1":  time(8, 30),     # BEA GDP — QUARTERLY!
    "INDPRO": time(9, 15),     # Fed G.17 mid-month
    "RSAFS":  time(8, 30),     # Census mid-month
    "HOUST":  time(8, 30),
    "UMCSENT": time(10, 0),    # UMich
    "M2SL":   time(13, 0),     # H.6 ~1:00 PM ET
    "WALCL":  time(16, 30),    # H.4.1 Thursday — CRITICAL: a Thursday RTH-close 16:00 bar must NOT use this week's level
    "RRPONTSYD": time(13, 30), # NY Fed TOMO
    "DFF":    time(9, 0),      # NY Fed next-business-day
    "FEDFUNDS": time(9, 0),    # Monthly aggregate, posted ~1st BD of next month
    "SOFR":   time(8, 0),      # NY Fed previous-BD
}

def feature_visible_at(series_id: str, observation_date: date, now_et: datetime) -> bool:
    """Returns True iff the value is publicly available at now_et."""
    release_dt = datetime.combine(observation_date + timedelta(days=1), FRED_RELEASE_TOD_ET[series_id], tzinfo=ZoneInfo("America/New_York"))
    return now_et >= release_dt
```

**5. CPI/PCE annual seasonal-factor revisions silently rewrite 5y of history.** Every February (CPI) and August (PCE), BLS/BEA rebenchmarks seasonal factors. ALFRED captures revision rows with distinct `realtime_start`. **MUST use vintage queries** (`get_series_all_releases`), NEVER current snapshot. Same applies to UNRATE annual revision and PAYEMS annual benchmark (Q1).

**6. GDELT theme codes — 6 REFUTED, must remove or fix:**

| Original code | Status | Fix |
|---|---|---|
| `EPU_CATS_MONETARY_POLICY` | REFUTED | Remove — not in GKG-MASTER-THEMELIST |
| `EPU_POLICY_FEDERAL_RESERVE` | REFUTED | Remove |
| `EPU_UNCERTAINTY` | REFUTED | Remove |
| `EPU_ECONOMY_HISTORIC` | REFUTED | Remove |
| `TAX_WEAPONS_BOMB` | REFUTED | Replace with `TAX_WEAPONS` (canonical) |
| `WB_2432_FRAGILITY` | REFUTED | Replace with `WB_2432_FRAGILITY_CONFLICT_AND_VIOLENCE` |

**7. GDELT publication latency = 30min, not 15min.** GDELT 2.0 publishes every 15 min, but slot `15:00-15:15` lands ~15:30 UTC and timestamps reflect article scrape times within the slot. Use `news.DATE <= T_close - 30min` (was 15min). Bump `NEWS_LATENCY_MIN = 30` for GDELT.

**8. WGC URL was WRONG.** Actual:
- Quarterly time series since 2000: `https://www.gold.org/download/8052`
- Latest official reserves: `https://www.gold.org/download/7739`

WGC is **MONTHLY**, not quarterly. ~2-month IMF reporting lag. **NO PUBLIC VINTAGE ARCHIVE** — must self-snapshot weekly into immutable store.

**9. AI-GPR daily index is NOT real-time** — 30-day lag confirmed (today=2026-05-04, latest row = 2026-03-31). Treat as monthly with ~30-day lag, NOT same-day.

**10. GPR monthly has no vintage archive** — Caldara/Iacoviello revise history when methodology updates (e.g., 2026-04-24 revision). Self-snapshot weekly with fetch-date keying for honest backtests.

**11. pandas-ta has confirmed look-ahead bugs** (`bukosabino/ta#181`): KAMA, Ichimoku (`visual=True`), KST, DPO, TRIX, Vortex. Forbid these. RSI / MACD / BBANDS proper are causal IF `min_periods` respected and no `bfill`. Add growing-window-stability test for every indicator: `f(close[:N])[-1] == f(close[:N+k])[N-1]` for k in [1, 5, 50].

**12. Multi-symbol Alpaca pagination INTERLEAVES symbols.** Returned page is sorted by symbol first, then timestamp. A partial page is NOT all-symbols-up-to-T. **MUST drain all pages before constructing time-aligned panel.**

**13. `adjustment="all"` is retroactive.** Today's split-adjusted history embeds future split knowledge into past dates. For BACKTESTING:
- Either use `asof=<as_of_date>` parameter to retrieve historically-adjusted snapshots, OR
- Pull `raw` prices + corporate actions feed and apply adjustments forward only up to simulated decision time.

V1 default: pull `adjustment="all"` for training (single split-adjusted history) but acknowledge in DEVIATION section. Real fix is forward-only adjustment.

**14. CFTC 2025 government shutdown.** Caused multi-month gap in COT publication (announcements 9138-25 + 9147-25). Some 2025 weeks have irregular release cadence. Either skip those weeks in training OR explicitly flag with `irregular_release=True`. Verify each row's release date against CFTC Special Announcements page.

**15. CFTC contract identifier**: `GOLD - COMMODITY EXCHANGE INC.`, contract code `088691`. Confirmed exact string. Holiday-Friday → Monday release.

**16. WALCL Thursday 4:30 PM ET release.** A Thursday RTH-close (16:00) bar MUST NOT use that week's level. Use prior week's. Strict timestamp comparator, not date.

**17. ICSA Thursday 8:30 AM ET embargoed release.** Bar at 09:30 RTH open OK, bar at 08:00 pre-market NOT.

### Mandatory Test Suite for Leakage (`tests/test_no_leakage.py`)

doc 04 owns the implementation. The test list is the contract:

```python
def test_bar_visibility_is_bar_end()                # §1
def test_news_uses_created_at_not_updated_at()      # §2
def test_news_t_visible_buffer_60s()                # §2
def test_dff_replaces_fedfunds_for_daily()          # §3
def test_fred_release_tod_table_complete()          # §4
def test_fred_uses_alfred_realtime_period()         # §5
def test_fred_pit_cache_matches_alfred_api()        # §5
def test_gdelt_theme_codes_in_master_list()         # §6
def test_gdelt_buffer_30min_not_15()                # §7
def test_gdelt_uses_file_publish_ts()               # §7
def test_wgc_url_is_correct_self_snapshot()         # §8
def test_aigpr_treated_as_monthly_lag()             # §9
def test_gpr_uses_self_snapshot_not_live()          # §10
def test_no_pandas_ta_kama_ichimoku_kst_dpo_trix()  # §11
def test_indicators_growing_window_stability()      # §11
def test_multisymbol_pagination_drained()           # §12
def test_no_split_adjusted_leakage_in_backtest()    # §13
def test_cftc_2025_shutdown_gap_handled()           # §14
def test_cot_t_visible_is_friday_330pm_et()         # §15
def test_cot_holiday_friday_uses_monday_release()   # §15
def test_walcl_thursday_visibility_after_1630_et()  # §16
def test_icsa_thursday_visibility_after_0830_et()   # §17
def test_anchor_dates_precede_train_period()        # doc 04
def test_no_minutes_until_event_features()          # doc 04
def test_features_never_reference_close_t_plus_1()  # label hygiene
def test_release_ts_lte_t_visible_all_rows()        # universal — catches §3, §4, §7, §15, §16, §17 simultaneously
def test_shuffled_label_baseline_auc_near_50()      # global sanity check
def test_universe_static_no_delistings()            # survivorship
```

**Highest-leverage test:** `test_release_ts_lte_t_visible_all_rows` — single assertion catches §3, §4, §7, §15, §16, §17 simultaneously if every source carries an explicit `release_ts` column.

### Hard Rule (V1)

> Every feature row carries a `t_visible: pd.Timestamp` column representing "earliest moment this row was publicly available." Every join uses `t_visible <= prediction_time` with strict `<` (no exact matches). Any source that does not produce `t_visible` is forbidden.

This rule is enforced by `test_release_ts_lte_t_visible_all_rows` and is a CI gate.

---

**Decision rule (re-run before you start coding):**
```bash
# Verify current free space before deciding
df -h /System/Volumes/Data
```
- IF total data footprint > 60% of CURRENT free space → push raw GDELT to BigQuery, keep snapshots local
- ELSE keep everything local (current verdict)

### What Lives Where (current verdict: mostly local)

```
LOCAL ~/Desktop/Coding Stuff/Side Projects/ML-Trading/data/         ~10-12 GB total
├── raw/
│   ├── alpaca_bars_GLD_30min.parquet            2 MB
│   ├── alpaca_bars_etfs_30min.parquet           ~20 MB (SPY/QQQ/IWM/GDX/SLV/XLF/XLE/XLK/XLU)
│   ├── alpaca_news_GLD.parquet                  ~150 MB
│   ├── fred_*_all_releases.parquet              ~120 MB (~30 series ALFRED cubes)
│   ├── brent_daily.parquet, wti_daily.parquet   <1 MB each
│   ├── gpr_monthly.parquet                       <1 MB
│   ├── cftc_cot_gold_weekly.parquet              <1 MB
│   ├── wgc_central_bank_quarterly.parquet        <1 MB
│   └── calendar_events_v1.parquet                <1 MB (deterministic schedule)
├── snapshots/
│   ├── v1_<hash>.parquet                        ~1.5 - 2.5 GB (expanded feature set)
│   └── v1_<hash>_meta.json
├── embeddings/
│   └── v1_<hash>_qwen3-emb-4b.npy               ~25-130 MB (memmap fp16)
├── anchors/v1.npz                                ~64 KB
└── checkpoints/                                   ~400 MB total

LOCAL ~/.cache/huggingface/                       ~5 GB
└── Qwen3-Embedding-4B (auto-downloaded once)

LOCAL ~/.config/nanogld/                       ~MB
├── .env.paper, .env.live
└── state.sqlite (live cycle audit, ~2 MB/year)

BIGQUERY nanogld-data:gold_news.gkg_5y         ~5-10 GB
└── Filtered GDELT GKG materialized table (FREE under 10 GB BigQuery storage tier)
```

GDELT GKG raw stays in BigQuery (already there natively, free). Joined snapshot stays local (read 50-100× per training run, latency matters). Everything else is small enough to live local.

### Cost Plan ($0 target — verified)

```
BIGQUERY storage 5-10 GB:           FREE (under 10 GB tier)
BIGQUERY queries 5y extract:        ~931 GB ONE-TIME (under 1 TB free tier)
BIGQUERY queries subsequent:        ~5-10 GB each, runs fine under monthly free tier
NETWORK egress:                     $0 (we read into Python, not external transfer)
─────────────────────────────────────────────────────────────────────
TOTAL MONTHLY COST:                 $0
```

### Mandatory Catastrophe Mitigations (all $0)

```
1. maximum_bytes_billed=1_100_000_000_000 on every BigQuery call
   (1.1 TB hard cap — can't accidentally scan 21 TB unfiltered table)

2. GCP Console > IAM & Admin > Quotas
   "Query usage per day per user" → 1024 GiB
   (caps daily scan at 1 TB)

3. ALWAYS dry-run first:
   client.query(SQL, job_config=QueryJobConfig(dry_run=True))
   prints scan estimate before real execution

4. GCP Console > Billing > Budgets & Alerts
   $1 threshold email alert (early warning)

5. Service account least-privilege:
   roles/bigquery.jobUser + roles/bigquery.dataViewer
   NO Owner role on the data project

Worst case if all 5 followed: ~$5-10 for one-time mistake.
Worst case if NONE followed: ~$130 for one accidental SELECT *.
```

### Files You Create

```
src/nanogld/data/
├── __init__.py
├── alpaca_bars.py          # Alpaca historical bars pull (5y GLD 30min)
├── alpaca_etfs.py          # NEW V1 expansion — 9-ETF basket (SPY/QQQ/IWM/GDX/SLV/XLF/XLE/XLK/XLU)
├── alpaca_news.py          # Alpaca News API (Benzinga, since 2015)
├── gdelt.py                # GDELT 2.0 GKG via BigQuery (themes, materialize once)
├── fred.py                 # FRED + ALFRED for vintage-correct macro (34 series)
├── yfinance_helpers.py     # Brent/WTI daily (with curl_cffi wrapper)
├── gpr.py                  # GPR Index monthly download from matteoiacoviello.com
├── cot.py                  # NEW V1 expansion — CFTC COT weekly disaggregated for COMEX gold
├── wgc.py                  # NEW V1 expansion — World Gold Council quarterly central bank flows
├── calendar_events.py      # NEW V1 expansion — deterministic FOMC/CPI/NFP/GDP/JOLTS/PCE schedule
├── join.py                 # Point-in-time-correct joiner with 15min news latency
├── schema.py               # Pydantic schemas for validation (extended for new sources)
├── snapshot.py             # Hashing + parquet writing + meta.json
└── cli.py                  # `python -m nanogld.data build` entrypoint

data/                       # gitignored — do not commit
├── raw/                    # one parquet per source
├── snapshots/              # immutable hashed joined parquets + meta.json
└── README.md               # how to reproduce data

tests/
├── test_pit.py             # GOLDEN FIXTURE — single most important test in project
├── test_join_schema.py     # schema validation
└── test_snapshot_hash.py   # determinism check
```

### Files You DO NOT Touch

- `src/nanogld/features/` — doc 04
- `src/nanogld/embed/` — doc 03
- `src/nanogld/model/` — doc 05
- `src/nanogld/training/` — doc 05
- Any doc-NN.md other than this one
- `.pre-commit-config.yaml`, `pyproject.toml`, `.gitignore` — doc 01 owns these (you can ASK them to add a dep, don't edit yourself)

### Stable Interface You Publish (other docs read against this)

`data/snapshots/v1_<hash>.parquet` with the schema documented in this doc's "Dataset Schema" section. doc 04 reads this. doc 03 reads this. If you change the schema, update this doc, ping STATUS.md, AskUserQuestion before shipping.

### Acceptance Criteria

You're done when:

1. ✅ `python -m nanogld.data build` produces `data/snapshots/v1_<sha256_first_16>.parquet` with full 5y of joined data + accompanying `_meta.json`
2. ✅ `pytest tests/test_pit.py` passes (golden fixture for point-in-time joiner — extended to cover ETF bars, COT release-time, and calendar event leakage)
3. ✅ `pytest tests/test_join_schema.py` passes (every column matches schema, no NaN in non-nullable cols)
4. ✅ `pytest tests/test_snapshot_hash.py` passes (running build twice on same input produces identical hash)
5. ✅ Row count is approximately 16K bars (5y × 252 days × 13 RTH bars/day = ~16,380; allow ±5% for holidays)
6. ✅ News coverage report shows ≥30% of bars have ≥1 Alpaca News article + ≥30% have ≥1 GDELT event in window
7. ✅ FRED ALFRED vintage cubes saved for ALL 34 V1 series (full curve + TIPS + breakevens + macro full bundle)
8. ✅ Equity ETF basket pulled for all 9 symbols, row counts within ±5% of GLD
9. ✅ CFTC COT parquet has ≥260 weekly rows (5y × 52w) with non-null `release_ts`
10. ✅ WGC quarterly parquet has ≥20 rows (5y × 4q) with non-null `release_ts`
11. ✅ Calendar events parquet covers all 7 event types over 2021-2026
12. ✅ A README in `data/` explains how another developer reproduces the pipeline

Hand off to doc 04 (feature engineering) by updating STATUS.md with the snapshot hash + meta.json path.

### Spawn Nia Agents When You Need To

Don't guess on:
- Alpaca SDK API quirks ('TimeFrame.Minute_30' doesn't exist — verify each call signature)
- GDELT BigQuery cost (always dry-run first — `maximum_bytes_billed` cap is mandatory)
- FRED API rate limits (~120/min — community-confirmed, not officially documented)
- Whether yfinance broke this week (it breaks every 2-3 months)

```python
# Spawn pattern
Agent(
    description="Verify [specific API behavior]",
    prompt="""Today is 2026-05-01. Verify [claim] using nia search web,
    nia github (search Alpaca/GDELT/FRED issues), direct curl to docs.
    Specifically: [questions]. Output VERIFIED/REFUTED with citations.""",
)
```

### Common Pitfalls (Nia-verified — DO read before starting)

1. **`TimeFrame.Minute_30` is NOT a class attribute.** Use `TimeFrame(30, TimeFrameUnit.Minute)`.
2. **Alpaca bars default to unadjusted prices.** ALWAYS pass `adjustment="all"`.
3. **Free tier is IEX-only, ~2.5% of US volume.** Build resilience to occasional gap bars.
4. **Latest 15-min of intraday is gated on free tier.** Live trading must run on bars ≥15 min old.
5. **Alpaca News is Benzinga ONLY** (Reuters/Bloomberg paywalled 2024). Don't claim multi-source.
6. **`NewsClient` REQUIRES api keys** (despite stale PyPI docs).
7. **GDELT themes live on `gkg_partitioned`, NOT events table.** JOIN events on URL.
8. **Many 'standard' GDELT theme codes are wrong** — verified canonical codes are in this doc. Use them.
9. **`get_series_as_of_date` returns DataFrame of ALL revisions** — must `groupby('date').tail(1)` to collapse.
10. **yfinance 30m bars STILL capped at 60 days.** Do NOT try to pull 5y of 30m via yfinance — use Alpaca historical instead.
11. **GCP billing setup is the biggest first-day friction.** ~30-60min wall clock. Set custom 1024 GiB/day quota cap.
12. **One accidental `SELECT * FROM gkg` (non-partitioned) = 21 TB scan = $130** if billing is on. Always set `maximum_bytes_billed`.

### Hand-off Protocol

When you're done:

1. Update STATUS.md with: snapshot hash, parquet path, row count, time range, news coverage stats
2. Run a "data sanity" notebook (`notebooks/01_explore_data.ipynb`) showing: equity curve of GLD, news count over time, NaN/missing report, vintage-vs-current comparison for one FRED series
3. Tag the snapshot in git: `git tag data-v1-<hash>` (not pushed; for your own reference)

Now read the implementation specifics below.

---

# 01 — Data Pipeline (FINAL — verified against live APIs April 2026)

**Status:** ✅ Complete, implementation-ready, claims verified via Nia research subagents
**Owner:** samsiavoshian
**Implementation effort:** 4-5 days (revised up from 2-3 after 2026-05-04 dataset expansion: equity basket + full Treasury curve + macro bundle + COT + WGC + calendar)
**Last verified:** 2026-05-04 (dataset expansion — owner directive)

## Goal

Pull 5 years of 30-minute GLD bars + 5 years of timestamped news + macro + geopolitical features. All aligned with strict point-in-time discipline. Save as immutable hashed parquet snapshots.

## Data Sources (verified)

```
PRIMARY:
├── Alpaca historical bars         GLD 30m, since 2016, IEX feed (free Basic tier)
├── Alpaca News API                Benzinga only, since 2015, real-time live + historical
├── GDELT 2.0 GKG (BigQuery)       Themed news events, free 1TB/mo BigQuery quota
└── FRED + ALFRED                  DGS10, DGS2, T5YIE, VIXCLS, DTWEXBGS, DCOILBRENTEU, DCOILWTICO

SUPPLEMENTARY:
├── yfinance (BZ=F, CL=F daily)    Brent + WTI futures (DAILY only — 30m capped at 60d)
├── matteoiacoviello.com (GPR)     Geopolitical Risk Index, monthly Excel download
└── RSS feeds (live trading only)  Reuters/Bloomberg/FT business RSS for live cycle (not historical)
```

## Source 1 — Alpaca Historical Bars (replaces yfinance for GLD 30m)

```python
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from datetime import datetime
import os

client = StockHistoricalDataClient(
    os.environ["ALPACA_API_KEY"],
    os.environ["ALPACA_API_SECRET"],
)

# CRITICAL: TimeFrame.Minute_30 is NOT a class attribute. Construct explicitly.
# CRITICAL: adjustment="all" or splits/dividends silently break models.
req = StockBarsRequest(
    symbol_or_symbols=["GLD"],
    timeframe=TimeFrame(30, TimeFrameUnit.Minute),
    start=datetime(2021, 4, 24),
    end=datetime(2026, 4, 24),
    adjustment="all",        # split + dividend adjusted
    feed="iex",              # free tier — ~2.5% of US volume
    limit=None,              # SDK auto-paginates via next_page_token
)
bars = client.get_stock_bars(req).df
bars = bars.loc["GLD"] if "GLD" in bars.index.get_level_values(0) else bars
```

**Schema validation (must run before using):**
- Columns: `[open, high, low, close, volume, trade_count, vwap]`
- Timestamps strictly monotonic UTC
- **CRITICAL: `timestamp` is bar START — bar at `t=09:30:00Z` covers `[09:30, 10:00)` half-open interval**
- ~13 RTH bars/day × 252 days/yr × 5 yr ≈ **16K rows** (not 87K — yfinance/futures math doesn't apply)
- Expect occasional NaN/missing bars on IEX feed (low volume)
- **Visibility rule (V1 hard rule):** `t_visible = timestamp + 30min`. Use `t_visible` everywhere downstream, never `timestamp`.

**Free Basic tier limits:**
- Historical depth: back to **2016** (5y from 2026 is fine)
- Latest 15-min of intraday data **gated** on free tier (live trading must run on bars ≥15 min old)
- Rate limit: **200 req/min** per API key
- Per-page cap: 10,000 rows; SDK auto-paginates

## Source 2 — Alpaca News API (Benzinga only)

```python
from alpaca.data.historical.news import NewsClient
from alpaca.data.requests import NewsRequest

# CRITICAL: NewsClient REQUIRES api keys (despite stale PyPI docs claiming otherwise)
news_client = NewsClient(
    os.environ["ALPACA_API_KEY"],
    os.environ["ALPACA_API_SECRET"],
)

req = NewsRequest(
    symbols="GLD",
    start=datetime(2021, 4, 24),
    end=datetime(2026, 4, 24),
    include_content=True,
    exclude_contentless=True,
    limit=50,                  # max per page is 50 articles, paginate via next_page_token
)
news_df = news_client.get_news(req).df
```

- Source: Benzinga firehose (Reuters NOT included — single source)
- Historical depth: back to **2015**
- Real-time on free tier (no 15-min gate — only PRICE data has the gate)
- Symbol-filtered query works for GLD
- **CRITICAL: field is `created_at` (NOT `published_at` — does not exist).** Never join on `updated_at` (drifts forward on edits → guaranteed leak).
- **Visibility rule (V1 hard rule):** `t_visible = created_at + 60s` (safety buffer for wire-clock skew).

## Source 3 — GDELT 2.0 GKG via BigQuery (NOT events table)

**CRITICAL CORRECTION:** themes live on GKG, not events table. Pipeline:
1. Filter `gkg_partitioned` by themes
2. JOIN to `events_partitioned` on `SOURCEURL = DocumentIdentifier` for Goldstein/EventCode features

### Setup (1-2 hrs first time)

```bash
# 1. Create GCP project + enable BigQuery
gcloud projects create nanogld-data
gcloud services enable bigquery.googleapis.com

# 2. Attach billing account (REQUIRED even for free tier — biggest friction point)

# 3. Set hard cost cap
# Console > IAM & Admin > Quotas > "Query usage per day per user" > 1024 GiB

# 4. Local auth (avoids service-account key on dev machine)
gcloud auth application-default login
```

### Query Pattern (verified theme codes)

```sql
-- Materialize ONCE (5y of GKG ≈ 931 GB, fits free 1TB/month tier exactly once)
CREATE TABLE `nanogld-data.gold_news.gkg_5y` AS
SELECT
  PARSE_TIMESTAMP('%Y%m%d%H%M%S', CAST(g.DATE AS STRING)) AS pub_ts_utc,
  g.DocumentIdentifier AS url,
  g.V2Themes,
  g.V2Tone,
  g.V2Locations
FROM `gdelt-bq.gdeltv2.gkg_partitioned` AS g
WHERE g._PARTITIONTIME BETWEEN TIMESTAMP('2021-04-24') AND TIMESTAMP('2026-04-24')
  AND TranslationInfo = ''                         -- English-only v1
  AND (
    -- GOLD-related (verified against GKG-MASTER-THEMELIST.TXT, 2026-05-04)
    REGEXP_CONTAINS(g.V2Themes, r'WB_2936_GOLD|ECON_GOLDPRICE|WB_2937_SILVER|SLFID_MINERAL_RESOURCES')
    -- MONETARY (verified — 4 EPU codes REFUTED in V4 audit; removed)
    OR REGEXP_CONTAINS(g.V2Themes, r'ECON_INTEREST_RATES|ECON_INFLATION|ECON_CENTRALBANK|WB_1235_CENTRAL_BANKS|WB_444_MONETARY_POLICY')
    -- CONFLICT (V4 corrections: WB_2432_FRAGILITY -> WB_2432_FRAGILITY_CONFLICT_AND_VIOLENCE; TAX_WEAPONS_BOMB -> TAX_WEAPONS)
    OR REGEXP_CONTAINS(g.V2Themes, r'ARMEDCONFLICT|WB_2433_CONFLICT_AND_VIOLENCE|WB_2432_FRAGILITY_CONFLICT_AND_VIOLENCE|TERROR|SANCTIONS|TAX_WEAPONS|MARITIME_INCIDENT')
    -- ECONOMIC STRESS (V4: EPU_ECONOMY_HISTORIC REFUTED — removed)
    OR REGEXP_CONTAINS(g.V2Themes, r'ECON_BANKRUPTCY|ECON_TRADE_DISPUTE|ECON_DEBT')
  );
```

### Python Client (verified pattern)

```python
from google.cloud import bigquery

client = bigquery.Client()  # picks up ADC automatically

job_config = bigquery.QueryJobConfig(
    maximum_bytes_billed=1_100_000_000_000,  # ~1.1 TB hard cap (safety)
    use_query_cache=True,
)

# CRITICAL: dry-run BEFORE every real query
dry = client.query(SQL, job_config=bigquery.QueryJobConfig(
    query_parameters=job_config.query_parameters, dry_run=True
))
print(f"Will scan {dry.total_bytes_processed/1e9:.1f} GB")
assert dry.total_bytes_processed < 1.0e12, "Query too big — abort"

# Then execute, stream to Arrow (10-50x faster than REST for large results)
df = client.query(SQL, job_config=job_config).result().to_dataframe(
    create_bqstorage_client=True,
)
```

### After Materialization, Query Local Table

```sql
-- Subsequent queries scan our local materialized table, NOT the public GKG.
-- ~5-10 GB instead of 931 GB. Cheap.
SELECT pub_ts_utc, url, V2Themes, V2Tone
FROM `nanogld-data.gold_news.gkg_5y`
WHERE pub_ts_utc BETWEEN TIMESTAMP('2024-01-01') AND TIMESTAMP('2024-12-31');
```

### GDELT Pitfalls (from verification research)

1. **UTC only.** All GDELT timestamps UTC. Convert in pandas only AFTER pulling, never in SQL on the partition column (kills pruning).
2. **`_PARTITIONTIME` filter required for partition pruning.** Filter on `DATE` column does NOT prune.
3. **30-min batching delay (V4 update).** GDELT 2.0 publishes every 15 minutes, but slot `15:00-15:15` lands ~15:30 UTC and timestamps inside reflect article scrape times within the slot. **Use 30-min buffer, not 15-min.** `news.DATE <= T_close - 30min`.
4. **Multiple events per article.** ONE `SOURCEURL` → many event rows (one per Actor1-action-Actor2 dyad). Dedupe on URL in features.
5. **`V2Themes` is semicolon-delimited with comma char-offsets.** Use `REGEXP_CONTAINS`, not equality.
6. **Single accidental `SELECT * FROM gkg` (non-partitioned) = 21 TB scan = $130 if billing on.** Always set `maximum_bytes_billed`.
7. **Multilingual.** Filter `TranslationInfo = ''` (empty string for English-only originals — `IS NULL` is wrong, column is populated).
8. **DATE field semantics (V4):** `DATE` is the article's published date per V2.1 codebook, NOT URL crawl time and NOT BigQuery availability time. Article was on the open web at `DATE`. BigQuery row appears 15-60 min later. Conservative: gate on `t_visible = DATE + 30min` AND require `_PARTITIONTIME <= bar_close_utc` for the BigQuery-availability semantic.
9. **Visibility rule (V1 hard rule):** `t_visible = max(DATE + 30min, _PARTITIONTIME)`. Use this everywhere.

## Source 4 — FRED + ALFRED (vintage-correct macro)

### Series IDs (V1 expanded 2026-05-04 — 30 series)

**Treasury curve + TIPS + breakevens (11 series — real rates are #1 gold driver):**

| Series ID | Description | Vintage horizon | Revisions |
|-----------|-------------|-----------------|-----------|
| `DGS3MO` | 3-Month Treasury Constant Maturity | 2006+ | Never |
| `DGS6MO` | 6-Month Treasury Constant Maturity | 2006+ | Never |
| `DGS2` | 2-Year Treasury Constant Maturity | 2006+ | Never |
| `DGS5` | 5-Year Treasury Constant Maturity | 2006+ | Never |
| `DGS10` | 10-Year Treasury Constant Maturity | 2006+ | Never |
| `DGS30` | 30-Year Treasury Constant Maturity | 2006+ | Never |
| `DFII5` | 5-Year TIPS Real Yield | 2006+ | Rare |
| `DFII10` | 10-Year TIPS Real Yield (DIRECT real-rate signal) | 2006+ | Rare |
| `T5YIE` | 5-Year Breakeven Inflation Rate | 2006+ | Occasional |
| `T10YIE` | 10-Year Breakeven Inflation Rate | 2006+ | Occasional |
| `T5YIFR` | 5-Year, 5-Year Forward Inflation Expectation | 2006+ | Occasional |

**FX + market vol (3 series):**

| Series ID | Description | Vintage horizon | Revisions |
|-----------|-------------|-----------------|-----------|
| `DTWEXBGS` | Nominal Broad USD Trade-Weighted Index | 2006+ | Rare |
| `VIXCLS` | VIX close | 2006+ | Rare |

**Oil (2 series, daily):**

| Series ID | Description | Vintage horizon | Revisions |
|-----------|-------------|-----------------|-----------|
| `DCOILBRENTEU` | Brent crude spot | 2006+ | Occasional |
| `DCOILWTICO` | WTI crude spot | 2006+ | Occasional |

**Macro economic — full bundle (19 series, varying frequency, ALL vintage-correct via ALFRED):**

Labor (5):
| Series ID | Description | Frequency | Release lag |
|-----------|-------------|-----------|-------------|
| `UNRATE` | Civilian Unemployment Rate | Monthly | ~1st Fri next month |
| `PAYEMS` | Total Nonfarm Payrolls | Monthly | ~1st Fri next month |
| `ICSA` | Initial Jobless Claims (weekly — vintage matters) | Weekly Thursday | ~1 day |
| `CCSA` | Continued Jobless Claims | Weekly Thursday | ~1 day |
| `JTSJOL` | JOLTS Job Openings | Monthly | ~1 month |

Inflation (4):
| Series ID | Description | Frequency | Release lag |
|-----------|-------------|-----------|-------------|
| `CPIAUCSL` | CPI, All Urban Consumers, all items | Monthly | ~mid-month |
| `CPILFESL` | Core CPI (ex food + energy) | Monthly | ~mid-month |
| `PCEPI` | PCE Price Index | Monthly | ~end of month |
| `PCEPILFE` | Core PCE (Fed's preferred inflation gauge) | Monthly | ~end of month |

Growth + sentiment (5):
| Series ID | Description | Frequency | Release lag |
|-----------|-------------|-----------|-------------|
| `GDPC1` | Real GDP, chained 2017 dollars | Quarterly | ~1 month after quarter |
| `INDPRO` | Industrial Production Index | Monthly | ~mid-month |
| `RSAFS` | Retail Sales | Monthly | ~mid-month |
| `HOUST` | Housing Starts | Monthly | ~mid-month |
| `UMCSENT` | University of Michigan Consumer Sentiment | Monthly | ~end of month, prelim mid-month |

Money + Fed (6 — V4 added DFF, kept FEDFUNDS for monthly aggregates):
| Series ID | Description | Frequency | Release time | Notes |
|-----------|-------------|-----------|--------------|-------|
| `M2SL` | M2 Money Supply | Monthly (NOT weekly — V4 correction) | ~1:00 PM ET 4th Tuesday | H.6 release |
| `WALCL` | Total Fed Assets (Fed balance sheet) | Weekly Wednesday level | Thursday ~4:30 PM ET | H.4.1 — Thu RTH-close 16:00 bar must NOT use this week's level |
| `RRPONTSYD` | Overnight Reverse Repo | Daily | ~1:30 PM ET | NY Fed TOMO |
| `DFF` | **Daily** Effective Fed Funds Rate (V4 — replaces FEDFUNDS for daily features) | Daily BD | Next BD ~9:00 AM ET | H.15 |
| `FEDFUNDS` | Monthly avg Fed Funds Rate (kept for monthly aggregates ONLY) | Monthly | ~1st BD next month | DO NOT use as daily — would silently use a value that does not exist until first BD of next month |
| `SOFR` | Secured Overnight Financing Rate | Daily | Next BD ~8:00 AM ET | NY Fed |

**Total: 35 FRED series (V4 update; vs 7 in pre-expansion plan).** Breakdown: 6 nominal curve + 5 TIPS/breakevens + 2 FX/vol + 2 oil + 5 labor + 4 inflation + 5 growth + 6 money/Fed = 35.

**Critical:** ALL series use ALFRED `get_series_all_releases` → groupby tail(1) for vintage discipline. Inflation/GDP/labor monthly prints get heavily revised; using non-vintage data = look-ahead leak that destroys validity.

### Verified Code Pattern

```python
from fredapi import Fred
import pandas as pd
import os

fred = Fred(api_key=os.environ["FRED_API_KEY"])  # ~120 req/min limit per key

def pit_series(series_id: str, as_of: str) -> pd.Series:
    """
    Return daily series as known on `as_of` (point-in-time-correct).
    
    CRITICAL: get_series_as_of_date returns DataFrame of ALL revisions up to as_of,
    NOT a clean Series. Must groupby('date').tail(1) to collapse.
    """
    df = fred.get_series_as_of_date(series_id, as_of)
    if df.empty:
        return pd.Series(dtype=float, name=series_id)
    pit = (
        df.sort_values("realtime_start")
          .groupby("date").tail(1)
          .set_index("date")["value"]
          .astype(float)
          .sort_index()
    )
    pit.name = series_id
    return pit
```

### Bulk Backfill Pattern (recommended for 30min granularity)

```python
# Pull entire vintage cube once per series, persist, slice locally during backtest.
# Avoids per-tick API calls.

FRED_SERIES_V1 = [
    # Treasury curve
    "DGS3MO", "DGS6MO", "DGS2", "DGS5", "DGS10", "DGS30",
    # TIPS + breakevens
    "DFII5", "DFII10", "T5YIE", "T10YIE", "T5YIFR",
    # FX + vol
    "DTWEXBGS", "VIXCLS",
    # Oil
    "DCOILBRENTEU", "DCOILWTICO",
    # Labor
    "UNRATE", "PAYEMS", "ICSA", "CCSA", "JTSJOL",
    # Inflation
    "CPIAUCSL", "CPILFESL", "PCEPI", "PCEPILFE",
    # Growth + sentiment
    "GDPC1", "INDPRO", "RSAFS", "HOUST", "UMCSENT",
    # Money + Fed (V4: DFF added — daily fed funds; FEDFUNDS kept for monthly aggregates only)
    "M2SL", "WALCL", "RRPONTSYD", "DFF", "FEDFUNDS", "SOFR",
]

# 35 series total. 35 calls × 0.5s sleep = ~18s wall. ALFRED cubes ~3 MB each = ~100-120 MB.
for series_id in FRED_SERIES_V1:
    df_all = fred.get_series_all_releases(series_id)
    df_all.to_parquet(f"data/raw/fred_{series_id.lower()}_all_releases.parquet")
    time.sleep(0.5)  # under 120 req/min limit
```

Then at training time, build vintage lookup locally:

```python
def vintage_lookup(df_all: pd.DataFrame, T: pd.Timestamp) -> pd.Series:
    """Get latest-known value per observation date as of time T."""
    visible = df_all[df_all["realtime_start"] <= T]
    return (
        visible.sort_values("realtime_start")
               .groupby("date").tail(1)
               .set_index("date")["value"]
               .astype(float)
               .sort_index()
    )
```

### FRED Pitfalls

1. **`get_series_as_of_date` returns ALL revisions, not collapsed series.** Must groupby tail(1).
2. **ALFRED horizon = 2006-present.** Pre-2006 backtests have NO vintages.
3. **Release lag.** DGS10 for date D published next business day ~4pm ET. Gate visibility by `realtime_start`, not `date`.
4. **Weekends/holidays = NaN.** Don't ffill silently — decide explicitly.
5. **Rate limit 120/min per key.** Add 0.5s sleep on bulk loops.

## Source 5 — yfinance (Brent + WTI daily ONLY)

```python
import yfinance as yf
from curl_cffi import requests

# Wrap in curl_cffi to dodge Yahoo rate limiting (default since v1.2)
session = requests.Session(impersonate="chrome")

# Daily Brent + WTI for 5y
brent_daily = yf.Ticker("BZ=F", session=session).history(period="5y", interval="1d")
wti_daily = yf.Ticker("CL=F", session=session).history(period="5y", interval="1d")
```

**Pin: `yfinance==1.3.0`** in `pyproject.toml`. v1.3.0 fixed an April 2026 dividends breakage; do not auto-upgrade. (V4 verified: 1.4.x not yet released.)

**Pitfalls (V4 audit):**
- 30m bars **capped at 60 days** by Yahoo — never going to work for 5y. We don't try.
- BZ=F and CL=F are continuous front-month futures. Yahoo handles rolls but injects phantom returns at roll dates. yfinance does NOT mitigate this. Either use back-adjusted contracts or filter roll-day returns yourself.
- Brent trades on ICE (UK calendar, close 8:00 PM London = 15:00/16:00 ET), WTI on NYMEX (US calendar, close 5:00 PM CT = 17:00/18:00 ET). Different holidays. Use `pandas_market_calendars` to align sessions.
- **`period="5y"` includes today's PARTIAL bar.** Live test: returns last index = `2026-05-04 13:00 ET` for queries before close. For training, drop today's row. For inference, mark current-day OHLC `provisional=True` and use yesterday's settled close as the lagged feature.
- **`.history()` returns timezone `America/New_York`** (NOT UTC, NOT exchange-local). Yahoo pre-converts to ET. So a 14:00 ET 30m bar for BZ=F covers 14:00-14:30 ET = 19:00-19:30 London time. Document and convert to UTC immediately on ingest.
- **Lag Brent by ≥1 bar** when used as a feature for US RTH sessions — Brent close lands AFTER GLD close, so same-bar joining leaks future Brent info into past US bars.
- **Visibility rule (V1 hard rule):** `t_visible = settlement_ts_per_contract` (CL ~14:30 ET, BZ ~15:00 ET in summer). Daily settlements only. No 30m treatment.

## Source 6 — GPR Index (Caldara & Iacoviello, monthly) — V4 CORRECTED

```python
import pandas as pd

# Live download URL (V4 verified 2026-05-04 — file has 1516 rows, 1900-01 to 2026-04, 115 columns)
GPR_URL = "https://www.matteoiacoviello.com/gpr_files/data_gpr_export.xls"

# AI-GPR DAILY URL (V4 verified — but data lags ~30 days, NOT real-time)
AIGPR_DAILY_URL = "https://www.matteoiacoviello.com/ai_gpr_files/ai_gpr_data_daily.csv"

# Cache locally with FETCH-DATE keying (no public vintage archive — methodology revisions
# silently rewrite history; e.g. 2026-04-24 revision expanded bilateral index)
def fetch_and_snapshot_gpr():
    fetch_ts = datetime.utcnow().isoformat()
    for name, url in [("monthly", GPR_URL), ("aigpr_daily", AIGPR_DAILY_URL)]:
        content = requests.get(url, timeout=60).content
        sha = hashlib.sha256(content).hexdigest()[:16]
        path = f"data/raw/gpr/{name}_{fetch_ts}_{sha}.bin"
        Path(path).write_bytes(content)
```

**Update cadence and visibility (V4):**
- **Monthly GPR:** ~3-7 days lag into new month. Visibility = day 7 of month t+1.
- **AI-GPR daily: NOT real-time. ~30-day lag.** Today (2026-05-04) latest = 2026-03-31. Treat as monthly with ~30-day lag, OR contact authors for live feed.
- **No public vintage archive.** Caldara/Iacoviello revise history when methodology updates. Self-snapshot weekly with fetch-date keying. For backtests, key features by snapshot fetch date — never by current download.

**Pitfalls (V4):**
- "AI-GPR daily" name is misleading — it's a CSV updated infrequently, not a daily feed. Do NOT use as same-day feature.
- Treat both files as needing weekly self-snapshot for vintage discipline.

## Source 7 — RSS Feeds (live trading cycle ONLY, no historical)

```python
import feedparser

LIVE_RSS_FEEDS = {
    "reuters_business": "https://www.reuters.com/business/feed/",
    "bloomberg_macro":  "https://feeds.bloomberg.com/markets/news.rss",
    "ft_markets":       "https://www.ft.com/markets?format=rss",
}

# Live cycle only. NOT used in historical backfill.
def fetch_recent_rss(window_min: int = 30) -> list[dict]:
    cutoff = datetime.utcnow() - timedelta(minutes=window_min)
    headlines = []
    for name, url in LIVE_RSS_FEEDS.items():
        feed = feedparser.parse(url)
        for entry in feed.entries:
            pub = parse_rss_date(entry.published)
            if pub > cutoff:
                headlines.append({
                    "source": name,
                    "ts_utc": pub,
                    "headline": entry.title,
                })
    return headlines
```

For historical 5y backfill, RSS is dead — feeds only carry recent items. We get historical news from Alpaca News (Benzinga 2015+) and GDELT GKG (2015+).

## Source 8 — Alpaca Equity ETF Basket (V1 expansion 2026-05-04)

Same client + endpoint as Source 1, just a multi-symbol pull. Captures risk-on/off (SPY/QQQ/IWM), gold-specific cross-correlations (GDX/SLV), and sector regime (XLF/XLE/XLK/XLU).

```python
ETF_BASKET = {
    # Broad equity (risk-on/off)
    "SPY": "S&P 500",
    "QQQ": "Nasdaq-100",
    "IWM": "Russell 2000 (small caps)",
    # Gold-specific cross-references
    "GDX": "VanEck Gold Miners (direct gold cross-correlation)",
    "SLV": "iShares Silver Trust (gold-silver ratio numerator)",
    # Sector ETFs (factor regime)
    "XLF": "Financials",
    "XLE": "Energy",
    "XLK": "Technology",
    "XLU": "Utilities (rate-sensitive defensive)",
}

req = StockBarsRequest(
    symbol_or_symbols=list(ETF_BASKET.keys()),  # 9 symbols, single batched call
    timeframe=TimeFrame(30, TimeFrameUnit.Minute),
    start=datetime(2021, 4, 24),
    end=datetime(2026, 4, 24),
    adjustment="all",        # split + dividend adjusted
    feed="iex",              # free tier
    limit=None,              # SDK auto-paginates
)
etf_bars = client.get_stock_bars(req).df
# multi-index (symbol, timestamp) → reshape per ETF on disk
for sym in ETF_BASKET:
    etf_bars.loc[sym].to_parquet(f"data/raw/alpaca_bars_{sym}_30min.parquet")
```

**Pitfalls (Nia-verify before relying on):**
- All 9 ETFs are NYSE/NASDAQ listed and IEX-feed eligible on free tier — verify on first pull.
- GDX/SLV occasionally have low IEX volume causing missing 30m bars; same resilience pattern as GLD applies.
- All ETFs follow NYSE calendar (`pandas_market_calendars.get_calendar('NYSE')`) — same RTH gating as GLD.
- Storage: ~2 MB × 9 = ~20 MB raw. Trivial.
- Rate limit: 200 req/min applies across symbols, not per-symbol. SDK handles pagination across the full multi-symbol batch.

**Schema (per ETF, identical to GLD):**
- Columns: `[open, high, low, close, volume, trade_count, vwap]`
- Same row count as GLD (~16K bars per 5y).
- Aligned on the same NYSE 30min calendar — joins cleanly without resampling.

## Source 9 — CFTC COT (Commitments of Traders, weekly gold futures) — V4 CORRECTED

**V4 corrections:**
- Contract code confirmed `088691`, contract name = `GOLD - COMMODITY EXCHANGE INC.`
- Historical zip URL pattern is NOT stable — must parse `cftc.gov/MarketReports/CommitmentsofTraders/HistoricalCompressed/index.htm` at build time to extract per-year zip filenames. Do NOT hardcode `com_disagg_txt_{year}.zip`.
- Backup data path (Socrata): `https://publicreporting.cftc.gov/Commitments-of-Traders/Disaggregated-Futures-Only/72hh-3qpy` if zip URLs break.
- **2025 government shutdown caused multi-month publication gap.** Non-Friday catch-up cadence through early 2026. Verify each row's release date against CFTC Special Announcements page; flag irregular releases with `irregular_release=True`.
- Holiday-Friday → Monday release. Read schedule, do NOT compute Friday-from-report-date.

Source: https://www.cftc.gov/dea/futures/deacmesf.htm — weekly CSV/TXT, free, public.

Every Friday 3:30 PM ET (sharp; empirically file write often lags 5-15 min), CFTC publishes positions as of the previous Tuesday 4:00 PM ET. Disaggregated COT report gives managed money / commercial / nonreportable net longs and open interest.

```python
import pandas as pd
import io
import requests

# Disaggregated COT (preferred over legacy — separates managed money from swap dealers)
COT_DISAGG_URL_TEMPLATE = (
    "https://www.cftc.gov/dea/newcot/f_disagg.txt"      # current year
)
COT_DISAGG_HISTORICAL_TEMPLATE = (
    "https://www.cftc.gov/files/dea/history/com_disagg_txt_{year}.zip"  # historical years
)

def fetch_cot_gold_5y(start_year: int = 2021, end_year: int = 2026) -> pd.DataFrame:
    """Pull weekly COT, filter to gold, return tidy frame.
    Columns: report_date_ts, market_pos_dt, oi_open_interest, mm_net_long, mm_pct_oi, ...
    """
    rows = []
    for yr in range(start_year, end_year + 1):
        url = COT_DISAGG_HISTORICAL_TEMPLATE.format(year=yr)
        zip_bytes = requests.get(url, timeout=60).content
        # ... unzip, parse, filter to "GOLD - COMMODITY EXCHANGE INC."
        df_yr = parse_cot_csv(zip_bytes, contract="GOLD - COMMODITY EXCHANGE INC.")
        rows.append(df_yr)
    return pd.concat(rows).sort_values("report_date_ts")

cot = fetch_cot_gold_5y()
cot.to_parquet("data/raw/cftc_cot_gold_weekly.parquet")
```

**Critical fields to extract (per arXiv:2305.05186 + CFTC docs):**
- `Open_Interest_All` — open interest in contracts
- `M_Money_Positions_Long_All`, `M_Money_Positions_Short_All` — managed money longs/shorts
- `Prod_Merc_Positions_Long_All`, `Prod_Merc_Positions_Short_All` — commercial (producer/merchant)
- `NonRept_Positions_Long_All`, `NonRept_Positions_Short_All` — non-reportable (small spec)
- `Report_Date_as_YYYY-MM-DD` — Tuesday 4 PM ET reference date
- Derived `release_ts` — Friday 3:30 PM ET (or next BD if Friday is a holiday) + 30min safety buffer

(Field names contain UNDERSCORES, not spaces — V4 correction.)

**Point-in-time discipline for COT:**
- Feature visible at bar T must satisfy `release_ts < T_close`. NOT `report_date < T_close` (that leaks ~3 trading days).
- Conservative gate: `release_ts = next_friday_after(report_date_tuesday).at(20:00 UTC)` (3:30 PM ET + 30 min buffer; UTC = 19:30 winter / 20:30 summer; pick conservative 20:00 UTC).
- Holiday-Friday: roll to next BD's 20:00 UTC.
- 2025 shutdown gap: rows in that window get `irregular_release=True`; either drop or train with explicit indicator.

**Pitfalls:**
- File format changed format in 2017 (legacy → disaggregated). Use disaggregated for full 5y.
- Historical zips fail silently for pending year — wrap in try/except.
- Field names with spaces; the parser must handle quoted CSV.
- Holiday weeks shift release to Monday — handle missing Friday gracefully.

**Storage: ~260 weekly rows × 5y × ~20 fields = <1 MB.**

## Source 10 — World Gold Council (central bank flows) — V4 CORRECTED

**V4 corrections (2026-05-04 audit):**
- WGC is **MONTHLY**, not quarterly (V1 plan was wrong)
- Direct URLs (no form gate):
  - Quarterly time series since 2000: `https://www.gold.org/download/8052`
  - Latest official reserves: `https://www.gold.org/download/7739`
- Release lag is ~2 months (IMF IFS reporting cascade)
- **NO PUBLIC VINTAGE ARCHIVE** — WGC overwrites the latest snapshot in place. Each release contains REVISIONS of prior months as more countries report to IMF. Without self-snapshotting we leak future revisions into past dates.

```python
WGC_QUARTERLY_TIMESERIES = "https://www.gold.org/download/8052"
WGC_LATEST_RESERVES      = "https://www.gold.org/download/7739"

def fetch_and_snapshot_wgc():
    """Pull both files, store with retrieval timestamp into immutable cache.
    Run weekly via cron. Each downloaded file becomes a vintage row.
    """
    fetch_ts = datetime.utcnow().isoformat()
    for name, url in [("quarterly_ts", WGC_QUARTERLY_TIMESERIES),
                      ("latest_reserves", WGC_LATEST_RESERVES)]:
        content = requests.get(url, timeout=60).content
        sha = hashlib.sha256(content).hexdigest()[:16]
        path = f"data/raw/wgc/{name}_{fetch_ts}_{sha}.xlsx"
        Path(path).write_bytes(content)
    # Maintain index keyed by fetch_ts for vintage lookups during backtests
```

**Vintage discipline (mandatory):**
- For training, key each WGC feature to the snapshot file that existed at `t_visible`.
- Self-snapshot WEEKLY — fast-reporters reveal data within days, slow-reporters within months.
- Conservative gate: `release_ts = first_business_day(report_month + 2_months) at 12:00 UTC` (London noon).

**Schema (latest_reserves file):**
- Country
- Holdings (tonnes)
- % of total reserves
- Net purchases derived as Δ vs prior period

**Pitfalls:**
- IMF cascade: a "March 2026" WGC release contains China-Mar (timely), US-Feb (US reports later), Russia-Jan (sanctioned, irregular). Treat each (country, period) cell as having its own as-of date if you can; else gate the entire row by the slowest reporter (~2 months conservative).
- Net purchases flip negative when central banks sell — do NOT clip.
- Press release vs file refresh: same morning London time (~09:00 UTC), within the same day.

**Storage: weekly snapshots × 5y × 2 files × ~500 KB each ≈ 250 MB if kept indefinitely. V1 keep last 12 weeks = ~12 MB.**

## Source 11 — Calendar Event Schedule (deterministic, no API)

Pre/post-release windows for high-impact macro releases that move gold. Built deterministically from published BLS/BEA/Fed calendars, then merged onto bars.

```python
# Hard-code release calendars from official sources for 2021-2026:
# - FOMC meeting dates: federalreserve.gov/monetarypolicy/fomccalendars.htm (8 per year)
# - CPI release: BLS calendar — 2nd or 3rd Tuesday of month, 8:30 AM ET
# - NFP release: BLS calendar — 1st Friday of month, 8:30 AM ET
# - GDP advance: BEA — last Thursday of month following quarter end, 8:30 AM ET
# - JOLTS: BLS — early in month, 10:00 AM ET
# - PCE: BEA — last business day of month following data month, 8:30 AM ET
# - FOMC minutes: 3 weeks after each meeting, 2:00 PM ET

CALENDAR_EVENTS = {
    "FOMC": load_fomc_dates(2021, 2026),       # ~40 events × 5y
    "CPI": load_cpi_dates(2021, 2026),          # ~60 events × 5y
    "NFP": load_nfp_dates(2021, 2026),          # ~60 events × 5y
    "GDP": load_gdp_dates(2021, 2026),          # ~20 events × 5y
    "JOLTS": load_jolts_dates(2021, 2026),      # ~60 events × 5y
    "PCE": load_pce_dates(2021, 2026),          # ~60 events × 5y
    "FOMC_minutes": load_fomc_minutes_dates(2021, 2026),  # ~40 events × 5y
}

# For each bar, compute event proximity features (doc 04 owns the feature side; doc 02 just persists the schedule).
calendar = build_calendar_dataframe(CALENDAR_EVENTS)
calendar.to_parquet("data/raw/calendar_events_v1.parquet")
```

**Schema (one row per event):**
- `event_type` (FOMC/CPI/NFP/GDP/JOLTS/PCE/FOMC_minutes)
- `event_ts_utc` (release timestamp)
- `tier` (1 = market-moving like NFP/CPI/FOMC; 2 = secondary like JOLTS)

**Pitfalls:**
- Calendars sometimes change (Fed shifts FOMC date for emergencies — March 2020 cut). Verify with /browse before training; no live API for "official" calendar.
- Daylight Saving shifts release-time UTC by 1 hour in spring/fall. Always store UTC, never ET.
- Some FOMC days have a `decision` and a `press conference` ~30 min later — represent as two separate events if useful; default V1 = single event at decision time (14:00 ET).

**Storage: ~340 events × 5y × 3 cols = trivial.**

## Source 12 — FNSPID Historical News Corpus (V4 expansion 2026-05-04)

**Single biggest free-news win** for filling the historical pre-2021 gap. arXiv:2402.06698, CC BY 4.0.

```python
# 15.7M news articles, 1999-2023, multi-source: Reuters, NASDAQ, Benzinga, Lenta + 4 stock-news sites
# Already on HuggingFace, parquet format, ~50 GB
# Filterable by ticker — pull GLD-relevant rows + any commodity / macro tickers

from datasets import load_dataset
fnspid = load_dataset("Zihan1004/FNSPID", split="train")
# Filter to: GLD, commodity miners (NEM, GOLD, FNV, AEM), gold ETFs, macro tickers (TLT, IEF, UUP, GDX, SLV)
relevant = fnspid.filter(lambda x: x["symbol"] in {"GLD", "GDX", "SLV", "NEM", "GOLD", "FNV", "AEM", "TLT", "IEF", "UUP"})
relevant.to_parquet("data/raw/fnspid_gold_relevant.parquet")
```

**Schema:** `[symbol, date, title, body, source, url, sentiment]`. Source field is one of `{Reuters, NASDAQ, Benzinga, Lenta, Cnnmoney, FT_unverified, Marketwatch, Yahoo}`. Map to `SOURCE_REGISTRY` (see doc 03).

**License:** CC BY 4.0 — free with attribution. Cite arXiv:2402.06698 in README.

**Leakage:** FNSPID timestamps are **date-precise only** (`YYYY-MM-DD`), no time-of-day. Conservative gate: feature visible at first RTH bar of date+1. This is a 1-day buffer, more conservative than minute-level Alpaca/Kitco articles. Document this in `t_visible` calculation.

**Storage:** ~5 GB filtered to gold-relevant tickers + 10y window.

## Source 13 — Kitco News Scraper (V4 expansion 2026-05-04)

Free site, no API. Date-slugged URLs make 10y backfill feasible.

```python
# URL pattern: kitco.com/news/category/markets/{page}, articles at /news/article/YYYY-MM-DD/{slug}
# RSS for live cycle: kitco.com/news/category/markets/rss

KITCO_RSS = {
    "markets":     "https://www.kitco.com/news/category/markets/rss",
    "mining":      "https://www.kitco.com/news/category/mining/rss",
    "commodities": "https://www.kitco.com/news/category/commodities/rss",
}

def scrape_kitco_archive(start_date: date, end_date: date):
    """Crawl date-slug archive page-by-page. Throttle 1 req/2s. Respect robots.txt."""
    # Kitco: full body + author + minute-precise pub_ts in HTML
    # Use BeautifulSoup4. requests with User-Agent header.
    # Persist as: (article_id, source_id=SOURCE_REGISTRY['kitco'], created_at, title, body, url)
    ...
```

**Bias-tier:** `industry_bullish`. LAFTR head will learn the per-source prior.

**Storage:** ~1 GB for 10y of Kitco markets + commodities articles.

**Pitfalls:**
- ToS reserves rights — research-use should be fine, redistribution prohibited.
- robots.txt allows crawl; rate-limit aggressively (1 req/2s minimum).
- No mature open-source scraper found via Nia search. Build small custom scraper.

## Source 14 — Investing.com Gold Scraper (V4 expansion 2026-05-04)

Aggregator with the largest archive of the user's listed sources (~250K articles confirmed via Apify actor `glitch_404/investing-scraper`). Mostly neutral wire syndication.

```python
INVESTING_GOLD_URL = "https://www.investing.com/commodities/gold-news"

# Reference implementations (consult before writing):
# - npm: investing-com-api (MIT license)
# - GitHub: alvarobartt/investpy (Pandas-style historical lib)
# - Apify: glitch_404/investing-scraper (commercial, 250K archive claim)

# Cloudflare anti-bot is aggressive — use curl_cffi browser impersonation (already pinned in doc 02)
def scrape_investing_gold(...):
    # Article schema: title, body, author, asset_tags (XAU/USD, GLD), pub_ts (minute precision)
    ...
```

**Bias-tier:** `aggregator_neutral`. Best signal-to-bias ratio of the user's gold-specific sources.

**ToS:** "Explicit prior written permission" required for redistribution. Research-use grey zone — cite source, throttle aggressively (≤ 1 req/3s), expect Cloudflare friction.

**Storage:** ~2 GB for 10y.

## Source 15 — BullionVault Author-Pages Scraper (V4 expansion 2026-05-04)

Strongly bullish dealer marketing. Use as bias-extreme feature, NOT raw signal.

```python
BULLIONVAULT_AUTHOR_PAGES = [
    "https://www.bullionvault.com/gold-news/users/adrian-ash",     # head of research
    "https://www.bullionvault.com/gold-news/users/gold-report",
    # Discover others via /gold-news main page pagination
]
```

**Bias-tier:** `dealer_bullish`. LAFTR will heavily down-weight in inference.

**No news API exists** — `bullionvault.com/help/xml_api.html` is for trading data only.

**Storage:** ~500 MB.

**Pitfalls:**
- robots FAQ at `bullionvault.com/help/FAQs/FAQs_bots.html` — review before scraping.
- RSSHub had a known-broken BullionVault gold-news route (gitcode 2025-05). Custom scraper needed.

## Source 16 — Central Bank Speeches + FOMC Statements (V4 expansion 2026-05-04)

Free, public domain (US 17 USC §105 + ECB free research). Highest-impact news class for gold.

```python
# Pre-built HF datasets (5min download each):
# - samchain/bis_central_bank_speeches  (1997-2023+, 90+ central banks)
# - istat-ai/ECB-FED-speeches            (1996-2025, ECB + FED, 30 MB parquet)

from datasets import load_dataset
bis_speeches = load_dataset("samchain/bis_central_bank_speeches")
ecb_fed_speeches = load_dataset("istat-ai/ECB-FED-speeches")

# fomc/statements GitHub repo: pre-cleaned FOMC text 1994+
# git clone https://github.com/fomc/statements ~/Downloads/fomc-statements

# US Treasury press releases (scrape):
# https://home.treasury.gov/news/press-releases  (paginated archive 1995+)

# CFTC commissioner speeches:
# https://www.cftc.gov/PressRoom/SpeechesTestimony  (paginated archive 1990s+)
```

**Schema unifier:** `(article_id, source_id, bias_tier_id="central_bank_official", t_visible, title, body)`. `t_visible` = official press-release timestamp (every speech's URL has a published-on date; minute-precision available for FOMC press releases via Fed's calendar).

**Bias-tier:** `central_bank_official` (Fed/ECB/BIS) or `government_official` (Treasury/CFTC).

**Storage:** ~200 MB for everything pre-cleaned + scrapes.

## Source 17 — Reddit Arctic Shift Dumps (V4 expansion 2026-05-04)

Pushshift successor. Free torrents through 2026-04. Retail sentiment proxy.

```python
# Dumps: https://github.com/ArthurHeitmann/arctic_shift (releases page lists monthly torrents)
# Subreddits to ingest:
SUBREDDITS = ["Gold", "wallstreetbets", "investing", "Goldandsilverstackers", "Commodities"]

# Each torrent is per-subreddit per-month, JSON-lines compressed with zstd.
# Filter to gold-relevant submissions + comments via keyword match (gold|GLD|silver|SLV|gdx|comex|xau|bullion)
```

**Bias-tier:** `retail_social`. Often counter-indicator at extremes (r/wallstreetbets euphoria precedes pullbacks).

**Storage:** ~5 GB for filtered 10y window.

**Pitfalls:**
- Reddit ToS gray zone for academic distribution — Arctic Shift author still distributes. Use at own risk.
- Each post + comment has UTC timestamp — fine for `t_visible`.
- Comments outnumber posts ~50:1 — consider keeping only posts + top-comment for V1.

## Source 18 — Kaggle Gold-Labeled Sentiment (V4 expansion 2026-05-04)

Direct gold-labeled headlines. Small N but useful for training a gold-relevance classifier (binary: is this article gold-relevant?).

```python
# Two datasets:
# - kaggle.com/datasets/ankurzing/sentiment-analysis-in-commodity-market-gold (CC0/CC BY)
# - kaggle.com/datasets/romanfonel/precious-metals-history-since-2000-with-news (CC)
# - Mirror: huggingface.co/datasets/SaguaroCapital/sentiment-analysis-in-commodity-market-gold
```

**Use:** training labels for a binary "is this gold-relevant?" classifier that filters the bigger corpora. NOT a feature input.

**Storage:** ~50 MB.

## DEFERRED news sources (V4 — owner can re-prioritize after V1 baseline)

| Source | Reason for defer |
|---|---|
| **Reuters Gold** | $1/wk paywall + Reuters Connect API enterprise-only ($5K-$50K+/yr). Plan: revisit when funded. CNBC syndicates Reuters wire for partial free coverage in the meantime. |
| **Financial Times Gold** | **LEGAL BLOCKER** — `ft.com/robots.txt` explicitly prohibits ML/AI use. DO NOT scrape. Revisit only if FT Datamining License is licensed formally. |
| **Trading Economics news** | News content is shallow + leaks Reuters wire (which we'd already have via CNBC syndication). Indicator API already in doc 02. |
| **FXStreet** | Best minute-ts technicals-embedded news but B2B-paid via Acuity Trading. Add when budget allows. |
| **Metals Daily** | Sharps Pixley aggregator — heavy syndication overlap with primary sources. Skip unless archive depth confirmed past 2-3y. |
| **ACLED conflict events** | Geopolitical supplement to GDELT. Free for non-commercial only — Arena is commercial in V2+. Skip until license review. |
| **Bloomberg.com free articles** | Limited free content + News/Media Alliance lawsuit pressure (BREIN forced takedowns 2025). Skip. |
| **Common Crawl + Bloomberg/CNBC/FT filters** | Same legal pressure. Skip in V1. |
| **NewsAPI.org / NewsCatcher** | Aggregator APIs, $50-500/mo. Worth evaluating after V1 baseline. |
| **Twitter/X full-archive search** | Pro tier $200/mo limited, Enterprise $42K+/yr. Skip. |
| **WGC Goldhub research articles (full body)** | Already have download/8052 + 7739 statistics. Research-article full-body scraping is ToS-gray. V1: title + summary only via Goldhub research-library page. |

## Point-in-Time Discipline (CRITICAL)

**The Rule:** every feature for bar at time T uses data with timestamp **strictly less than** T_close, and accounts for publication latency.

```python
NEWS_LATENCY_MIN = 15   # matches Alpaca free tier IEX 15-min price gate
                        # (Alpaca News itself is real-time, but live IEX price is +15min,
                        #  so end-to-end retail decision lag is ~15 min)

def news_window_for_bar(bar_close_utc: pd.Timestamp) -> tuple[pd.Timestamp, pd.Timestamp]:
    """News visible at bar T's close: published in [T-30min, T-news_latency)."""
    end = bar_close_utc - pd.Timedelta(minutes=NEWS_LATENCY_MIN)
    start = end - pd.Timedelta(minutes=30)
    return start, end
```

For FRED data, gate by `realtime_start ≤ T_close`, not by observation `date`.

## Snapshot Hashing (immutable artifacts)

```python
import hashlib

def snapshot_hash(df: pd.DataFrame) -> str:
    """Deterministic SHA256 of the joined dataframe."""
    return hashlib.sha256(df.to_csv().encode()).hexdigest()[:16]

snapshot = join_everything(...)
hash_id = snapshot_hash(snapshot)
snapshot.to_parquet(f"data/snapshots/v1_{hash_id}.parquet", compression="zstd")

# Meta JSON (track everything that produced this snapshot)
meta = {
    "snapshot_version": "v1",
    "snapshot_hash": hash_id,
    "created_utc": datetime.utcnow().isoformat() + "Z",
    "row_count": len(snapshot),
    "time_range_utc": [snapshot.index.min().isoformat(), snapshot.index.max().isoformat()],
    "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip(),
    "data_sources": [
        {"name": "alpaca_bars", "feed": "iex", "since": "2016", "license": "MIT-equivalent", "symbols": ["GLD"]},
        {"name": "alpaca_etfs", "feed": "iex", "since": "2016",
         "symbols": ["SPY", "QQQ", "IWM", "GDX", "SLV", "XLF", "XLE", "XLK", "XLU"]},
        {"name": "alpaca_news", "source": "benzinga", "since": "2015"},
        {"name": "gdelt_gkg", "version": "2.0", "snapshot_table": "nanogld-data.gold_news.gkg_5y"},
        {"name": "fred_alfred", "n_series": 34, "series": [
            "DGS3MO", "DGS6MO", "DGS2", "DGS5", "DGS10", "DGS30",
            "DFII5", "DFII10", "T5YIE", "T10YIE", "T5YIFR",
            "DTWEXBGS", "VIXCLS", "DCOILBRENTEU", "DCOILWTICO",
            "UNRATE", "PAYEMS", "ICSA", "CCSA", "JTSJOL",
            "CPIAUCSL", "CPILFESL", "PCEPI", "PCEPILFE",
            "GDPC1", "INDPRO", "RSAFS", "HOUST", "UMCSENT",
            "M2SL", "WALCL", "RRPONTSYD", "FEDFUNDS", "SOFR",
        ]},
        {"name": "yfinance", "version": "1.3.0", "tickers": ["BZ=F", "CL=F"]},
        {"name": "gpr_index", "url": "https://www.matteoiacoviello.com/gpr_files/data_gpr_export.xls"},
        {"name": "cftc_cot", "report": "disaggregated_futures_only", "contract": "GOLD - COMMODITY EXCHANGE INC.",
         "frequency": "weekly", "release_dow": "Friday 15:30 ET"},
        {"name": "wgc_central_bank", "url": "https://www.gold.org/goldhub/data/quarterly-central-bank-statistics",
         "frequency": "quarterly", "release_lag_weeks": 6},
        {"name": "calendar_events_v1", "events": ["FOMC", "CPI", "NFP", "GDP", "JOLTS", "PCE", "FOMC_minutes"],
         "source": "deterministic schedule built from BLS/BEA/Fed calendars"},
    ],
    "schema_version": "v1.0.0",
}

with open(f"data/snapshots/v1_{hash_id}_meta.json", "w") as f:
    json.dump(meta, f, indent=2)
```

## Golden Fixture Test (NON-NEGOTIABLE)

```python
def test_point_in_time_correctness():
    """Hand-built dataset where the right answer is known. Single most important test."""
    bars = pd.DataFrame([
        {"ts": "2024-01-15 14:00:00", "close": 100.0},
        {"ts": "2024-01-15 14:30:00", "close": 100.5},
        {"ts": "2024-01-15 15:00:00", "close": 100.3},
    ])
    news = pd.DataFrame([
        {"ts": "2024-01-15 14:25:00", "headline": "EARLY"},
        {"ts": "2024-01-15 14:35:00", "headline": "AFTER_BAR_2_CLOSE"},
    ])
    
    joined = join_with_pit_discipline(bars, news, latency_min=15)
    
    # Bar at 14:30 close → news must be < 14:30 - 15min = 14:15
    # Both news items are AFTER 14:15, so neither joins
    assert "EARLY" not in joined.loc[bar_idx_14_30, "news_headlines"]
    
    # Bar at 15:00 close → news must be < 15:00 - 15min = 14:45
    # "EARLY" at 14:25 → joins. "AFTER_BAR_2_CLOSE" at 14:35 → joins.
    assert "EARLY" in joined.loc[bar_idx_15_00, "news_headlines"]
    
    # No future leakage anywhere
    for bar_t, row in joined.iterrows():
        for headline_ts in row.news_timestamps:
            assert headline_ts + timedelta(minutes=15) < bar_t.close_time
```

Run on every CI / pre-commit. If it fails, do not commit.

## File Layout

```
data/
├── raw/
│   ├── alpaca_bars_GLD_30min_2021_2026.parquet
│   ├── alpaca_bars_SPY_30min.parquet           # NEW V1 — equity basket
│   ├── alpaca_bars_QQQ_30min.parquet
│   ├── alpaca_bars_IWM_30min.parquet
│   ├── alpaca_bars_GDX_30min.parquet           # gold miners
│   ├── alpaca_bars_SLV_30min.parquet           # silver
│   ├── alpaca_bars_XLF_30min.parquet           # sector ETFs
│   ├── alpaca_bars_XLE_30min.parquet
│   ├── alpaca_bars_XLK_30min.parquet
│   ├── alpaca_bars_XLU_30min.parquet
│   ├── alpaca_news_GLD_2021_2026.parquet
│   ├── gdelt_gkg_macro_5y.parquet              # exported from materialized BigQuery table
│   ├── fred_*_all_releases.parquet             # 34 series × 1 file each
│   ├── brent_daily.parquet
│   ├── wti_daily.parquet
│   ├── gpr_monthly.parquet
│   ├── cftc_cot_gold_weekly.parquet            # NEW V1 — COT disaggregated
│   ├── wgc_central_bank_quarterly.parquet      # NEW V1 — WGC flows
│   └── calendar_events_v1.parquet              # NEW V1 — deterministic event schedule
├── snapshots/
│   ├── v1_<hash>.parquet
│   └── v1_<hash>_meta.json
└── tests/
    └── test_data_join.py                       # golden fixture (extended for new sources)
```

## Day-by-Day Implementation (REVISED for verified pitfalls)

| Day | Block | Task | Output |
|-----|-------|------|--------|
| 1 AM | Setup | Alpaca paper account + keys + verify SDK works | `.env.paper` populated |
| 1 AM | Setup | FRED API key (free, instant) | `.env` updated |
| 1 PM | Setup | GCP project + billing + BigQuery + custom 1024 GiB/day quota | Working `bq` CLI |
| 1 PM | Pull | Alpaca historical bars 5y GLD 30min | `data/raw/alpaca_bars_GLD_*.parquet` |
| 1 PM | Pull | Alpaca ETF basket 5y 30min (SPY/QQQ/IWM/GDX/SLV/XLF/XLE/XLK/XLU) | `data/raw/alpaca_bars_<sym>_30min.parquet` × 9 |
| 2 AM | Pull | Alpaca News 5y (Benzinga, paginate via 50-cap) | `data/raw/alpaca_news_*.parquet` |
| 2 AM | Pull | FRED ALFRED 34 series, full vintage cubes (~17s wall, ~120 MB) | `data/raw/fred_*_all_releases.parquet` |
| 2 PM | Pull | yfinance Brent + WTI daily + GPR Excel | `data/raw/{brent,wti,gpr}_*.parquet` |
| 2 PM | Pull | CFTC COT weekly disaggregated for COMEX gold | `data/raw/cftc_cot_gold_weekly.parquet` |
| 2 PM | Pull | WGC central bank quarterly flows (browse-verify URL first) | `data/raw/wgc_central_bank_quarterly.parquet` |
| 3 AM | Pull | GDELT GKG 5y materialize via BigQuery (~931 GB scan, dry-run first!) | BQ table + parquet export |
| 3 AM | Build | Calendar event schedule (deterministic, hard-coded BLS/BEA/Fed dates) | `data/raw/calendar_events_v1.parquet` |
| 3 PM | Build | Joiner with point-in-time discipline (now ~12 sources) | `src/data/join.py` |
| 3 PM | Test | Golden fixture test (extended for COT release-time + calendar event) | `tests/test_data_join.py` (passes) |
| 4 AM | Run | Full join, snapshot, hash, meta JSON | `data/snapshots/v1_<hash>.parquet` |
| 4 PM | Validate | Schema validator + sanity plots (per-source coverage, NaN counts, vintage spot-checks) | Plots + report |

Total: **4-5 days** realistic (was 4 days pre-expansion; +1 day for ETF basket pull validation, COT parser, WGC URL discovery, calendar build).

## Top 10 Pitfalls (VERIFIED, all from research subagents)

1. **`TimeFrame.Minute_30` doesn't exist.** Use `TimeFrame(30, TimeFrameUnit.Minute)`.
2. **Alpaca bars default to unadjusted prices.** Always `adjustment="all"`.
3. **Alpaca free tier is IEX-only**, expect occasional empty/missing bars during low-volume periods. Build resilience.
4. **Latest 15min of intraday gated on free tier.** Live trading must run on bars ≥15 min old.
5. **Alpaca News is Benzinga ONLY.** Single source. Don't claim "Reuters + Benzinga" anywhere.
6. **`NewsClient` requires keys** despite stale docs.
7. **PDT applies to paper.** $100 live account is PDT-restricted from day 1 — max 3 day-trades per 5 business days unless equity > $25K.
8. **GDELT themes live on GKG**, not events table. Use `gkg_partitioned`. Many original theme codes were wrong (`MIL_CONFLICT`, `ECON_RECESSION`, `WB_654` — all corrected above).
9. **`get_series_as_of_date` returns DataFrame of all revisions, not a clean Series.** Must groupby tail(1) to collapse.
10. **yfinance 30m bars permanently capped at 60 days.** No workaround. Use Alpaca for 30m.

## Open Questions / TODOs

- [ ] After `pip install`, verify `alpaca-py>=0.43,<1.0`, `yfinance==1.3.0`, `fredapi==0.5.2`, `google-cloud-bigquery>=3.40`
- [ ] After GCP setup, verify dry-run estimate matches reality on a 1-month GKG slice (catches partition-prune mistakes early)
- [x] Confirm `T5YIE` (breakeven) is what we want — V1 expansion settles this: pull BOTH T5YIE + T10YIE + T5YIFR. doc 04 builds derived features from all three.
- [ ] Decide: do we backfill Brent/WTI 30m at all? Probably no — daily ffill is fine at our scale. Leave as 30m TODO if model regresses on smoothing.
- [ ] AI-GPR daily index (https://www.matteoiacoviello.com/ai_gpr.html) — worth fetching as additional feature?
- [ ] Verify all 9 ETFs in the basket (SPY/QQQ/IWM/GDX/SLV/XLF/XLE/XLK/XLU) return clean 5y of 30m bars on Alpaca free IEX feed before continuing — spawn Nia subagent if any fails.
- [ ] Verify CFTC disaggregated zip URLs are still live for years 2021-2026 (CFTC has rotated paths in past). Use `/browse` to confirm.
- [ ] Verify WGC central-bank-quarterly CSV exact URL (it changes — last verified URL pattern is form-based; may require `/browse` to extract direct download link).
- [ ] FOMC emergency dates outside scheduled meetings (e.g., March 2020 emergency cut) — V1 hard-codes scheduled-only. If model misses regime breaks, add emergency dates.

### Deferred specialty signals (owner declined 2026-05-04 — re-ask after V1 baseline)

These were surfaced to owner on 2026-05-04 dataset expansion, owner did NOT select. Track here so they don't get lost:

- [ ] **Gold-IV / credit / bond vol bundle** — GVZ (CBOE Gold VIX, direct gold IV), HY OAS (BAMLH0A0HYM2), IG OAS (BAMLC0A0CM), MOVE (bond vol). All free from FRED + CBOE. Adds ~4 series. Re-ask if model misses vol-spike regimes.
- [ ] **USD cross-rates / crypto / industrial metals bundle** — DEXUSEU, DEXJPUS, DEXCHUS, DEXUSUK direct FX; BTC-USD/ETH-USD daily; HG=F (Dr. Copper), SI=F silver futures, PL=F platinum, PA=F palladium. All free from FRED/yfinance. Re-ask if alt-store-of-value regime not captured by SLV/GDX/DTWEXBGS combination.
