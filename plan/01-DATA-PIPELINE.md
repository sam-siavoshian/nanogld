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
- Estimated data footprint: **~7-8 GB**
- Utilization: 8 / 40 = **20% of budget** → KEEP LOCAL

**Decision rule (re-run before you start coding):**
```bash
# Verify current free space before deciding
df -h /System/Volumes/Data
```
- IF total data footprint > 60% of CURRENT free space → push raw GDELT to BigQuery, keep snapshots local
- ELSE keep everything local (current verdict)

### What Lives Where (current verdict: mostly local)

```
LOCAL ~/Desktop/Coding Stuff/Side Projects/ML-Trading/data/         ~7-8 GB total
├── raw/
│   ├── alpaca_bars_GLD_30min.parquet            2 MB
│   ├── alpaca_news_GLD.parquet                  ~150 MB
│   ├── fred_*_all_releases.parquet              ~20 MB
│   ├── brent_daily.parquet, wti_daily.parquet   <1 MB each
│   └── gpr_monthly.parquet                       <1 MB
├── snapshots/
│   ├── v1_<hash>.parquet                        ~500 MB - 1 GB
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
├── alpaca_news.py          # Alpaca News API (Benzinga, since 2015)
├── gdelt.py                # GDELT 2.0 GKG via BigQuery (themes, materialize once)
├── fred.py                 # FRED + ALFRED for vintage-correct macro
├── yfinance_helpers.py     # Brent/WTI daily (with curl_cffi wrapper)
├── gpr.py                  # GPR Index monthly download from matteoiacoviello.com
├── join.py                 # Point-in-time-correct joiner with 15min news latency
├── schema.py               # Pydantic schemas for validation
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

- `src/nanogld/features/` — doc 02
- `src/nanogld/embed/` — doc 04
- `src/nanogld/model/` — doc 03
- `src/nanogld/training/` — doc 05
- Any doc-NN.md other than this one
- `.pre-commit-config.yaml`, `pyproject.toml`, `.gitignore` — doc 10 owns these (you can ASK them to add a dep, don't edit yourself)

### Stable Interface You Publish (other docs read against this)

`data/snapshots/v1_<hash>.parquet` with the schema documented in this doc's "Dataset Schema" section. Doc 02 reads this. Doc 04 reads this. If you change the schema, update this doc, ping STATUS.md, AskUserQuestion before shipping.

### Acceptance Criteria

You're done when:

1. ✅ `python -m nanogld.data build` produces `data/snapshots/v1_<sha256_first_16>.parquet` with full 5y of joined data + accompanying `_meta.json`
2. ✅ `pytest tests/test_pit.py` passes (golden fixture for point-in-time joiner)
3. ✅ `pytest tests/test_join_schema.py` passes (every column matches schema, no NaN in non-nullable cols)
4. ✅ `pytest tests/test_snapshot_hash.py` passes (running build twice on same input produces identical hash)
5. ✅ Row count is approximately 16K bars (5y × 252 days × 13 RTH bars/day = ~16,380; allow ±5% for holidays)
6. ✅ News coverage report shows ≥30% of bars have ≥1 Alpaca News article + ≥30% have ≥1 GDELT event in window
7. ✅ FRED ALFRED vintage cubes saved for all 7 series (DTWEXBGS, DGS10, DGS2, T5YIE, VIXCLS, DCOILBRENTEU, DCOILWTICO)
8. ✅ A README in `data/` explains how another developer reproduces the pipeline

Hand off to doc 02 (feature engineering) by updating STATUS.md with the snapshot hash + meta.json path.

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
**Implementation effort:** 2-3 days (revised up from 1.5-2 after pitfall discovery)
**Last verified:** 2026-04-30

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
- ~13 RTH bars/day × 252 days/yr × 5 yr ≈ **16K rows** (not 87K — yfinance/futures math doesn't apply)
- Expect occasional NaN/missing bars on IEX feed (low volume)

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
    -- GOLD-related (CORRECTED theme codes)
    REGEXP_CONTAINS(g.V2Themes, r'WB_2936_GOLD|ECON_GOLDPRICE|WB_2937_SILVER|SLFID_MINERAL_RESOURCES')
    -- MONETARY (verified)
    OR REGEXP_CONTAINS(g.V2Themes, r'ECON_INTEREST_RATES|ECON_INFLATION|ECON_CENTRALBANK|EPU_CATS_MONETARY_POLICY|EPU_POLICY_FEDERAL_RESERVE|WB_1235_CENTRAL_BANKS|WB_444_MONETARY_POLICY|EPU_UNCERTAINTY')
    -- CONFLICT (CORRECTED — original codes don't exist)
    OR REGEXP_CONTAINS(g.V2Themes, r'ARMEDCONFLICT|WB_2433_CONFLICT_AND_VIOLENCE|WB_2432_FRAGILITY|TERROR|SANCTIONS|TAX_WEAPONS_BOMB|MARITIME_INCIDENT')
    -- ECONOMIC STRESS (replacement for non-existent ECON_RECESSION)
    OR REGEXP_CONTAINS(g.V2Themes, r'EPU_ECONOMY_HISTORIC|ECON_BANKRUPTCY|ECON_TRADE_DISPUTE|ECON_DEBT')
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
3. **15-min batching delay.** GDELT publishes every 15 minutes; events appear in BigQuery ~15-45 min after wall clock.
4. **Multiple events per article.** ONE `SOURCEURL` → many event rows (one per Actor1-action-Actor2 dyad). Dedupe on URL in features.
5. **`V2Themes` is semicolon-delimited with comma char-offsets.** Use `REGEXP_CONTAINS`, not equality.
6. **Single accidental `SELECT * FROM gkg` (non-partitioned) = 21 TB scan = $130 if billing on.** Always set `maximum_bytes_billed`.
7. **Multilingual.** Filter `TranslationInfo = ''` for English-only v1.

## Source 4 — FRED + ALFRED (vintage-correct macro)

### Series IDs (verified, with corrections)

| Series ID | Description | Vintage horizon | Revisions in practice |
|-----------|-------------|-----------------|----------------------|
| `DTWEXBGS` | Nominal Broad USD Trade-Weighted Index | 2006+ | Rare |
| `DGS10` | 10Y Treasury Constant Maturity | 2006+ | Essentially never |
| `DGS2` | 2Y Treasury Constant Maturity | 2006+ | Essentially never |
| `T5YIE` | **5-Year Breakeven Inflation Rate** (NOT forward — that's T5YIFR) | 2006+ | Occasional |
| `VIXCLS` | VIX close | 2006+ | Rare |
| `DCOILBRENTEU` | Brent crude spot | 2006+ | Occasional |
| `DCOILWTICO` | WTI crude spot | 2006+ | Occasional |

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

for series_id in ["DTWEXBGS", "DGS10", "DGS2", "T5YIE", "VIXCLS", "DCOILBRENTEU", "DCOILWTICO"]:
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

**Pin: `yfinance==1.3.0`** in `pyproject.toml`. v1.3.0 fixed an April 2026 dividends breakage; do not auto-upgrade.

**Pitfalls:**
- 30m bars **capped at 60 days** by Yahoo — never going to work for 5y. We don't try.
- BZ=F and CL=F are continuous front-month futures. Yahoo handles rolls but injects phantom returns at roll dates. For our use (daily features only, ffilled to 30min), the noise is acceptable.
- Brent trades on ICE (UK calendar), WTI on NYMEX (US calendar). Different holidays. Use `pandas_market_calendars` to align sessions.

## Source 6 — GPR Index (Caldara & Iacoviello, monthly)

```python
import pandas as pd

# Live download URL (verified Apr 2026)
GPR_URL = "https://www.matteoiacoviello.com/gpr_files/data_gpr_export.xls"

# Cache locally — only updates monthly
gpr = pd.read_excel(GPR_URL)
gpr.to_parquet("data/raw/gpr_monthly.parquet")
```

Update cadence: monthly batch, daily resolution within batch, ~1 month publication lag. Plan feature pipeline around this lag.

Bonus: AI-GPR daily index (LLM-generated, 1960-present) at https://www.matteoiacoviello.com/ai_gpr.html. Worth using as higher-resolution feature.

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
        {"name": "alpaca_bars", "feed": "iex", "since": "2016", "license": "MIT-equivalent"},
        {"name": "alpaca_news", "source": "benzinga", "since": "2015"},
        {"name": "gdelt_gkg", "version": "2.0", "snapshot_table": "nanogld-data.gold_news.gkg_5y"},
        {"name": "fred_alfred", "series": ["DTWEXBGS", "DGS10", "DGS2", "T5YIE", "VIXCLS", "DCOILBRENTEU", "DCOILWTICO"]},
        {"name": "yfinance", "version": "1.3.0", "tickers": ["BZ=F", "CL=F"]},
        {"name": "gpr_index", "url": "https://www.matteoiacoviello.com/gpr_files/data_gpr_export.xls"},
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
│   ├── alpaca_news_GLD_2021_2026.parquet
│   ├── gdelt_gkg_macro_5y.parquet            # exported from materialized BigQuery table
│   ├── fred_dtwexbgs_all_releases.parquet
│   ├── fred_dgs10_all_releases.parquet
│   ├── fred_dgs2_all_releases.parquet
│   ├── fred_t5yie_all_releases.parquet
│   ├── fred_vixcls_all_releases.parquet
│   ├── fred_dcoilbrenteu_all_releases.parquet
│   ├── fred_dcoilwtico_all_releases.parquet
│   ├── brent_daily.parquet
│   ├── wti_daily.parquet
│   └── gpr_monthly.parquet
├── snapshots/
│   ├── v1_<hash>.parquet
│   └── v1_<hash>_meta.json
└── tests/
    └── test_data_join.py                      # golden fixture
```

## Day-by-Day Implementation (REVISED for verified pitfalls)

| Day | Block | Task | Output |
|-----|-------|------|--------|
| 1 AM | Setup | Alpaca paper account + keys + verify SDK works | `.env.paper` populated |
| 1 AM | Setup | FRED API key (free, instant) | `.env` updated |
| 1 PM | Setup | GCP project + billing + BigQuery + custom 1024 GiB/day quota | Working `bq` CLI |
| 1 PM | Pull | Alpaca historical bars 5y GLD 30min (`adjustment="all"`, `feed="iex"`) | `data/raw/alpaca_bars_*.parquet` |
| 2 AM | Pull | Alpaca News 5y (Benzinga, paginate via 50-cap) | `data/raw/alpaca_news_*.parquet` |
| 2 AM | Pull | FRED ALFRED 7 series, full vintage cubes | `data/raw/fred_*_all_releases.parquet` |
| 2 PM | Pull | yfinance Brent + WTI daily + GPR Excel | `data/raw/{brent,wti,gpr}_*.parquet` |
| 3 AM | Pull | GDELT GKG 5y materialize via BigQuery (~931 GB scan, dry-run first!) | BQ table + parquet export |
| 3 PM | Build | Joiner with point-in-time discipline | `src/data/join.py` |
| 3 PM | Test | Golden fixture test | `tests/test_data_join.py` (passes) |
| 4 AM | Run | Full join, snapshot, hash, meta JSON | `data/snapshots/v1_<hash>.parquet` |
| 4 PM | Validate | Schema validator + sanity plots (price coverage, news coverage by month, NaN counts) | Plots + report |

Total: 4 days realistic (revised from 1.5-2 in skeleton).

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
- [ ] Confirm `T5YIE` (breakeven) is what we want, not `T5YIFR` (forward) — both available on ALFRED, picking based on whether breakeven or forward aligns with our gold thesis
- [ ] Decide: do we backfill Brent/WTI 30m at all? Probably no — daily ffill is fine at our scale. Leave as 30m TODO if model regresses on smoothing.
- [ ] AI-GPR daily index (https://www.matteoiacoviello.com/ai_gpr.html) — worth fetching as additional feature?
