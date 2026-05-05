# data/

Generated artifacts for nanoGLD. Everything in `data/raw/`, `data/snapshots/`,
`data/embeddings/`, and `data/anchors/` is gitignored — content-addressed and
reproducible from `python -m nanogld.data build`.

## Quickstart

```bash
# 1. Populate ~/.config/nanogld/.env.paper (see docs/SETUP.md).
# 2. Authenticate GCP for BigQuery (one-time).
gcloud auth application-default login

# 3. Pull every source + run the joiner + write a hashed snapshot.
make data         # equivalent to: python -m nanogld.data build

# 3a. Pull only the no-key sources (calendar / COT / WGC / GPR / yfinance):
python -m nanogld.data build --skip-keyed

# 4. (One-time, ~931 GB scan) Materialize 5y of GDELT GKG into BigQuery:
python -m nanogld.data pull gdelt_materialize
# Then pull the materialized table back as a parquet:
python -m nanogld.data pull gdelt
```

## Layout

```
data/
├── raw/                                       per-source primaries
│   ├── alpaca_bars_GLD_30min.parquet          GLD 30m × 5y (Alpaca)
│   ├── alpaca_bars_<SYM>_30min.parquet × 9    SPY/QQQ/IWM/GDX/SLV/XLF/XLE/XLK/XLU
│   ├── alpaca_news_GLD.parquet                Benzinga news (Alpaca)
│   ├── gdelt_gkg_5y.parquet                   exported from BigQuery materialized table
│   ├── fred_<series>_all_releases.parquet × 35  ALFRED vintage cubes
│   ├── brent_daily.parquet, wti_daily.parquet yfinance daily futures
│   ├── gpr_combined.parquet                   Caldara-Iacoviello GPR + AI-GPR
│   ├── cftc_cot_gold_weekly.parquet           COT disaggregated, gold contract 088691
│   ├── wgc_central_bank_monthly.parquet       WGC central-bank flows (when /browse-fetched)
│   ├── calendar_events_v1.parquet             FOMC/CPI/NFP/GDP/JOLTS/PCE schedule 2021-2026
│   ├── fnspid_gold_relevant.parquet           HF FNSPID filtered to gold tickers
│   ├── kitco_news_recent.parquet              Kitco RSS recent
│   ├── investing_gold_news.parquet            investing.com gold-news scrape
│   ├── bullionvault_news.parquet              BullionVault author pages
│   ├── central_bank_news.parquet              BIS/ECB-FED HF + Treasury/CFTC scrapes
│   ├── reddit_gold_filtered.parquet           Arctic Shift filtered (owner downloads dumps)
│   ├── kaggle_gold_labeled.parquet            Kaggle gold-labeled (HF mirror)
│   ├── gpr/                                   self-snapshot vault keyed by fetch-ts + sha
│   └── wgc/                                   ditto
├── snapshots/
│   ├── v1_<sha>.parquet                       bar-aligned joined snapshot
│   └── v1_<sha>_meta.json                     full source manifest + git commit
├── embeddings/                                doc 03 fills
└── anchors/                                   doc 04 fills
```

## Hard rules (V4 — silently kills models if violated)

Every source emits `release_ts` + `t_visible` columns; the joiner enforces
`release_ts <= t_visible` before joining and uses strict `<` asof
(no exact matches) on `bar_close_utc`. The 28 mandatory tests under
`tests/test_no_leakage.py` are CI gates.

| # | Rule | Check |
|---|------|-------|
| §1 | Bar visibility = bar END (`timestamp + 30min`) | `test_bar_visibility_is_bar_end` |
| §2 | Alpaca News uses `created_at`, NOT `published_at`/`updated_at` | `test_news_uses_created_at_not_updated_at` |
| §3 | `DFF` for daily Fed Funds, NOT `FEDFUNDS` (monthly) | `test_dff_replaces_fedfunds_for_daily` |
| §4 | FRED release-tod table covers every series | `test_fred_release_tod_table_complete` |
| §5 | `get_series_all_releases` (ALFRED), never current snapshot | `test_fred_uses_alfred_realtime_period` |
| §6 | GDELT theme codes from V4-corrected list | `test_gdelt_theme_codes_in_master_list` |
| §7 | GDELT 30-min buffer (NOT 15) | `test_gdelt_buffer_30min_not_15` |
| §8 | WGC URL `gold.org/download/{8052,7739}` (form-walled) | `test_wgc_url_is_correct_self_snapshot` |
| §9 | AI-GPR has 30-day lag | `test_aigpr_treated_as_monthly_lag` |
| §10 | GPR self-snapshot weekly with fetch-ts vintage | `test_gpr_uses_self_snapshot_not_live` |
| §11 | pandas-ta KAMA/Ichimoku/KST/DPO/TRIX/Vortex banned | `test_no_pandas_ta_kama_ichimoku_kst_dpo_trix` |
| §12 | Multi-symbol Alpaca pagination drained (`limit=None`) | `test_multisymbol_pagination_drained` |
| §14 | Calendar features = binary windows ONLY | `test_no_minutes_until_event_features` |
| §15 | COT visibility = Friday 16:00 ET (15:30 + 30min buffer) | `test_cot_t_visible_is_friday_330pm_et` |
| §15 | Holiday-Friday rolls to next NYSE session | `test_cot_holiday_friday_uses_monday_release` |
| §16 | WALCL Thursday 16:30 ET visibility | `test_walcl_thursday_visibility_after_1630_et` |
| §17 | ICSA Thursday 08:30 ET visibility | `test_icsa_thursday_visibility_after_0830_et` |

## CLI

```bash
python -m nanogld.data list                  # show every source + output filename
python -m nanogld.data pull <name>           # one source (idempotent — skips if parquet exists)
python -m nanogld.data pull <name> --force   # force re-fetch
python -m nanogld.data join                  # re-run joiner from existing parquets
python -m nanogld.data build                 # full pull + join + snapshot
python -m nanogld.data build --skip-keyed    # only the no-key sources
```

## News-pipeline source matrix (2026-05 update)

| Source module | Bias tier | License | Default state | V1-window depth | Notes |
|---|---|---|---|---|---|
| `news_central_bank.py` | central_bank_official + government_official | public domain (US 17 USC §105) + ECB free-research | ✅ pulled (4988 rows, 683 in window) | ECB+FED 1996-2026 | regional Feds attempted (Cleveland/Chicago/NY/SF/Atlanta) — 0 rows so far, selectors per-Fed needed |
| `news_polygon.py` | mainstream_neutral | Polygon ToS — paid Starter+ for commercial; free non-commercial | gated NANOGLD_POLYGON_PAID=1 | TBD (5y if free tier supports `/v2/reference/news`, else paid only) | drop-in for dropped Alpaca News; Benzinga partner content baked in per Polygon June 2025 deal |
| `news_alpha_vantage.py` | per-outlet (mainstream_neutral / aggregator_neutral / retail_pundit) | AV free-tier ToS — non-commercial; paid for commercial | needs ALPHA_VANTAGE_API_KEY | 4y back to 2022-03 (multi-day backfill journal) | sentiment scores stored under body field as JSON |
| `news_kitco.py` | industry_bullish | Kitco ToS grey-zone (existing IA captures = fair use) | Wayback CDX backfill | TBD (long soak ~5 hr full 5y at 2 s/req) | 5y article URL pattern `kitco.com/news/article/*` |
| `news_investing.py` | aggregator_neutral | research-use grey | Wayback CDX backfill | TBD | `investing.com/news/commodities-news/*` glob |
| `news_bullionvault.py` | dealer_bullish | research-use grey | Wayback CDX backfill | TBD (smaller corpus) | LAFTR head down-weights at inference per doc 03 |
| `news_fnspid.py` | per-outlet | **CC BY-NC-4.0** (NON-COMMERCIAL) | gated NANOGLD_NONCOMMERCIAL=1 | 1999-2023 (15.7M articles total; gold-relevant filter ~50-200 MB) | parallel `Dataset.filter(num_proc=4)` for 10× speed |
| `news_multisource.py` | per-outlet | NON-COMMERCIAL "Other" | gated NANOGLD_NONCOMMERCIAL=1 + needs HF_TOKEN (gated dataset) | 1990-2025 (57M rows; Reuters/Bloomberg/Benzinga subsets) | HF dataset is HF-gated — owner adds HF_TOKEN |
| `news_reddit.py` | retail_social | Reddit ToS (academic distribution grey) | `open-index/arctic` HF mirror via DuckDB | ⚠️ HF mirror ends 2017-04 — does NOT cover V1 window | owner downloads post-2017 Arctic Shift `.jsonl.zst` to `data/raw/reddit/` for full coverage |
| `news_kaggle.py` | labeled_corpus | CC0 / CC BY (training corpus, NOT runtime feature) | ✅ pulled (8329 rows 2000-2019) | n/a — used only to train a binary gold-relevance classifier in doc 03 |
| `gdelt.py` | geopolitical themes | public domain | ✅ owner authed gcloud; needs BigQuery API enable in `nanogld` project | 5y (~931 GB scan from 1 TB free quota one-shot) | enable at https://console.developers.google.com/apis/api/bigquery.googleapis.com/overview?project=nanogld |

## Owner-action checklist (news pipeline)

| When | What | Time |
|---|---|---|
| Now | Sign up at `alphavantage.co/support/#api-key` (email-only, free, instant), paste `ALPHA_VANTAGE_API_KEY=...` into `~/.config/nanogld/.env.paper` | 2 min |
| Now | Add `HF_TOKEN=...` to `.env.paper` (free, huggingface.co/settings/tokens, read scope) — unblocks multi-source HF dataset which is gated | 2 min |
| Now | Enable BigQuery API in GCP `nanogld` project: https://console.developers.google.com/apis/api/bigquery.googleapis.com/overview?project=nanogld | 1 min |
| Optional | Decide: pay $29/mo Polygon News Stocks Starter (set `NANOGLD_POLYGON_PAID=1`) for clean Benzinga firehose, or keep all-free Wayback path | decision |
| ✅ DONE | gcloud ADC authed (~/.config/gcloud/application_default_credentials.json) |
| ✅ DECIDED | NANOGLD_NONCOMMERCIAL=1 default ON for V1 personal/research training |

## Open follow-ups

- **WGC** direct download URLs (gold.org/8052 + 7739) hit a HTML form-wall on
  bare GET. Owner extracts the real signed URL via `/browse` then patches
  `src/nanogld/data/wgc.py`.
- **CPI / JOLTS / PCE / GDP** dates are deterministic approximations for V1
  (CPI 12th of month, JOLTS 9th, PCE last BD). Owner `/browse`-verifies the
  exact BLS/BEA calendar before training (spec line 1054).
- **Reddit post-2017** — `open-index/arctic` HF mirror only covers 2005-2017.
  Owner downloads Arctic Shift torrents for 2018-2026 to `data/raw/reddit/`
  and writes a `news_reddit_local.py` parser (or extends existing module).
- **Regional Fed scrapers** (Cleveland/Chicago/NY/SF/Atlanta) currently use
  the generic `_scrape_index_page` selector heuristic which doesn't match
  any of those Fed sites' modern HTML. Per-Fed selector work needed (or
  Wayback CDX fallback per site).
- **GDELT BigQuery** — owner enables BigQuery API in the `nanogld` GCP
  project (one-click at the URL above), then `python -m nanogld.data pull
  gdelt_materialize` + `gdelt`.
- **Wayback CDX soak time** — per-source 5y full backfill at 2 s/req polite
  rate is 5+ hours wall time per source. Run overnight or during a coffee
  break; the cache resumes cleanly on re-run.
