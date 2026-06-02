# 02 — Feature Engineering

## ✅ STATUS: COMPLETE 2026-05-08

**681 features per bar in the unified dataset.** Composition:
- v1 base (635 cols): per-asset OHLCV+lag1, 4 log-ret horizons + 2 RV horizons + RS-vs-SPY + 30d gold-corr × 27 assets, plus 7 cross-asset ratios, 117 macro YoY/MoM, treasury term structure, COT, GPR, WGC, GDELT, calendar
- v2 engineered (46 cols, src/nanogld/features/v2_engineered.py): cross-asset interactions (4), volatility regime (5), calendar/event (7), regime+microstructure (8), macro term structure (6), momentum extensions (10), news×price (5), news anchor cosines (14)

**3 paranoid audit iterations × 8 agents each = 19 bugs caught and fixed** before shipping. Final state: zero leakage (max |corr| with next_log_return = 0.0117), zero inf, zero 100%-NaN cols, all rolling windows PIT-correct, all warmup periods correctly NaN.

**Output baked into unified dataset:** `data["features"]` is `(75993, 681) float32` inside `training_v1_unified.pt`. Column names in `data["feature_names"]`.

**Read `plan/HANDOFF.md` for full context.** Original spec retained below for archival.

---

## V1 HARD RULES (2026-05-08, additive to V1)

The V1 spec sheet (`plan-edit/V1-SPEC.md`) overrides any conflicting V1 line below. Hard rules specific to feature engineering:

1. **Triple-barrier labels with spread-adjusted neutral threshold.** Replaces fixed 5-bps threshold from V1. ATR-14 barriers, 1-bar timeout. See section "Triple-Barrier Labels (V1)".
2. **Half-hour-5 feature mandatory.** `gld_h5_log_return` + `gld_h5_x_vol_high` interaction. Gao-Han-Li-Zhou 2014 single-feature 5.43 Sharpe prior, GLD-specific.
3. **SimPSI / Wave-Mask augmentation only.** Naive jittering FORBIDDEN (Fons 2020 arXiv:2010.15111 net-negative on Sharpe). Manifold Mixup at hidden states only, never raw input.
4. **VSN feature gate at input is mandatory.** Variable Selection Network (Lim 2021, arXiv:1912.09363). Lives in `src/nanogld/model/`, but documented here because it shapes how the 681-feature input is consumed.
5. **Order of operations: decomp -> RevIN -> VSN -> patch projection -> backbone.** Series decomposition (24-bar MA kernel) splits trend + seasonal pre-RevIN. RevIN is per-channel (681 instances), not per-group.

---

## YOU ARE THE FEATURE ENGINEER AGENT

You own feature construction. You take immutable parquet snapshots from doc 02 + cached embeddings from doc 03 and produce the feature DataFrame that doc 05 (training) consumes.

**Read 00-OVERVIEW.md FIRST.** Project context is there.
**Read 02-DATA-PIPELINE.md schema section.** Your input is its output.
**Also read 00-OVERVIEW.md "Execution Mode" section before coding.**

### Execution Mode (short — full rules in 00-OVERVIEW.md)

**This doc IS the plan. Do not rewrite it.** Plan to *execute* what is written. If a claim is wrong, AskUserQuestion. Silent scope drift = fired.

- **Research with Nia** before guessing on libs / APIs / papers. Spawn a subagent: `nia papers` / `nia github` / `nia search` / `nia oracle` / `nia packages` / `nia tracer`.
- **Execution skills only:** `/investigate`, `/review`, `/qa`, `/qa-only`, `/cso`, `/benchmark`, `/ship`, `/land-and-deploy`, `/canary`, `/document-release`, `/retro`, `/learn`, `/browse`, `/setup-browser-cookies`.
- **NO planning skills:** `/office-hours`, `/plan-*`, `/autoplan`, `/design-*`, `/devex-review`. Planning is done.
- **Loop:** read doc → Nia for unknowns → code → `/review` → `/qa` → `/cso` (if security-sensitive) → `/ship`.
- **Escalate after 3 failed attempts.** AskUserQuestion. Bad work is worse than no work.

### Files You Create

```
src/nanogld/features/
├── __init__.py
├── price.py                # 12 price features (log returns, RSI, MACD, BB%B, etc.)
├── risk.py                 # 8 risk features (Garman-Klass vol, FOMC proximity)
├── macro.py                # ~12 short macro features (FX broad, oil, VIX, GPR)
├── geo.py                  # 10 geopolitical features (oil, GPR, GDELT aggregation)
├── sentiment.py            # 3 multi-dim sentiment scores (polarity + intensity + uncertainty)
├── equity.py               # NEW V1 expansion — 9-ETF basket features + cross-correlations + ratios
├── treasury.py             # NEW V1 expansion — full curve + TIPS + breakevens + spreads + butterfly
├── macro_bundle.py         # NEW V1 expansion — full 19-series macro (labor + inflation + growth + Fed)
├── cot.py                  # NEW V1 expansion — CFTC COT positioning features
├── wgc.py                  # NEW V1 expansion — WGC central bank flow features
├── calendar.py             # NEW V1 expansion — event proximity + cyclical (sin/cos) features
├── h5.py                   # V1 NEW — half-hour-5 intraday momentum (Gao 2014 prior, 2 dims)
├── spread.py               # V1 NEW — bid-ask spread feature (5-min trailing avg, 1 dim)
├── decomposition.py        # V1 NEW — 24-bar MA series decomposition (trend + seasonal)
├── labels.py               # V1: triple-barrier with spread-adjusted neutral (replaces 5-bps fixed)
├── normalize.py            # Rolling z-score with clip(-10, 10)
├── revin.py                # V1: RevIN per individual channel (681 instances, was per-group)
├── anchor_cosines.py       # Computes 4 anchor cosines per news source per bar
├── build.py                # End-to-end pipeline: snapshot → feature DataFrame
└── cli.py                  # `python -m nanogld.features build`

data/anchors/
└── v1.npz                  # Precomputed anchor embeddings (one-time)

tests/
├── test_no_leakage.py      # Modify raw[T+1], features[T] must not change
├── test_label_alignment.py # Labels match ±sign of next bar return
├── test_zscore_pit.py      # Z-score uses only past data
└── test_garman_klass.py    # Vol estimator math correctness
```

### Files You DO NOT Touch

- `src/nanogld/data/` — doc 02 owns
- `src/nanogld/embed/` — doc 03 owns
- `src/nanogld/model/` — doc 05
- Anything else in src/nanogld/

### Stable Interface You Publish

`build_feature_table(snapshot_path, embeddings_path, anchors_path) -> pd.DataFrame` — returns DataFrame with columns specified in this doc's "Per-Bar Input Vector" section. Total **~1000 dims per bar after V1 dataset expansion (2026-05-04)** (was ~804). Channel-group count grows from ~14 → ~25 — model architecture (doc 05) absorbs this with a wider input projection layer; no arch change needed.

### Acceptance Criteria

1. ✅ `python -m nanogld.features build` produces feature DataFrame from snapshot + cached embeddings
2. ✅ All 4 leakage tests pass — including new ones for equity ETFs, COT release-time, WGC release-time, calendar event windows
3. ✅ Class distribution at 5bps threshold roughly 28/44/28 (DOWN/FLAT/UP)
4. ✅ Z-scored features have mean ≈ 0, std ≈ 1, range bounded by `clip(-10, 10)`
5. ✅ Anchor cosines distribution looks reasonable (conflict_sim spikes during known events — verify on Russia/Ukraine 2022, Iran tensions 2024-25)
6. ✅ Feature build runs in <3 min for 16K bars (parquet I/O dominates; budget grew with expansion)
7. ✅ No NaN in final DataFrame except for the first ~3300 rows (rolling features warm-up grew from ~1000 due to YoY macro change features needing 1 year of history)
8. ✅ Equity feature sanity: GDX-GLD 30d corr in [0.4, 0.95], gold-silver ratio in [50, 100] historically
9. ✅ Treasury feature sanity: spread_10y_2y matches FRED's plot, real_rate_10y_direct matches DFII10 directly
10. ✅ Macro feature sanity: UNRATE_yoy spikes in March 2020 + 2022, CPI_yoy spikes 2021-2022
11. ✅ COT feature sanity: mm_net_long_pct_oi range [10%, 70%] historically
12. ✅ Calendar feature sanity: is_FOMC_release_window=1 on FOMC days only; cyclical encoders are continuous

### Spawn Nia Agents When You Need To

Specifically:
- pandas-ta-classic API (versions change). Verify `pta.rsi(...)` signature on current PyPI.
- GDELT theme aggregation patterns (most papers use sum, but per arXiv:2505.16136 mean tone + tone std beats counts)
- FRED ALFRED vintage lookup edge cases (weekends/holidays = NaN — decide explicitly)
- Whether your label threshold (5bps) gives the right class balance — A/B sweep 3/5/10 bps if curious

### Critical Corrections from Nia Verification (DO NOT REVERT)

1. **`ta` library is stale + has wrong API.** Use **`pandas-ta-classic==0.5.44`** (active fork, April 2026).
2. **Garman-Klass > Parkinson** for OHLC data (7.4× more efficient). We have OHLC; use GK.
3. **GDELT theme codes** — many "standard" codes don't exist. Verified canonical codes are in this doc. Use them.
4. **`pandas_market_calendars.get_calendar('NYSE')`** — NOT 'NYSEARCA' (not in canonical registry; GLD follows NYSE hours anyway).
5. **T5YIE = "5-Year Breakeven Inflation Rate"** (NOT forward — that's `T5YIFR`). We use breakeven.
6. **Add `clip(-10, 10)` after z-scoring** to bound outliers from near-zero std.
7. **L2-normalize 4096-dim news embeddings before projection** (StockTime pattern).
8. **Multi-dimensional sentiment > scalar polarity** per arXiv:2603.11408 (March 2026). Add intensity + uncertainty extracted via LLM prompts.

### V1 Updates (May 2026)

- News embedding dim changed: 4096 (earlier Llama-3.1-8B) → 256 (V1 Qwen3-Embedding-4B truncated via MRL). See doc 03 for details.
- Add 3 multi-dim sentiment features per news source (polarity / intensity / uncertainty) per arXiv:2603.11408.
- Anchor-cosine pattern unchanged but now uses Qwen3 embeddings.

### V4 Leakage Audit Corrections (2026-05-04 — MANDATORY)

5 Nia subagents verified every source. **Every fix below is a hard rule.** Read 02-DATA-PIPELINE.md "Verification Round 4" for the full list. Highlights for feature engineering:

1. **Bar visibility = bar END.** All `df.shift(1)` patterns in this doc assume bar-end indexing. With Alpaca's bar-START convention, the visibility column is `t_visible = bar.timestamp + 30min`. Re-read every `.shift(1)` in this doc as "shift to t_visible", not "shift to bar timestamp".
2. **Replace `FEDFUNDS` with `DFF` for DAILY features.** FEDFUNDS is monthly. Doc references using `FEDFUNDS` as a daily series are wrong — switch to `DFF`. Keep `FEDFUNDS` only for monthly aggregates.
3. **GDELT theme codes — 6 REFUTED:** drop `EPU_CATS_MONETARY_POLICY`, `EPU_POLICY_FEDERAL_RESERVE`, `EPU_UNCERTAINTY`, `EPU_ECONOMY_HISTORIC`, `TAX_WEAPONS_BOMB`. Replace `WB_2432_FRAGILITY` → `WB_2432_FRAGILITY_CONFLICT_AND_VIOLENCE`. Replace `TAX_WEAPONS_BOMB` → `TAX_WEAPONS`.
4. **GDELT buffer 30min not 15min.**
5. **News field is `created_at` (NOT `published_at`).** Use `t_visible = created_at + 60s`. Never join on `updated_at`.
6. **Anchor-cosine leakage:** anchor headlines must be either (a) hand-crafted templates with no event provenance, OR (b) sampled only from BEFORE train period. Otherwise the anchor set itself encodes future events.
7. **`pandas-ta` indicator audit:** KAMA, Ichimoku (`visual=True`), KST, DPO, TRIX, Vortex are **FORBIDDEN** (confirmed look-ahead bugs, `bukosabino/ta#181`). RSI, MACD, BBANDS proper are causal IFF `min_periods` respected and no `bfill`. Add growing-window stability test for every indicator: `f(close[:N])[-1] == f(close[:N+k])[N-1]` for k in [1, 5, 50]. Any indicator that fails is forward-looking.
8. **Calendar features must be BINARY windows only.** No `minutes_until_FOMC`, no `minutes_since_FOMC` raw features — only `is_within_30min_window` flags. Prevents calendar-pattern memorization that inflates CV.
9. **CPI/PCE annual seasonal revisions** silently rewrite 5y of history every Feb (CPI) and Aug (PCE). MUST use ALFRED `get_series_all_releases`, never current snapshot. UNRATE annual revision and PAYEMS Q1 benchmark have same risk.
10. **WALCL Thursday 4:30 PM ET** — a Thursday RTH-close 16:00 bar must NOT use that week's level. Use prior week's via release-time-aware as-of join.
11. **WGC monthly** (was thought quarterly) — self-snapshot weekly, no public vintage archive.
12. **AI-GPR is NOT real-time** — has ~30-day lag. Treat as monthly.

### Hand-off Protocol

When done:
1. Update STATUS.md with feature DataFrame stats (rows, columns, class balance)
2. Document any deviations from the spec in this doc's "Deviations" section
3. Notify doc 05 (training) that feature pipeline is stable

Now read the implementation specifics below.

---

# 02 — Feature Engineering

**Status:** ✅ Complete, implementation-ready, Nia-verified
**Last verified:** 2026-04-30

## CRITICAL CORRECTIONS (Nia verification)

- ❌ `ta==0.11.0` (stale, 17mo old, broken API calls in pseudocode) → ✅ **`pandas-ta-classic==0.5.44`** (active fork, April 2026)
- ❌ Parkinson volatility → ✅ **Garman-Klass** (uses OHLC, 7.4× more efficient than close-to-close, no extra cost since we have OHLC anyway)
- ❌ GDELT count + Goldstein only → ✅ also add **`gdelt_tone_mean`** + **`gdelt_tone_std`** (tone is GDELT's most validated signal)
- ❌ `mcal.get_calendar('NYSEARCA')` → ✅ **`mcal.get_calendar('NYSE')`** — NYSEARCA not in canonical registry; GLD follows NYSE hours/holidays anyway
- ❌ Anchor-cosine claimed novel → ✅ prior art exists (FinAnchor arXiv:2602.20859, FINEAS arXiv:2111.00526). Cite, don't claim novelty. Add anchor-cohesion test (intra-anchor pairwise cosine > 0.6)
- ❌ Z-score returns can blow up at near-zero std → ✅ add `clip(-10, 10)` after z-scoring
- ❌ News embeddings raw into projection → ✅ **L2-normalize 4096-dim before** projection + `LayerNorm(256)` after (StockTime pattern)
- ⚠️ Class imbalance is mild (28/44/28). Consider plain unweighted CE OR Cui-style β=0.999 effective-number weighting (arXiv:1901.05555). Default `len/(num_classes × N_i)` is fine but A/B compare. **V1 update:** focal loss gamma=3 replaces vanilla CE (Mukhoti 2020 arXiv:2002.09437) per spec section 5.1, doc 05 owns the loss config.
- ✅ **V1 PROMOTED:** the "5bps fixed threshold is approximation, triple-barrier TODO for v2" note is now resolved. Triple-barrier with ATR-14 barriers + spread-adjusted neutral threshold is the V1 default. See "Label Construction (V1)" section below.
**Owner:** samsiavoshian
**Implementation effort:** 1 day after data pipeline lands

## Per-Bar Input Vector (V1 expanded 2026-05-04, V1 +3 dims 2026-05-08)

```
Total: ~1000 dims per bar (was ~804 pre-expansion). 681 numeric features in
the unified dataset locked 2026-05-08; V1 adds 3 dims (h5 ×2 + spread ×1).

NUMERIC FEATURES (~232 dims pre-V1, +3 V1 = ~235):
├── Price features                  (12 dims)
├── Risk/volatility features        (8 dims)
├── Macro short (FX/VIX/oil/GPR)    (12 dims)
├── Geopolitical/GDELT events       (10 dims)
├── Equity ETF basket               (~72 dims — 9 ETFs × 8 features each)
├── Equity-derived ratios           (~9 dims — gold/silver, GDX/GLD, cross-correlations)
├── Treasury curve + TIPS           (~30 dims — 11 levels + 11 changes + 4 spreads + butterfly + 2 real-rate)
├── Macro bundle                    (~60 dims — 19 series × 3 features + 3 derived)
├── COT positioning                 (~6 dims — managed money / commercial / OI z-score / changes)
├── WGC central bank flows          (~3 dims — total + YoY + isPositive)
├── Calendar event features         (~10 dims — event proximity + cyclical sin/cos)
├── Half-hour-5 (V1 NEW)          (2 dims — gld_h5_log_return + gld_h5_x_vol_high)
└── Spread (V1 NEW)               (1 dim — gld_spread_bps_t, 5-min trailing avg)

NEWS EMBEDDINGS (V4: 1024 dims = 8 latent tokens × 128, see doc 03):
└── Bar-conditioned aggregation of per-article embeddings from 12+ sources
    via Per-source PMA pre-pool → FiLM Q-Former (K=8) → Flamingo gate
```

Sequence shape after stacking 64 bars: `(T=64, ~1000)`.

After learned input projection `Linear(~1000, 384)`: `(T=64, 384)`. Then transformer (doc 05 unchanged — projection layer just gets ~75K extra params, trivial).

Channel-group count for iTransformer-lite tokenization grows from ~14 → ~25 tokens (one token per channel group). doc 05's `~14 channel-group tokens` line should be read as "≥14"; agents owning doc 05 may resize without architectural change.

## Category 1 — Price Features (12 dims)

```python
def price_features(df: pd.DataFrame) -> pd.DataFrame:
    """All features computed from PAST bars only via .shift(1).rolling(...)."""
    out = pd.DataFrame(index=df.index)

    # Multi-horizon log returns (already shifted: return at T-1)
    log_close = np.log(df.close).shift(1)  # close lagged
    out['log_return_1']  = log_close.diff(1)
    out['log_return_4']  = log_close.diff(4)
    out['log_return_16'] = log_close.diff(16)
    out['log_return_48'] = log_close.diff(48)

    # Technical indicators via pandas-ta-classic (verified API).
    # V4 audit: KAMA, Ichimoku (visual=True), KST, DPO, TRIX, Vortex are FORBIDDEN — confirmed look-ahead bugs (bukosabino/ta#181).
    # RSI, MACD, BBANDS proper are causal IFF min_periods respected and no bfill applied.
    # Every indicator must pass growing-window stability test in tests/test_no_leakage.py.
    closed_lag = df.close.shift(1)  # all features use lagged close (= bar end) to avoid current-bar leakage
    out['rsi_14']      = pta.rsi(closed_lag, length=14)
    macd_df = pta.macd(closed_lag, fast=12, slow=26, signal=9)
    out['macd_signal'] = macd_df['MACDs_12_26_9']
    bbands_df = pta.bbands(closed_lag, length=20, std=2.0)
    out['bbands_pct']  = bbands_df['BBP_20_2.0']  # %B column

# FORBIDDEN INDICATORS (V4):
# - pta.kama  (np.roll wraps last value to front -> look-ahead)
# - pta.ichimoku(visual=True)  (shifts +26 forward into future)
# - pta.kst   (mean-fills with full close mean -> uses future)
# - pta.dpo, pta.trix, pta.vortex  (related lookahead patterns in upstream `ta`)

    # Microstructure-ish
    out['high_low_range_8'] = (df.high.shift(1) - df.low.shift(1)).rolling(8).mean()
    out['volume_zscore_20'] = (
        (df.volume.shift(1) - df.volume.shift(1).rolling(20).mean())
        / df.volume.shift(1).rolling(20).std()
    )
    out['close_open_ratio'] = (df.close.shift(1) / df.open.shift(1)) - 1

    # Session phase (4-dim one-hot encoded later)
    out['session_phase'] = df.timestamp.shift(1).apply(_get_session)

    return out
```

## Category 2 — Risk / Volatility Features (8 dims)

```python
def risk_features(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    log_returns = np.log(df.close / df.close.shift(1)).shift(1)

    out['realized_vol_8']    = log_returns.rolling(8).std()
    out['realized_vol_48']   = log_returns.rolling(48).std()
    out['realized_vol_240']  = log_returns.rolling(240).std()
    out['vol_ratio_short_long'] = out['realized_vol_8'] / out['realized_vol_48']
    out['vol_zscore_30d'] = (
        (out['realized_vol_48'] - out['realized_vol_48'].rolling(480).mean())
        / out['realized_vol_48'].rolling(480).std()
    )

    # Garman-Klass estimator (uses full OHLC, 7.4× more efficient than close-to-close)
    # CORRECTION: switched from Parkinson (high-low only) since we have full OHLC anyway.
    # Formula: 0.5 * (ln(H/L))^2 - (2*ln(2) - 1) * (ln(C/O))^2, then mean and sqrt.
    h_lag, l_lag, c_lag, o_lag = df.high.shift(1), df.low.shift(1), df.close.shift(1), df.open.shift(1)
    log_hl = np.log(h_lag / l_lag)
    log_co = np.log(c_lag / o_lag)
    gk_per_bar = 0.5 * log_hl**2 - (2 * np.log(2) - 1) * log_co**2
    out['garman_klass_8'] = np.sqrt(gk_per_bar.rolling(8).mean())

    # FOMC proximity
    out['days_since_FOMC'] = compute_days_since_last_fomc(df.timestamp.shift(1)) / 100
    out['is_FOMC_week'] = compute_is_fomc_week(df.timestamp.shift(1)).astype(float)

    return out
```

## Category 3 — Macro Features (6 dims)

```python
def macro_features(df: pd.DataFrame, fred_data: dict) -> pd.DataFrame:
    """fred_data is dict of vintage-correct daily series, keyed by FRED ID."""
    out = pd.DataFrame(index=df.index)

    # Forward-fill daily macro to 30min bars, using ALFRED vintage-correct values
    dxy_at_T = vintage_lookup(fred_data['DTWEXBGS'], df.timestamp.shift(1))
    dgs10_at_T = vintage_lookup(fred_data['DGS10'], df.timestamp.shift(1))
    dgs2_at_T = vintage_lookup(fred_data['DGS2'], df.timestamp.shift(1))
    vix_at_T = vintage_lookup(fred_data['VIXCLS'], df.timestamp.shift(1))
    # T5YIE = 5-Year Breakeven Inflation Rate (NOT forward). T5YIFR is forward expectation.
    # Verified via Nia: ALFRED has both, picking breakeven (cleaner signal for gold).
    inflation_5y_at_T = vintage_lookup(fred_data['T5YIE'], df.timestamp.shift(1))

    # Compute features
    out['dxy_log_return_5d'] = np.log(dxy_at_T / dxy_at_T.shift(240))  # 240 30min bars = 5 trading days
    out['dgs10']             = dgs10_at_T / 10
    out['dgs2']              = dgs2_at_T / 10
    out['term_spread']       = (dgs10_at_T - dgs2_at_T)
    out['real_rate']         = dgs10_at_T - inflation_5y_at_T
    out['vix_level']         = vix_at_T / 30

    return out
```

## Category 4 — Geopolitical / Event Features (10 dims)

```python
def geo_features(df: pd.DataFrame, brent: pd.Series, wti: pd.Series, gpr: pd.Series, gdelt: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)

    # Oil proxies
    out['brent_log_return_1d'] = np.log(
        vintage_lookup(brent, df.timestamp.shift(1))
        / vintage_lookup(brent, df.timestamp.shift(49))
    )
    out['wti_log_return_1d'] = np.log(
        vintage_lookup(wti, df.timestamp.shift(1))
        / vintage_lookup(wti, df.timestamp.shift(49))
    )

    # Oil-gold correlation (rolling 30 days = 1440 30min bars)
    gold_returns = np.log(df.close / df.close.shift(1)).shift(1)
    brent_returns = np.log(brent / brent.shift(1))  # daily, fwd-filled
    out['oil_gold_corr_30d'] = gold_returns.rolling(1440).corr(brent_returns)

    # GDELT-derived event intensity in [T-1bar, T-news_latency)
    # Verified GDELT theme codes (canonical lookup):
    out['gdelt_goldstein_avg'] = aggregate_gdelt_field(gdelt, df.timestamp.shift(1), 'GoldsteinScale', agg='mean')
    # NEW: tone mean + std (per arXiv:2505.16136 — tone dispersion captures news disagreement, predicts vol spikes)
    out['gdelt_tone_mean'] = aggregate_gdelt_field(gdelt, df.timestamp.shift(1), 'V2Tone', agg='mean')
    out['gdelt_tone_std']  = aggregate_gdelt_field(gdelt, df.timestamp.shift(1), 'V2Tone', agg='std')
    out['gdelt_conflict_count'] = aggregate_gdelt_themes(gdelt, df.timestamp.shift(1),
        themes=['ARMEDCONFLICT', 'WB_2433_CONFLICT_AND_VIOLENCE', 'TERROR', 'TAX_WEAPONS_BOMB', 'SANCTIONS'])
    out['gdelt_oil_count'] = aggregate_gdelt_themes(gdelt, df.timestamp.shift(1),
        themes=['ENV_OIL', 'FUELPRICES', 'MARITIME_INCIDENT'])

    # GPR Index (monthly, fwd-filled)
    out['gpr_monthly'] = vintage_lookup(gpr, df.timestamp.shift(1))
    out['gpr_change_3m'] = out['gpr_monthly'] - out['gpr_monthly'].shift(2880)  # 3 months ≈ 2880 30min bars

    # Conflict-anchor cosine (computed AFTER news embeddings, so populated post-doc-04)
    out['conflict_sim_alpaca'] = NaN  # filled in Stage 2 of feature pipeline
    out['conflict_sim_gdelt']  = NaN

    return out
```

## Category 5 — News Embeddings (V4: 8 fused tokens × 128 dim = 1024 dims, see doc 03)

**V4 refactor (2026-05-04):** news pipeline expanded from 3 sources mean-pooled to 12+ sources per-article-embedded with bias-aware aggregation.

Per article, Qwen3-Embedding-4B (4-bit MLX, frozen) produces a 2560-dim vector → MRL-truncated to 256-dim → projected to d_model=128 inside the aggregator. Per-bar aggregation:

```
[N articles in 30-min window]
  → 256-d MRL embeddings
  → +source_id_emb +time_offset_emb (per-article tokens)
  → group by source → PerSourcePMA (Set Transformer, 2 seeds/src)
  → Bar-conditioned FiLM Q-Former (K=8 latent queries adapt to current price/vol regime)
  → 8 fused news tokens × 128 = 1024 dims per bar
```

Plus the LAFTR adversarial head fights per-source bias (industry-bullish Kitco / dealer-bullish BullionVault / etc. don't trick the model).

doc 03 owns the source registry + aggregator. doc 04 imports the aggregator and wires its output into the bar-level feature stream.

## Category 6 — Equity ETF Basket (~72 dims) — V1 expansion 2026-05-04

```python
ETF_BASKET = ["SPY", "QQQ", "IWM", "GDX", "SLV", "XLF", "XLE", "XLK", "XLU"]

def equity_features(etf_bars: dict[str, pd.DataFrame], gld_bars: pd.DataFrame) -> pd.DataFrame:
    """Per-ETF features + cross-correlations + ratios.
    All inputs are 30m bars on the same NYSE calendar — no resampling needed."""
    out = pd.DataFrame(index=gld_bars.index)
    spy_log_returns = np.log(etf_bars["SPY"].close / etf_bars["SPY"].close.shift(1)).shift(1)
    gld_log_returns = np.log(gld_bars.close / gld_bars.close.shift(1)).shift(1)

    for sym in ETF_BASKET:
        df = etf_bars[sym]
        log_close = np.log(df.close).shift(1)
        log_returns = log_close.diff(1)
        out[f"{sym}_logret_1"]  = log_returns
        out[f"{sym}_logret_4"]  = log_close.diff(4)
        out[f"{sym}_logret_16"] = log_close.diff(16)
        out[f"{sym}_logret_48"] = log_close.diff(48)
        out[f"{sym}_vol_8"]     = log_returns.rolling(8).std()
        out[f"{sym}_vol_48"]    = log_returns.rolling(48).std()
        # Relative strength vs SPY (skip self for SPY)
        out[f"{sym}_rs_spy"] = log_returns - spy_log_returns if sym != "SPY" else 0.0
        # 30-day rolling correlation with GLD (1440 30m bars ≈ 30 trading days)
        out[f"{sym}_corr_gld_30d"] = log_returns.rolling(1440).corr(gld_log_returns)

    return out  # 9 × 8 = 72 dims
```

## Category 7 — Equity-Derived Ratios (~9 dims) — V1 expansion 2026-05-04

```python
def equity_ratio_features(etf_bars: dict[str, pd.DataFrame], gld_bars: pd.DataFrame) -> pd.DataFrame:
    """Cross-asset ratios known to predict gold."""
    out = pd.DataFrame(index=gld_bars.index)

    # Gold-Silver ratio (centuries-old gold value indicator; mean-reverts ~75:1)
    gsr = (gld_bars.close.shift(1) / etf_bars["SLV"].close.shift(1))
    out["gold_silver_ratio"]     = gsr
    out["gold_silver_ratio_logret_1d"]  = np.log(gsr / gsr.shift(48))   # 1 day = 48 30m bars (RTH); use 13 if intraday-only
    out["gold_silver_ratio_logret_5d"]  = np.log(gsr / gsr.shift(48*5))

    # GDX/GLD ratio (miners leverage gold price; spread predicts mean-reversion)
    gdx_gld = (etf_bars["GDX"].close.shift(1) / gld_bars.close.shift(1))
    out["gdx_gld_ratio"]         = gdx_gld
    out["gdx_gld_ratio_logret_1d"] = np.log(gdx_gld / gdx_gld.shift(48))
    out["gdx_gld_ratio_logret_5d"] = np.log(gdx_gld / gdx_gld.shift(48*5))

    # Stocks-vs-gold cross-correlations (regime indicators)
    spy_ret = np.log(etf_bars["SPY"].close / etf_bars["SPY"].close.shift(1)).shift(1)
    qqq_ret = np.log(etf_bars["QQQ"].close / etf_bars["QQQ"].close.shift(1)).shift(1)
    iwm_ret = np.log(etf_bars["IWM"].close / etf_bars["IWM"].close.shift(1)).shift(1)
    gld_ret = np.log(gld_bars.close / gld_bars.close.shift(1)).shift(1)
    out["spy_gld_corr_30d"] = spy_ret.rolling(1440).corr(gld_ret)
    out["qqq_gld_corr_30d"] = qqq_ret.rolling(1440).corr(gld_ret)
    out["iwm_gld_corr_30d"] = iwm_ret.rolling(1440).corr(gld_ret)

    return out  # 9 dims
```

## Category 8 — Treasury Curve + TIPS + Breakevens (~30 dims) — V1 expansion 2026-05-04

```python
TREASURY_NOMINAL = ["DGS3MO", "DGS6MO", "DGS2", "DGS5", "DGS10", "DGS30"]   # 6 nominal points
TREASURY_TIPS    = ["DFII5", "DFII10"]                                       # 2 TIPS points
BREAKEVENS       = ["T5YIE", "T10YIE", "T5YIFR"]                             # 3 inflation expectation series

def treasury_features(df: pd.DataFrame, fred_data: dict) -> pd.DataFrame:
    """Full curve + TIPS + breakevens + spreads + butterfly + real rates.
    Real rates are the #1 gold price driver (gold pays no yield)."""
    out = pd.DataFrame(index=df.index)
    ts_lag = df.timestamp.shift(1)

    # 11 series — levels (vintage-correct)
    levels = {}
    for series_id in TREASURY_NOMINAL + TREASURY_TIPS + BREAKEVENS:
        levels[series_id] = vintage_lookup(fred_data[series_id], ts_lag)
        out[f"{series_id}_level"] = levels[series_id] / 10  # scale roughly
        out[f"{series_id}_change_1d"] = levels[series_id] - levels[series_id].shift(48)  # 1 day in 30m bars (RTH only)

    # Term spreads (recession + risk-on indicators)
    out["spread_10y_2y"]   = levels["DGS10"] - levels["DGS2"]    # classic recession indicator
    out["spread_30y_10y"]  = levels["DGS30"] - levels["DGS10"]   # long-end slope
    out["spread_5y_2y"]    = levels["DGS5"] - levels["DGS2"]     # belly slope
    out["spread_10y_3m"]   = levels["DGS10"] - levels["DGS3MO"]  # NY Fed recession indicator

    # Butterfly (curve curvature)
    out["butterfly_2_5_10"] = 2 * levels["DGS5"] - levels["DGS2"] - levels["DGS10"]

    # Real rates — both proxies (DFII10 directly + DGS10 - T10YIE) for redundancy
    out["real_rate_10y_direct"]    = levels["DFII10"]                            # direct from TIPS
    out["real_rate_10y_breakeven"] = levels["DGS10"] - levels["T10YIE"]          # nominal - breakeven

    return out  # 11 levels + 11 1d-changes + 4 spreads + 1 butterfly + 2 real-rate = 29 dims (call it ~30)
```

## Category 9 — Macro Bundle (~60 dims) — V1 expansion 2026-05-04

```python
MACRO_LABOR     = ["UNRATE", "PAYEMS", "ICSA", "CCSA", "JTSJOL"]
MACRO_INFLATION = ["CPIAUCSL", "CPILFESL", "PCEPI", "PCEPILFE"]
MACRO_GROWTH    = ["GDPC1", "INDPRO", "RSAFS", "HOUST", "UMCSENT"]
MACRO_FED       = ["M2SL", "WALCL", "RRPONTSYD", "FEDFUNDS", "SOFR"]

def macro_bundle_features(df: pd.DataFrame, fred_data: dict) -> pd.DataFrame:
    """Each release moves gold. All vintage-correct via ALFRED."""
    out = pd.DataFrame(index=df.index)
    ts_lag = df.timestamp.shift(1)

    for series_id in MACRO_LABOR + MACRO_INFLATION + MACRO_GROWTH + MACRO_FED:
        level = vintage_lookup(fred_data[series_id], ts_lag)
        out[f"{series_id}_level"]    = level
        # Monthly series: 1 month in 30m bars ≈ 21 trading days × 13 bars = 273 (use 273); approximate
        # Choose horizons that match release cadence:
        if series_id in MACRO_LABOR + MACRO_INFLATION + MACRO_GROWTH:
            # Monthly — YoY change in % form. ~12 months ≈ 12 × 273 = 3276 30m RTH bars/year
            out[f"{series_id}_yoy"] = (level / level.shift(3276)) - 1
            out[f"{series_id}_mom"] = (level / level.shift(273)) - 1
        else:
            # Daily / weekly Fed series — use shorter horizons
            out[f"{series_id}_change_1w"]  = level - level.shift(13 * 5)   # 1 trading week
            out[f"{series_id}_change_4w"]  = level - level.shift(13 * 5 * 4)

    # Derived signals
    out["icsa_4w_ma"] = vintage_lookup(fred_data["ICSA"], ts_lag).rolling(4).mean()
    out["real_fedfunds"] = vintage_lookup(fred_data["FEDFUNDS"], ts_lag) - (
        vintage_lookup(fred_data["CPIAUCSL"], ts_lag).pct_change(periods=12) * 100
    )  # Real Fed funds rate — Bernanke's "shadow rate" proxy
    out["m2_yoy"] = (vintage_lookup(fred_data["M2SL"], ts_lag).pct_change(periods=52) * 100)  # weekly to YoY

    return out  # 19 × ~3 + 3 derived ≈ 60 dims
```

**CRITICAL leakage trap:** the YoY shift uses `3276` only as an approximation. The vintage-correct lookup must guarantee `realtime_start ≤ ts_lag` for ALL revisions. Test this in `test_no_leakage.py` for every macro series.

## Category 10 — CFTC COT Positioning (~6 dims) — V1 expansion 2026-05-04

```python
def cot_features(df: pd.DataFrame, cot: pd.DataFrame) -> pd.DataFrame:
    """COT data = positioning extremes predict reversals.
    Release rule: feature visible at bar T iff cot.release_ts < T_close."""
    out = pd.DataFrame(index=df.index)

    # As-of join — find the most recent COT release strictly before each bar's close
    cot_pit = cot.sort_values("release_ts")
    bar_release_lookup = pd.merge_asof(
        df[["timestamp"]].rename(columns={"timestamp": "T_close"}),
        cot_pit,
        left_on="T_close",
        right_on="release_ts",
        direction="backward",
        allow_exact_matches=False,  # strict <, not ≤
    )

    out["cot_mm_net_long_pct_oi"]      = bar_release_lookup["mm_net_long_pct_oi"]
    out["cot_comm_net_long_pct_oi"]    = bar_release_lookup["comm_net_long_pct_oi"]
    out["cot_nonrep_net_long_pct_oi"]  = bar_release_lookup["nonrep_net_long_pct_oi"]
    out["cot_oi_zscore_52w"]           = bar_release_lookup["oi_total"].rolling(52).apply(
        lambda x: (x.iloc[-1] - x.mean()) / (x.std() + 1e-8)
    )
    out["cot_mm_change_2w"]            = bar_release_lookup["mm_net_long"].diff(2)
    out["cot_mm_change_12w"]           = bar_release_lookup["mm_net_long"].diff(12)

    return out  # 6 dims
```

**Why these 6:** academic + practitioner consensus is that managed-money net long as % of OI predicts mean-reversion at extremes (90+ %ile = bearish for gold near term). 12-week change captures positioning trend. Open interest z-score captures conviction.

## Category 11 — WGC Central Bank Flows (~3 dims) — V1 expansion 2026-05-04

```python
def wgc_features(df: pd.DataFrame, wgc: pd.DataFrame) -> pd.DataFrame:
    """WGC quarterly central bank net purchases. Slow signal, structurally bullish for gold."""
    out = pd.DataFrame(index=df.index)

    # Same release-time as-of join as COT
    wgc_pit = wgc.sort_values("release_ts")
    lookup = pd.merge_asof(
        df[["timestamp"]].rename(columns={"timestamp": "T_close"}),
        wgc_pit,
        left_on="T_close",
        right_on="release_ts",
        direction="backward",
        allow_exact_matches=False,
    )

    out["wgc_total_net_purchase_tonnes_q"] = lookup["net_purchase_total_tonnes"]
    out["wgc_total_net_purchase_yoy"]      = lookup["net_purchase_total_tonnes"] - lookup["net_purchase_total_tonnes"].shift(4)
    out["wgc_is_net_buyer_q"]              = (lookup["net_purchase_total_tonnes"] > 0).astype(float)

    return out  # 3 dims
```

## Category 12 — Calendar Event Features (~10 dims) — V1 expansion 2026-05-04

```python
def calendar_features(df: pd.DataFrame, calendar: pd.DataFrame) -> pd.DataFrame:
    """Event proximity + cyclical encodings. NO LOOKAHEAD — event timestamps are deterministic ahead-of-time."""
    out = pd.DataFrame(index=df.index)
    ts = df.timestamp

    # Event proximity (signed minutes to nearest event of each type, clipped to ±2 days)
    for evt_type in ["FOMC", "CPI", "NFP", "GDP", "JOLTS", "PCE"]:
        evt_times = calendar[calendar.event_type == evt_type].event_ts_utc.values
        signed_min = signed_minutes_to_nearest(ts, evt_times)  # negative = past, positive = future
        out[f"is_{evt_type.lower()}_release_window"] = (signed_min.abs() <= 24 * 60).astype(float)  # within 1 day
        # Note: we do NOT include a "minutes-until-future-event" raw feature because that is technically
        # known information (calendar is published) but creates pseudo-leakage in correlation tests.
        # The window indicator suffices.

    # Cyclical encodings (sin/cos to avoid wraparound discontinuity)
    out["dow_sin"]   = np.sin(2 * np.pi * ts.dt.dayofweek / 5.0)         # M-F
    out["dow_cos"]   = np.cos(2 * np.pi * ts.dt.dayofweek / 5.0)
    out["hour_sin"]  = np.sin(2 * np.pi * ts.dt.hour / 24.0)
    out["hour_cos"]  = np.cos(2 * np.pi * ts.dt.hour / 24.0)
    out["month_sin"] = np.sin(2 * np.pi * ts.dt.month / 12.0)
    out["month_cos"] = np.cos(2 * np.pi * ts.dt.month / 12.0)

    # Special days
    out["is_options_expiry_day"]    = is_third_friday(ts).astype(float)
    out["is_quarter_end_window"]    = is_last_3_days_of_quarter(ts).astype(float)

    return out  # 6 event windows + 6 cyclical + 2 special = ~10 dims (ish; trim if collinear)
```

**Why event windows beat raw "minutes-to-event":** the model can learn higher vol + reversion patterns around CPI/NFP/FOMC; but `minutes_until_NFP` injects deterministic future-time information that breaks point-in-time discipline in subtle ways (it's "information about when the future is", which the model could exploit to memorize calendar artifacts). Window indicators are simpler and safer.

## Category 13 — Half-Hour-5 Intraday Momentum (V1 NEW, 2 dims)

Gao-Han-Li-Zhou 2014 found a single feature, the log return of the 5th RTH half-hour (~11:30-12:00 ET), predicts the last half-hour with **Sharpe 5.43 on GLD specifically**, concentrated on high-vol days. This is the published apples-to-apples intraday GLD record. Gating it through a vol-tercile interaction captures the concentration effect.

```python
def h5_features(df: pd.DataFrame, vol_tercile_high: pd.Series) -> pd.DataFrame:
    """Half-hour-5 intraday momentum (Gao 2014 prior).
    H5 = the 5th RTH half-hour bar (approx 11:30-12:00 ET).
    NaN outside RTH-5 window propagates forward to end of day.
    """
    out = pd.DataFrame(index=df.index)

    # Identify H5 bar by ET time of day
    et_times = df.timestamp.dt.tz_convert("America/New_York")
    is_h5 = (et_times.dt.hour == 11) & (et_times.dt.minute == 30)

    # 30-min log return at H5 bar; forward-fill within day, NaN at SOD
    h5_logret = np.where(
        is_h5,
        np.log(df.close.shift(1) / df.close.shift(2)),  # close at H5 / close at H4
        np.nan,
    )
    h5_logret = pd.Series(h5_logret, index=df.index).groupby(et_times.dt.date).ffill()

    out["gld_h5_log_return"]  = h5_logret.astype("float32")
    out["gld_h5_x_vol_high"]  = (h5_logret * vol_tercile_high.astype("float32")).astype("float32")

    return out  # 2 dims
```

**Why this matters:** the 681-feature ensemble is wide. A single feature with published Sharpe 5.43 on the same instrument we're trading is a non-negotiable inclusion. The interaction term `r_h5 * vol_tercile_high` lets the model attend to it primarily on high-vol days where the prior holds. Reference: Gao-Han-Li-Zhou 2014 (cited in V1 spec section 4.1).

## Category 14 — Spread Feature (V1 NEW, 1 dim)

```python
def spread_feature(quotes: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    """5-min trailing average bid-ask spread in bps. Used by triple-barrier neutral
    threshold and by sizing layer (doc 07). Quote feed must be PIT-correct: every
    quote carries a t_visible = quote.timestamp + epsilon."""
    out = pd.DataFrame(index=df.index)

    quotes = quotes.sort_values("t_visible")
    spread_bps = ((quotes.ask - quotes.bid) / ((quotes.ask + quotes.bid) / 2)) * 10_000

    # 5-minute trailing avg, joined as-of bar close
    spread_trailing = spread_bps.rolling("5min", on=quotes.t_visible).mean()
    joined = pd.merge_asof(
        df[["timestamp"]].rename(columns={"timestamp": "T_close"}),
        pd.DataFrame({"t_visible": quotes.t_visible, "spread_5min_avg": spread_trailing}),
        left_on="T_close",
        right_on="t_visible",
        direction="backward",
        allow_exact_matches=False,
    )

    out["gld_spread_bps_t"] = joined["spread_5min_avg"].astype("float32")
    return out  # 1 dim
```

**Why:** triple-barrier labeling needs a per-bar neutral threshold so micro-moves inside the spread don't get labeled directional. Sizing layer (doc 07) also gates positions when spread spikes (cost-aware adjustment).

## Series Decomposition (V1 NEW, no new columns)

Pre-VSN, pre-RevIN: split each of the 681 channels into trend + seasonal via 24-bar moving-average kernel. xLSTMTime (Alharthi & Mahmood, arXiv:2407.10240) requires this; Autoformer-style.

```python
def decompose_series(x: np.ndarray, kernel: int = 24) -> tuple[np.ndarray, np.ndarray]:
    """Causal moving-average decomposition. center=False so the kernel only sees past.
    x shape: (T, 681). Returns (trend, seasonal) each (T, 681)."""
    df = pd.DataFrame(x)
    trend = df.rolling(kernel, min_periods=1, center=False).mean().to_numpy()
    seasonal = x - trend
    return trend, seasonal
```

Both trend and seasonal feed RevIN separately, then sum back. Lives in `src/nanogld/features/decomposition.py` (or under `src/nanogld/model/` since it's coupled to RevIN). Order of operations: **decomp -> RevIN per-channel -> VSN -> patch projection -> backbone**.

## RevIN per-Channel (V1 upgrade)

V1: RevIN per channel-group (~14 to ~25 groups).

V1: RevIN per individual channel (681 instances of learnable affine, one per feature). Trivial cost (~1.4K extra params), strictly more expressive. Justification: Huang & Yang ESWA 2026 — per-feature RevIN drops RMSE 50% / MAPE 54% on cross-market stock data.

```python
class RevINPerChannel(nn.Module):
    """Per-channel reversible instance norm. 681 learnable (gamma, beta) pairs."""
    def __init__(self, num_channels: int = 681, eps: float = 1e-5):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(num_channels))
        self.beta  = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps
        self._mean = None
        self._std  = None

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C=681)
        self._mean = x.mean(dim=1, keepdim=True)  # (B, 1, C)
        self._std  = x.std(dim=1, keepdim=True) + self.eps
        x = (x - self._mean) / self._std
        return x * self.gamma + self.beta

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.beta) / self.gamma
        return x * self._std + self._mean
```

Lives in `src/nanogld/model/revin.py` per V1 file layout (model layer). Documented here because feature engineering decides what one channel is and how 681 channels are ordered.

## VSN Feature Gate at Input (V1 NEW)

Variable Selection Network (Lim 2021, TFT, arXiv:1912.09363) sits at the input layer **after** RevIN, **before** patch projection. Math:

```
gate_i = softmax_i(GRN(x_i))      for i in {1..681}
x_gated = gate * x                # element-wise across 681 channels
```

GRN (Gate Residual Network) = 2-layer MLP with ELU activation + GLU + LayerNorm. Cost: ~2M extra params (one GRN per feature group, parameter-shared inside group).

**Why pay 2M params:** VLSTM (LSTM + VSN) hits 2.40 Sharpe on the Saly-Kaufmann/Wood/Zohren 2026 benchmark (arXiv:2603.01820), versus plain LSTM at 1.48 Sharpe. **+0.92 Sharpe delta from VSN alone.** Even half that delta on our setup pays for itself many times over.

This is a model-side artifact (file lives in `src/nanogld/model/vsn.py`), but documented here because it shapes what 681 features mean to the backbone. Doc 05 owns the implementation.

## Conflict-Anchor Cosine (semantic features beyond raw embeddings)

```python
def build_anchor_embeddings(llama, anchor_texts: dict[str, list[str]]) -> dict[str, np.ndarray]:
    """
    For each named anchor (conflict, monetary, dollar, recession),
    embed ~20 historical headlines, mean-pool across them.
    Returns dict[name -> 4096-dim vector]. Saved as .npz (NOT pickle).
    """
    anchors = {}
    for name, texts in anchor_texts.items():
        embs = [embed_text(llama, t) for t in texts]
        v = np.stack(embs).mean(axis=0)
        anchors[name] = v / np.linalg.norm(v)  # normalize for cosine
    return anchors

# Save (use numpy native, not pickle)
np.savez("data/anchors/v1.npz", **anchors)

# Load
anchors_npz = np.load("data/anchors/v1.npz")
anchors = {k: anchors_npz[k] for k in anchors_npz.files}

# At each bar T, for each news source emb:
def compute_anchor_cosines(news_emb_normalized: np.ndarray, anchors_dict: dict) -> dict[str, float]:
    return {name: float(news_emb_normalized @ anchor) for name, anchor in anchors_dict.items()}
```

**Anchor sets — V4 LEAKAGE FIX (mandatory):** anchor texts MUST be either:
- **(a) hand-crafted templated phrases with no event provenance** (preferred V1 default), OR
- **(b) sampled exclusively from BEFORE the training window** (e.g., 2015-2020 for a 2021-2026 train window).

If anchors are sampled from the full corpus, the anchor set itself encodes future events and leaks 2024+ semantics into 2017 cosine values. Detection: `assert max(anchor.pub_ts) < min(train.pub_ts)`. If hand-crafted, no provenance check needed.

V1 default: hand-crafted templates.

```python
ANCHOR_TEMPLATES = {
    "conflict":  ["central banks face geopolitical tensions",
                  "military escalation in resource-rich region",
                  "sanctions imposed on commodity exporter",
                  ...],  # 20 templated phrases, no event-specific names
    "monetary":  ["central bank tightens policy rate",
                  "inflation print exceeds expectations",
                  "Federal Reserve signals dovish pivot",
                  ...],
    "dollar":    ["US dollar strengthens against major currencies",
                  "currency intervention announced",
                  ...],
    "recession": ["yield curve inversion deepens",
                  "unemployment claims rise sharply",
                  ...],
}
```

Each gets ONE anchor vector per news source (computed once, frozen). Per bar = 4 cheap cosine features per source = potentially 12 extra features. Currently included as 2 of the 10 geo features (conflict_sim_alpaca, conflict_sim_gdelt). Can expand later if useful.

## Label Construction (V1: Triple-Barrier replaces fixed 5-bps threshold)

V1 used `label = sign(next_log_return)` thresholded at fixed 5 bps. V1 replaces this with López de Prado's triple-barrier method ("Advances in Financial Machine Learning", Ch. 3) plus a spread-adjusted neutral guard from the TLOB lesson.

**Triple-barrier rule per bar T:**
- Compute ATR-14 at bar T close.
- Set `barrier_up = +1.0 * ATR_14`, `barrier_down = -1.0 * ATR_14`, `timeout = 1 bar (30 min)`.
- Step forward in time. Label `+1` if up barrier hit first, `-1` if down barrier hit first, `0` if timeout reached without either touched.
- **Spread-adjusted neutral guard:** even if a barrier is touched, force `label = 0` when `|return| < spread_t` (move smaller than the spread is unactionable).

```python
def triple_barrier_labels(
    df: pd.DataFrame,
    spread_bps: pd.Series,
    atr_period: int = 14,
    barrier_mult: float = 1.0,
    timeout_bars: int = 1,
) -> pd.DataFrame:
    """López de Prado triple-barrier labels with spread-adjusted neutral guard.
    Returns DataFrame with label_triple_barrier (-1/0/+1), barrier_up, barrier_down."""
    out = pd.DataFrame(index=df.index)

    # ATR-14 at bar T close (PIT: only past bars used)
    high_low   = (df.high.shift(1) - df.low.shift(1)).abs()
    high_close = (df.high.shift(1) - df.close.shift(2)).abs()
    low_close  = (df.low.shift(1)  - df.close.shift(2)).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr_14 = tr.rolling(atr_period).mean()

    barrier_up   = barrier_mult * atr_14
    barrier_down = -barrier_mult * atr_14

    out["barrier_up"]   = barrier_up.astype("float32")
    out["barrier_down"] = barrier_down.astype("float32")

    # 30-min log return between bar T close and bar T+1 close
    next_log_return = np.log(df.close.shift(-1) / df.close)

    # Spread per bar in absolute log-return units (bps -> ratio)
    spread_abs = (spread_bps / 10_000).astype("float32")

    # Triple-barrier with timeout=1 bar: simply check next return vs barriers
    label = pd.Series(0, index=df.index, dtype="int8")  # 0 = timeout/neutral
    label[next_log_return >=  barrier_up.values]   = 1
    label[next_log_return <= barrier_down.values] = -1

    # Spread-adjusted neutral: micro-moves inside the spread are unactionable
    label[next_log_return.abs() < spread_abs] = 0

    out["label_triple_barrier"] = label  # int8 in {-1, 0, +1}
    # CE-friendly mapping for the 3-class head: {-1, 0, +1} -> {0, 1, 2}
    out["label_ce"] = (label + 1).astype("int8")
    return out
```

**Three new columns:**
- `label_triple_barrier` (int8 in {-1, 0, +1}, mapped to {0, 1, 2} as `label_ce` for 3-class CE head)
- `barrier_up` (float32 ATR-scaled, kept for backtest cross-check)
- `barrier_down` (float32 ATR-scaled)

**Why triple-barrier over 5-bps fixed:**
- ATR-scaled barriers adapt to regime. A 5-bps move during 2020 March chaos is noise; a 5-bps move during 2017 calm is signal. Fixed thresholds collapse that asymmetry into a single class boundary.
- Spread-adjusted neutral kills the "predict a 4-bps move when spread is 8 bps" trap. The TLOB paper showed this matters: micro-direction predictions inside the spread don't translate to PnL.
- López de Prado's method is the practitioner standard for exactly this reason. Reference: AFML Ch. 3, plus the V1 spec sheet section 4.5.

**Class distribution sanity (expected on 30min GLD):**
- DOWN (-1): ~25-30%
- NEUTRAL (0): ~40-50% (depends on ATR / spread regime)
- UP (+1): ~25-30%

Class-weighted CE weights = `len(df) / (3 * class_count)` per class. Or use focal loss gamma=3 (V1 default per spec section 5.1, doc 05 owns the loss config).

## Normalization (Point-in-Time Z-Scoring)

```python
def rolling_zscore(series: pd.Series, lookback: int = 1000) -> pd.Series:
    """Z-score against rolling-past mean/std. Never global stats."""
    mean = series.shift(1).rolling(lookback).mean()
    std = series.shift(1).rolling(lookback).std()
    z = (series - mean) / (std + 1e-8)
    return z.clip(-10, 10)   # bound outliers from near-zero std denominators
```

Apply to ALL continuous features. Categorical features (session_phase, is_FOMC_week) skip this.

News embeddings are NOT z-scored per-feature (they're 256-dim each). They're already roughly normalized by the LLM and the learned projection layer.

## Final Pipeline (V1)

```python
def build_feature_table(snapshot_path: str, llama_embeddings_path: str, anchors_path: str, quotes_path: str) -> pd.DataFrame:
    raw = pd.read_parquet(snapshot_path)

    # Raw feature engineering
    price = price_features(raw)
    risk = risk_features(raw)
    macro = macro_features(raw, fred_data=...)
    geo = geo_features(raw, brent=..., wti=..., gpr=..., gdelt=...)

    # V1 expansion (2026-05-04)
    equity = equity_features(etf_bars=..., gld_bars=raw)
    equity_ratios = equity_ratio_features(etf_bars=..., gld_bars=raw)
    treasury = treasury_features(raw, fred_data=...)
    macro_full = macro_bundle_features(raw, fred_data=...)
    cot = cot_features(raw, cot_data=...)
    wgc = wgc_features(raw, wgc_data=...)
    cal = calendar_features(raw, calendar=...)

    # V1 expansion (2026-05-08)
    quotes = pd.read_parquet(quotes_path)
    spread = spread_feature(quotes, raw)                  # 1 dim: gld_spread_bps_t
    vol_tercile_high = compute_vol_tercile_high(risk["realized_vol_48"])
    h5 = h5_features(raw, vol_tercile_high)               # 2 dims: gld_h5_log_return + gld_h5_x_vol_high

    # Concat
    numeric = pd.concat([
        price, risk, macro, geo,                    # existing 36 dims
        equity, equity_ratios,                       # V1: 72 + 9 = 81 dims
        treasury,                                    # V1: ~30 dims
        macro_full,                                  # V1: ~60 dims
        cot, wgc, cal,                               # V1: 6 + 3 + 10 = 19 dims
        spread, h5,                                  # V1: 1 + 2 = 3 dims
    ], axis=1)

    # Z-score (per-channel, point-in-time). Note: per-channel RevIN happens later
    # at the model-input layer (src/nanogld/model/revin.py); this z-score keeps
    # the feature table on a sane scale before training-time RevIN.
    for col in numeric.columns:
        if col not in CATEGORICAL_COLS:
            numeric[col] = rolling_zscore(numeric[col], lookback=1000)

    # One-hot session phase
    numeric = pd.get_dummies(numeric, columns=['session_phase'])

    # Load precomputed news embeddings
    news_emb = pd.read_parquet(llama_embeddings_path)  # cols: alpaca_emb_<0..4095>, gdelt_emb_<0..4095>, rss_emb_<0..4095>

    # Anchor cosines (fill conflict_sim_alpaca, conflict_sim_gdelt)
    anchors_npz = np.load(anchors_path)  # safe numpy-native .npz format
    anchors = {k: anchors_npz[k] for k in anchors_npz.files}
    geo['conflict_sim_alpaca'] = compute_cosine(news_emb['alpaca_emb_*'], anchors['conflict'])
    geo['conflict_sim_gdelt']  = compute_cosine(news_emb['gdelt_emb_*'],  anchors['conflict'])

    # Combine: numeric + news embeddings (raw 4096-dim, projection happens in model)
    features = pd.concat([numeric, news_emb], axis=1)

    # V1 Labels: triple-barrier with spread-adjusted neutral
    label_table = triple_barrier_labels(
        raw,
        spread_bps=spread["gld_spread_bps_t"],
        atr_period=14,
        barrier_mult=1.0,
        timeout_bars=1,
    )
    features = pd.concat([features, label_table], axis=1)

    # Drop rows with NaN (rolling features at start of dataset)
    features = features.dropna()

    return features
```

**Note on order of operations at training time** (doc 05 owns this; documented here for clarity):

```
features (681 dim, z-scored) -> series_decompose(kernel=24) -> trend, seasonal
                              -> RevIN_per_channel(trend) + RevIN_per_channel(seasonal)
                              -> sum back
                              -> VSN gate (681 -> 681, learnable softmax weights per feature)
                              -> patch projection (P=4, S=4, T_bars=64 -> 16 patches)
                              -> backbone (10 transformer + 2 sLSTM, FiLM regime conditioning)
```

## Augmentation (V1: SimPSI + Wave-Mask, naive jittering BANNED)

V1 used jittering (Gaussian noise sigma=0.02) + magnitude warping. V1 bans naive jittering: Fons 2020 (arXiv:2010.15111) showed jittering is **net negative on Sharpe** in financial time series because it destroys the dominant-frequency components that carry the trend signal.

V1 uses two PIT-safe, spectral-preserving augmentations:

1. **SimPSI** (Ryu et al. AAAI 2024, arXiv:2312.05790). Reweights augmentation strength per frequency component so dominant components are preserved. Math:
   - `X_freq = FFT(x)`
   - `mask = softmax(|X_freq|^2 / tau)` -- low-energy bins get more aug, high-energy preserved
   - `x_aug = IFFT(X_freq * (1 + epsilon * (1 - mask) * noise))`

2. **Wave-Mask** (Arabi 2024, arXiv:2408.10951). Masks DWT (discrete wavelet transform) coefficients at random scales, then inverse-DWT. Both PIT-safe (no time-axis shifts).

3. **Manifold Mixup** (alpha=0.2, kept from V1) -- applied at hidden states only, **never on raw input**. Lives in doc 05 training loop.

Forbidden in V1:
- Jittering (Gaussian additive noise on raw input) -- net-negative Sharpe per Fons 2020.
- Time warping / scaling -- breaks PIT alignment with labels.
- Window slicing with offset -- same PIT break.

```python
def simpsi_augment(x: np.ndarray, epsilon: float = 0.1, tau: float = 1.0) -> np.ndarray:
    """SimPSI spectral-preserving augmentation. x shape (T, C). PIT-safe."""
    X = np.fft.rfft(x, axis=0)
    energy = np.abs(X) ** 2
    mask = softmax(energy / tau, axis=0)  # high energy -> high mask -> preserved
    noise = np.random.randn(*X.shape) + 1j * np.random.randn(*X.shape)
    X_aug = X * (1 + epsilon * (1 - mask) * noise)
    return np.fft.irfft(X_aug, n=x.shape[0], axis=0)
```

Lives in `src/nanogld/training/augment.py` (doc 05 owns implementation; documented here because it shapes feature semantics during training).

## Validation Tests

```python
def test_no_future_leakage(features, raw):
    """For every row T, no value in features[T] depends on raw[T:]."""
    # Modify raw[T+1] arbitrarily, recompute features[T], ensure unchanged
    ...

def test_label_alignment_triple_barrier(features, raw, spread):
    """For each row T, features.label_triple_barrier[T] is +1 iff
    next_log_return >= barrier_up[T] AND |next_log_return| >= spread[T].
    Mirror checks for -1 and 0."""
    ...

def test_z_score_no_global_leakage(features, raw):
    """Z-score at row T uses only past data."""
    ...

def test_h5_feature_pit(features, raw):
    """gld_h5_log_return is NaN before the first H5 bar of each day; it carries
    the same value forward within a day; it resets at SOD next day."""
    ...

def test_spread_pit(features, quotes):
    """gld_spread_bps_t at bar T uses only quotes with t_visible < bar T close."""
    ...

def test_atr_14_pit(raw):
    """ATR-14 at bar T uses only bars [T-14, T-1]; never bar T itself."""
    ...
```

## Open Questions / TODOs

- [ ] Threshold sweep (3 / 5 / 10 bps) — picks which one in week 1 day 5 implementation
- [ ] Decide on session phase boundaries (Asia 18:00-02:00 ET? US 09:30-16:00 ET?)
- [x] ✅ Confirmed via Nia: T5YIE has ALFRED vintage data back to 2006. T5YIE = 5-Year Breakeven (NOT forward — that's T5YIFR). Doc updated.
- [x] ✅ Confirmed via Nia: all 7 FRED series (DTWEXBGS, DGS10, DGS2, T5YIE, VIXCLS, DCOILBRENTEU, DCOILWTICO) have ALFRED vintage from 2006+
- [x] ✅ Corrected GDELT theme codes — original codes (MIL_CONFLICT, ECON_RECESSION, WB_654, TAX_FNCACT_BOMBING) don't exist; replaced with verified canonical codes

### V1 Expansion TODOs (2026-05-04)

- [ ] Verify ALL 27 newly added FRED series have ALFRED vintage cubes back to 2021 (DGS3MO, DGS6MO, DGS5, DGS30, DFII5, DFII10, T10YIE, T5YIFR + 19 macro). Spawn Nia subagent on day 2.
- [ ] Re-baseline rolling z-score lookback for monthly macro features — 1000-bar default (≈ 3 months) is too short for series with quarterly cadence. Use 3276 (1 year) for macro YoY features.
- [ ] Decide channel-group binning when count grows from ~14 → ~25: (a) split to ~25 fine-grained groups, OR (b) keep ~14 by merging similar series into wider groups. Affects model token count + speed. Default V1 = (a) finer = better, eat the modest compute cost.
- [ ] Validate that `merge_asof` with `allow_exact_matches=False` is actually strict-`<`. Some pandas versions handle exact equality differently. Add explicit assertion in golden fixture test.
- [ ] Calendar event windows for emergency FOMC dates (e.g., March 15 2020, March 23 2020). Hard-code those dates explicitly.
- [ ] Consider adding "minutes_until_next_FOMC" with the assertion that the calendar is fully published >2 weeks before each meeting (so the feature is genuinely PIT-correct). Default: skip per leakage-conservatism principle.
