# 02 — Feature Engineering

## YOU ARE THE FEATURE ENGINEER AGENT

You own feature construction. You take immutable parquet snapshots from doc 01 + cached embeddings from doc 04 and produce the feature DataFrame that doc 05 (training) consumes.

**Read 00-OVERVIEW.md FIRST.** Project context is there.
**Read 01-DATA-PIPELINE.md schema section.** Your input is its output.
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
├── macro.py                # 6 macro features (DXY, DGS10, DGS2, etc.)
├── geo.py                  # 10 geopolitical features (oil, GPR, GDELT aggregation)
├── sentiment.py            # 3 multi-dim sentiment scores (polarity + intensity + uncertainty)
├── labels.py               # 3-class label construction with threshold sweep
├── normalize.py            # Rolling z-score with clip(-10, 10)
├── revin.py                # RevIN instance norm per channel-group
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

- `src/nanogld/data/` — doc 01 owns
- `src/nanogld/embed/` — doc 04 owns
- `src/nanogld/model/` — doc 03
- Anything else in src/nanogld/

### Stable Interface You Publish

`build_feature_table(snapshot_path, embeddings_path, anchors_path) -> pd.DataFrame` — returns DataFrame with columns specified in this doc's "Per-Bar Input Vector" section. Total ~804 dims per bar (V1: ~803 with multi-dim sentiment replacing single).

### Acceptance Criteria

1. ✅ `python -m nanogld.features build` produces feature DataFrame from snapshot + cached embeddings
2. ✅ All 4 leakage tests pass
3. ✅ Class distribution at 5bps threshold roughly 28/44/28 (DOWN/FLAT/UP)
4. ✅ Z-scored features have mean ≈ 0, std ≈ 1, range bounded by `clip(-10, 10)`
5. ✅ Anchor cosines distribution looks reasonable (conflict_sim spikes during known events — verify on Russia/Ukraine 2022, Iran tensions 2024-25)
6. ✅ Feature build runs in <2 min for 16K bars (parquet I/O dominates)
7. ✅ No NaN in final DataFrame except for the first ~1000 rows (rolling features warm-up)

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

- News embedding dim changed: 4096 (earlier Llama-3.1-8B) → 256 (V1 Qwen3-Embedding-4B truncated via MRL). See doc 04 for details.
- Add 3 multi-dim sentiment features per news source (polarity / intensity / uncertainty) per arXiv:2603.11408.
- Anchor-cosine pattern unchanged but now uses Qwen3 embeddings.

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
- ⚠️ Class imbalance is mild (28/44/28). Consider plain unweighted CE OR Cui-style β=0.999 effective-number weighting (arXiv:1901.05555). Default `len/(num_classes × N_i)` is fine but A/B compare.
- ⚠️ 5bps label threshold is research approximation. For live trading, consider triple-barrier with vol-scaled barriers (Lopez de Prado AFML Ch.3) — TODO for v2.
**Owner:** samsiavoshian
**Implementation effort:** 1 day after data pipeline lands

## Per-Bar Input Vector (final)

```
Total: ~804 dims per bar

├── Price features            (12 dims)
├── Risk/volatility features  (8 dims)
├── Macro features            (6 dims)
├── Geopolitical/event feats  (10 dims)
├── Alpaca news embedding     (256 dims, projected from 4096 via learned linear)
├── GDELT news embedding      (256 dims, projected from 4096 via learned linear)
└── RSS news embedding        (256 dims, projected from 4096 via learned linear)
```

Sequence shape after stacking 64 bars: `(T=64, 804)`.

After learned input projection `Linear(804, 384)`: `(T=64, 384)`. Then transformer.

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
    
    # Technical indicators via pandas-ta-classic (verified API)
    # CORRECTION: original `ta` lib pseudocode was wrong — that API doesn't exist.
    # Use pandas-ta-classic==0.5.44 (active fork, Apr 2026).
    closed_lag = df.close.shift(1)  # all features use lagged close to avoid current-bar leakage
    out['rsi_14']      = pta.rsi(closed_lag, length=14)
    macd_df = pta.macd(closed_lag, fast=12, slow=26, signal=9)
    out['macd_signal'] = macd_df['MACDs_12_26_9']
    bbands_df = pta.bbands(closed_lag, length=20, std=2.0)
    out['bbands_pct']  = bbands_df['BBP_20_2.0']  # %B column
    
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

## Category 5 — News Embeddings (768 dims, see doc 04)

Per source per bar, Llama-3.1-8B-4bit produces a 4096-dim mean-pooled vector. A learned `Linear(4096, 256)` (inside the TinyTransformer, not the LLM) projects it down. 3 sources × 256 = 768 dims.

Computation lives in doc 04.

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

**Anchor sets (each ~20 historical headlines):**
- `conflict`: military / war / sanctions / Middle East / Strait of Hormuz / etc.
- `monetary`: Fed / rate cut / rate hike / FOMC / Powell speech / inflation print
- `dollar`: DXY / USD strength / currency intervention
- `recession`: recession / unemployment / GDP contraction / yield curve inversion

Each gets ONE anchor vector per news source (computed once, frozen). Per bar = 4 cheap cosine features per source = potentially 12 extra features. Currently included as 2 of the 10 geo features (conflict_sim_alpaca, conflict_sim_gdelt). Can expand later if useful.

## Label Construction

```python
THRESHOLD_BPS = 5  # tune in {3, 5, 10}

def make_labels(df: pd.DataFrame, threshold_bps: int) -> pd.Series:
    """Labels for bar T are based on bar T+1's return."""
    next_log_return = np.log(df.close.shift(-1) / df.close)
    threshold = threshold_bps / 10_000
    
    labels = pd.Series(np.full(len(df), 1, dtype=int), index=df.index)  # default flat=1
    labels[next_log_return > threshold] = 2   # up
    labels[next_log_return < -threshold] = 0  # down
    
    return labels
```

**Class distribution at threshold = 5bps on 30min GLD typically:**
- DOWN (0): ~28%
- FLAT (1): ~44%
- UP (2): ~28%

Class-weighted cross-entropy weights = `len(df) / (3 * class_count)` per class.

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

## Final Pipeline

```python
def build_feature_table(snapshot_path: str, llama_embeddings_path: str, anchors_path: str) -> pd.DataFrame:
    raw = pd.read_parquet(snapshot_path)
    
    # Raw feature engineering
    price = price_features(raw)
    risk = risk_features(raw)
    macro = macro_features(raw, fred_data=...)
    geo = geo_features(raw, brent=..., wti=..., gpr=..., gdelt=...)
    
    # Concat
    numeric = pd.concat([price, risk, macro, geo], axis=1)
    
    # Z-score
    for col in numeric.columns:
        if col not in CATEGORICAL_COLS:
            numeric[col] = rolling_zscore(numeric[col], lookback=1000)
    
    # One-hot session phase
    numeric = pd.get_dummies(numeric, columns=['session_phase'])
    
    # Load precomputed news embeddings
    news_emb = pd.read_parquet(llama_embeddings_path)  # cols: alpaca_emb_<0..4095>, gdelt_emb_<0..4095>, rss_emb_<0..4095>
    
    # Anchor cosines (fill conflict_sim_alpaca, conflict_sim_gdelt)
    anchors_npz = np.load(anchors_path)  # safe numpy native, not pickle
    anchors = {k: anchors_npz[k] for k in anchors_npz.files}
    geo['conflict_sim_alpaca'] = compute_cosine(news_emb['alpaca_emb_*'], anchors['conflict'])
    geo['conflict_sim_gdelt']  = compute_cosine(news_emb['gdelt_emb_*'],  anchors['conflict'])
    
    # Combine: numeric + news embeddings (raw 4096-dim, projection happens in model)
    features = pd.concat([numeric, news_emb], axis=1)
    
    # Labels
    features['label'] = make_labels(raw, threshold_bps=5)
    
    # Drop rows with NaN (rolling features at start of dataset)
    features = features.dropna()
    
    return features
```

## Validation Tests

```python
def test_no_future_leakage(features, raw):
    """For every row T, no value in features[T] depends on raw[T:]."""
    # Modify raw[T+1] arbitrarily, recompute features[T], ensure unchanged
    ...

def test_label_alignment(features, raw):
    """For each row T, features.label[T] should match np.sign(raw.close[T+1] - raw.close[T]) thresholded."""
    ...

def test_z_score_no_global_leakage(features, raw):
    """Z-score at row T uses only past data."""
    ...
```

## Open Questions / TODOs

- [ ] Threshold sweep (3 / 5 / 10 bps) — picks which one in week 1 day 5 implementation
- [ ] Decide on session phase boundaries (Asia 18:00-02:00 ET? US 09:30-16:00 ET?)
- [x] ✅ Confirmed via Nia: T5YIE has ALFRED vintage data back to 2006. T5YIE = 5-Year Breakeven (NOT forward — that's T5YIFR). Doc updated.
- [x] ✅ Confirmed via Nia: all 7 FRED series (DTWEXBGS, DGS10, DGS2, T5YIE, VIXCLS, DCOILBRENTEU, DCOILWTICO) have ALFRED vintage from 2006+
- [x] ✅ Corrected GDELT theme codes — original codes (MIL_CONFLICT, ECON_RECESSION, WB_654, TAX_FNCACT_BOMBING) don't exist; replaced with verified canonical codes
