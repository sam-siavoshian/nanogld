# 04 — News Embedding

## YOU ARE THE NEWS EMBEDDING AGENT

You own the news-text-to-vector pipeline. You set up Qwen3-Embedding-4B (swap from Llama-3.1-8B-mean-pool), embed all news for all bars once, cache to disk. You also compute the anchor embeddings used by doc 04 for semantic features.

**Read 00-OVERVIEW.md FIRST.**
**Read 02-DATA-PIPELINE.md** for the parquet input schema (alpaca_headlines, gdelt_headlines, rss_headlines columns).
**Also read 00-OVERVIEW.md "Execution Mode" section before coding.**

### Execution Mode (short — full rules in 00-OVERVIEW.md)

**This doc IS the plan. Do not rewrite it.** Plan to *execute* what is written. If a claim is wrong, AskUserQuestion. Silent scope drift = fired.

- **Research with Nia** before guessing on libs / APIs / papers. Spawn a subagent: `nia papers` / `nia github` / `nia search` / `nia oracle` / `nia packages` / `nia tracer`.
- **Execution skills only:** `/investigate`, `/review`, `/qa`, `/qa-only`, `/cso`, `/benchmark`, `/ship`, `/land-and-deploy`, `/canary`, `/document-release`, `/retro`, `/learn`, `/browse`, `/setup-browser-cookies`.
- **NO planning skills:** `/office-hours`, `/plan-*`, `/autoplan`, `/design-*`, `/devex-review`. Planning is done.
- **Loop:** read doc → Nia for unknowns → code → `/review` → `/qa` → `/cso` (if security-sensitive) → `/ship`.
- **Escalate after 3 failed attempts.** AskUserQuestion. Bad work is worse than no work.

### Files You Create (V4 — expanded)

```
src/nanogld/embed/
├── __init__.py
├── qwen3_embedder.py       # Qwen3-Embedding-4B wrapper (4-bit MLX preferred)
├── source_registry.py      # NEW V4 — SOURCE_REGISTRY + bias-tier table
├── precompute.py           # Embed every article individually, save (article_id, source_id, bias_tier, t_visible, emb_256) parquet
├── aggregator.py           # NEW V4 — PerSourcePMA + BarConditionedQuery + Q-Former-lite + Flamingo gate
├── adversary.py            # NEW V4 — LAFTR-style adversarial head with gradient reversal
├── anchors.py              # Computes 4 anchor embeddings from V4 hand-crafted templates
├── live_embed.py           # Single-bar embedding for live cycle (doc 08 imports)
├── cache.py                # SHA256-keyed cache (model + prompt + text → embedding)
└── cli.py                  # `python -m nanogld.embed precompute`

data/embeddings/
├── v1_<hash>_articles.parquet          # NEW V4 — per-article (article_id, source_id, bias_tier, t_visible, emb_256)
└── v1_<hash>_meta.json                  # model_id, prompt_template, MRL truncation dim, source_registry_hash

data/anchors/
└── v1.npz                               # 4 anchor vectors × 256 dim (V4 hand-crafted templates only)

tests/
├── test_embedding_determinism.py        # Same input + Q4 model → same embedding (rtol 1e-3)
├── test_anchor_cohesion.py              # V4: intra-anchor cosine > 0.7 (was 0.6 — templates are tighter)
├── test_anchor_no_event_provenance.py   # NEW V4 — anchor texts have no event-specific named entities
├── test_semantic_alignment.py           # Conflict text > benign text on conflict anchor
├── test_source_registry_complete.py     # NEW V4 — every article in the corpus has a known source_id
├── test_laftr_adversary_drops.py        # NEW V4 — adversary accuracy → 1/n_sources after 30% training
└── test_aggregator_no_news_token.py     # NEW V4 — N=0 article bars produce stable representation via NO_NEWS token
```

### Files You DO NOT Touch

- Anything outside `src/nanogld/embed/`, `data/embeddings/`, `data/anchors/`
- Other doc files

### Stable Interface You Publish (V4 expanded)

```python
# doc 04 / doc 05 read precomputed embeddings via:
articles = pd.read_parquet("data/embeddings/v1_<hash>_articles.parquet")
# Columns:
#   article_id: str (sha256 of source + url + created_at)
#   source_id: int  (from SOURCE_REGISTRY)
#   bias_tier_id: int  (from BIAS_TIERS)
#   t_visible: pd.Timestamp UTC
#   emb_256: np.ndarray fp16 (256,)
#   bar_id_window: int  (foreign key to bars table — first bar where this article is visible)

# doc 05 imports the aggregator:
from nanogld.embed.aggregator import BarConditionedNewsAggregator
agg = BarConditionedNewsAggregator(
    d_in=256, d_model=128, K=8, K_per_src=2, n_sources=25, d_bar=16
)
news_tokens, gate = agg(articles_per_bar, src_ids, dt_to_bar, mask, bar_feat)
# news_tokens: [B, K=8, 128]   gate: scalar (Flamingo tanh-gate)

# doc 05 imports the LAFTR adversary:
from nanogld.embed.adversary import AdversarialDebiasingHead, GradientReversalLayer

# doc 08 (live trading) imports for single-bar embedding:
from nanogld.embed.live_embed import embed_articles_live
articles_emb = embed_articles_live(articles_in_window: list[Article]) -> pd.DataFrame
# Returns the same per-article schema as the precomputed parquet, for live cycle.

# doc 04 reads anchors (V4 templates):
anchors_npz = np.load("data/anchors/v1.npz")
# Keys: 'conflict', 'monetary', 'dollar', 'recession'. Each (256,) normalized.
# V4 source = ANCHOR_TEMPLATES_V4 (hand-crafted, no event provenance).
```

### Acceptance Criteria (V4 expanded)

1. ✅ `python -m nanogld.embed precompute` runs in <120min on M4 mini (was <60min — corpus grew from 3 sources to 12+ sources, ~10× more articles)
2. ✅ Per-article parquet ~500 MB - 2 GB (depending on article volume; per-row 256 fp16 + metadata)
3. ✅ All determinism tests pass (use Q4 GGUF for bit-exact)
4. ✅ Anchor cohesion test: intra-anchor pairwise cosine > **0.7** (V4 — was 0.6) for all 4 anchor sets
5. ✅ Anchor-no-event-provenance test passes — no anchor text contains a country, person, or event name
6. ✅ Semantic alignment test: a held-out "central bank tightens policy" headline has cosine > 0.5 to monetary anchor; a held-out "Apple announces new iPhone" has cosine < 0.3 to all 4 anchors
7. ✅ `live_embed.py` produces per-article embedding for a 30-min live window in <500ms (M4 Pro inference)
8. ✅ Anchor embeddings versioned in git via hash in filename (NEVER overwrite)
9. ✅ Source registry test: every article has a known `source_id` in `SOURCE_REGISTRY`. Unknown sources fall through to `UNK` and get logged.
10. ✅ LAFTR adversary test on a 1-week training run: adversary accuracy drops below `2 / n_sources` and direction-AUC stays within 5% of no-adversary baseline.
11. ✅ NO_NEWS token test: a bar with zero articles produces a stable, deterministic representation that the model can learn to interpret as "no signal" (gate path absorbs it via tanh init=0).

### Spawn Nia Agents When You Need To

- **MLX-LM Qwen3-Embedding-4B integration** — verify exact API for sentence-level embeddings (NOT generation tokens)
- **Q4 GGUF bit-determinism** — verify `llama-cpp-python` Qwen3 embedding mode produces stable outputs across runs
- **Anchor headline selection** — what 20 headlines best represent "geopolitical military conflict" vs "monetary policy"? Spawn agent to suggest representative texts
- **MRL truncation dim trade-off** — confirm 256-dim retains ≥99% MTEB quality on retrieval (256 is recommended; 128 saves more space)

### V1 Critical Decision (DO NOT REVERT)

**Switched from `meta-llama/Llama-3.1-8B-Instruct-4bit` mean-pool to `Qwen/Qwen3-Embedding-4B` 4-bit MLX.** Reasons in this doc's "V1 PIVOT" section. Key wins: 45× faster (18K vs 400 tok/s), Apache 2.0 license (vs Meta Community), MTEB-en 74.6 (vs ~64), MRL truncatable 2560→256.

If you find Qwen3-Embedding-4B doesn't fit your hardware or has a bug, fallback options (with documented quality cost):
- `Qwen/Qwen3-Embedding-0.6B` — 44K tok/s, 70.7 MTEB (-4 pts), 900MB RAM
- `google/embeddinggemma-300m` — even smaller, ~58 MTEB (significant drop)

Document the choice + reason in this doc's "Deviations" section.

### Encouragement to Research

Embedding models change MONTHLY. Before precomputing 87K bars, run `nia search web "MTEB leaderboard May 2026"` and check if a better model has dropped in the last 2 weeks. If it has, run a 1000-bar A/B between Qwen3-Embedding-4B and the new contender BEFORE committing to the full precompute. The agent's recommendation is based on May 1 — by the time you implement, May 15+ may have something better.

### Hand-off Protocol

1. Update STATUS.md with: embedding cache hash, total size, model used, time-to-precompute
2. Add anchor headlines to `data/anchors/v1_anchors.json` for reproducibility
3. Notify doc 04 (features) that embeddings + anchors are cached and ready

Now read the spec below.

---

# 04 — News Embedding

**Status:** ✅ Complete, implementation-ready, Nia-verified
**Last verified:** 2026-04-30

## V4 News Pipeline Refactor (2026-05-04 — owner directive)

5 Nia agents audited every news-source the user listed (Kitco, Metals Daily, BullionVault, Investing.com, CNBC, Reuters, FT, Trading Economics, FXStreet) plus prior plan sources (Alpaca News, GDELT, RSS) and surveyed available free 10y datasets + 2024-2026 SOTA on multi-document aggregation + bias debiasing. Outcome: **major news-pipeline expansion + architectural upgrade to the news fuser.**

### Source Verdicts (from V4 audit — all checks against live URLs / API docs / robots.txt 2026-05-04)

| Source | Verdict | Reason | Bias-tier |
|---|---|---|---|
| **Alpaca News (Benzinga)** | KEEP | Already in plan; free, 2015+, 130+/day | `mixed_retail` |
| **GDELT 2.0 GKG** | KEEP | Already in plan (with V4 leakage fixes for theme codes + 30min buffer) | `aggregator` (per-row source carries its own tier) |
| **Kitco News** | **ADD** | Free scrape, date-slug archive 10y, minute-ts, full body. Bullish-industry — needs source-conditioning. | `industry_bullish` |
| **Investing.com Gold** | **ADD** | Free, 250K-article archive, neutral aggregator. ToS-risky → throttle + cite. | `aggregator_neutral` |
| **BullionVault** | **ADD WITH CAUTION** | Free, dealer marketing — strongly bullish. Use as bias-extreme feature. | `dealer_bullish` |
| **CNBC** | **ADD** | Free RSS rolling + Wayback for 10y backfill. Full body. US-equity tilt. | `mainstream_equity_bias` |
| **WGC Goldhub research library** | KEEP (already in data plan) | Free download, industry trade body | `industry_bullish` |
| **FNSPID** (HF Zihan1004/FNSPID, CC BY 4.0) | **ADD — biggest free win** | 15.7M news articles 1999-2023, multi-source (Reuters, NASDAQ, Benzinga, Lenta). Fills pre-2021 historical gap. | per-row tier from source field |
| **Central bank speeches** (HF samchain/bis_central_bank_speeches + istat-ai/ECB-FED-speeches) | **ADD** | Free, public domain (US gov 17 USC §105 + ECB free-research). Rate decisions move gold harder than equity news. | `central_bank_official` |
| **fomc/statements** (GitHub) | **ADD** | FOMC text 1994+ pre-cleaned | `central_bank_official` |
| **Federal Reserve / ECB / Treasury / CFTC press releases** | **ADD** | All free, US gov public domain. CFTC commissioner speeches are gold-relevant (commodity regulator). | `government_official` |
| **Arctic Shift Reddit dumps** (Pushshift successor) | **ADD** | Free torrents through 2026-04. r/Gold, r/wallstreetbets, r/investing. | `retail_social` |
| **Kaggle ankurzing gold sentiment + Precious Metals 2000+** | **ADD** | Direct gold-labeled headlines (small, but gold-keyword bootstrap for relevance classifier) | n/a (training labels) |
| **Reuters Gold** | **DEFER (paid)** | $1/wk consumer paywall + Reuters Connect API enterprise-only. CNBC syndicates Reuters wire for partial free coverage. Reuters Connect when funded. | `mainstream_neutral` (when added) |
| **Financial Times Gold** | **REFUTED — LEGAL BLOCKER** | `ft.com/robots.txt` explicitly bans ML/AI use: "We expressly prohibit any use of our content or data for any machine learning or artificial intelligence". DO NOT scrape. | n/a |
| **Trading Economics news** | DEFER | Aggregator that re-publishes Reuters wire — duplicates we'd already have. Indicator API already in doc 02. | n/a |
| **FXStreet** | **DEFER (paid B2B)** | Best minute-ts technicals-embedded news but B2B-paid via Acuity Trading. Free RSS limited. Add when budget allows. | `retail_technical_bias` (when added) |
| **Metals Daily** | SKIP | Sharps Pixley aggregator, syndication overlap with originators, archive depth unconfirmed | `dealer_bullish` (if added) |

### News count: 3 → 12+ sources

Pre-V4 plan: 3 sources (Alpaca News + GDELT + RSS for live).
V4 plan: **12 historical sources + 3 live sources** (RSS still used for live cycle only).

### Bias-Aware Source Registry

The 5-tier bias scheme above must be encoded as a static registry in code. Every article ingested gets tagged with its source's bias tier at ingestion time. The model SEES the tier (as an embedding) and uses LAFTR to learn the per-source prior + subtract it.

```python
# src/nanogld/embed/source_registry.py — V1 hard-coded
SOURCE_REGISTRY = {
    # source_id_str: (numeric_id, bias_tier, public_domain_yes_no)
    "alpaca_benzinga":         (0,  "mixed_retail",          False),  # licensed via Alpaca
    "gdelt:reuters":           (1,  "mainstream_neutral",    False),
    "gdelt:bloomberg":         (2,  "mainstream_neutral",    False),
    "gdelt:cnbc":              (3,  "mainstream_equity_bias",False),
    "gdelt:wsj":               (4,  "mainstream_neutral",    False),
    "gdelt:other":             (5,  "aggregator",            False),
    "kitco":                   (6,  "industry_bullish",      False),
    "investing_com":           (7,  "aggregator_neutral",    False),
    "bullionvault":            (8,  "dealer_bullish",        False),
    "cnbc":                    (9,  "mainstream_equity_bias",False),
    "wgc_research":            (10, "industry_bullish",      False),
    "fnspid:reuters":          (11, "mainstream_neutral",    False),  # CC BY 4.0 dataset
    "fnspid:nasdaq":           (12, "mainstream_neutral",    False),
    "fnspid:benzinga":         (13, "mixed_retail",          False),
    "central_bank:fed":        (14, "central_bank_official", True),
    "central_bank:ecb":        (15, "central_bank_official", True),
    "central_bank:bis":        (16, "central_bank_official", True),
    "central_bank:other":      (17, "central_bank_official", True),
    "government:treasury":     (18, "government_official",   True),
    "government:cftc":         (19, "government_official",   True),
    "reddit:gold":             (20, "retail_social",         False),
    "reddit:wsb":              (21, "retail_social",         False),
    "reddit:investing":        (22, "retail_social",         False),
    "kaggle:gold_labeled":     (23, "training_labels_only",  False),  # used for classifier pretraining, not feature input
    "UNK":                     (24, "aggregator",            False),
}

BIAS_TIERS = [
    "mainstream_neutral",
    "mainstream_equity_bias",
    "mixed_retail",
    "aggregator",
    "aggregator_neutral",
    "industry_bullish",
    "dealer_bullish",
    "central_bank_official",
    "government_official",
    "retail_social",
    "retail_technical_bias",
    "training_labels_only",
]  # 12 tiers; learned embedding nn.Embedding(12, 8)
```

### LAFTR Adversarial Debiasing Head

**Why:** filtering Kitco / BullionVault / WGC throws away signal (industry chatter often LEADS moves — WGC publishes Q4 demand data before retail flows arrive). Right move: keep them, tag them, force the model to learn `E[y | text, source]` and subtract per-source prior so a wave of correlated bullish stories from gold-aligned outlets gets down-weighted automatically.

**Recipe (LAFTR, Madras+ 2018, arXiv:1802.06309):**

```python
class AdversarialDebiasingHead(nn.Module):
    """LAFTR-style adversary that predicts source from the news fusion representation Z.
    Predictor minimizes L_pred - alpha*L_adv. Gradient reversal via DANN trick (Ganin 2015).
    """
    def __init__(self, d_model: int, n_sources: int):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, n_sources),
        )
    def forward(self, z, alpha=1.0):
        z_rev = GradientReversalLayer.apply(z, alpha)
        return self.classifier(z_rev)  # logits over source IDs

# In training loop:
y_hat = direction_head(z)                 # GLD up/flat/down
s_hat = adv_head(z, alpha=alpha)          # source-id prediction with grad reversal
L = ce(y_hat, y)                           # main task
# Adversary loss subtracted via grad reversal — written as +ce(s_hat, s_id) which becomes
# subtraction during backward through GradientReversalLayer:
L_adv = ce(s_hat, source_id_majority)     # majority source for that bar's news set
L_total = L + L_adv                        # grad reversal makes adversary fight predictor
```

Anneal `alpha`: 0.0 → 0.1 → 1.0 over first 30% of training. Verify success: adversary accuracy should drop toward `1 / num_sources` (random) while `y_hat` direction-AUC holds. If `y_hat` collapses, lower `alpha`. Implementation references: arXiv:1505.07818 (DANN gradient reversal), arXiv:1801.07593 (Zhang+ adversarial debiasing).

Plus inverse-frequency reweighting:

```python
# src/nanogld/training/sample_weights.py
sample_w = 1.0 / source_count_30d_rolling[source_id]   # rolling 30-day per-source frequency
```

So a Kitco article (60/day) gets weighted ~12× lower than a Reuters article (5/day).

### Architectural Upgrade: News Fuser

Pre-V4 plan: Perceiver-Resampler-lite K=16 + Flamingo gated cross-attn (arXiv:2204.14198 + arXiv:2107.14795). Still SOTA primitive in 2026. Three upgrades from 2025-2026 financial multimodal literature:

**Upgrade 1: K=16 → K=8 latent queries.** Sweet spot for 30M backbone with 256-d MRL inputs. Direct ablation evidence from Perceiver-IO (arXiv:2103.03206), Perceiver-VL (arXiv:2211.11701), SparseFormer (ICLR 2024 f5537b8d8fd126c7fe9d7429b181b1eb), Jumbo-token (arXiv:2502.15021). K=8 ≈ 8 latent axes (sentiment / surprise / sector / macro-vs-micro / source-credibility / staleness / horizon / direction) — fits the project's information bandwidth without overfitting on 5y window with sparse news bars. K=16 was overprovisioned.

**Upgrade 2: Bar-conditioned FiLM on latent queries.** Biggest 2025-26 jump per arXiv:2504.13522 (CMTF Cross-Modal Temporal Fusion, Pei+ Cartlidge 2025) and arXiv:2512.00293 (FiCoTS, Dec 2025): condition the news-pooling latents on bar features (price returns, realized vol, volume z-score). Concretely:

```python
class BarConditionedQuery(nn.Module):
    def __init__(self, K, d_model, d_bar):
        super().__init__()
        self.Q = nn.Parameter(torch.randn(K, d_model) * 0.02)
        self.film = nn.Linear(d_bar, 2 * d_model)  # gamma, beta

    def forward(self, bar_feat):  # bar_feat: [B, d_bar]
        gamma_beta = self.film(bar_feat)
        gamma, beta = gamma_beta.chunk(2, dim=-1)  # each [B, d_model]
        Q = self.Q.unsqueeze(0).expand(bar_feat.size(0), -1, -1)
        return Q * (1 + gamma.unsqueeze(1)) + beta.unsqueeze(1)  # [B, K, d_model]
```

This makes the SAME news article get pooled differently in a high-vol vs low-vol regime — the model asks different questions of the news depending on what's happening in price.

**Upgrade 3: Per-source PMA pre-pool (Set Transformer, Lee+ 2019, arXiv:1810.00825).** Two seed vectors per source token-pool fold N articles within a source down to 2 tokens before cross-source fusion. Cheap (O(N_per_src)). Explainable (we know which source contributed which token). Handles "50 Reuters wire copies in one bar" (FinGPT dissemination problem, arXiv:2412.10823) without drowning the model.

```python
class PerSourcePMA(nn.Module):
    def __init__(self, d_model, n_heads=4, K_per_src=2):
        super().__init__()
        self.seeds = nn.Parameter(torch.randn(K_per_src, d_model) * 0.02)
        self.attn  = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
    def forward(self, articles, mask):  # articles: [B, N_max_per_src, d_model]
        seeds = self.seeds.unsqueeze(0).expand(articles.size(0), -1, -1)
        kpm = ~mask
        out, _ = self.attn(seeds, articles, articles, key_padding_mask=kpm)
        return out  # [B, K_per_src, d_model]
```

**Final architecture (V4):**

```
[N articles in 30-min bar]
      ↓ Qwen3-Embedding-4B (frozen) → 2560-d → MRL truncate to 256
      ↓ Linear(256, d_model=128) + add nn.Embedding(n_sources, 128)(src_id)
      ↓ + sin/cos embedding of (article.created_at - bar_close_ts)
[per-article tokens: B × N_max × 128]
      ↓ group by source_id → PerSourcePMA(K_per_src=2)
[per-source tokens: B × (S × 2) × 128]
      ↓ BarConditionedQuery(K=8) ← FiLM(bar_feat[price_ret, vol, volume_z])
      ↓ Q-Former-lite: cross-attn → self-attn → FFN
[8 fused news tokens: B × 8 × 128]
      ↓ Flamingo tanh-gated cross-attn (gate init = 0)
[main 30M backbone consumes news at every (or every-other) layer]
      ↓ + LAFTR adversarial head on z (bar-level news representation)
      ↓ + inverse-frequency-reweighted CE on direction
[loss = ce(direction) + ce(adversarial_source) via grad reversal]
```

Key params added: PMA seeds (~5K) + FiLM (~2K) + Q-Former cross/self/FFN (~200K) + adversary head (~10K) = ~220K extra. Trivial vs 30M backbone.

**Defer to V2:** late-interaction top-4 raw-article residual in final backbone layer (ColBERT-style). Helps high-news bars but adds memory. Skip in V1.

### Anchor Text Refactor (V4 — leakage fix from Phase 1)

Anchors must be **hand-crafted templated phrases with NO event provenance**. The prior plan's anchors ("Russia invades Ukraine", "Israel strikes Iranian nuclear facilities") are SPECIFIC EVENTS — if those events postdate the train window, the anchor set leaks future semantics into past cosines. Replace:

```python
ANCHOR_TEMPLATES_V4 = {
    "conflict":  [
        "central banks face geopolitical tensions in resource regions",
        "military escalation disrupts commodity supply chain",
        "sanctions imposed on major commodity exporter",
        "naval incident in critical shipping corridor",
        "armed conflict in oil-producing region threatens stability",
        "diplomatic relations break down between major powers",
        "terror attack disrupts energy infrastructure",
        "border tensions escalate into military buildup",
        "international force deployed to conflict zone",
        "weapons shipment intercepted in volatile region",
        "ceasefire negotiations stall amid renewed violence",
        "economic warfare measures expand against state actor",
        "infrastructure attack disrupts global trade routes",
        "rebel forces seize strategic port",
        "embargo imposed on energy exports",
        "missile strike threatens regional escalation",
        "civil war escalates in resource-rich nation",
        "naval blockade restricts commodity flows",
        "armed insurgency disrupts mining operations",
        "regional power signals military readiness",
    ],
    "monetary":  [
        "central bank tightens policy rate amid inflation pressures",
        "Federal Reserve signals dovish pivot toward easing",
        "inflation print exceeds market expectations",
        "policymakers debate balance sheet runoff pace",
        "central bank holds rate steady citing labor market",
        "rate cut expectations build amid weakening data",
        "policy minutes reveal hawkish tone among committee",
        "dot plot shifts to fewer cuts than expected",
        "central bank governor signals end of tightening cycle",
        "quantitative easing announced in response to crisis",
        "policy makers signal patience on rate decisions",
        "real interest rates push toward multi-decade high",
        "central bank intervenes in currency markets",
        "policy committee splits on next rate move",
        "forward guidance abandoned in favor of data dependence",
        "yield curve control adjustments under discussion",
        "core inflation accelerates above target",
        "wage growth pressures complicate rate path",
        "policy stance described as restrictive by committee",
        "emergency rate move executed by central bank",
    ],
    "dollar":    [
        "US dollar strengthens against major currencies",
        "currency intervention announced by finance ministry",
        "dollar weakens amid risk-on equity rally",
        "trade-weighted dollar index hits multi-year extreme",
        "carry trade flows pressure emerging market currencies",
        "Asian currency intervention defends against dollar surge",
        "European central bank signals concern over currency strength",
        "dollar funding stress emerges in repo markets",
        "Treasury official comments on dollar policy",
        "swap line activation eases dollar liquidity",
        "reserve currency status debated by policymakers",
        "global dollar shortage signals systemic stress",
        "currency basket reweighting reduces dollar share",
        "petro-dollar arrangement under negotiation",
        "central bank reserves diversify away from dollar",
        "exchange rate volatility spikes on policy divergence",
        "trade-deficit data weakens dollar near-term",
        "interest rate differential drives dollar flows",
        "safe-haven demand boosts dollar amid crisis",
        "currency war rhetoric escalates between blocs",
    ],
    "recession": [
        "yield curve inversion deepens to multi-year extreme",
        "unemployment claims rise sharply for consecutive weeks",
        "manufacturing PMI contracts below expansion threshold",
        "consumer sentiment plunges to recession-era lows",
        "industrial production declines for consecutive months",
        "credit spreads widen amid risk aversion",
        "leading economic indicators signal contraction",
        "labor market shows broad-based weakening",
        "housing market activity collapses on rate shock",
        "GDP growth turns negative in advance estimate",
        "corporate earnings guidance broadly cut",
        "high-yield bond defaults accelerate in cyclical sectors",
        "money supply contraction signals tightening cycle",
        "real personal income declines for consecutive quarters",
        "wholesale inventories build amid weak final demand",
        "small business optimism falls to multi-year lows",
        "freight indicators signal goods sector weakness",
        "service sector activity slows abruptly",
        "Sahm rule recession indicator triggers",
        "consumer credit growth stalls amid stress",
    ],
}
```

Anchor cohesion test on these templates is now stricter: intra-anchor pairwise cosine > 0.7 (was 0.6) since templates are more semantically homogeneous than real headlines. doc 04 acceptance criteria updated.

### Hand-off / dataset additions

doc 02 (data pipeline) gets new sources 12-18 for the historical news corpus. doc 04 imports the source registry. doc 05 (training) implements LAFTR head + inverse-frequency reweighting. This doc owns the embedder + aggregator only.

---

## V1 PIVOT — Switch from Llama-3.1-8B-Mean-Pool to Qwen3-Embedding-4B (May 2026)

After 7-agent research found that 2026 brought purpose-built embedding models that crush generative-LLM mean-pool on every relevant axis. We switch to **`Qwen/Qwen3-Embedding-4B`** in 4-bit MLX.

### Why this swap is non-negotiable

| Metric | Earlier (Llama-3.1-8B mean-pool) | V1 (Qwen3-Embedding-4B) | Delta |
|--------|----------------------------|------------------------|-------|
| M4 mini throughput | ~400 tok/s | **~18,000 tok/s** | **45×** |
| RAM (4-bit) | ~5 GB | **~2.5 GB** | -50% |
| MTEB-en avg score | ~64 (LLM2Vec) | **74.60** | +10 pts |
| MTEB-en retrieval | ~58 | **68.46** | +10 pts |
| Embedding dim | 4096 | **2560 (MRL truncatable to 256)** | smaller cache |
| License | Meta Community License | **Apache 2.0** | commercial-clean |
| Context length | 4K tokens | **32K tokens** | 8× |
| Instruction-aware | No | **Yes** | +1-5% quality from prompts |
| Bit-determinism via Q4 GGUF | No | **Yes** (integer matmul) | reproducible cache |
| Precompute time (87K × 3 sources) | ~24 hrs | **~30 min** | 48× |

### Model Specs (for downstream architecture)

```
Model:           Qwen/Qwen3-Embedding-4B (4-bit MLX or GGUF)
Backbone:        Qwen3-4B-base (decoder-only, contrastive-trained for retrieval)
Params:          4B
Hidden dim:      2560 (MRL — can truncate to {32, 64, 128, 256, 512, 1024, 2560})
Context:         32K tokens
Pooling:         Built-in (model returns sentence embeddings directly, NOT raw hidden states)
License:         Apache 2.0
HF Hub:          Qwen/Qwen3-Embedding-4B
Quantized:       Qwen/Qwen3-Embedding-4B-GGUF (Q8_0, F16) — official
                 mlx-community/Qwen3-Embedding-4B-4bit — community, verified
Throughput:      ~18K tok/s on M4 mini in 4-bit MLX
RAM footprint:   ~2.5 GB (4-bit)
```

### Updated Pipeline

```python
# src/embed/qwen3_embedder.py
from sentence_transformers import SentenceTransformer
import torch

class Qwen3NewsEmbedder:
    """Replaces earlier Llama-3.1-8B mean-pool. ~45× faster, +10pts MTEB."""
    
    def __init__(self, model_id: str = "Qwen/Qwen3-Embedding-4B", truncate_dim: int = 256):
        self.model = SentenceTransformer(
            model_id,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device="mps",
        )
        self.truncate_dim = truncate_dim   # MRL: 2560 → 256, smaller cache, near-identical quality
    
    def embed_news_for_bar(self, headlines: list[str], source: str) -> np.ndarray:
        """Embed up to 5 headlines per source. Use instruction-aware prompts."""
        if not headlines:
            return self._no_news_vec(source)
        text = f"[{source.upper()}] " + " [SEP] ".join(headlines[:5])
        emb = self.model.encode(
            text,
            prompt_name="document",     # instruction prefix: "Represent this document for retrieval:"
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        # MRL truncation
        return emb[: self.truncate_dim]
    
    def embed_anchor(self, anchor_text: str) -> np.ndarray:
        """Anchor embedding uses query-side prompt (slightly different, +1-2% on retrieval)."""
        emb = self.model.encode(
            anchor_text,
            prompt_name="query",
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return emb[: self.truncate_dim]


# Storage updated: 256-dim instead of 4096-dim
# 87K bars × 3 sources × 256 floats × 2 bytes (FP16) = ~133 MB on disk (vs 4-5 GB before)
```

### MRL Truncation Strategy

Qwen3-Embedding-4B was contrastive-trained with **Matryoshka Representation Learning** — truncating the embedding to first N dims gives you a usable embedding at all N ∈ {32, 64, 128, 256, 512, 1024, 2560}. Trade-off:

| Truncate dim | Approx MTEB retention | Cache size (87K × 3 × dim × 2B) |
|--------------|----------------------|-------------------------------|
| 32 | ~85% | 16 MB |
| 64 | ~92% | 32 MB |
| 128 | ~97% | 65 MB |
| **256** | **~99%** | **133 MB** ← recommended sweet spot |
| 512 | ~99.5% | 265 MB |
| 2560 | 100% | 1.3 GB |

We use **256-dim** as default. Already fits in our `Linear(256, D=384)` projection. doc 04 / 03 don't need to change — same downstream tensor shapes.

### Throughput Math

87K bars × 3 sources × ~150 tokens (5 headlines × 30 tokens avg) = ~39M tokens to embed.

- Earlier (Llama-3.1-8B mean-pool, ~400 tok/s): **~27 hours**. Overnight runs.
- V1 (Qwen3-Embedding-4B 4-bit MLX, ~18K tok/s): **~36 minutes**. Iterate same-day.

For **anchor embeddings** (one-time, ~80 anchors at ~30 tokens each): ~0.13 seconds. Negligible.

### Apache 2.0 License Implications

Llama 3.1 was under Meta's Llama 3.1 Community License — restricted commercial use above 700M MAU, "Acceptable Use Policy" restrictions. Apache 2.0 is permissive: we can ship anything we want. Important if nanoGLD ever evolves into a paid product or open-source artifact.

### Bit-Determinism via Q4 GGUF

MPS BF16 matmul is **NOT bit-deterministic** across runs (FMA reduction order varies — Valori arXiv:2512.22280, Karnam blog). For cosine queries we don't need bit-exact (cosine still > 0.9999), but for **cache hit-rate via SHA256 keying**, bit-exactness matters.

**Solution:** use the **Q4 GGUF** quantized weights via `llama.cpp` Python bindings. Integer matmul is deterministic. Slight quality loss (~0.5-1 pt MTEB) for full reproducibility.

```python
# Optional: deterministic path via llama.cpp bindings
# from llama_cpp import Llama
# llm = Llama(model_path="Qwen3-Embedding-4B-Q4_K_M.gguf", embedding=True)
# emb = llm.embed("[ALPACA] Fed dovish surprise...")
```

For week 1 ship: use 4-bit MLX (faster, near-deterministic). Migrate to GGUF if cache-hit-rate matters in production.

### Anchor Embedding Update

Anchors get the same model + the `query` prompt prefix (small quality boost):

```python
ANCHOR_PROMPTS = {
    "conflict": "Represent this query about geopolitical military conflict for retrieval:",
    "monetary": "Represent this query about Federal Reserve monetary policy for retrieval:",
    "dollar":   "Represent this query about USD dollar strength for retrieval:",
    "recession": "Represent this query about recession and economic contraction for retrieval:",
}
```

### Fallback Path (Qwen3-Embedding-0.6B if 4B is too slow)

If the M4 mini is also doing nanoGLD training simultaneously (memory pressure):

| Spec | Qwen3-Embedding-0.6B |
|------|---------------------|
| Throughput | ~44K tok/s |
| RAM | 900 MB |
| MTEB-en avg | 70.70 (-4 pts vs 4B) |
| Recommendation | Use only if 4B doesn't co-exist with training |

### What we drop

- ❌ Llama-3.1-8B-Instruct-4bit (replaced)
- ❌ HuggingFace transformers + MPS for 8B model (replaced by sentence-transformers Qwen3 + MLX)
- ❌ Manual mean-pool with attention mask (Qwen3-Embedding handles pooling internally)
- ❌ MLX-LM CaptureWrapper pattern (no longer needed — Qwen3-Embedding-4B exposes embeddings cleanly)
- ❌ Llama 3.1 license acceptance step
- ❌ tokenizer.padding_side / pad_token configuration (sentence-transformers handles)

### What we keep

- ✅ L2-normalize before projection (already done by `normalize_embeddings=True`)
- ✅ Learnable [NO_NEWS] token per source (still needed when no news)
- ✅ Per-source tanh gate init=0 (Flamingo trick)
- ✅ Anchor-cosine semantic features (exact same pattern, different model)
- ✅ FinAnchor / FINEAS / Steck et al. citations
- ✅ memmap fp16 .npy storage (now ~133 MB instead of 4 GB)
- ✅ Anchor cohesion validation test
- ✅ Pin: `sentence-transformers>=5.0`, `transformers>=4.51`, `mlx-lm>=0.20`

### Citations (verified by Nia agent)

- Qwen3-Embedding blog: https://qwenlm.github.io/blog/qwen3-embedding/
- HF Hub: https://huggingface.co/Qwen/Qwen3-Embedding-4B
- MLX 4-bit benchmarks: https://github.com/jakedahn/qwen3-embeddings-mlx
- MTEB leaderboard May 2026: https://huggingface.co/spaces/mteb/leaderboard
- MRL paper: arXiv:2205.13147 (Kusupati et al.)
- MPS non-determinism: arXiv:2512.22280, https://adityakarnam.com/mlx-non-determinism-apple-silicon/

## CRITICAL CORRECTIONS (Nia round 2 — kept)

- ❌ Llama-3.1-8B-FP16 path → ✅ **DROP**. 16GB Mac mini cannot fit 8B-FP16 (16GB just for weights). Use **`mlx-community/Llama-3.1-8B-Instruct-4bit`** (~5GB on disk) OR **Llama-3.2-3B-BF16** if going transformers route.
- ❌ "Use HF transformers + MPS for simplicity" → ⚠️ no clean 4-bit path on MPS (`bitsandbytes` is CUDA-only). **Two real options:**
  - **(a) MLX-LM with CaptureWrapper pattern** ([ml-explore/mlx#3285](https://github.com/ml-explore/mlx)). Wrap `model.model.layers[i]`, swap, run forward, restore. ~30 lines, fast (~30-50 t/s on M4 Pro), proven pattern.
  - **(b) HF transformers + Llama-3.2-3B-Instruct in BF16** (skip 4-bit). Slower per-token but no MLX-specific code. Embeddings are 3072-dim instead of 4096.
  - **Recommendation: (a) MLX path.** ~6-8 hr precompute vs ~24 hr for transformers+MPS+8B.
- ❌ A/B test deferred → ✅ **A/B 3B vs 8B on 1000-bar slice BEFORE full precompute**. If 3B is within 5% on cosine-to-anchors correlation with returns, take the 3-4× speedup.
- ❌ Storage as parquet → ✅ **memmap fp16 `.npy`** OR **zarr**. 87K × 3 × 4096 fp16 = ~2GB on disk (vs 4.3GB fp32 parquet). Random access by `bar_idx` is instant.
- ❌ MPS determinism not tested → ✅ **add batched-vs-unbatched determinism test** ([PyTorch #170837](https://github.com/pytorch/pytorch/issues/170837): batched MPS SDPA produces inconsistent results). Pin `torch>=2.12` for the fix.
- ❌ Tokenizer defaults → ✅ explicitly set `tokenizer.padding_side="right"` and `tokenizer.pad_token = tokenizer.eos_token` (Llama has no default pad token).
- ❌ MISSING `huggingface-cli login` step → ✅ Llama-3.1 is gated. Must accept license + login before download.
- ❌ Anchor-cosine claimed novel → ✅ prior art: **FinAnchor** (arXiv:2602.20859), **FINEAS** (arXiv:2111.00526). Cite. Add anchor-cohesion test (intra-anchor pairwise cosine > 0.6).
- ⚠️ Add **L2-normalize** to 4096-dim mean-pooled vectors before projection (StockTime pattern). Stabilizes scale across articles of varying length.
- ⚠️ FinE5 (arXiv:2502.10990) and FinBERT2 (arXiv:2506.06335) are stronger finance-domain alternatives — note as "considered but Llama wins on macro/geopolitical breadth"
- ⚠️ Pin: `transformers==5.7.0`, `huggingface_hub==1.13.0`, `torch>=2.12`, `mlx-lm==0.31.3`
**Owner:** samsiavoshian
**Implementation effort:** 0.5 day setup + ~6-24 hrs overnight precompute

## Goal

Convert each piece of news text per source per bar into a fixed-dim vector that captures semantic meaning. Use frozen Llama-3.1-8B-4bit running locally on Mac mini. Cache to disk so the LLM never coexists with TinyTransformer training (memory headroom).

## Why Llama-3.1-8B and Not Smaller

User upgraded from Llama-3.2-1B to 8B in the the pivot. Tradeoffs:

| Model | Memory (4-bit) | Tokens/sec on M4 mini | Embedding quality |
|-------|---------------|-----------------------|-------------------|
| Llama-3.2-1B-4bit | ~750MB | ~150 t/s | Good for general text, weaker on finance-specific |
| Llama-3.2-3B-4bit | ~2GB | ~80 t/s | Better, broader knowledge |
| **Llama-3.1-8B-4bit** | **~5GB** | **~40 t/s** | **Best, financial news in pretraining** |

8B fits Mac mini's 16GB unified memory comfortably (LLM only, not coexisting with training). Inference quality meaningfully better for financial / geopolitical text. Tradeoff: precompute time. At 40 t/s with ~512 tokens/embedding × 87,500 bars × 3 sources = ~3.4M tokens to embed = ~24 hrs. Run overnight 2-3 nights or batch process in chunks.

**If precompute is too slow:** fall back to Llama-3.2-3B-4bit (~6 hrs total). Document the choice.

## Setup

```bash
# Install MLX-LM
pip install mlx-lm

# Download model (one time, ~5GB, takes ~10min)
huggingface-cli download mlx-community/Llama-3.1-8B-Instruct-4bit
```

## Embedding Pipeline (HuggingFace Transformers + MPS)

MLX-LM does not natively expose hidden states without patching. Use HuggingFace `transformers` with `output_hidden_states=True` instead. Slower than MLX (~10-20 t/s vs 40 t/s) but works trivially.

```python
# src/embed/precompute.py
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

def precompute_embeddings(
    snapshot_path: str,
    output_path: str,
    model_id: str = "meta-llama/Llama-3.1-8B-Instruct",  # use HF Hub
    batch_size: int = 4,
):
    """
    For each bar, embed news from each source (Alpaca, GDELT, RSS) separately.
    Save as parquet: cols = [bar_idx, alpaca_emb_*, gdelt_emb_*, rss_emb_*]
    """
    df = pd.read_parquet(snapshot_path)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,  # MPS-compatible quantization is limited; FP16
        device_map="mps",
    )
    # Switch to inference mode (no grads, no dropout)
    model.requires_grad_(False)
    
    results = []
    for i in range(0, len(df), batch_size):
        batch = df.iloc[i : i + batch_size]
        embs_per_bar = {"bar_idx": batch.index.tolist()}
        
        for source in ["alpaca", "gdelt", "rss"]:
            texts = [
                build_text_for_bar(row, source)
                for _, row in batch.iterrows()
            ]
            with torch.no_grad():
                tokens = tokenizer(
                    texts, return_tensors="pt", padding=True,
                    truncation=True, max_length=512,
                ).to("mps")
                outputs = model(**tokens, output_hidden_states=True)
                last_hidden = outputs.hidden_states[-1]   # (B, seq, 4096)
                
                # Mean-pool, masking out padding
                mask = tokens.attention_mask.unsqueeze(-1).float()
                summed = (last_hidden * mask).sum(dim=1)
                counts = mask.sum(dim=1).clamp(min=1)
                pooled = (summed / counts).cpu().numpy()  # (B, 4096)
            
            for j in range(len(batch)):
                embs_per_bar[f"{source}_emb"] = pooled[j]
        
        results.extend([
            dict(bar_idx=embs_per_bar["bar_idx"][j], **embs_per_bar)
            for j in range(len(batch))
        ])
        
        if i % 1000 == 0:
            print(f"Embedded {i}/{len(df)} bars")
    
    pd.DataFrame(results).to_parquet(output_path)
```

## Text Construction Per Source (V4 — per-article, NOT per-source-bucket)

V4 architectural change: we DO NOT concatenate top-5 headlines per source into one text blob anymore. The new aggregator (per-source PMA → bar-conditioned Q-Former) wants PER-ARTICLE embeddings so it can learn dissemination weighting (FinGPT pattern, arXiv:2412.10823) and source attribution. Embed each article individually:

```python
def build_text_for_article(article) -> str:
    """Build embedding input for ONE article. Source-tagging happens at the embedding-token
    level via SOURCE_REGISTRY[source].numeric_id, NOT via inline string prefix.
    """
    title = article.title.strip()
    body = (article.body or "").strip()[:1500]   # Qwen3-Embedding context budget
    if not (title or body):
        return ""    # caller skips empty articles
    if title and body:
        return f"{title}\n\n{body}"
    return title or body

# Per bar T, for the news set in window [bar_close - lookback, bar_close - news_latency):
articles_in_window = pd.merge_asof(
    bars_table[["bar_id", "close_ts_utc"]],
    all_news_table.sort_values("t_visible"),
    left_on="close_ts_utc", right_on="t_visible",
    direction="backward", allow_exact_matches=False,
)  # all sources joined by t_visible discipline (see doc 02 V4 hard rule #6)

# For each article, produce:
#   text_emb (256-d MRL-truncated Qwen3 output)
#   source_id (from SOURCE_REGISTRY)
#   bias_tier (from SOURCE_REGISTRY)
#   dt_to_bar = bar_close - article.created_at  → sin/cos encoding (2-d)
# These four pieces become the ARTICLE-LEVEL TOKEN that the Q-Former consumes.

NO_NEWS_TOKEN = "no_news_in_window"  # learned token in the aggregator, NOT a text input
```

For backward-compat with the older 3-source mean-pool approach (deprecated), keep `build_text_for_bar` available behind `--legacy_per_source_concat` flag during the V4 transition cut. Default V4 = per-article.

## Anchor Embedding Computation (frozen, computed once)

```python
# src/embed/anchors.py
ANCHOR_TEXTS = {
    "conflict": [
        "Russia invades Ukraine, military forces cross border",
        "Israel strikes Iranian nuclear facilities",
        "Iran threatens to close Strait of Hormuz",
        "Houthi rebels attack Red Sea shipping",
        # ~20 total per anchor
    ],
    "monetary": [
        "Federal Reserve raises interest rates by 25bps",
        "Powell signals dovish pivot in FOMC press conference",
    ],
    "dollar": [
        "DXY surges past 105 on hawkish Fed comments",
    ],
    "recession": [
        "GDP contracts second consecutive quarter",
    ],
}

def compute_anchors(model, tokenizer) -> dict[str, np.ndarray]:
    anchors = {}
    for name, texts in ANCHOR_TEXTS.items():
        embs = []
        for text in texts:
            with torch.no_grad():
                tokens = tokenizer(
                    text, return_tensors="pt",
                    truncation=True, max_length=512,
                ).to("mps")
                outputs = model(**tokens, output_hidden_states=True)
                last_hidden = outputs.hidden_states[-1]
                mask = tokens.attention_mask.unsqueeze(-1).float()
                pooled = ((last_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)).cpu().numpy()
                embs.append(pooled[0])
        v = np.stack(embs).mean(axis=0)
        anchors[name] = v / np.linalg.norm(v)
    
    # Save as numpy native (.npz), NOT pickle
    np.savez("data/anchors/v1.npz", **anchors)
    return anchors
```

## Caching Strategy

```
data/
├── snapshots/
│   └── v1_<hash>.parquet               ← raw joined data (doc 02)
├── embeddings/
│   ├── v1_<hash>_llama-3.1-8b.parquet  ← per-source embeddings, ~5GB
│   └── v1_<hash>_meta.json             ← timestamp, model used, count
└── anchors/
    └── v1.npz                           ← anchor vectors (small, ~64KB)
```

Embedding cache file size estimate: 87,500 bars × 3 sources × 4096 floats × 4 bytes (FP32) = ~4.3GB raw. Save as parquet with compression → ~2-3GB on disk.

## Why Not Fine-Tune the LLM (recap)

- Hardware: 16GB Mac mini, 4-bit quantized 8B = inference fits but training does not (gradients + optimizer state would 4x memory)
- Methodology: pretrained LLM already has rich finance/macro knowledge; fine-tuning on 87K examples either helps marginally or breaks it (catastrophic forgetting)
- Karpathy mode: not needed. Frozen embeddings + train-your-own-transformer-on-top is the right pattern (same one used in image classification with frozen ResNet)
- Literature: research subagents found no convincing case where LLM fine-tuning beats frozen-embedding + downstream model on financial direction prediction at our data scale

## Validation Tests

```python
def test_embedding_determinism():
    """Same text gives same embedding (within FP16 precision)."""
    e1 = embed_text(model, tokenizer, "Fed raises rates 25bps")
    e2 = embed_text(model, tokenizer, "Fed raises rates 25bps")
    np.testing.assert_allclose(e1, e2, rtol=1e-3)

def test_semantic_anchor_alignment():
    """Conflict-themed news should have high cosine with conflict anchor."""
    conflict_text = "Iran closes Strait of Hormuz amid escalating tensions"
    benign_text = "Apple announces new iPhone features"
    
    anchors_npz = np.load("data/anchors/v1.npz")
    e_conflict = embed_text(model, tokenizer, conflict_text)
    e_benign = embed_text(model, tokenizer, benign_text)
    
    e_conflict_norm = e_conflict / np.linalg.norm(e_conflict)
    e_benign_norm = e_benign / np.linalg.norm(e_benign)
    
    sim_conflict = e_conflict_norm @ anchors_npz['conflict']
    sim_benign = e_benign_norm @ anchors_npz['conflict']
    
    assert sim_conflict > sim_benign + 0.1, "Conflict text should align with conflict anchor more than benign text"
```

## Open Questions / TODOs

- [ ] Verify Llama-3.1-8B-Instruct fits in Mac mini 16GB unified with FP16 (vs needing 4-bit)
- [ ] If MPS FP16 too slow, switch to Llama-3.2-3B (faster, smaller embeddings 3072-dim)
- [ ] Decide if RSS source is worth historical-data difficulty (RSS is forward-only)
- [ ] Confirm anchor texts are representative (review the ~20 headlines per anchor with user)
