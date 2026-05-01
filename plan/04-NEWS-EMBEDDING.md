# 04 — News Embedding

## YOU ARE THE NEWS EMBEDDING AGENT

You own the news-text-to-vector pipeline. You set up Qwen3-Embedding-4B (V4 swap from Llama-3.1-8B-mean-pool), embed all news for all bars once, cache to disk. You also compute the anchor embeddings used by doc 02 for semantic features.

**Read 00-OVERVIEW.md FIRST.**
**Read 01-DATA-PIPELINE.md** for the parquet input schema (alpaca_headlines, gdelt_headlines, rss_headlines columns).
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
src/nanogld/embed/
├── __init__.py
├── qwen3_embedder.py       # Qwen3-Embedding-4B wrapper (4-bit MLX preferred)
├── precompute.py           # Embed all 87K bars × 3 sources, save to memmap
├── anchors.py              # Compute 4 anchor embeddings (conflict, monetary, dollar, recession)
├── live_embed.py           # Single-bar embedding for live cycle (doc 09 imports)
├── cache.py                # SHA256-keyed cache (model + prompt + text → embedding)
└── cli.py                  # `python -m nanogld.embed precompute`

data/embeddings/
├── v1_<hash>_qwen3-emb-4b.npy        # memmap fp16 (87K, 3, 256) ≈ 133MB
└── v1_<hash>_meta.json                # model_id, prompt_template, MRL truncation dim

data/anchors/
└── v1.npz                              # 4 anchor vectors × 256 dim (small)

tests/
├── test_embedding_determinism.py       # Same input + Q4 model → same embedding (rtol 1e-3)
├── test_anchor_cohesion.py             # Intra-anchor cosine > 0.6 (sanity check)
└── test_semantic_alignment.py          # Conflict text > benign text on conflict anchor
```

### Files You DO NOT Touch

- Anything outside `src/nanogld/embed/`, `data/embeddings/`, `data/anchors/`
- Other doc files

### Stable Interface You Publish

```python
# Doc 02 reads precomputed embeddings via:
embeddings = np.load("data/embeddings/v1_<hash>_qwen3-emb-4b.npy", mmap_mode='r')
# Shape: (n_bars, 3, 256). dtype=fp16.

# Doc 09 (live trading) imports for single-bar embedding:
from nanogld.embed.live_embed import embed_news_live
emb = embed_news_live(headlines: dict[str, list[str]]) -> np.ndarray  # (3, 256)

# Doc 02 reads anchors:
anchors_npz = np.load("data/anchors/v1.npz")
# Keys: 'conflict', 'monetary', 'dollar', 'recession'. Each (256,) normalized.
```

### Acceptance Criteria

1. ✅ `python -m nanogld.embed precompute` runs in <60min on M4 mini
2. ✅ Embedding cache file is ~130-200 MB (memmap fp16)
3. ✅ All determinism tests pass (use Q4 GGUF for bit-exact)
4. ✅ Anchor cohesion test: intra-anchor pairwise cosine > 0.6 for all 4 anchor sets
5. ✅ Semantic alignment test: "Iran closes Strait of Hormuz" has cosine to conflict anchor > 0.5; "Apple announces new iPhone" has cosine < 0.3
6. ✅ `live_embed.py` produces single-bar embedding in <100ms (Mac Pro inference)
7. ✅ Anchor embeddings versioned in git via hash in filename (NEVER overwrite)

### Spawn Nia Agents When You Need To

- **MLX-LM Qwen3-Embedding-4B integration** — verify exact API for sentence-level embeddings (NOT generation tokens)
- **Q4 GGUF bit-determinism** — verify `llama-cpp-python` Qwen3 embedding mode produces stable outputs across runs
- **Anchor headline selection** — what 20 headlines best represent "geopolitical military conflict" vs "monetary policy"? Spawn agent to suggest representative texts
- **MRL truncation dim trade-off** — confirm 256-dim retains ≥99% MTEB quality on retrieval (256 is recommended; 128 saves more space)

### V4 Critical Decision (DO NOT REVERT)

**Switched from `meta-llama/Llama-3.1-8B-Instruct-4bit` mean-pool to `Qwen/Qwen3-Embedding-4B` 4-bit MLX.** Reasons in this doc's "V4 PIVOT" section. Key wins: 45× faster (18K vs 400 tok/s), Apache 2.0 license (vs Meta Community), MTEB-en 74.6 (vs ~64), MRL truncatable 2560→256.

If you find Qwen3-Embedding-4B doesn't fit your hardware or has a bug, fallback options (with documented quality cost):
- `Qwen/Qwen3-Embedding-0.6B` — 44K tok/s, 70.7 MTEB (-4 pts), 900MB RAM
- `google/embeddinggemma-300m` — even smaller, ~58 MTEB (significant drop)

Document the choice + reason in this doc's "Deviations" section.

### Encouragement to Research

Embedding models change MONTHLY. Before precomputing 87K bars, run `nia search web "MTEB leaderboard May 2026"` and check if a better model has dropped in the last 2 weeks. If it has, run a 1000-bar A/B between Qwen3-Embedding-4B and the new contender BEFORE committing to the full precompute. The agent's recommendation is based on May 1 — by the time you implement, May 15+ may have something better.

### Hand-off Protocol

1. Update STATUS.md with: embedding cache hash, total size, model used, time-to-precompute
2. Add anchor headlines to `data/anchors/v1_anchors.json` for reproducibility
3. Notify doc 02 (features) that embeddings + anchors are cached and ready

Now read the spec below.

---

# 04 — News Embedding

**Status:** ✅ Complete, implementation-ready, Nia-verified
**Last verified:** 2026-04-30

## V4 PIVOT — Switch from Llama-3.1-8B-Mean-Pool to Qwen3-Embedding-4B (May 2026)

After 7-agent research found that 2026 brought purpose-built embedding models that crush generative-LLM mean-pool on every relevant axis. We switch to **`Qwen/Qwen3-Embedding-4B`** in 4-bit MLX.

### Why this swap is non-negotiable

| Metric | V3 (Llama-3.1-8B mean-pool) | V4 (Qwen3-Embedding-4B) | Delta |
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
    """Replaces V3's Llama-3.1-8B mean-pool. ~45× faster, +10pts MTEB."""
    
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

We use **256-dim** as default. Already fits in our `Linear(256, D=384)` projection. Doc 02 / 03 don't need to change — same downstream tensor shapes.

### Throughput Math

87K bars × 3 sources × ~150 tokens (5 headlines × 30 tokens avg) = ~39M tokens to embed.

- V3 (Llama-3.1-8B mean-pool, ~400 tok/s): **~27 hours**. Overnight runs.
- V4 (Qwen3-Embedding-4B 4-bit MLX, ~18K tok/s): **~36 minutes**. Iterate same-day.

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

### What V4 Drops

- ❌ Llama-3.1-8B-Instruct-4bit (replaced)
- ❌ HuggingFace transformers + MPS for 8B model (replaced by sentence-transformers Qwen3 + MLX)
- ❌ Manual mean-pool with attention mask (Qwen3-Embedding handles pooling internally)
- ❌ MLX-LM CaptureWrapper pattern (no longer needed — Qwen3-Embedding-4B exposes embeddings cleanly)
- ❌ Llama 3.1 license acceptance step
- ❌ tokenizer.padding_side / pad_token configuration (sentence-transformers handles)

### What V4 Keeps

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

User upgraded from Llama-3.2-1B to 8B in the V2.1 pivot. Tradeoffs:

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

## Text Construction Per Source

```python
def build_text_for_bar(row, source: str) -> str:
    """Build the input text for embedding one news source for one bar."""
    if source == "alpaca":
        headlines = row["alpaca_headlines"]   # list[str]
    elif source == "gdelt":
        headlines = row["gdelt_headlines"]
    elif source == "rss":
        headlines = row["rss_headlines"]
    
    if not headlines:
        return f"[NO_NEWS_{source.upper()}]"
    
    top_5 = headlines[:5]
    text = f"[{source.upper()}] " + " [SEP] ".join(top_5)
    return text
```

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
│   └── v1_<hash>.parquet               ← raw joined data (doc 01)
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
