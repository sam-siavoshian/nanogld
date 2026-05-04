# 03 — Model Architecture

## YOU ARE THE ML SYSTEMS ENGINEER AGENT

You own the nanoGLD transformer code itself. You write the model from scratch in raw PyTorch (Karpathy mode — NO HuggingFace `Trainer`, NO Unsloth, NO TRL). You implement the V1 architecture spec locked below.

**Read 00-OVERVIEW.md FIRST.** Project context.
**You DO NOT train.** That's doc 06. You build the model class + verify forward pass shapes.
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
src/nanogld/model/
├── __init__.py
├── rms_norm.py             # Llama-style RMSNorm (no mean centering, no bias)
├── rope.py                 # Real-form RoPE (NEVER torch.view_as_complex on MPS)
├── swiglu.py               # SwiGLU FFN, hidden = round(8D/3, 64)
├── attention.py            # CausalSelfAttention with QK-Norm + per-head gating + value residuals
├── tda.py                  # [A/B candidate] Threshold Differential Attention
├── sype.py                 # [A/B candidate] Symplectic Position Embedding
├── revin.py                # Reversible Instance Norm per channel-group (Kim ICLR 2022)
├── news_fuser.py           # Perceiver-Resampler-lite + Flamingo-gated cross-attn
├── tiny_trader.py          # Main nanoGLDV1 class wiring everything together
├── baselines.py            # DLinear, TSMixer, TimeMixer, xLSTMTime stubs (full impls in doc 08)
└── cli.py                  # `python -m nanogld.model summary` prints param count + arch table

tests/
├── test_forward_pass.py    # Synthetic input, verify (B, T_groups, D) → (B, 3) shape
├── test_param_count.py     # Verify ~24-60M depending on config
├── test_rope_correctness.py # Real-form RoPE matches reference impl on small example
└── test_attention_no_causal.py # Verify is_causal=False (encoder-only)
```

### Files You DO NOT Touch

- Anything in `src/nanogld/data/`, `features/`, `embed/`, `training/`, `backtest/`, `sizing/`, `live/`
- Doc files other than this one
- Training scripts (you provide the model class; doc 06 writes the loop)

### Stable Interface You Publish

```python
# Other docs (esp. doc 06) instantiate:
model = nanoGLDV1(
    numeric_dim: int = 36,
    n_news_queries: int = 8,
    D: int = 384,
    num_heads: int = 6,
    num_layers: int = 12,
    T_bars: int = 64,
    n_classes: int = 3,
    dropout: float = 0.2,
    drop_path: float = 0.15,
)

# Forward signature
def forward(self, channel_inputs: dict[str, torch.Tensor], 
            news_embeddings: torch.Tensor,    # (B, n_sources, 256) — V1: Qwen3 truncated
            news_mask: torch.Tensor) -> torch.Tensor:  # (B, n_sources)
    return logits  # (B, 3)
```

If you change this signature, update STATUS.md + AskUserQuestion before shipping.

### Acceptance Criteria

1. ✅ `pytest tests/test_forward_pass.py` passes — synthetic input produces correct output shape
2. ✅ `pytest tests/test_param_count.py` passes — model has expected param count (within 5% of spec)
3. ✅ `python -m nanogld.model summary` prints param breakdown table
4. ✅ Forward pass on M4 Pro MPS at B=32, T=64 takes <100ms (sanity benchmark)
5. ✅ `pytest tests/test_rope_correctness.py` passes — RoPE matches Karpathy nanoGPT reference impl
6. ✅ Backward pass succeeds without NaN on synthetic input (gradient flow check)
7. ✅ No use of `torch.view_as_complex` (crashes MPS — verified bug)
8. ✅ `.contiguous()` called on Q/K/V before SDPA (PyTorch #181133 workaround)

### Spawn Nia Agents When You Need To

Especially:
- **TDA implementation details** (arXiv:2601.12145) — paper has math but no reference code; spawn agent to find blog/repo implementations
- **SyPE Sp(2,R) symplectic group implementation** (arXiv:2602.08983) — verify the math before coding
- **Per-head gating + value residuals** (IMU-1 arXiv:2602.02522) — paper has spec; verify your impl matches
- **Latest PyTorch SDPA on MPS bug list** (PyTorch issue tracker) — week-to-week changes

### Critical V1 Architecture Decisions (DO NOT REVERT)

1. **ENCODER-only** (drop causal mask). Bidirectional context strictly better for next-bar classification.
2. **Channel-group tokens** (~14 group tokens via iTransformer-lite), NOT 64 per-bar tokens.
3. **RMSNorm + SwiGLU + RoPE + QK-Norm + no-bias** — Llama 3 / Qwen 3 consensus stack.
4. **Per-head gating + value residuals** — IMU-1 recipe, ~50K extra params, big sample efficiency gain.
5. **Partial RoPE** — apply RoPE to 10% of head_dim only.
6. **head_dim = 64** (D=384, num_heads=6). NOT head_dim=32 (Llama 3 / Qwen consensus).
7. **dropout = 0.2** (small data regime). NOT 0.1.
8. **Param count math: ~24-60M depending on D and num_layers.** Don't over-engineer; verify with summary.
9. **Loss is set in doc 06 (3-class CE).** Your model outputs raw logits — never apply softmax in forward.

### A/B Candidates (post-baseline)

These are coded as alternative components but NOT used in the default V1 build:
- **TDA in 1 attention block** — replace `CausalSelfAttention` middle block, compare val Sharpe
- **SyPE replaces RoPE** — single hyperparameter swap
- **Muon optimizer** for 2D weights (doc 06 owns this — model code unchanged)

If your default model fails to converge, before going A/B, first verify:
- Param count within 5% of spec
- LR schedule is working (Schedule-Free should plateau around peak LR after warmup)
- No silent CPU fallback on MPS (run with `TORCH_LOGS=fallback`)

### Hand-off Protocol

1. Update STATUS.md with: model param count, forward pass benchmark, MPS dtype, any deviations
2. Notify doc 06 (training) that model class is stable
3. Document A/B candidate results IN this doc once they're tested

Now read the architecture spec below.

---

# 03 — Model Architecture

**Status:** ✅ Complete, implementation-ready, Nia-verified
**Last verified:** 2026-04-30

## ARCHITECTURE (V1) — 6-Agent 2026 Deep Research (May 2026)

After 7-agent earlier-draft research, ran 6 more agents on 2026-specific releases (Llama 4, Gemma 4, Qwen 3.5, embedding SOTA, time-series foundation models, architecture innovations Jan-May 2026, finance papers, training optimizations). Critical 2026 findings below — V1 ADDS these to earlier draft.

### CRITICAL ADDITIONS FROM 2026 PAPERS

#### A. **FORECAST COLLAPSE RULE** (arXiv:2604.00064, March 2026) — non-negotiable

Paper proves: on weak-conditional-structure data (financial returns), Transformer expressivity **increases variance WITHOUT reducing bias**, producing larger errors than linear baselines on a majority of windows. Direct empirical warning.

**Project rule: NEVER use MSE on returns. Use:**
- 3-class classification with cross-entropy (current default — keep)
- Quantile regression loss (alternative for continuous magnitude estimation)
- NEVER squared loss on raw returns

This is now a hard rule across doc 05, doc 06, doc 08. Loss-function choice is no longer up for debate.

#### B. **per-head gating + value residuals** (IMU-1 paper, arXiv:2602.02522, Jan 2026) — adopt

IMU-1 demonstrates **56× sample efficiency** on 430M model (matches teachers trained on 72B tokens). Validated each component via ablation. Recipe: QK-norm (we have) + **per-head gating** (NEW) + **value residuals** (NEW) + LayerNorm scaling + NorMuon optimizer + muP.

```python
class CausalSelfAttentionV1(nn.Module):
    """V1 adds per-head gating + value residual (IMU-1 recipe)."""
    def __init__(self, D, num_heads, max_seq, dropout=0.2):
        super().__init__()
        # ... existing qkv, proj, q_norm, k_norm, RoPE buffers ...
        # NEW: per-head gating (sigmoid scalar per head, learned)
        self.head_gate = nn.Parameter(torch.zeros(num_heads))
        # NEW: value residual (MLP that produces a residual to add to V from previous layer's V)
        self.value_residual = nn.Linear(D, D, bias=False)
    
    def forward(self, x, prev_v=None):
        # ... existing QK-Norm + RoPE + SDPA ...
        out_per_head = ...   # (B, H, T, head_dim)
        
        # Per-head gating
        gates = torch.sigmoid(self.head_gate).view(1, -1, 1, 1)  # (1, H, 1, 1)
        out_per_head = out_per_head * gates
        
        # Value residual (skip connection across attention layers on V)
        v_residual = self.value_residual(prev_v) if prev_v is not None else 0
        # ... combine into output ...
```

Per-head gating: ~H new params. Value residuals: 1 Linear per block. Total cost ~50K extra params on 12-block 384-dim model. Trivial. Expected gain: **significant on small data** per IMU-1 ablations.

#### C. **TDA — Threshold Differential Attention** (arXiv:2601.12145, Jan 2026) — A/B candidate

Sink-free attention with row-wise extreme-value thresholding. **>99% exact zeros, theoretical O(1) spurious survivors.** Designed precisely for low-SNR signals. Pure PyTorch (no custom kernel needed). MPS-compatible.

```python
class ThresholdDifferentialAttention(nn.Module):
    """TDA: differential attention with row-wise thresholding (arXiv:2601.12145)."""
    def __init__(self, D, num_heads, theta_init=0.5):
        super().__init__()
        # Two parallel softmax branches (subtract one from other)
        self.attn_a = MultiHeadAttention(D, num_heads)
        self.attn_b = MultiHeadAttention(D, num_heads)
        self.lambda_init = nn.Parameter(torch.full((num_heads,), 0.8))
        self.theta = nn.Parameter(torch.full((1,), theta_init))  # threshold
    
    def forward(self, x):
        a = self.attn_a(x)  # (B, H, T, T) softmax weights
        b = self.attn_b(x)
        diff = a - self.lambda_init.view(1, -1, 1, 1) * b
        # Row-wise thresholding (length-dependent gate)
        T = diff.size(-1)
        threshold = self.theta * (1.0 / math.sqrt(T))   # length-dependent
        diff = torch.where(diff.abs() < threshold, torch.zeros_like(diff), diff)
        # Renormalize per row
        diff = diff / diff.sum(-1, keepdim=True).clamp(min=1e-8)
        return diff @ V  # standard attention output
```

**Plan:** A/B replace ONE attention block with TDA, compare val Sharpe. If wins, replace all.

#### D. **SyPE — Symplectic Position Embedding** (arXiv:2602.08983, Feb 2026) — A/B vs RoPE

Paper **proves RoPE cannot represent non-affine temporal warping** (cyclic vs trending markets, vol clusters). Replaces SO(2) rotation with Sp(2,R) symplectic group + adaptive warp module. **Specifically motivated by financial cycles.** SOTA on multivariate forecasting per paper.

```python
class SymplecticPositionEmbedding(nn.Module):
    """SyPE: replaces RoPE for non-stationary time (arXiv:2602.08983)."""
    def __init__(self, head_dim, max_seq):
        super().__init__()
        # Sp(2,R) symplectic group has 3 parameters per pair (vs 1 for SO(2) rotation)
        # Plus adaptive warp module (small MLP)
        self.warp_mlp = nn.Sequential(
            nn.Linear(2, 16), nn.GELU(), nn.Linear(16, 3)
        )
        # ... full impl per paper ...
    
    def forward(self, x, position_ids):
        # Apply position-dependent symplectic transformation
        ...
```

Plan: A/B SyPE vs RoPE on val Sharpe. If financial cycles matter, SyPE wins.

#### E. **Partial RoPE** (arXiv:2603.11611, March 2026) — free win

Applying RoPE to **only ~10% of head dims** matches full RoPE convergence. ~10× KV-cache savings. Not load-bearing for us at T=64-256 but free quality at minimal cost.

```python
def apply_partial_rope(x, cos, sin, frac: float = 0.10):
    """Apply RoPE only to first frac× of head dims."""
    head_dim = x.size(-1)
    rope_dim = int(head_dim * frac) // 2 * 2   # even
    rotated_part = apply_rope(x[..., :rope_dim], cos[..., :rope_dim//2], sin[..., :rope_dim//2])
    return torch.cat([rotated_part, x[..., rope_dim:]], dim=-1)
```

#### F. **xLSTM dominates 2026 finance benchmark** (arXiv:2603.01820, March 2026)

**Most nanoGLD-relevant 2026 finding.** Large-scale benchmark on 2010-2025 daily futures across commodities, equities, bonds, FX. **xLSTM has highest breakeven transaction cost buffer and best downside-adjusted Sharpe** of all sequence models tested. Pure Transformer underperforms.

**Implication:** add xLSTM (xLSTMTime — separate paper, code released 2025) as a **mandatory baseline** alongside DLinear, TSMixer, TimeMixer. doc 08 update.

If our 24-60M Transformer can't beat xLSTM at ~10M params on val Sharpe, ship xLSTM. Same logic as TLOB's "MLP matches transformer" finding from earlier draft.

#### G. **Hybrid attention NOT recommended at our scale** (Qwen 3.5 production endorsement)

Qwen 3.5 (early 2026) ships with **75% Gated DeltaNet linear-attention + 25% softmax**. Strong production signal. But Olmo Hybrid + Long-Context Aware Upcycling ablations show gains are **marginal at T<256**. Skip for our T=64. Reconsider only if context expands to T≥1024.

#### H. **FlashAttention 4 (Blackwell-only)** — useless on MPS

FA4 (arXiv:2603.05451, March 2026) is CUDA Blackwell-targeted. PyTorch SDPA on MPS remains the right call for us.

#### I. **Mamba-3 released ICLR 2026** (arXiv:2603.15569) — skip at our scale

Trapezoidal SSM + complex state + MIMO. Half state size of Mamba-2. MPS-feasible but no CUDA selective-scan kernel = slow. At T=64-256 with 24M params, Transformer still wins on speed AND quality. Skip.

### Updated V1 Architecture Spec

```
nanoGLD V1 (May 2026)
═══════════════════════════════════════════════════════════════════
Backbone:        ENCODER-only transformer 
Tokenization:    Channel-group (iTransformer-lite, ~14 tokens) 
Per-block:       RMSNorm + SwiGLU + RoPE + QK-Norm + no-bias 
ADDITIONS:    
  • Per-head gating (IMU-1) — sigmoid scalar per head, learned
  • Value residuals (IMU-1) — Linear shortcut on V across blocks
  • Partial RoPE (apply to 10% of head_dim, leave rest unrotated)
  • [A/B candidate] TDA in 1+ blocks — sink-free attention
  • [A/B candidate] SyPE replaces RoPE — symplectic position for cycles
News fusion:     Perceiver-Resampler-lite + Flamingo-gated cross-attn 
News embedder:   Qwen/Qwen3-Embedding-4B 4-bit MLX (V1 — replaces Llama-3.1-8B)
Loss:            3-class CE with class weights + label smoothing 0.1
                 NEVER MSE on returns (forecast-collapse rule, arXiv:2604.00064)
Pretrain:        SSL masked-bar reconstruction → linear-probe → LLRD fine-tune 
Optimizer:       SAM ρ=0.05 wrapping AdamW 
                 [A/B candidate] NorMuon — IMU-1 paper, free at our scale
EMA weights:     decay=0.999 
Regularization:  Dropout 0.2 + stochastic depth 0.15 + label smoothing 0.1 
Mandatory baselines (doc 08 — UPDATED):
  • DLinear (~10K params)
  • TSMixer (~2M)
  • TimeMixer (~5M)
  • xLSTMTime (~10M) — NEW per arXiv:2603.01820 finding
  • XGBoost (committed config)
```

### Loss Function Hard Rule (V1)

> **NEVER use squared loss (MSE) on raw returns. Use 3-class cross-entropy or quantile loss.**
> 
> Per arXiv:2604.00064 (March 2026): on weak-conditional-structure data, Transformer expressivity increases variance without reducing bias. Squared-loss training is provably worse than linear baselines on majority of windows in noisy financial regimes.

This rule propagates to doc 06 and 06. Already aligned (we use 3-class CE).

### A/B Test Hierarchy (when implementation begins)

Test in this order (cheapest→most invasive):

1. **Adopt:** per-head gating + value residuals + partial RoPE (IMU-1, ~50K params, no breaks)
2. **A/B:** TDA in one block (replace `CausalSelfAttention` in middle block, compare val Sharpe over 5 seeds)
3. **A/B:** SyPE replaces RoPE (single hyperparameter swap, compare on val Sharpe)
4. **Skip unless context grows:** hybrid linear+softmax attention (Qwen 3.5 pattern)
5. **Skip unless 24M Transformer fails:** xLSTMTime as full architectural pivot

Decision criterion at each gate: ≥0.1 Sharpe improvement OOS, seed-averaged across 5 seeds. Otherwise revert.

## ARCHITECTURE — earlier-draft research (May 2026)

The 5 corrections from previous Nia rounds stand. PLUS major architecture pivots from 7 parallel research agents (modern LLM SOTA + time-series transformers + multimodal fusion + attention 2026 + small-data training + empirical SOTA + MPS optimization). Cross-agent consensus drove these decisions.

### Major Pivots

1. **DECODER-ONLY → ENCODER-ONLY** — drop causal mask. Bidirectional context is strictly better for next-bar 3-class classification (no autoregressive rollout). Free win, all time-series transformer literature converges on this. Reference: PatchTST, iTransformer, MOIRAI all encoder-only.

2. **PER-BAR TOKENS → CHANNEL-GROUP TOKENS** (iTransformer-lite). Instead of 64 time-step tokens of 384 dims, use 6-10 channel-group tokens each summarizing T=64 timesteps. Groups: price_OHLCV, macro, geopolitical, news_a/b/c (post-Resampler), multi-scale aggregates (2hr, 4hr, daily). Reference: iTransformer arXiv:2310.06625, MOIRAI, M5 results.

3. **Modern stack: RMSNorm + SwiGLU + RoPE + QK-Norm + no-bias.** Llama 3 / Qwen 3 / Mistral consensus. QK-Norm is the single biggest stability win at small scale. Reference: Raschka "The Big LLM Architecture Comparison" April 2026.

4. **Multimodal fusion upgrade**: replace `3×Linear(4096,256)` with **Perceiver-Resampler-lite + Flamingo-gated cross-attention**. 8 learnable queries cross-attend to 3 frozen news vectors → 8 fused news tokens. Tanh gate init=0 = identity at start. Multi-stage: news self-attn first, then gated cross-attn into encoder mid-depth. Reference: Flamingo arXiv:2204.14198, BLIP-2 arXiv:2301.12597.

5. **L2-norm news + learnable [NO_NEWS] token + per-source tanh gate + 15% modality dropout** — completes the multimodal fusion stack.

6. **RevIN instance norm per channel-group on input** — Kim et al. ICLR 2022. ~half of PatchTST's empirical gains came from this single trick.

7. **Two-stage training**: SSL pretrain (MLM on same 5y unlabeled) → linear-probe → LLRD fine-tune. Documented 25-50% gain on small-data classification. See doc 06.

8. **Skip Time-LLM / Chronos / Lag-Llama / TimesFM as backbone** — Tan et al. NeurIPS 2024 (arXiv:2406.16964) ablation showed ablating the LLM matches Time-LLM. None of the foundation models cleanly ingest 804-dim multivariate input with news embeddings.

9. **MANDATORY BASELINES** — ship the simpler one if it ties:
   - DLinear (1-layer linear) — sanity floor
   - TSMixer (~2M MLP-mixer) — TLOB paper showed MLP can match transformer
   - TimeMixer (~5M, multi-scale MLP)
   - XGBoost (committed config, doc 08)

   **If nanoGLD 24M doesn't beat all 4 by ≥0.2 Sharpe OOS, SHIP THE BASELINE.**

10. **MPS pin: PyTorch 2.11.0** (has SDPA fix #174945). FP32 everywhere. No autocast, no torch.compile, no quantization. `.contiguous()` Q/K/V before SDPA. DataLoader `num_workers=0`.

### Empirical Bar (from Agent 6 research)

**No published transformer beats XGBoost/MLP on single-asset 30min direction with honest walk-forward + DSR + costs.** TLOB paper: "an MLP matches our transformer." Cleanest gold result = **Forecast-to-Fill** (arXiv:2511.08571) — NOT a transformer, Sharpe 2.88, MDD 0.52%, walk-forward 5y. This is the bar to beat.

Targets:
- **Minimum viable**: 38% accuracy (3-class), beat XGBoost on same features, Sharpe > 0 after 5bps round-trip, walk-forward
- **Real claim**: Sharpe > 1.0, hit rate > 52%, DSR > 1.0, beats all 4 baselines by ≥0.2 Sharpe
- **Forecast-to-Fill tier**: Sharpe > 2.5 — almost no transformer paper here

## CRITICAL CORRECTIONS (Nia round 2 — kept from prior verification)

- ❌ Manual attention math (Q@K.T / sqrt(D) + masked_fill + softmax) → ✅ **`F.scaled_dot_product_attention(q, k, v, dropout_p=p, is_causal=True)`** — fused MPS kernel, 1.5-3× faster, fewer lines, drops causal_mask buffer. nanoGPT does this when SDPA is available.
- ❌ heads=12, head_dim=32 → ✅ **heads=6, head_dim=64** at D=384. head_dim=32 is unusually small (Llama 3 uses 128 even at 8B; sub-1B sweet spot is 64). Same total params, better-conditioned per-head subspace.
- ❌ Default dropout=0.1 → ✅ **dropout=0.2** (small data ~16K samples + transformer overfits fast). Modern LLMs use 0.0 because pretraining data is effectively infinite — opposite regime from us. Sweep 0.0-0.3 in training.
- ❌ Param count "~24M total" with "9.45M news projections" → ✅ **~18M total**, news projections are **3.15M total** (3 × 1.05M, NOT 3 × 1.05M × 3). To hit 50M target, raise to D=512, 10 layers OR D=768, 8 layers.
- ❌ Standard residual init → ✅ **scaled residual init** for `*proj.weight` layers: `std=0.02 / sqrt(2 * num_layers)`. GPT-2 trick, keeps activation variance stable as depth grows.
- ❌ NewsProjection raw output → ✅ add **`LayerNorm(projected_dim)` after each per-source Linear** (handles variable news-embedding magnitude across articles)
- ❌ `torch.compile` on MPS — try it → ✅ **DON'T**. PyTorch issue #171764: produces NaN loss on MPS where eager converges fine. Stay eager.
- ⚠️ SwiGLU vs GELU MLP: marginal gain (1-3% ppl) at this scale. Skip for v1 (Karpathy mode = type GELU, ship). future candidate if val loss plateaus.
- ⚠️ MPS BF16 autocast partial in 2026 (PyTorch #139386, #97236, #84516). Phase 1 = FP32 only. Phase 2 = autocast forward only (NEVER manual `.to(bfloat16)` on weights — explicit anti-pattern in PyTorch docs).
**Owner:** samsiavoshian
**Implementation effort:** 1 day (day 6 of week 1)

## Karpathy-Mode Discipline

You write every line. No HuggingFace `Trainer`. No Unsloth. No `transformers.AutoModel`. Raw PyTorch on MPS. The point is to understand attention, residuals, layer norm, and decoder-only causal masking by typing them.

## Specs

```
Architecture:        ENCODER-only transformer (NO causal mask, bidirectional)
                     Channel-group tokenization (iTransformer-lite, 6-10 group tokens)
Parameters:          ~24-60M
Hidden dim D:        384 (or 512 for ~60M variant)
Num layers:          12 (deeper > wider per Cerebras/Levine empirical rule, Qwen 3 evidence)
Num heads:           6  (head_dim = 64 — Llama 3 / Qwen consensus, NOT 32)
MLP hidden:          round(8D/3, 64) = 1024 at D=384  (SwiGLU, NOT 4D GELU)
Group tokens:        6-10 (price, macro, geo, news×3 post-Resampler, multi_scale×3)
Time context:        T=64 30min bars summarized INSIDE each channel token's projection
Output:              3 logits via mean-pool over group tokens → Linear(D, 3)
Dropout:             0.2 (small-data regime); + stochastic depth p=0.15
Norm:                RMSNorm (NOT LayerNorm), pre-norm, eps=1e-6, + QK-Norm
Position encoding:   RoPE real-form (NOT view_as_complex on MPS) for time tokens
                     Learned positional for channel-group identity
Activation:          SwiGLU (NOT GELU) — Shazeer 2020 / Llama / Qwen consensus
Bias:                NO biases anywhere (Llama / Qwen / Mistral consensus)
RevIN:               instance norm per channel-group on input (Kim ICLR 2022)
```

Param count check:
- Per attention: 4 × D² = 4 × 384² = 590K
- Per MLP: 2 × D × mlp_hidden = 2 × 384 × 1536 = 1.18M
- Per block (attn + mlp): ~1.77M
- 8 blocks: ~14M
- Input projection 804 → 384: 308K
- Position embedding: 64 × 384 = 24K
- Output head 384 → 3: 1.2K
- News-source projections (3 × Linear(4096, 256)): 3.15M each = 9.45M

**Total: ~24M params** (less than the 50-100M target — we can scale up by raising D or num_layers if budget allows).

To hit 50M: try `D=512, num_layers=10, mlp_hidden=2048`. Calc: per-block ~3.1M × 10 = 31M, plus input/projections, total ~40-45M. Matches target.

To hit 100M: try `D=768, num_layers=12`. ~85-100M. Will fit on Mac mini 16GB but tight during training.

**Recommendation:** start at D=384 / 8 layers / ~24M total. Train it, see if val loss converges. Scale up only if val loss hasn't converged AND OOS Sharpe is improving with capacity (it usually doesn't on financial data — see prior warning).

## Code (~80 lines)

```python
# src/model/tiny_trader.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    def __init__(self, D: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert D % num_heads == 0
        self.D = D
        self.H = num_heads
        self.head_dim = D // num_heads
        self.qkv = nn.Linear(D, 3 * D, bias=False)
        self.proj = nn.Linear(D, D)
        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)
        # Causal mask buffer (no learnable params, registered for device-move)
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(1024, 1024)).view(1, 1, 1024, 1024),
            persistent=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.H, self.head_dim)
        q, k, v = qkv.unbind(dim=2)            # each (B, T, H, head_dim)
        q = q.transpose(1, 2)                  # (B, H, T, head_dim)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        scores = q @ k.transpose(-2, -1) / (self.head_dim ** 0.5)
        mask = self.causal_mask[:, :, :T, :T]
        scores = scores.masked_fill(mask == 0, float("-inf"))
        attn = F.softmax(scores, dim=-1)
        attn = self.attn_dropout(attn)

        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, D)
        return self.proj_dropout(self.proj(out))


class Block(nn.Module):
    def __init__(self, D: int, num_heads: int, mlp_hidden: int, dropout: float = 0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(D)
        self.attn = CausalSelfAttention(D, num_heads, dropout)
        self.ln2 = nn.LayerNorm(D)
        self.mlp = nn.Sequential(
            nn.Linear(D, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, D),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))   # pre-norm
        x = x + self.mlp(self.ln2(x))
        return x


class NewsProjection(nn.Module):
    """Projects each news source's 4096-dim raw Llama embedding down to 256."""
    def __init__(self, num_sources: int = 3, raw_dim: int = 4096, projected_dim: int = 256):
        super().__init__()
        self.projs = nn.ModuleList([
            nn.Linear(raw_dim, projected_dim) for _ in range(num_sources)
        ])

    def forward(self, news_raw: torch.Tensor) -> torch.Tensor:
        """news_raw: (B, T, num_sources * raw_dim) → (B, T, num_sources * projected_dim)"""
        B, T, total = news_raw.shape
        num_sources = len(self.projs)
        per_source = total // num_sources
        chunks = news_raw.chunk(num_sources, dim=-1)
        projected = [proj(chunk) for proj, chunk in zip(self.projs, chunks)]
        return torch.cat(projected, dim=-1)


class nanoGLD(nn.Module):
    def __init__(
        self,
        numeric_dim: int = 36,        # price + risk + macro + geo features
        news_raw_dim_per_source: int = 4096,
        num_news_sources: int = 3,
        news_projected_dim: int = 256,
        D: int = 384,
        num_heads: int = 12,
        num_layers: int = 8,
        T: int = 64,
        dropout: float = 0.1,
        num_classes: int = 3,
    ):
        super().__init__()
        self.T = T
        self.news_proj = NewsProjection(num_news_sources, news_raw_dim_per_source, news_projected_dim)
        # Total per-bar input dim after projection
        total_in = numeric_dim + num_news_sources * news_projected_dim   # e.g. 36 + 3*256 = 804
        self.input_proj = nn.Linear(total_in, D)
        self.pos_embed = nn.Parameter(torch.zeros(1, T, D))
        self.blocks = nn.ModuleList([
            Block(D, num_heads, 4 * D, dropout) for _ in range(num_layers)
        ])
        self.ln_f = nn.LayerNorm(D)
        self.head = nn.Linear(D, num_classes)
        self.dropout = nn.Dropout(dropout)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.trunc_normal_(m.weight, std=0.02)
        elif isinstance(m, nn.LayerNorm):
            nn.init.zeros_(m.bias)
            nn.init.ones_(m.weight)

    def forward(self, numeric: torch.Tensor, news_raw: torch.Tensor) -> torch.Tensor:
        """
        numeric:  (B, T, numeric_dim)            ← price + risk + macro + geo
        news_raw: (B, T, num_sources * 4096)     ← concatenated raw embeddings
        Returns logits: (B, num_classes)
        """
        news_projected = self.news_proj(news_raw)            # (B, T, 768)
        x = torch.cat([numeric, news_projected], dim=-1)     # (B, T, 804)
        x = self.input_proj(x)                                # (B, T, D)
        x = x + self.pos_embed[:, : x.size(1), :]
        x = self.dropout(x)

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        # Use final-token representation for direction prediction
        return self.head(x[:, -1, :])                         # (B, num_classes)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
```

## Forward-Pass Sanity Test

```python
def test_forward_pass():
    model = nanoGLD()
    B, T = 4, 64
    numeric = torch.randn(B, T, 36)
    news_raw = torch.randn(B, T, 3 * 4096)
    logits = model(numeric, news_raw)
    assert logits.shape == (B, 3)
    print(f"Param count: {count_params(model):,}")  # ~24M with default config
```

## Memory Estimate (Mac mini 16GB)

For batch B=32, T=64, D=384:

- Activations per layer: ~B × T × D × 4 bytes (FP32) = 32 × 64 × 384 × 4 = 3MB per tensor, with attention scores adding ~B × H × T² × 4 = 32 × 12 × 64² × 4 = 6MB per layer
- 8 layers + storage for backward: ~150-300MB activations
- Parameters: 24M × 4 bytes = 96MB (FP32) or 48MB (FP16)
- AdamW optimizer state: 2× params = 192MB FP32 or 96MB FP16
- Gradients: 1× params = 96MB FP32
- News raw embeddings batch: B × T × 3 × 4096 × 4 = 32 × 64 × 12288 × 4 = 100MB

Total: ~600MB-1GB peak during training. Fits trivially on 16GB Mac mini.

If we scale to D=768 / 12 layers / ~100M params: ~3-5GB peak. Still fits but no margin for OS.

## Mixed Precision on MPS

```python
# Apple Silicon MPS doesn't support full autocast yet, but we can use bfloat16 weights
model = model.to(device='mps', dtype=torch.float32)  # weights FP32 for stability
# OR for memory savings, FP16 weights with FP32 master copy via torch.amp.autocast (limited MPS support)
```

Current state: MPS autocast support is partial. Recommend FP32 weights for stability, bfloat16 only if OOM.

## Position Encoding Choice

**Why learned positional embedding (not sinusoidal):**
- T=64 is fixed throughout the project (we never extend sequence length)
- Learned positions have ~24K params (negligible)
- Easier to debug than sinusoidal (no math errors)
- For variable T, RoPE would be better, but we don't need that

## Output Choice (final-token vs mean-pool)

We use the **final token's hidden state** to produce logits. Alternative: mean-pool across all tokens.

**Why final token wins for autoregressive forecasting:**
- The final token has had attention over all previous tokens (causal mask)
- It's the "summary" of all 64 bars from the perspective of "predict the next bar"
- Standard pattern in GPT-style models
- Mean-pool would give equal weight to bar 1 (oldest, least relevant) and bar 64 (most recent)

## Why Not MoE / Bigger / Fancier

User mentioned MoE earlier. MoE shines when:
- Total params > 1B
- Multiple "expert" specializations are obvious (different domains, languages)
- You can amortize router overhead across many tokens

For our case (~50-100M, single domain, single asset), MoE adds complexity for no meaningful gain. Skip.

## V1 Reference Implementation (Modern Stack)

```python
# src/model/tiny_trader_v3.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class RMSNorm(nn.Module):
    """Llama-style RMSNorm. No mean centering, no bias. ~7-15% faster than LayerNorm."""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x):
        norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return norm * self.weight


def precompute_rope_cache(head_dim: int, max_seq: int, theta: float = 10000.0):
    """Real-form RoPE cache. NEVER use torch.view_as_complex on MPS (crashes)."""
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    t = torch.arange(max_seq).float()
    angles = torch.outer(t, freqs)  # (max_seq, head_dim/2)
    return torch.cos(angles), torch.sin(angles)


def apply_rope(x, cos, sin):
    """Real-form RoPE. x: (..., seq, head_dim)."""
    x1, x2 = x[..., 0::2], x[..., 1::2]
    cos = cos[: x.size(-2), :]
    sin = sin[: x.size(-2), :]
    rotated = torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
    return rotated.flatten(-2)


class CausalSelfAttention(nn.Module):
    """Encoder-only (NO causal mask) MHA + QK-Norm + RoPE + SDPA."""
    def __init__(self, D: int, num_heads: int, max_seq: int, dropout: float = 0.2):
        super().__init__()
        assert D % num_heads == 0
        self.D, self.H, self.head_dim = D, num_heads, D // num_heads
        self.qkv = nn.Linear(D, 3 * D, bias=False)
        self.proj = nn.Linear(D, D, bias=False)
        # QK-Norm: stabilizes attention at small scale (Qwen 3 / Gemma 3)
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)
        self.dropout = dropout
        # Pre-compute RoPE cache (real-form, MPS-safe)
        cos, sin = precompute_rope_cache(self.head_dim, max_seq)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        qkv = self.qkv(x).view(B, T, 3, self.H, self.head_dim)
        q, k, v = qkv.unbind(dim=2)              # (B, T, H, head_dim)
        # QK-Norm BEFORE RoPE
        q = self.q_norm(q)
        k = self.k_norm(k)
        # RoPE on Q and K (not V)
        q = apply_rope(q, self.rope_cos, self.rope_sin)
        k = apply_rope(k, self.rope_cos, self.rope_sin)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        # SDPA — encoder-only, NO causal mask. .contiguous() per PyTorch #181133
        out = F.scaled_dot_product_attention(
            q.contiguous(), k.contiguous(), v.contiguous(),
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,
        )
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        return self.proj(out)


class SwiGLU(nn.Module):
    """Llama-style SwiGLU FFN. hidden = round(8D/3, 64). 3 matrices, no bias."""
    def __init__(self, D: int, dropout: float = 0.2):
        super().__init__()
        hidden = ((int(8 * D / 3) + 63) // 64) * 64  # round to multiple of 64
        self.w_gate = nn.Linear(D, hidden, bias=False)
        self.w_up = nn.Linear(D, hidden, bias=False)
        self.w_down = nn.Linear(hidden, D, bias=False)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        return self.dropout(self.w_down(F.silu(self.w_gate(x)) * self.w_up(x)))


class Block(nn.Module):
    def __init__(self, D, num_heads, max_seq, dropout=0.2, drop_path=0.15):
        super().__init__()
        self.ln1 = RMSNorm(D)
        self.attn = CausalSelfAttention(D, num_heads, max_seq, dropout)
        self.ln2 = RMSNorm(D)
        self.mlp = SwiGLU(D, dropout)
        self.drop_path = drop_path  # stochastic depth probability
    
    def forward(self, x):
        if self.training and torch.rand(1).item() < self.drop_path:
            return x  # skip block (stochastic depth)
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class RevIN(nn.Module):
    """Reversible Instance Normalization per channel group (Kim ICLR 2022)."""
    def __init__(self, num_features: int, eps: float = 1e-5, affine: bool = True):
        super().__init__()
        self.eps = eps
        self.affine = affine
        if affine:
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias = nn.Parameter(torch.zeros(num_features))
    
    def forward(self, x, mode='norm'):
        if mode == 'norm':
            self.mean = x.mean(dim=-2, keepdim=True).detach()
            self.std = x.std(dim=-2, keepdim=True).detach() + self.eps
            x = (x - self.mean) / self.std
            if self.affine:
                x = x * self.weight + self.bias
        elif mode == 'denorm':
            if self.affine:
                x = (x - self.bias) / self.weight
            x = x * self.std + self.mean
        return x


class NewsFuser(nn.Module):
    """Perceiver-Resampler-lite + Flamingo-gated cross-attn (Agent 3 recommendation)."""
    def __init__(self, d_news=4096, d_model=384, n_sources=3, n_queries=8, n_heads=4):
        super().__init__()
        self.no_news = nn.Parameter(torch.randn(n_sources, d_news) * 0.02)
        self.proj = nn.ModuleList([nn.Linear(d_news, d_model, bias=False) for _ in range(n_sources)])
        self.alpha = nn.Parameter(torch.zeros(n_sources))   # init=0, opens gradually
        self.news_sa = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, batch_first=True, norm_first=True
        )
        self.queries = nn.Parameter(torch.randn(n_queries, d_model) * 0.02)
        self.cross = nn.MultiheadAttention(d_model, num_heads=n_heads, batch_first=True, bias=False)
    
    def forward(self, news_raw: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """news_raw: (B, n_sources, 4096), mask: (B, n_sources) 1=present 0=absent"""
        B, S, _ = news_raw.shape
        # F: substitute learned [NO_NEWS] for absent sources
        x = torch.where(mask.unsqueeze(-1).bool(), news_raw, self.no_news)
        # E: L2-normalize before projection
        x = F.normalize(x, dim=-1)
        # Project per source
        x = torch.stack([self.proj[s](x[:, s]) for s in range(S)], dim=1)  # (B, S, D)
        # C: per-source tanh gate, init=0
        x = x * torch.tanh(self.alpha).view(1, S, 1)
        # G: news self-attn (sources fuse together)
        x = self.news_sa(x)
        # B: Perceiver queries cross-attend to fused news
        q = self.queries.unsqueeze(0).expand(B, -1, -1)
        out, _ = self.cross(q, x, x)
        return out  # (B, n_queries, D)


class nanoGLDV1(nn.Module):
    """Encoder-only, channel-group tokenization, full modern stack."""
    def __init__(
        self,
        numeric_dim: int = 36,             # price + risk + macro + geo
        news_raw_dim: int = 4096,
        n_news_sources: int = 3,
        n_news_queries: int = 8,
        D: int = 384,
        num_heads: int = 6,                # head_dim = 64
        num_layers: int = 12,
        T_bars: int = 64,
        n_classes: int = 3,
        dropout: float = 0.2,
        drop_path: float = 0.15,
    ):
        super().__init__()
        # RevIN per channel group
        self.revin_numeric = RevIN(numeric_dim)
        # Channel-group projections (each summarizes T=64 timesteps × features → 1 token of D)
        # In practice each "group" is its own time-collapsing Linear(T × group_dim, D)
        self.channel_groups = nn.ModuleDict({
            'price':       nn.Linear(T_bars * 12, D, bias=False),  # 12 price feats × 64 bars
            'macro':       nn.Linear(T_bars * 6, D, bias=False),
            'geo':         nn.Linear(T_bars * 10, D, bias=False),
            'risk':        nn.Linear(T_bars * 8, D, bias=False),
            'multi_2hr':   nn.Linear(T_bars // 4 * 12, D, bias=False),
            'multi_daily': nn.Linear(T_bars // 13 * 12, D, bias=False),
        })
        # News fuser → 8 query tokens
        self.news_fuser = NewsFuser(d_news=news_raw_dim, d_model=D,
                                     n_sources=n_news_sources, n_queries=n_news_queries)
        # Total tokens: 6 channel groups + 8 news queries = 14 tokens
        n_tokens = len(self.channel_groups) + n_news_queries
        self.pos_embed = nn.Parameter(torch.zeros(1, n_tokens, D))
        # Encoder blocks
        self.blocks = nn.ModuleList([
            Block(D, num_heads, max_seq=n_tokens, dropout=dropout, drop_path=drop_path)
            for _ in range(num_layers)
        ])
        self.ln_f = RMSNorm(D)
        self.head = nn.Linear(D, n_classes, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            # Scaled init on output projections (GPT-2 trick)
            for name, _ in self.named_parameters():
                if 'proj.weight' in name or 'w_down.weight' in name:
                    nn.init.trunc_normal_(m.weight, std=0.02 / math.sqrt(2 * len(self.blocks)))
    
    def forward(self, channel_inputs: dict, news_raw: torch.Tensor, news_mask: torch.Tensor):
        """
        channel_inputs: dict[group_name -> (B, T_group * features) flattened]
        news_raw: (B, 3, 4096)
        news_mask: (B, 3)
        """
        # Build channel-group tokens
        tokens = []
        for name, proj in self.channel_groups.items():
            tokens.append(proj(channel_inputs[name]).unsqueeze(1))  # (B, 1, D)
        # Add news query tokens
        news_tokens = self.news_fuser(news_raw, news_mask)  # (B, 8, D)
        x = torch.cat(tokens + [news_tokens], dim=1)  # (B, n_tokens, D)
        # Add learned positional embedding
        x = x + self.pos_embed
        x = self.dropout(x)
        # Encoder blocks
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        # Mean-pool across all tokens (vs final-token in decoder-only)
        return self.head(x.mean(dim=1))   # (B, n_classes)
```

Param count check at D=384, 12 layers, 6 heads:
- Per block: 4×D² (attn) + 3×D×(8D/3) (SwiGLU) ≈ 590K + 1.18M ≈ 1.77M
- 12 blocks: ~21.2M
- Channel projections: ~15M (depends on T)
- NewsFuser: ~4.5M
- **Total: ~40-50M params.** Bigger than earlier 24M target because we kept Channel projections fat. Tune by reducing T-collapse or D.

## Open Questions / TODOs

- [ ] Decide final size (24M / 50M / 100M) after first training run convergence check
- [ ] Test MPS bfloat16 weights vs FP32 — does training still converge?
- [ ] Profile attention compute — does flash-attention help on MPS? (Probably not yet supported)
