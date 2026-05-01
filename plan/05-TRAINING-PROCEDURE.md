# 05 — Training Procedure

## YOU ARE THE TRAINING ENGINEER AGENT

You own the training loop. You take the model class from doc 03, the feature DataFrame from doc 02, and produce trained model checkpoints + walk-forward validation results.

**Read 00-OVERVIEW.md FIRST.**
**Read 02-FEATURE-ENGINEERING.md** schema (input shape).
**Read 03-MODEL-ARCHITECTURE.md** model class signature.
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
src/nanogld/training/
├── __init__.py
├── walk_forward.py         # Walk-forward CV splitter (4 folds at 5y, 1-week embargo)
├── dataset.py              # PyTorch Dataset wrapping feature DataFrame + cached embeddings
├── losses.py               # Class-weighted CE + label smoothing (NEVER MSE on returns)
├── optimizer.py            # Schedule-Free AdamW config + Friendly-SAM wrapper
├── ema.py                  # Exponential moving average of weights (decay=0.999)
├── augmentation.py         # Jittering + magnitude warping + Manifold Mixup
├── checkpoint.py           # Save/load with metadata (snapshot hash, fold, seed)
├── ssl_pretrain.py         # MAE-on-masked-bars (Stage 1 of 3-stage training)
├── linear_probe.py         # Stage 2: freeze encoder, train head only
├── llrd_finetune.py        # Stage 3: layer-wise LR decay full fine-tune
├── train_fold.py           # End-to-end one fold (called by walk_forward driver)
├── seeds.py                # Comprehensive RNG seeding (torch + mps + numpy + random)
└── cli.py                  # `python -m nanogld.training run --config configs/v4.yaml`

tests/
├── test_walk_forward.py    # Verify no train/val/test overlap, embargo respected
├── test_dataset.py         # Sample-by-sample shape verification
└── test_seed.py            # Same seed → same loss curve (within MPS non-determinism floor)

checkpoints/                # gitignored
├── fold_0_seed_42_ema.pt
├── fold_1_seed_42_ema.pt
└── ...
```

### Files You DO NOT Touch

- `src/nanogld/model/` — doc 03 owns the architecture; you wire it into the training loop
- `src/nanogld/embed/` — doc 04
- `src/nanogld/features/` — doc 02
- `src/nanogld/backtest/`, `sizing/`, `live/` — downstream consumers

### Stable Interface You Publish

```python
# Doc 06 (backtest) loads your checkpoints:
checkpoint = torch.load("checkpoints/fold_3_seed_42_ema.pt", weights_only=True)
model.load_state_dict(checkpoint['ema_state_dict'])

# Doc 09 (live) loads the SAME format
```

### Acceptance Criteria

1. ✅ `python -m nanogld.training run` completes 4 walk-forward folds
2. ✅ Loss decreases across epochs (no NaN, no plateau-at-init)
3. ✅ Val accuracy beats class-prior baseline (44% if always-flat) on at least 3/4 folds
4. ✅ Walk-forward unit test passes (no overlap, embargo respected)
5. ✅ Checkpoints saved with EMA state, metadata includes snapshot hash + fold + seed
6. ✅ Total training time < 12 hrs on M4 Mac mini for 4 folds × 1 seed
7. ✅ wandb workspace public, runs named `fold{N}_seed{N}` for X-thread material

### Spawn Nia Agents When You Need To

Especially:
- **Schedule-Free AdamW integration with PyTorch** — verify the train()/eval() calls and warmup_steps interaction with our walk-forward
- **Friendly-SAM impl details** — paper has math; spawn agent to find reference repo / usable implementation
- **MTS-JEPA implementation details** for Phase 2 — paper is from Feb 2026, code may not be public
- **MPS-specific seeding** — `torch.mps.manual_seed()` exists in PyTorch 2.11; verify it works as expected
- **wandb logging cadence on macOS** (avoid daemon issues with multiprocess)

### V4 Critical Decisions (DO NOT REVERT)

1. **Schedule-Free AdamW REPLACES "AdamW + cosine + warmup"** — Defazio ICLR 2025
2. **Friendly-SAM REPLACES vanilla SAM** — filters gradient noise, drop-in
3. **3-stage training: SSL pretrain → linear-probe → LLRD fine-tune**
4. **EMA decay=0.999** — deployed model is EMA, NOT raw
5. **dropout 0.2 + stoch depth 0.15 + label smoothing 0.1** — small-data regime
6. **NEVER MSE on returns** (forecast-collapse rule arXiv:2604.00064) — 3-class CE only
7. **Walk-forward = 4 folds at 5y** (NOT 6-8 — math: floor((60-48)/3) = 4)
8. **PyTorch 2.11.0 pinned, FP32, num_workers=0**

### A/B Candidates (after baseline ships)

These are alternative components you can test if baseline plateaus:
- **Muon optimizer** for 2D weights (DeepSeek V4 / Kimi-2 production)
- **MTS-JEPA** replaces MAE pretrain (Phase 2)
- **Cross-asset transfer** SPY → GLD via LLRD (bonus experiment)

A/B gate: ≥0.1 Sharpe improvement OOS (val), seed-averaged 5 seeds.

### Hand-off Protocol

1. Update STATUS.md with: best fold val accuracy, EMA checkpoint paths, training time, wandb workspace URL
2. Notify doc 06 (backtest) that checkpoints are ready
3. Document any deviations IN this doc

Now read the implementation specifics.

---

# 05 — Training Procedure

**Status:** ✅ Complete, implementation-ready, Nia-verified
**Last verified:** 2026-04-30

## V5.1 — Library addition: `accelerate==1.13.x` (May 2026)

Owner lifted "raw PyTorch only" for non-architectural infrastructure. Single library added: **`accelerate>=1.13,<2.0`** (HuggingFace).

**What it gives us (5 LOC):**
- Device-agnostic code (no `if mps else cpu` branches)
- Free checkpoint state save/load with optimizer + scheduler state
- `accelerator.set_seed()` covers torch + cuda + mps + numpy + random in one call
- Gradient accumulation handled by `accelerator.accumulate(model)`

**What it does NOT give (skip):**
- Mixed precision on MPS (near-zero gain, stay FP32)
- Distributed training (we are single-device)

**Pattern:**
```python
from accelerate import Accelerator

accelerator = Accelerator(mixed_precision="no")  # FP32 on MPS
model, optimizer, train_loader, val_loader = accelerator.prepare(
    model, optimizer, train_loader, val_loader
)
accelerator.set_seed(42)

# In loop:
for batch in train_loader:
    with accelerator.accumulate(model):
        loss = compute_loss(model, batch)
        accelerator.backward(loss)
        optimizer.step()
        optimizer.zero_grad()

# Checkpoint
accelerator.save_state("checkpoints/fold_3/")
```

**`transformers.Trainer` — optional, deferred decision.** If our Friendly-SAM `training_step` override fights Trainer assumptions (it might), stick with raw loop + accelerate. If clean override works, Trainer brings free EarlyStopping + W&B + checkpointing callbacks (~50 LOC override). Decide during implementation.

**Skipped permanently (MPS reasons):**
- Unsloth — requires Triton, no Mac support
- bitsandbytes 4-bit — CUDA-only
- `torch.compile` — still NaN on MPS in 2026
- `lightning-thunder` — NVIDIA-focused, no MPS
- torchtitan — multi-GPU LLM pretraining, overkill

**`peft==0.19.x` deferred** until/if embedder fine-tune day arrives (currently month 4+ per Agent E recommendation).

## TRAINING STACK V4 — 2026 SOTA (May 2026, 6-agent research)

After V3 stack went live, ran another deep-research pass on training/optimization papers from Jun 2025 to May 2026. Several adopted improvements below.

### Major V4 Changes

#### 1. Schedule-Free AdamW REPLACES "AdamW + cosine + warmup"

**Defazio et al., ICLR 2025 outstanding paper, arXiv:2405.15682.** Won MLCommons AlgoPerf 2024 across all workloads. Removes the LR schedule entirely via momentum-based iterate averaging. **Anytime-optimal** — checkpoint at any step is the best the model can be at that point.

```python
# Old V3:
# optimizer = torch.optim.AdamW(...)
# scheduler = cosine_lr_schedule(optimizer, warmup_steps, total_steps)

# New V4:
import schedulefree
optimizer = schedulefree.AdamWScheduleFree(
    model.parameters(),
    lr=1e-4,
    betas=(0.9, 0.95),
    weight_decay=0.1,
    warmup_steps=300,
)

# In training loop:
optimizer.train()      # call at start of train phase
loss.backward()
optimizer.step()

# When validating, call optimizer's eval-mode method (switches to averaged weights)
# (See schedulefree docs — there is a corresponding state-switch call before validation)
```

Why for us: 16K samples means we ablate epoch count constantly. Schedule-Free removes "what if I train one more epoch" anxiety. Anytime-optimal = any saved checkpoint is the best model at that step.

Repo: https://github.com/facebookresearch/schedule_free

**Fallback if Schedule-Free underperforms:** WSD (Warmup-Stable-Decay), arXiv:2601.09000 (Jan 2026, ICML 2025). Linear warmup → flat plateau → 8-20% step cooldown. Compute-agnostic, lets you branch checkpoints.

#### 2. Friendly-SAM (F-SAM) REPLACES vanilla SAM

**arXiv:2403.12350 + revisited arXiv:2603.10048 (March 2026).** Decomposes SAM perturbation into "full gradient" + "stochastic noise"; only ascends along full-gradient component. Better generalization at same ρ.

Why for us: 16K samples = high gradient noise. Vanilla SAM amplifies noise into perturbation step. F-SAM filters it.

#### 3. Muon optimizer for 2D weight matrices (A/B candidate)

**Keller Jordan blog 2024 + arXiv:2502.16982 (Moonshot) + arXiv:2603.17970 MUD (March 2026).** Newton-Schulz orthogonalization on 2D gradients. AdamW handles embeddings/heads/norms. **DeepSeek V4 (April 2026) + Kimi-2 ship Muon in production.**

For 24-60M scale: gains modest (most documented wins at 300M+), but Newton-Schulz is matmul-only → MPS-compatible, no CUDA kernel needed. ~52% of AdamW's FLOPs to reach same loss at 1.5B GPT-2 XL.

A/B test: Muon vs Schedule-Free AdamW. If Muon wins by ≥0.5pp val accuracy, adopt.

Repo: https://github.com/KellerJordan/Muon

#### 4. MTS-JEPA SSL pretrain (Phase 2 candidate)

**arXiv:2602.04643 (Feb 2026), Multi-variate Time Series JEPA.** Predict masked target EMBEDDINGS (not raw values) from context embeddings. Avoids MAE pixel-level reconstruction wastage AND contrastive collapse problem.

For us: V3 plan uses MAE (reconstruct masked OHLCV). JEPA produces smoother representations that transfer better to small-data classification. Larger code change (~1-2 days), highest-ceiling change in the list.

**Plan:** Pilot MTS-JEPA after Schedule-Free + F-SAM are stable. Compare val Sharpe vs MAE pretrain.

#### 5. Cross-Asset Transfer Learning (cheap experiment)

**Springer 2025 cross-asset transfer transformer.** Pretrain on broad-market index (SPY 30min, far more data), fine-tune on GLD with LLRD. Maps onto our existing LLRD pipeline. Documented gains over single-asset baselines.

Add as bonus experiment after baseline ships.

#### 6. Conformal Prediction for sizing (deployment-side, see doc 07)

Wrap softmax probabilities with split-conformal calibration on a held-out slice. Produces calibrated prediction intervals. Use **interval width** to size positions. See doc 07 for full implementation.

#### 7. Multi-Dimensional Sentiment (per arXiv:2603.11408, March 2026)

Paper found on WTI crude (gold-adjacent commodity): **intensity + uncertainty matter MORE than polarity.** Currently doc 02 conflates these — embed news, project, done.

Update: extract THREE sentiment scores per news item using LLM prompts:
- Polarity: -1 (bearish) to +1 (bullish)
- Intensity: 0 (mild) to 1 (urgent/breaking)
- Uncertainty: 0 (certain) to 1 (speculative/conditional)

Add as 3 cheap features in `geo_features` table (arXiv:2603.11408 found these dominate via SHAP).

### Updated V4 Training Stack Summary

```
Optimizer:      Schedule-Free AdamW (β=0.9, β2=0.95, wd=0.1, lr=1e-4, warmup_steps=300)
                [A/B against Muon-for-2D + AdamW-for-rest]
LR schedule:    NONE (Schedule-Free handles this) OR WSD if SF underperforms
Sharpness:      Friendly-SAM ρ=0.05 (replace vanilla SAM)
EMA:            decay=0.999 (V3 — kept)
Regularization: dropout 0.2 + stoch depth 0.15 + label smoothing 0.1 (V3 — kept)
Augmentation:   jittering σ=0.02 + magnitude warping (V3 — kept)
Manifold Mixup: at hidden states α=0.2 (V3 — kept)
Modality dropout: 15% on news embeddings (V3 — kept)
SSL pretrain:   MAE on masked bars (V3 default)
                [A/B with MTS-JEPA — Phase 2, higher ceiling]
Linear-probe → LLRD fine-tune: V3 — kept
Cross-asset transfer: SPY→GLD as bonus experiment after baseline
Conformal calibration: split-CP on val fold for sizing (deploy-side, see doc 07)
Loss:           3-class CE + label smoothing 0.1 (NEVER MSE on returns — forecast-collapse rule from doc 03)
Mixed precision: FP32 weights (V3 — kept; FP8 H100-only, unstable for small models)
```

### What V4 Skips (saved investigation)

- Sophia optimizer — at <150M scale ties Signum, barely beats AdamW
- FP8 mixed precision — H100-only, unstable for small models per NVIDIA + Axolotl 2025-2026 docs
- Knowledge distillation from foundation model — no useful teacher available yet
- Pure Mamba / xLSTM as backbone — covered in doc 03 (skip at our scale)

## SMALL-DATA TRAINING STACK V3 (kept for reference)

For 16K labeled samples, regularization beats capacity. Stack is non-negotiable. Each piece has documented ROI.

### Stage 1 — SSL Pretraining (MUST, single biggest win)

Masked autoencoder on same 5y unlabeled bars BEFORE classification fine-tune. Documented 25-50% gain on small-data classification (Financial Fine-tuning paper, SSL4TS literature).

```python
# Pretrain head: reconstruct masked bars
class MAEHead(nn.Module):
    def __init__(self, D, target_dim):
        super().__init__()
        self.proj = nn.Linear(D, target_dim, bias=False)
    def forward(self, hidden):
        return self.proj(hidden)

def mask_bars(x, mask_ratio=0.20):
    """Mask 20% of input bars for reconstruction objective."""
    B, T, F = x.shape
    n_mask = int(T * mask_ratio)
    mask_idx = torch.randperm(T)[:n_mask]
    x_masked = x.clone()
    x_masked[:, mask_idx, :] = 0  # or learned [MASK] token
    return x_masked, mask_idx, x[:, mask_idx, :]  # target = original at mask positions
```

Train MAE for ~10 epochs. Save encoder weights. Don't train classification head yet.

### Stage 2 — Linear-probe (5-10 epochs)

Freeze encoder. Train ONLY classification head on direction labels. Sanity check that pretrained features are useful.

```python
for p in model.encoder.parameters():
    p.requires_grad = False
# train head only, fast convergence
```

### Stage 3 — Full fine-tune with LLRD (layer-wise LR decay 0.85)

Unfreeze encoder. Lower layers get smaller LR (preserve general features), upper layers + head get full LR.

```python
def llrd_param_groups(model, base_lr=1e-4, decay=0.85):
    groups = []
    n_layers = len(model.blocks)
    for i, block in enumerate(model.blocks):
        lr = base_lr * (decay ** (n_layers - i - 1))
        groups.append({'params': block.parameters(), 'lr': lr})
    groups.append({'params': model.head.parameters(), 'lr': base_lr})
    return groups
```

### SAM Optimizer (MUST — 2× compute, big gain)

SAMformer (ICML 2024) showed SAM is the unlock for transformers on time series — **14.33% MSE improvement over TSMixer**, 4× fewer parameters.

```python
# pip install sam-pytorch  OR git submodule davda54/sam
from sam import SAM

base_optim = torch.optim.AdamW(model.parameters(), lr=1e-4, betas=(0.9, 0.95), weight_decay=0.1)
optim = SAM(model.parameters(), base_optim, rho=0.05, adaptive=False)

# In training loop
def closure():
    loss = F.cross_entropy(model(x), y)
    loss.backward()
    return loss

# First step
loss = F.cross_entropy(model(x), y)
loss.backward()
optim.first_step(zero_grad=True)
# Second step
F.cross_entropy(model(x), y).backward()
optim.second_step(zero_grad=True)
```

### EMA Weights (MUST — cheap, robust to label noise)

```python
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn

ema_model = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(0.999))

# In training loop after each batch
ema_model.update_parameters(model)

# Deploy ema_model, NOT model
```

### Stochastic Depth + Label Smoothing (MUST — free regularization)

- Stochastic depth p=0.15 (in Block.forward)
- Label smoothing ε=0.1 in F.cross_entropy

### Manifold Mixup (recommended — at hidden states, NOT raw input)

```python
def manifold_mixup(hidden, labels, alpha=0.2):
    """Mixup at intermediate hidden state, preserves temporal structure."""
    lam = np.random.beta(alpha, alpha)
    perm = torch.randperm(hidden.size(0))
    mixed = lam * hidden + (1 - lam) * hidden[perm]
    return mixed, labels, labels[perm], lam
```

Apply ONLY at hidden states (between blocks), NEVER on raw input (destroys temporal structure).

### Augmentation: Jittering + Magnitude Warping (MUST)

```python
def augment(x, jitter_sigma=0.02, mag_warp_sigma=0.02):
    if np.random.rand() < 0.5:
        x = x + torch.randn_like(x) * jitter_sigma  # jittering
    if np.random.rand() < 0.5:
        # Magnitude warping (smooth random scaling)
        knot_count = 4
        knots = torch.randn(knot_count) * mag_warp_sigma + 1.0
        warp_curve = F.interpolate(knots.view(1, 1, -1), size=x.size(-2), mode='linear').view(-1, 1)
        x = x * warp_curve
    return x
```

NEVER raw-input mixup — destroys temporal structure.

### TTA at validation (K=5)

```python
def tta_predict(model, x, K=5):
    preds = []
    for _ in range(K):
        x_aug = augment(x)
        preds.append(F.softmax(model(x_aug), dim=-1))
    return torch.stack(preds).mean(dim=0)
```

Use for validation and offline backtest. Skip for live (latency).

### Modality Dropout (15% on news embeddings)

```python
def modality_dropout(news_mask, p=0.15, training=True):
    """Randomly drop news sources during training to force robustness."""
    if not training:
        return news_mask
    drop = torch.rand_like(news_mask.float()) < p
    return news_mask & ~drop
```

### Implementation Order (by ROI)

1. EMA + stochastic depth + label smoothing + jittering — cheap, immediate signal (~1 day)
2. SSL pretrain → linear-probe → LLRD fine-tune (the big one, multi-day)
3. SAM optimizer (2× compute, validate gain)
4. Manifold Mixup + TTA (polish)

Expected total stack lift: +0.3-0.6 Sharpe at val (baseline ~0.5-1.0). Compounding is where the gain lives, not any single trick.

## CRITICAL CORRECTIONS (Nia round 2 — kept)

- ❌ "~6-8 folds at 5y" → ✅ **4 folds at 5y** (math: floor((60-48)/3) = 4). To get 6 folds, shorten train window 3y→2.5y.
- ❌ Peak LR `3e-4` default → ✅ **`1e-4` default** (3e-4 is too aggressive for 24M model on 16K samples). Sweep up to 3e-4 only after LR-range test on fold 0.
- ❌ Patience 3 → ✅ **patience 5** with `min_delta=1e-4` (small val sets are noisy; 3 fires too early)
- ❌ Manual `p.data.to(torch.bfloat16)` weight cast → ✅ **autocast forward only**, keep weights FP32. PyTorch docs explicitly warn against the manual cast pattern with autocast.
- ❌ Only `torch.manual_seed(seed)` → ✅ **set_seed helper** that seeds `torch + torch.mps + numpy + random`. MPS dropout still non-deterministic — document as known limitation.
- ❌ Vague wandb auth → ✅ read `WANDB_API_KEY` env var with boot-time validation, fail loudly if missing, support `mode='offline'` for airgapped runs
- ❌ TODO "try gradient accumulation" → ✅ **delete** — arXiv:2507.07101 says gradient accumulation isn't justified for transformer LM training unless multi-device
- ⚠️ MPS bf16 transformer training has known correctness issues (PyTorch #139386, #97236, #84516). Phase 2 only after fold-0 numerical equivalence test (FP32 vs bf16 logits, abs diff < 1e-2)
**Owner:** samsiavoshian
**Implementation effort:** 1 day after model + features land

## Locked Specs

| Component | Choice | Why |
|-----------|--------|-----|
| Optimizer | AdamW | Per-param adaptive LR, decoupled weight decay, standard for transformers |
| betas | (0.9, 0.95) | Llama / GPT convention; β2=0.95 makes second-moment more responsive |
| Peak LR | **1e-4** (default) | 3e-4 too aggressive for 24M model on 16K samples; sweep 1e-4 → 3e-4 after LR-range test |
| Weight decay | 0.1 | Standard transformer regularization |
| Weight decay groups | bias + LayerNorm + pos_embed = no decay | Standard practice |
| LR schedule | Cosine with linear warmup | Smoother than step decay, empirically better |
| Warmup | 10% of total steps | Standard |
| Min LR | 0.1 × peak | Decay floor |
| Loss | Class-weighted cross-entropy | Handles ~30/40/30 class imbalance |
| Class weights | Inverse-frequency (`N / (3 * N_i)`) | Equivalent to sklearn 'balanced' |
| Label smoothing | 0.0 default, 0.1 if overconfident | Try if val accuracy plateaus |
| Gradient clipping | max_norm 1.0 | Prevents gradient explosion on outlier batches |
| Dropout | 0.1 (in attention + MLP) | Set in model |
| Mixed precision | FP32 phase 1, bfloat16 phase 2 if needed | MPS support partial |
| Batch size | 32 train, 64 val/test | Fits 16GB Mac mini comfortably |
| Epochs per fold | 20 max with early stopping | Usually converges in 5-12 |
| Early stopping | **Patience 5, min_delta=1e-4** | Small val sets noisy; 3 fires too early |
| Random seed | 42 default, multi-seed in week 4+ | Single-seed for week 1, 5-seed for confidence |

## Walk-Forward Cross-Validation

Anchored walk-forward with 1-week embargo:
- Train: 3 years (anchored from data start)
- Val: 6 months (after embargo)
- Test: 6 months (after embargo)
- Step: 3 months (slide forward)

For 5y of data: **4 folds exactly** (60mo - 48mo overhead) / 3mo step = 4. To get 6 folds, shorten train window from 3y to 2.5y.

```python
@dataclass
class Fold:
    fold_idx: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    val_start: pd.Timestamp
    val_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


def walk_forward_splits(
    timestamps: pd.Series,
    train_years: float = 3.0,
    val_months: float = 6.0,
    test_months: float = 6.0,
    step_months: float = 3.0,
    embargo_days: int = 7,
) -> Iterator[Fold]:
    """Anchored walk-forward. Train always starts at data_start, extends each step."""
    embargo = pd.Timedelta(days=embargo_days)
    train_dur = pd.Timedelta(days=train_years * 365)
    val_dur = pd.Timedelta(days=val_months * 30)
    test_dur = pd.Timedelta(days=test_months * 30)
    step = pd.Timedelta(days=step_months * 30)
    
    data_start = timestamps.min()
    data_end = timestamps.max()
    
    fold_idx = 0
    train_end = data_start + train_dur
    
    while True:
        val_start = train_end + embargo
        val_end = val_start + val_dur
        test_start = val_end + embargo
        test_end = test_start + test_dur
        
        if test_end > data_end:
            break
        
        yield Fold(fold_idx, data_start, train_end, val_start, val_end, test_start, test_end)
        fold_idx += 1
        train_end += step
```

## Optimizer Configuration

```python
def configure_optimizer(model: nn.Module, lr: float = 3e-4, weight_decay: float = 0.1) -> torch.optim.AdamW:
    """AdamW with no decay on biases / LayerNorm / pos_embed."""
    decay_params = []
    no_decay_params = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.endswith('.bias') or 'norm' in name.lower() or 'pos_embed' in name:
            no_decay_params.append(p)
        else:
            decay_params.append(p)
    
    return torch.optim.AdamW(
        [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': no_decay_params, 'weight_decay': 0.0},
        ],
        lr=lr, betas=(0.9, 0.95), eps=1e-8,
    )
```

## LR Schedule

```python
import math

def cosine_lr_schedule(optimizer, warmup_steps: int, total_steps: int, min_lr_ratio: float = 0.1):
    """Linear warmup, then cosine decay to min_lr_ratio * peak."""
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        cosine = 0.5 * (1 + math.cos(math.pi * progress))
        return min_lr_ratio + (1 - min_lr_ratio) * cosine
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
```

## Class Weights

```python
def compute_class_weights(labels: torch.Tensor, num_classes: int = 3) -> torch.Tensor:
    """weight_i = N_total / (num_classes * N_i). Equivalent to sklearn 'balanced'."""
    counts = torch.bincount(labels, minlength=num_classes).float()
    N = counts.sum()
    return N / (num_classes * counts)
```

## Checkpoint Tracker (early stopping)

```python
class CheckpointTracker:
    def __init__(self, output_dir: str, patience: int = 3):
        self.output_dir = output_dir
        self.patience = patience
        self.best_val_loss = float('inf')
        self.epochs_no_improve = 0
        self.best_path = None
    
    def step(self, model, optimizer, val_loss: float, epoch: int) -> bool:
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.epochs_no_improve = 0
            path = f"{self.output_dir}/checkpoint_epoch{epoch}_val{val_loss:.4f}.pt"
            torch.save({
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'epoch': epoch,
                'val_loss': val_loss,
            }, path)
            self.best_path = path
            return False
        else:
            self.epochs_no_improve += 1
            return self.epochs_no_improve >= self.patience
```

## Full Training Loop (one fold)

```python
import torch
import torch.nn.functional as F
import wandb

def train_one_fold(
    model: nn.Module,
    train_loader, val_loader, test_loader,
    device: str = 'mps',
    num_epochs: int = 20,
    peak_lr: float = 3e-4,
    weight_decay: float = 0.1,
    warmup_pct: float = 0.1,
    grad_clip: float = 1.0,
    label_smoothing: float = 0.0,
    patience: int = 3,
    wandb_run=None,
) -> dict:
    model = model.to(device)
    
    # Class weights from training set only (no val/test peeking)
    train_labels = torch.cat([batch[2] for batch in train_loader])
    class_weights = compute_class_weights(train_labels).to(device)
    
    optimizer = configure_optimizer(model, peak_lr, weight_decay)
    total_steps = len(train_loader) * num_epochs
    warmup_steps = int(total_steps * warmup_pct)
    scheduler = cosine_lr_schedule(optimizer, warmup_steps, total_steps)
    tracker = CheckpointTracker('checkpoints/', patience=patience)
    
    for epoch in range(num_epochs):
        model.requires_grad_(True)
        
        epoch_train_loss = 0
        epoch_train_correct = 0
        epoch_train_total = 0
        
        for batch_idx, (numeric, news_raw, labels) in enumerate(train_loader):
            numeric = numeric.to(device)
            news_raw = news_raw.to(device)
            labels = labels.to(device)
            
            logits = model(numeric, news_raw)
            loss = F.cross_entropy(logits, labels, weight=class_weights, label_smoothing=label_smoothing)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            scheduler.step()
            
            with torch.no_grad():
                preds = logits.argmax(dim=-1)
                epoch_train_loss += loss.item() * len(labels)
                epoch_train_correct += (preds == labels).sum().item()
                epoch_train_total += len(labels)
            
            if wandb_run and batch_idx % 50 == 0:
                wandb_run.log({
                    'train/loss_step': loss.item(),
                    'train/lr': scheduler.get_last_lr()[0],
                    'step': epoch * len(train_loader) + batch_idx,
                })
        
        train_loss = epoch_train_loss / epoch_train_total
        train_acc = epoch_train_correct / epoch_train_total
        
        # Validate
        with torch.no_grad():
            val_loss = 0
            val_correct = 0
            val_total = 0
            for numeric, news_raw, labels in val_loader:
                numeric, news_raw, labels = numeric.to(device), news_raw.to(device), labels.to(device)
                logits = model(numeric, news_raw)
                loss = F.cross_entropy(logits, labels, weight=class_weights)
                val_loss += loss.item() * len(labels)
                preds = logits.argmax(dim=-1)
                val_correct += (preds == labels).sum().item()
                val_total += len(labels)
            val_loss /= val_total
            val_acc = val_correct / val_total
        
        if wandb_run:
            wandb_run.log({
                'train/loss_epoch': train_loss, 'train/acc_epoch': train_acc,
                'val/loss': val_loss, 'val/acc': val_acc,
                'epoch': epoch,
            })
        
        print(f"Epoch {epoch}: train_loss={train_loss:.4f} train_acc={train_acc:.3f} val_loss={val_loss:.4f} val_acc={val_acc:.3f}")
        
        if tracker.step(model, optimizer, val_loss, epoch):
            print(f"Early stopping at epoch {epoch}")
            break
    
    # Test on best checkpoint
    best_state = torch.load(tracker.best_path)
    model.load_state_dict(best_state['model_state'])
    
    with torch.no_grad():
        test_loss = 0
        test_correct = 0
        test_total = 0
        all_logits = []
        for numeric, news_raw, labels in test_loader:
            numeric, news_raw, labels = numeric.to(device), news_raw.to(device), labels.to(device)
            logits = model(numeric, news_raw)
            loss = F.cross_entropy(logits, labels, weight=class_weights)
            test_loss += loss.item() * len(labels)
            preds = logits.argmax(dim=-1)
            test_correct += (preds == labels).sum().item()
            test_total += len(labels)
            all_logits.append(logits.cpu())
        test_loss /= test_total
        test_acc = test_correct / test_total
    
    return {
        'best_val_loss': tracker.best_val_loss,
        'test_loss': test_loss,
        'test_acc': test_acc,
        'best_checkpoint': tracker.best_path,
        'test_logits': torch.cat(all_logits),
    }
```

## Walk-Forward Driver

```python
def run_walk_forward(snapshot_path, embeddings_path, anchors_path, num_seeds: int = 1):
    features = build_feature_table(snapshot_path, embeddings_path, anchors_path)
    
    fold_results = []
    for fold in walk_forward_splits(features.timestamp):
        for seed in range(num_seeds):
            torch.manual_seed(seed)
            
            train_data = features[fold.train_start:fold.train_end]
            val_data = features[fold.val_start:fold.val_end]
            test_data = features[fold.test_start:fold.test_end]
            
            train_loader = make_loader(train_data, batch_size=32, shuffle=True)
            val_loader = make_loader(val_data, batch_size=64, shuffle=False)
            test_loader = make_loader(test_data, batch_size=64, shuffle=False)
            
            model = nanoGLD()
            wandb_run = wandb.init(project='nanogld', name=f'fold{fold.fold_idx}_seed{seed}')
            
            result = train_one_fold(model, train_loader, val_loader, test_loader, wandb_run=wandb_run)
            result['fold'] = fold.fold_idx
            result['seed'] = seed
            fold_results.append(result)
            
            wandb_run.finish()
    
    return fold_results
```

## Mixed Precision Strategy

**Phase 1 (week 1, default):** all FP32 weights + activations. Slower but works. ~1.5GB model memory for 24M params, 5-10 hrs training time per fold.

**Phase 2 (if memory or speed limits hit):** bfloat16 weights + activations on MPS. ~50% memory + ~30% speedup. Keep optimizer states FP32.

```python
# Phase 2 (only if needed)
model = nanoGLD().to('mps')
for p in model.parameters():
    p.data = p.data.to(torch.bfloat16)
optimizer = configure_optimizer(model, ...)  # AdamW state stays FP32
```

If MPS-specific bugs surface (silent CPU fallback for some ops), profile with `torch.profiler` and revert affected layers to FP32.

## Implementation Day Plan (Day 7 of week 1)

| Hour | Task |
|------|------|
| 1 | Walk-forward splitter + tests |
| 2 | PyTorch Dataset + DataLoader |
| 3 | Hook up training loop with wandb |
| 4 | Run 1 fold smoke test (subsample data, 2 epochs, verify shapes + loss decrease) |
| 5-7 | Run 1 full fold |
| 8-10 | Inspect val curves, tune LR / batch / dropout if needed |
| 11-14 | Launch all folds (overnight) |
| Day 8 AM | Review fold results, save model artifacts |

## Open Questions / TODOs

- [ ] If single-seed OOS results are unstable across folds, add 5-seed averaging for reproducibility
- [ ] Decide whether to publish wandb workspace publicly (yes for X thread, double-check no secrets in run names)
- [ ] Profile MPS forward pass — does flash-attention kernel (if available) help?
- [ ] If batch 32 is too small for stable gradients on imbalanced data, try batch 64 or 128 with gradient accumulation
- [ ] Consider gradient accumulation only if memory limits force batch < 32
