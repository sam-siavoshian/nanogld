# 07 — Sizing + Exits + Risk

## YOU ARE THE POSITION-SIZING + EXITS + RISK AGENT

You own how positions are sized AND how they are exited:
1. **Stage 2 sizing** — vol-target × Kelly-lite × calibrated conformal probability shrinkage.
2. **Per-trade stop-loss** — vol-scaled SL with client-side polling on Alpaca.
3. **Profit-take logic** — partial profit ladders + trailing stop.
4. **Drawdown circuit-breaker** — hard re-entry rules.

You start when doc 05 (backtest) hands off a backtest engine + a calibrated model from doc 05. You hand off to doc 06 (live) the live-deployable position-management module.

**Read 00-OVERVIEW.md FIRST.** Project context. Read doc 05's calibration interface (temperature + APS quantiles) and doc 05's backtest engine before coding.

### Execution Mode (short — full rules in 00-OVERVIEW.md)

**This doc IS the plan. Do not rewrite it.** Plan to *execute*. If a claim is wrong, AskUserQuestion. Silent scope drift = fired.

- **Research with Nia** before guessing.
- **Execution skills only:** /investigate, /review, /qa, /cso, /benchmark, /ship.
- **Loop:** read doc → Nia for unknowns → code → /review → /qa → /cso (live $) → /ship.
- **Escalate after 3 failed attempts.** AskUserQuestion.

### What You Build (top-level files)

```
src/nanogld/sizing/        # part 1 — Kelly-lite × vol-target × conformal shrinkage
src/nanogld/exits/         # part 2 — SL + profit-take + drawdown circuit-breaker
```

### Acceptance Criteria

You're done when:

1. ✅ Stage 2 sizer produces position size from calibrated probability + vol target (10% annual)
2. ✅ Conformal shrinkage uses APS p-values (NOT discrete set size — V4 correction)
3. ✅ Per-trade SL fires on Alpaca via client-side polling (Alpaca server-side OCO not reliable on free tier)
4. ✅ Profit-take ladders trigger correctly under backtest + paper-trade smoke
5. ✅ Drawdown circuit-breaker fires at -8% paper / -5% live with explicit re-entry rule
6. ✅ Sizing GATE: Stage 2 must beat Stage 1 by ≥0.2 Sharpe OOS (else ship Stage 1 fixed-share)
7. ✅ Hand-off artifact: position-mgmt module that doc 06 (live) imports

This doc is **2 parts** combined for one agent — read all of it before starting.

---


# Part 1 — Sizing (Stage 2)

_(was doc 05-SIZING-AND-EXITS.md before V5 merge — content unchanged)_

# 07 — Sizing Stage 2 (Vol-Target × Kelly-Lite)

## YOU ARE THE QUANT SIZING AGENT

You own the position sizing layer. You take model logits from doc 05 and produce the desired position multiplier (signed, clipped). You also implement the conformal calibration layer (V1 addition).

**Read 00-OVERVIEW.md FIRST.**
**Read 06-BACKTEST.md** for engine integration (Stage 2 plugs into backtest).
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
src/nanogld/sizing/
├── __init__.py
├── stage1.py                      # Fixed sizing: 1 share when argmax != flat
├── stage2.py                      # Vol-target × Kelly-lite × conformal
├── kelly_lite.py                  # 3-class Kelly-lite formula (heuristic, documented)
├── vol_target.py                  # 20-day rolling realized vol → multiplier
├── conformal.py                   # ConformalSizer (Wright 2026)
├── temperature_scaling.py         # Pre-conformal temperature calibration on val fold
├── drawdown_breaker.py            # Stateful DrawdownCircuitBreaker with re-entry rule
├── ablation.py                    # Stage 1 / 1.5a / 1.5b / Stage 2 ablation runner
└── cli.py                         # `python -m nanogld.sizing ablate --logits <path>`

tests/
├── test_stage2_formula.py         # Worked examples from spec
├── test_drawdown_signs.py         # Negative DD numbers, comparisons consistent
├── test_drawdown_recovery.py      # Halt at -15% recovers when equity climbs back
├── test_conformal_coverage.py     # 90% coverage achieved on val fold
└── test_ablation_isolation.py     # 1.5a vs 1.5b vs Stage 2 isolate components
```

### Files You DO NOT Touch

- Anything outside `src/nanogld/sizing/`
- The backtest engine (doc 06 owns) — you call its interface
- Other doc files

### Stable Interface You Publish

```python
from nanogld.sizing.stage2 import stage2_sizing
from nanogld.sizing.conformal import ConformalSizer
from nanogld.sizing.drawdown_breaker import DrawdownCircuitBreaker

# doc 06 (backtest) calls this per-bar:
size = stage2_sizing(
    probs: np.ndarray,                  # (3,) — softmax probs from model
    realized_vol_20d: float,            # annualized
    conformal_factor: float = 1.0,      # 1.0 high-conf, 0.5 medium, 0.0 low (from ConformalSizer)
    target_vol: float = 0.10,
    confidence_scale: float = 3.0,
    kelly_fraction: float = 0.5,
    position_limit: float = 1.0,
) -> float

# doc 08 (live) imports same interface
```

### Acceptance Criteria

1. ✅ `pytest tests/test_stage2_formula.py` passes (worked examples from doc match)
2. ✅ ConformalSizer achieves ≥90% coverage on held-out val fold
3. ✅ DrawdownCircuitBreaker re-entry test passes (halt at -15%, recover to -10%, exit halt mode)
4. ✅ Ablation table compares Stage 1 / 1.5a / 1.5b / Stage 2 with bootstrap CIs
5. ✅ Stage 2 must beat Stage 1 by ≥0.2 Sharpe OOS to ship Stage 2 (gate)
6. ✅ If neither 1.5a (Kelly only) nor 1.5b (vol only) wins solo, drop the losing component (simplicity wins)

### Spawn Nia Agents When You Need To

- **arch.bootstrap.StationaryBootstrap** for confidence intervals — verify current API
- **scipy.stats.LBFGS** for temperature scaling — alternative is `torch.optim.LBFGS` (simpler)
- **Conformal prediction TS extensions** — Wright 2026 (arXiv:2601.07852) for utility-weighted; pure Kato 2024 (arXiv:2410.16333) for portfolio variant. Pick the one that fits your data shape.
- **Drawdown circuit-breaker re-entry rules** — survey papers if our 5-day fallback rule isn't documented elsewhere

### V1 Critical Decisions (DO NOT REVERT)

1. **bars_per_year = 3276** (propagated from doc 06). Annualization MUST be `sqrt(3276)`.
2. **Drawdown sign consistency** — store as NEGATIVE number (-0.15 = 15% DD). Use `<=` comparisons throughout.
3. **Recovery rule** — exit halt mode when equity recovers above -10% line OR after 5 trading days flat (whichever first).
4. **Probability calibration BEFORE Kelly-lite** — Guo et al. show NN's are overconfident, Kelly-lite amplifies it. Temperature scaling on val fold first.
5. **Conformal Prediction REPLACES "just temperature scaling"** for V4 — Wright 2026 shows 30% lower decision loss.
6. **Half Kelly (0.5) is the default**. Quarter Kelly (0.25) is fallback if drawdowns alarm us.

### Hand-off Protocol

1. Update STATUS.md with: ablation table results, conformal coverage achieved, drawdown CB tested
2. doc 06 imports your `stage2_sizing` for backtest comparisons
3. doc 08 imports same function for live trading

Now read the implementation specifics.

---

# 07 — Sizing Stage 2 (Vol-Target × Kelly-Lite)

**Status:** ✅ Complete, implementation-ready, Nia-verified
**Last verified:** 2026-04-30

## CRITICAL CORRECTIONS (Nia verification)

- ❌ `sqrt(3276)` annualization → ✅ **`sqrt(3276)`** (NYSE RTH only, NOT 24/7 calendar). Same bug as doc 06. Causes strategy to **under-bet by ~57%** if uncorrected (vol_mult comes out 2.31× too small).
- ❌ Drawdown sign inconsistency (one block uses `<= -0.15`, another uses `> 0.15`) → ✅ canonicalize: drawdown stored as **negative number** (e.g. `-0.15` = 15% drawdown). Use `<=` comparisons throughout.
- ❌ Recovery lockout at -15% halt → ✅ add explicit **re-entry rule**: resume sizing once equity recovers above the -10% line OR after 5 trading days at flat (whichever first)
- ❌ MISSING probability calibration → ✅ **add temperature scaling** before Kelly-lite. Modern NNs systematically over-confident (Guo et al. arXiv:1706.04599). Without calibration, Kelly-lite amplifies bad probabilities. Single most important missing piece.
- ⚠️ 20-day rolling vol → consider **EWMA λ=0.94 (RiskMetrics)** as default (more responsive at 30min granularity)
- ⚠️ Half Kelly (0.5) is conventional but with ML-derived miscalibrated probs, **0.25 may be safer for v1**
- ⚠️ Kelly-lite formula `(max_prob - 0.33) × scale × kelly_fraction` is **heuristic, not published Kelly extension**. Label as such in code docstring. Real multinomial Kelly = Smoczynski-Tomkins 2010, requires explicit return distributions per class.
**Owner:** samsiavoshian
**Implementation effort:** 0.5 day

## The Formula

```python
size = (max_prob - 0.33) * confidence_scale * (target_annual_vol / realized_vol_annualized)
size = clip(size, -position_limit, +position_limit)
```

Components:

1. **Confidence weight (Kelly-lite):** `(max_prob - 0.33)` — uniform random would give 0.33 for a 3-class problem; anything above 0.33 is real signal. Multiply by `confidence_scale` (default 3, i.e. when max_prob = 0.66, signal weight = 1).

2. **Vol-target multiplier:** `target_annual_vol / realized_vol_annualized` — scales position so that *strategy* volatility is targeted, not raw position size. If gold realizes 15% vol and we target 10%, we hold 0.66× the size we'd otherwise.

3. **Sign:** `+1` if argmax is UP, `-1` if argmax is DOWN, 0 if FLAT.

4. **Clip:** position limit prevents blowups during extreme low-vol windows where the multiplier could go huge.

## Why This Beats Fixed Sizing (Per Literature)

- **Vol-targeting alone** captures most of the risk-adjusted return improvement (Zhang, Zohren, Roberts arXiv:1911.10107). Free lunch from 1990s practitioner literature.
- **Kelly-lite confidence scaling** converts well-calibrated probabilities into proportional sizing — bets more when more confident.
- Combined: the strategy bets bigger when (a) the model is confident AND (b) the asset isn't already volatile. Both are statistically motivated.

## Hyperparameters (defaults to start, tune later)

```
target_annual_vol     = 0.10    (10% annualized strategy vol)
confidence_scale      = 3       (max signal weight at max_prob ≈ 0.66)
position_limit        = 1.0     (1× = full position)
realized_vol_lookback = 20 days (480 30min bars)
```

## Now Designed (post-deep-dive)

All resolved:
- ✅ Drawdown circuit-breaker: halve at -5% DD, quarter at -10%, halt at -15%
- ✅ Exit rules: position smoothing with enter/exit thresholds (deferred unless turnover > 10× annualized)
- ✅ Position smoothing: hysteresis (enter at confidence > 0.40, exit at < 0.35) — deferred until needed
- ✅ Fractional Kelly: 0.5× (half Kelly), the standard. Reduce to 0.25 if drawdowns spook
- ✅ Kelly correction for costs: implicit via the `confidence_scale × kelly_fraction` choice; revisit if cost-sensitivity hurts
- ✅ Reset rules: drawdown circuit-breaker covers it

## Final Stage 2 Code (with all components)

```python
import numpy as np

def stage2_sizing(
    probs: np.ndarray,                 # shape (3,) — [P_down, P_flat, P_up]
    realized_vol_20d: float,           # annualized realized vol of GLD
    target_vol: float = 0.10,          # 10% annualized strategy vol target
    confidence_scale: float = 3.0,     # maps (max_prob - 0.33) to position
    kelly_fraction: float = 0.5,       # half Kelly (safe default)
    position_limit: float = 1.0,
) -> float:
    """Returns desired position multiplier in [-position_limit, +position_limit]."""
    direction = np.argmax(probs)
    if direction == 1:  # FLAT
        return 0.0
    
    sign = 1.0 if direction == 2 else -1.0
    
    confidence = max(probs) - 0.33
    if confidence <= 0:
        return 0.0  # no edge, don't bet
    
    raw_size = confidence * confidence_scale * kelly_fraction
    vol_mult = target_vol / max(realized_vol_20d, 1e-3)
    size = sign * raw_size * vol_mult
    
    return float(np.clip(size, -position_limit, position_limit))


class DrawdownCircuitBreaker:
    """
    Drawdown control with explicit re-entry to avoid recovery lockout.
    DD stored as NEGATIVE number (e.g. -0.15 = 15% drawdown).
    
    States:
    - normal: full sizing
    - half: 50% sizing (DD <= -5%)
    - quarter: 25% sizing (DD <= -10%)
    - halted: 0% sizing (DD <= -15%) — until equity recovers above -10% line OR 5 trading days flat
    """
    def __init__(self, halt_threshold: float = -0.15, recovery_threshold: float = -0.10, max_halt_bars: int = 65):
        self.halt_threshold = halt_threshold       # negative
        self.recovery_threshold = recovery_threshold
        self.max_halt_bars = max_halt_bars         # 5 trading days × 13 bars
        self.halted = False
        self.bars_since_halt = 0
    
    def adjust(self, target_size: float, current_drawdown_pct: float) -> float:
        """current_drawdown_pct: negative number (peak - current) / peak."""
        # Re-entry logic
        if self.halted:
            self.bars_since_halt += 1
            if current_drawdown_pct >= self.recovery_threshold or self.bars_since_halt >= self.max_halt_bars:
                self.halted = False
                self.bars_since_halt = 0
            else:
                return 0.0
        
        # Trigger halt
        if current_drawdown_pct <= self.halt_threshold:
            self.halted = True
            self.bars_since_halt = 0
            return 0.0
        
        # Tiered de-risking
        if current_drawdown_pct <= -0.10:
            return target_size * 0.25
        elif current_drawdown_pct <= -0.05:
            return target_size * 0.5
        return target_size


# Backward-compat function (stateless, no recovery logic — use class above for production)
def adjust_for_drawdown(target_size: float, current_drawdown_pct: float) -> float:
    """Stateless helper. Halve at -5%, quarter at -10%, halt at -15%. NO recovery — use class for that."""
    if current_drawdown_pct <= -0.15:
        return 0.0
    elif current_drawdown_pct <= -0.10:
        return target_size * 0.25
    elif current_drawdown_pct <= -0.05:
        return target_size * 0.5
    return target_size


def smoothed_sizing(probs, current_position, target_size, enter_threshold=0.4, exit_threshold=0.35):
    """Hysteresis: stronger threshold to enter, weaker to exit. Reduces turnover."""
    confidence = max(probs) - 0.33
    
    if abs(current_position) < 1e-6:  # currently flat
        if confidence < enter_threshold:
            return 0.0
        return target_size
    else:  # currently in a position
        if confidence < exit_threshold:
            return 0.0
        return target_size
```

## V1 — Conformal Prediction for Sizing (May 2026, training-agent finding)

Beyond temperature scaling: wrap softmax with **split-conformal calibration** to produce CALIBRATED PREDICTION INTERVALS. Use INTERVAL WIDTH to size positions, not just point probabilities.

**Citations:**
- Kato 2024, "Conformal Predictive Portfolio Selection" arXiv:2410.16333
- Wright 2026, "Utility-Weighted Forecasting and Calibration" arXiv:2601.07852 (Jan 2026): **30% lower realized decision loss vs uncalibrated baseline.** Calibration as INPUT to constrained decision, not standalone metric.
- arXiv:2511.13608 (Nov 2025): conformal TS with weak-dependence guarantees.

```python
import math
import torch

class ConformalSizer:
    """
    Split-conformal calibration. Calibrate on held-out val fold.
    Returns position multiplier in [0, 1] based on prediction-set size.
    """
    def __init__(self, alpha: float = 0.1):
        self.alpha = alpha          # 1 - alpha = coverage (default 90%)
        self.q_hat = None           # learned quantile threshold
    
    def calibrate(self, val_logits: torch.Tensor, val_labels: torch.Tensor):
        """Compute the calibration quantile on val set."""
        probs = torch.softmax(val_logits, dim=-1)
        # Non-conformity score: 1 - prob(true class)
        true_class_probs = probs[range(len(val_labels)), val_labels]
        scores = 1.0 - true_class_probs
        n = len(scores)
        # Conformal quantile (Vovk's split CP)
        k = math.ceil((n + 1) * (1 - self.alpha)) / n
        self.q_hat = torch.quantile(scores, min(k, 1.0)).item()
    
    def confidence_factor(self, logits: torch.Tensor) -> float:
        """
        Returns confidence factor in {1.0, 0.5, 0.0} based on prediction set size.
        Singleton (high conf) → 1.0. Doubleton → 0.5. Triple → flat (0.0).
        """
        assert self.q_hat is not None, "Call calibrate() first"
        probs = torch.softmax(logits, dim=-1).squeeze()
        in_set = probs >= (1.0 - self.q_hat)
        set_size = in_set.sum().item()
        return {1: 1.0, 2: 0.5, 3: 0.0}.get(set_size, 0.0)


def stage2_sizing_v4_with_conformal(
    probs: np.ndarray,
    realized_vol_20d: float,
    conformal_factor: float,    # NEW from ConformalSizer.confidence_factor()
    target_vol: float = 0.10,
    confidence_scale: float = 3.0,
    kelly_fraction: float = 0.5,
    position_limit: float = 1.0,
) -> float:
    """V1 sizing: vol-target × Kelly-lite × conformal-confidence."""
    direction = np.argmax(probs)
    if direction == 1:  # FLAT
        return 0.0
    sign = 1.0 if direction == 2 else -1.0
    confidence = max(probs) - 0.33
    if confidence <= 0:
        return 0.0
    raw_size = confidence * confidence_scale * kelly_fraction
    vol_mult = target_vol / max(realized_vol_20d, 1e-3)
    # NEW: multiply by conformal factor (1.0 high-conf, 0.5 medium, 0.0 low)
    size = sign * raw_size * vol_mult * conformal_factor
    return float(np.clip(size, -position_limit, position_limit))
```

**Calibration data flow:**
1. Train model on train fold
2. Compute logits on val fold
3. `ConformalSizer.calibrate(val_logits, val_labels)` → fits `q_hat`
4. At test/live: `ConformalSizer.confidence_factor(test_logits)` → returns 1.0 / 0.5 / 0.0
5. Multiply into Stage 2 sizing

**Coverage guarantee:** with `alpha=0.1`, the prediction set contains the true class ≥90% of the time. Expected behavior: high-uncertainty bars (regime breaks, FOMC) → larger sets → smaller positions automatically.

**Empirical bound (Wright 2026):** ~30% lower realized decision loss vs uncalibrated baseline. Add this to weeks 5-9 implementation.

## Probability Calibration (REQUIRED before Kelly-lite)

Modern NNs are systematically overconfident (Guo et al. 2017, arXiv:1706.04599). If the model claims `max_prob = 0.70` but true frequency is 0.55, Kelly-lite over-bets by ~4×. Temperature scaling fixes this with one learned scalar.

```python
import torch
import torch.nn as nn

class TemperatureScaler(nn.Module):
    """One-parameter calibration. Fit on validation fold AFTER training."""
    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1))
    
    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.temperature
    
    def fit(self, val_logits: torch.Tensor, val_labels: torch.Tensor, max_iter: int = 50):
        """LBFGS fit on val set."""
        nll = nn.CrossEntropyLoss()
        optimizer = torch.optim.LBFGS([self.temperature], lr=0.01, max_iter=max_iter)
        
        def closure():
            optimizer.zero_grad()
            loss = nll(self.forward(val_logits), val_labels)
            loss.backward()
            return loss
        
        optimizer.step(closure)
        return self


# Usage in sizing pipeline
def calibrated_probs(model_logits: torch.Tensor, scaler: TemperatureScaler) -> np.ndarray:
    with torch.no_grad():
        scaled = scaler(model_logits)
        return torch.softmax(scaled, dim=-1).cpu().numpy()
```

Report **Expected Calibration Error (ECE)** on every fold's val set. Target: ECE < 0.05. If model is well-calibrated raw, T ≈ 1.0 (skip scaling). If T > 1.5, calibration was meaningfully wrong — applying scaling is non-optional.

## Worked Examples

```
Mild UP signal, mid vol:
  probs = [0.20, 0.30, 0.50], realized_vol = 0.15
  → confidence = 0.17, raw_size = 0.255, vol_mult = 0.667 → size = +0.17

Strong UP, low vol:
  probs = [0.10, 0.20, 0.70], realized_vol = 0.08
  → confidence = 0.37, raw_size = 0.555, vol_mult = 1.25 → size = +0.694

Strong DOWN, high vol:
  probs = [0.65, 0.20, 0.15], realized_vol = 0.30
  → confidence = 0.32, raw_size = 0.480, vol_mult = 0.333 → size = -0.16

Very low confidence:
  probs = [0.30, 0.40, 0.30], realized_vol = 0.15
  → direction = FLAT → size = 0
```

## Kelly Math Recap

Bernoulli case: `f* = (bp - q) / b`. Symmetric (b=1): `f* = 2p - 1`.

For p=0.55: f* = 0.10 (10% of bankroll).

We use heuristic 3-class Kelly: `confidence = max_prob - 0.33`, scaled by `confidence_scale × kelly_fraction`. Default `confidence_scale=3, kelly_fraction=0.5` maps high-confidence (max_prob = 0.66) to size ≈ 0.5 (half-Kelly equivalent).

## Vol-Targeting Math Recap

`size_multiplier = target_vol / realized_vol`. Targets constant strategy volatility instead of letting it drift with asset volatility. Per Zhang-Zohren-Roberts (arXiv:1911.10107), captures most of the risk-adjusted return improvement that DRL agents claim. Free Sharpe.

Realized vol estimator: 20-day rolling stdev annualized by `sqrt(3276)`. EWMA (λ=0.94, RiskMetrics) is the production-grade alternative; try if 20-day is too slow to adapt.

## Ablation Plan (mandatory)

| Variant | Confidence weighting | Vol-target | Tests |
|---------|---------------------|------------|-------|
| Stage 1 | NO                  | NO         | Baseline |
| Stage 1.5a | YES              | NO         | Kelly-lite alone |
| Stage 1.5b | NO               | YES        | Vol-target alone |
| Stage 2 | YES                 | YES        | Combined (the version we ship if it wins) |

Run all 4 backtests with bootstrap CIs. If Stage 1.5b alone matches Stage 2, drop Kelly-lite (vol-target is doing all the work; simplicity wins).

## Implementation Day Plan

| Hour | Task |
|------|------|
| 1 | `stage2_sizing` function + unit tests on synthetic probs |
| 2 | `adjust_for_drawdown` + DD running tracker |
| 3 | `smoothed_sizing` (deferred unless needed) |
| 4 | Apply to test fold predictions → equity curves |
| 5 | Stage 1.5a / 1.5b / Stage 2 ablation backtests |
| 6 | Bootstrap CIs on all variants |
| 7 | Comparison table; pick winner |

## Decision Gate

**Ship Stage 2** if its bootstrap CI lower bound on Sharpe > Stage 1's bootstrap CI upper bound (or close to it). Otherwise ship Stage 1 — the simpler version that ties is the better X-thread story than a complex version that ties.

## Backtest Comparison

In doc 06, we compare:
- **Stage 1** = always full size when not flat
- **Stage 2** = formula above
- **Vol-target only** (constant confidence weight) — to isolate Kelly-lite contribution
- **Kelly-lite only** (constant vol multiplier) — to isolate vol-target contribution

If Stage 2 doesn't beat its individual components by a meaningful margin, the combined formula adds complexity for nothing — defer to whichever component wins solo.

## Drawdown Circuit-Breaker (proposed)

```python
def adjust_for_drawdown(target_size: float, current_drawdown_pct: float) -> float:
    """If running drawdown > 5%, halve sizing. > 10%, quarter it. > 15%, halt."""
    if current_drawdown_pct > 0.15:
        return 0.0
    elif current_drawdown_pct > 0.10:
        return target_size * 0.25
    elif current_drawdown_pct > 0.05:
        return target_size * 0.5
    return target_size
```

This is risk management: not optimal in expectation, but prevents account blowup. With $100 capital, this matters.

## Open Questions for Deep-Dive

1. Realized vol lookback: 20 days vs 5 days vs EWMA? Tradeoff stability vs responsiveness.
2. Position limit: 1.0 (full size) or fractional (0.5)? On $100, 1.0 = $100 of GLD. Reasonable.
3. Should Stage 2 use vol of GOLD or vol of the STRATEGY's recent returns? Strategy vol targets the strategy's vol directly. Asset vol is simpler. Most practitioners use asset vol; revisit.
4. Round-trip turnover budget: target trades per day? Affects cost burn.

---

# Part 2 — Exits + Risk

_(was doc 05-SIZING-AND-EXITS.md before V5 merge — content unchanged. Note: doc 07's sizing math supersedes doc 07's where they conflict; this part is the authoritative sizing math.)_

# 08 — Exits, Sizing, and Per-Trade Risk

## YOU ARE THE QUANT RISK AGENT

You own the three V1 pieces that decide **how much capital is at risk on each bar** and **when each trade ends**:

1. **Confidence-aware position sizing** — translates the model's softmax `[P_down, P_flat, P_up]` into a signed exposure in `[-1, +1]`.
2. **Per-trade stop-loss** — protects against the within-bar tail (model is silent for 29 of every 30 minutes; FOMC / CPI / NFP can move GLD 1-2% in 5 minutes).
3. **Profit-taking / exit logic** — decides when an open position closes ahead of the next argmax flip.

This doc **supersedes the sizing math in doc 07** and **adds the missing per-trade exit layer** that was never specified. doc 07 stays as the public-facing "Stage 2 sizing" entry point but its formula and rationale are replaced by what's below. doc 06 (backtest) and doc 08 (live) gain the integration hooks listed at the end.

**Read 00-OVERVIEW.md FIRST.**
**Then read 06-BACKTEST.md, 07-SIZING-AND-EXITS.md, and 08-LIVE-TRADING.md** before touching code — those three docs are the consumers of the interfaces you publish here.

### Execution Mode

**This doc IS the plan. Do not rewrite it.** Plan to *execute* what is written. If a claim is wrong, AskUserQuestion. Silent scope drift = fired.

- **Research with Nia** before guessing on libs, APIs, or papers.
- **Execution skills only:** `/investigate`, `/review`, `/qa`, `/qa-only`, `/cso`, `/benchmark`, `/ship`, `/land-and-deploy`, `/canary`, `/document-release`, `/retro`, `/learn`, `/browse`. **`/cso` mandatory** before any code path that submits live orders.
- **Loop:** read doc → Nia for unknowns → code → `/review` → `/qa` → `/cso` (live path) → `/ship`.
- **Escalate after 3 failed attempts.**

### Files You Create

```
src/nanogld/sizing/                       # extends doc 07 ownership
├── stage2.py                             # REWRITE — signed-score formula (replaces (max_prob - 0.33))
├── conformal.py                          # MODIFY — emit set_size, not 3-bucket factor
├── temperature_scaling.py                # KEEP as-is from doc 07
├── kelly_lite.py                         # DELETE — multinomial-Kelly framing was wrong
└── __init__.py

src/nanogld/exits/                        # NEW directory you own end-to-end
├── __init__.py
├── atr.py                                # ATR(14) on 30-min RTH bars (Wilder)
├── stops.py                              # StopState class — hard + trailing + time
├── reentry.py                            # ReentryGate — block same-side until fresh signal
├── session.py                            # 15:55 ET session-flat + 09:35 ET re-eligibility
├── blackout.py                           # FOMC / CPI / NFP ±15 min calendar blackout
├── signal_decay.py                       # OPTIONAL exit: max_prob decays vs entry
├── backtest_overlay.py                   # Vectorized two-pass intrabar correction (doc 06 calls this)
└── live_overlay.py                       # Cycle-time stop polling for live (doc 08 calls this)

tests/
├── test_signed_score_sizing.py           # Worked examples in this doc
├── test_vol_target_cap.py                # σ_t → 0 caps at 3×, never explodes
├── test_conformal_continuous.py          # Set-size shrinkage continuous on val fold
├── test_atr_correctness.py               # Matches Wilder reference on a known series
├── test_hard_stop_touch.py               # low ≤ stop_px triggers, slightly above does not
├── test_trailing_stop.py                 # peak ratchets up; never down
├── test_time_stop.py                     # Force-flat at bar 390
├── test_reentry_gate.py                  # Stop at bar 5, block re-entry until 6 + new argmax
├── test_session_flat.py                  # 15:55 ET forces flat; 09:30-09:34 blocks entry
├── test_blackout_calendar.py             # FOMC 14:00-14:30 ET window blocks new entries
├── test_backtest_overlay_vs_live.py      # Live and vectorized agree to <5 bp on a synthetic year
└── test_signal_decay.py                  # max_prob below entry × floor closes position
```

### Files You DO NOT Touch

- `src/nanogld/{data,features,embed,model,training,backtest,live}/` — those agents own the surface area; you publish interfaces and they call them.
- Other doc files. If your interface changes break another doc's contract, AskUserQuestion before shipping.

### Stable Interfaces You Publish

```python
# Sizing — replaces stage2_sizing_v4_with_conformal in doc 07
from nanogld.sizing.stage2 import compute_target_position

target = compute_target_position(
    probs: np.ndarray,                # (3,) post-temperature-scaling softmax
    sigma_ewma: float,                # annualized EWMA λ=0.94 on 30-min returns
    sigma_20d: float,                 # annualized 20-day rolling stdev (floor)
    conformal_set_size: int,          # 1, 2, or 3 from ConformalSizer
    target_vol: float = 0.10,
    kelly_fraction: float = 0.25,
    vol_mult_cap: float = 3.0,
    min_signed_signal: float = 0.05,
    position_limit: float = 1.0,
) -> float                            # in [-position_limit, +position_limit]


# Exit overlay — used by both backtest and live
from nanogld.exits.stops import StopState

state = StopState(
    entry_px: float,
    side: int,                        # +1 long, -1 short
    atr_at_entry: float,
    hard_mult: float = 2.0,
    trail_mult: float = 1.5,
    max_bars: int = 390,
)
state.update(high: float, low: float, close: float, atr: float, bar_index: int)
state.should_exit() -> tuple[bool, str]  # (True, "hard" | "trail" | "time")
state.exit_price() -> float           # the price the engine should book the exit at


# Re-entry gate
from nanogld.exits.reentry import ReentryGate
gate = ReentryGate(prob_threshold: float = 0.55, cooldown_bars: int = 1)
gate.on_stop(side: int, bar_index: int)
gate.allow_entry(side: int, max_prob: float, bar_index: int) -> bool


# Session + blackout
from nanogld.exits.session import is_within_session, must_force_flat
from nanogld.exits.blackout import is_blacked_out

# Backtest overlay — single call from doc 06 engine after model produces target positions
from nanogld.exits.backtest_overlay import apply_exit_overlay
positions = apply_exit_overlay(
    target_positions: pd.Series,      # raw model-driven position (signed, in [-1, +1])
    bars: pd.DataFrame,               # OHLC at 30-min bars
    atr_14: pd.Series,
    probs: pd.DataFrame,              # columns: P_down, P_flat, P_up
    config: ExitConfig,
) -> pd.Series                        # corrected positions (after stops, session, blackout, re-entry)


# Live overlay — single call from doc 08 cycle just before order submission
from nanogld.exits.live_overlay import resolve_live_target
order_action = resolve_live_target(
    current_position_qty: float,
    current_entry_px: float | None,
    current_entry_atr: float | None,
    current_peak_px: float | None,
    bar_high: float,
    bar_low: float,
    bar_close: float,
    atr_now: float,
    probs: np.ndarray,
    sigma_ewma: float,
    sigma_20d: float,
    bar_index_in_trade: int,
    last_stop_bar: int | None,
    last_stop_side: int | None,
    now_et: datetime,
    config: ExitConfig,
) -> LiveAction                       # one of: HOLD, EXIT, ENTER(qty, side), REBALANCE(qty_delta)
```

### Acceptance Criteria

1. ✅ `pytest tests/test_signed_score_sizing.py` passes the four worked examples in this doc.
2. ✅ Vol-target multiplier capped at 3.0× on a synthetic σ→0 series; never explodes.
3. ✅ ConformalSizer.calibrate produces ≥90% empirical coverage on val fold (α=0.10).
4. ✅ ATR(14) implementation matches Wilder reference (test_atr_correctness golden values).
5. ✅ `apply_exit_overlay` and `resolve_live_target` agree to within 5 bp annualized on a 1-year synthetic replay (test_backtest_overlay_vs_live). This is the load-bearing test — without it, backtest and live diverge.
6. ✅ Hyperparameter sweep on val fold completed: `{hard_stop_mult, trail_stop_mult, reentry_prob_thresh}` 27 combos × 4 walk-forward folds.
7. ✅ Ablation table in this doc populated: 8 variants from "no stops, no signal decay" to "full V1 stack."
8. ✅ Stage 2 (full stack) must beat Stage 1 (fixed sizing, no exits) by ≥0.2 Sharpe OOS at 7 bp cost stress to ship. Otherwise ship Stage 1.
9. ✅ Forecast-to-Fill replication (separate baseline) lands within ±0.3 Sharpe of paper's 2.88 on a daily GLD replay (sanity check on the engine).

### Spawn Nia Agents When You Need To

- **Wilder ATR vs `pandas-ta-classic` `atr()`** — verify the modern package signature uses Wilder smoothing (RMA), not SMA.
- **`alpaca-py` `OrderClass.BRACKET` + fractional** — re-verify the "fractional ⇒ no bracket" constraint stays true in the alpaca-py version pinned at impl time. Forum thread cited error code 42210000; confirm still applies.
- **EWMA λ=0.94 on 30-min bars** — verify the annualization factor is `√3276` and that the EWMA implementation handles the per-day market-close-to-next-day-open gap correctly.
- **arXiv:2511.08571 v2/v3** — Forecast-to-Fill paper may have updated; check for any post-Nov-2025 changes to the ATR multipliers or Kelly fraction.

### V1 Critical Decisions (DO NOT REVERT)

1. **Signed score `s = P_up - P_down`**, not `max_prob - 0.33`. Magnitude-only formula collapses two distinct signals (clean UP vs mixed UP/DOWN) into the same size.
2. **Quarter-Kelly (0.25) at V1 launch.** Half-Kelly only after 6 months of OOS calibration evidence shows the model is well-calibrated.
3. **`target_vol = 0.10` annualized.** F2F uses 0.15 on gold *futures*; GLD is unleveraged, so 0.10 is the conservative cash-equity equivalent. Test 0.15 in A/B if 0.10 underbets.
4. **`vol_mult` capped at 3.0.** With `sigma_target = 0.10` this means realized vol below 3.3% annualized never multiplies further.
5. **Continuous conformal shrinkage**, not 3-bucket. Set-size 1 → λ=1.0; set-size 2 → λ=0.5; set-size 3 → λ=0.0. Identical numerically to doc 07's three-bucket form, but the function signature exposes the underlying primitive (set size) so future work can swap in continuous variants.
6. **`hard_stop = 2.0 × ATR(14)` and `trail_stop = 1.5 × ATR(14)`.** Match Forecast-to-Fill exactly. Sweep on val fold but default to 2.0 / 1.5.
7. **Time-stop = 390 bars.** F2F uses 30 trading days on daily bars; 30 days × 13 RTH bars = 390 30-min bars. This is the safety net, not the primary exit.
8. **No fixed take-profit.** Empirical literature (F2F, Baur-Dimpfl, positive-skew trend research) plus the 5-bp cost gate make a price-level TP net-negative. Optional signal-decay exit (`max_prob[t] < entry_max_prob × floor`) is the only TP-like rule, and ships only if val-fold sweep shows ≥30 bp annualized lift.
9. **Session-flat at 15:55 ET, re-eligible at 09:35 ET.** No overnight GLD positions. Eliminates Sunday-night CME gold spike risk and after-hours news risk. Costs us any genuine overnight model edge but at $100 capital that's not worth a 3% gap.
10. **News blackout ±15 min around scheduled FOMC, CPI, NFP releases.** Calendar already wired in doc 04 features. No new entries during blackout; existing positions can hold but stops apply normally.
11. **Re-entry gate after stop-out:** block same-side re-entry until either (a) ≥1 bar elapsed AND `max_prob ≥ 0.55`, or (b) model argmax has flipped to a different side. Prevents stop-then-re-enter-then-stop loops on news bars.
12. **Live ≠ broker bracket.** Alpaca rejects bracket orders for fractional positions (error 42210000), and at $100 capital with GLD ~$200 every position is fractional. Stops are enforced **client-side** in `cycle.py` by polling current price each cycle and submitting a market exit when triggered. The broker is a dumb execution venue.
13. **Drawdown circuit-breaker thresholds (-5% / -10% / -15%) stay**, but they're now scaled to quarter-Kelly. At quarter-Kelly a -15% portfolio drawdown means the model is materially broken, so halt-then-investigate is correct behavior.
14. **`min_signed_signal = 0.05`.** Below 5 percentage points of P_up minus P_down, the model is noise. Stay flat.
15. **`min_notional = $1.00`.** Alpaca minimum. If the sized position dollarizes to <$1, stay flat.

---

# 08 — Exits, Sizing, and Per-Trade Risk

**Status:** ✅ Complete, V1 spec, research-backed
**Last verified:** 2026-05-04 (4 parallel research agents — sizing, stop-loss, profit-taking, Forecast-to-Fill replication + Alpaca constraints)

## Why This Doc Exists

The owner flagged on 2026-05-04 that three load-bearing pieces of the trading system were never properly designed:

1. **Confidence sizing** — doc 07 sketches a formula but uses a heuristic (`max_prob - 0.33`) that throws away signed information, a `confidence_scale = 3` that was guessed, a `kelly_fraction = 0.5` that the literature suggests is too aggressive for ML-derived probabilities, and cites a fabricated "30% lower decision loss" number (verified absent from arXiv:2601.07852 Wright 2026).
2. **Per-trade stop-loss** — never designed. doc 08 line 198 says "stop-loss handled at strategy level via drawdown circuit-breaker (not order-level)." The drawdown circuit-breaker is a *portfolio-level* protection, not a per-trade one. A single trade can lose 5-10% before the circuit-breaker fires.
3. **Profit-taking** — never designed. Implicit assumption: model re-decision next bar closes the trade. True 95% of the time. False on news bars where price moves more in 5 minutes than the model can react to in 30.

This doc fixes all three with literature-backed defaults and explicit val-fold tuning protocols.

## Three Pillars

```
                    ┌─────────────────────────────────┐
                    │   model logits (3-class)        │
                    └────────────────┬────────────────┘
                                     │
              ┌──────────────────────▼──────────────────────┐
              │  PILLAR 1 — SIZING                          │
              │  signed score → vol-target → conformal      │
              │  output: target_position ∈ [-1, +1]         │
              └──────────────────────┬──────────────────────┘
                                     │
              ┌──────────────────────▼──────────────────────┐
              │  PILLAR 2 — STOP-LOSS / EXITS               │
              │  hard ATR stop + trailing ATR + time-stop   │
              │  + re-entry gate + session-flat + blackout  │
              │  output: corrected_position                 │
              └──────────────────────┬──────────────────────┘
                                     │
              ┌──────────────────────▼──────────────────────┐
              │  PILLAR 3 — PROFIT-TAKING                   │
              │  optional: signal-decay exit                │
              │  default: model re-decision IS the TP       │
              └──────────────────────┬──────────────────────┘
                                     │
                              order submission
                          (Alpaca client-side stops)
```

## Glossary

- **Signed score:** `s = P_up - P_down` ∈ [-1, +1]. Captures direction and conviction.
- **Vol multiplier:** `min(target_vol / sigma_t, vol_mult_cap)`. Caps blowups in low-vol regimes.
- **Conformal set size:** the number of classes in the prediction set produced by split-conformal calibration at α=0.10. 1 = high confidence, 3 = no confidence.
- **ATR(14):** 14-bar Wilder-smoothed average true range computed on 30-min RTH bars.
- **R distance:** initial risk per share = |entry_px - hard_stop_px| = `2 × ATR_at_entry`.
- **Trade:** maximal run of bars during which sign(position) is constant and non-zero. The system thinks in trades for the purpose of exits, but in continuous positions for the purpose of sizing.

---

## PILLAR 1 — Confidence-Aware Position Sizing (V2)

### The Formula

```python
def compute_target_position(
    probs: np.ndarray,                # post-temperature-scaling softmax, shape (3,) [P_down, P_flat, P_up]
    sigma_ewma: float,                # annualized EWMA λ=0.94 on 30-min returns
    sigma_20d: float,                 # annualized 20-day rolling stdev (floor)
    conformal_set_size: int,          # 1, 2, or 3
    target_vol: float = 0.10,
    kelly_fraction: float = 0.25,
    vol_mult_cap: float = 3.0,
    min_signed_signal: float = 0.05,
    position_limit: float = 1.0,
) -> float:
    """
    Returns desired position multiplier in [-position_limit, +position_limit].
    Output is signed: positive = long, negative = short, 0 = flat.
    """
    p_down, p_flat, p_up = probs
    s = p_up - p_down

    # Gate 1: signed signal too weak
    if abs(s) < min_signed_signal:
        return 0.0

    # Gate 2: conformal set says model abstains
    lambda_conf = {1: 1.0, 2: 0.5, 3: 0.0}[conformal_set_size]
    if lambda_conf == 0.0:
        return 0.0

    # Vol multiplier with EWMA primary + 20d floor + hard cap
    sigma_t = max(sigma_ewma, 0.5 * sigma_20d, 1e-3)
    vol_mult = min(target_vol / sigma_t, vol_mult_cap)

    raw = s * kelly_fraction * lambda_conf * vol_mult
    return float(np.clip(raw, -position_limit, position_limit))
```

### Why It Beats the Doc-07 Formula

| Concern | doc 07 (`max_prob - 0.33`) | doc 07 (`P_up - P_down`) |
|---|---|---|
| `[0.20, 0.30, 0.50]` (clean UP) | size = `0.17 × scale` | `s = +0.30` |
| `[0.40, 0.10, 0.50]` (mixed UP/DOWN) | size = `0.17 × scale` (same!) | `s = +0.10` (correctly smaller) |
| Sign info | derived from argmax separately | naturally signed |
| `confidence_scale` | guessed at 3, fragile | unnecessary; signed score is in [-1, +1] |
| Low-vol blowup | `1e-3` floor still allows 100× multiplier | hard cap at 3.0× |

**Citation:** Lim, Zohren, Roberts arXiv:1904.04912 (Deep Momentum Networks) and the Forecast-to-Fill paper (arXiv:2511.08571) both use signed forecasts in `[-1, +1]` scaled by `target_vol / sigma_t`. The `(max_prob - 0.33)` form is a retail heuristic with no literature support.

### Worked Examples

```
EX 1 — Strong signed UP, mid vol, set-size 1
  probs = [0.10, 0.20, 0.70], σ_ewma = 0.12, σ_20d = 0.15, set_size = 1
  s = 0.60, λ_conf = 1.0
  σ_t = max(0.12, 0.5 * 0.15, 1e-3) = 0.12
  vol_mult = min(0.10 / 0.12, 3.0) = 0.833
  raw = 0.60 * 0.25 * 1.0 * 0.833 = 0.125
  target_position = +0.125  (12.5% long)

EX 2 — Mixed UP/DOWN, low vol, set-size 1 (set narrow but signal not clean)
  probs = [0.40, 0.10, 0.50], σ_ewma = 0.05, σ_20d = 0.08, set_size = 1
  s = 0.10, λ_conf = 1.0
  σ_t = max(0.05, 0.04, 1e-3) = 0.05
  vol_mult = min(0.10 / 0.05, 3.0) = 2.0
  raw = 0.10 * 0.25 * 1.0 * 2.0 = 0.05
  target_position = +0.05  (correctly tiny — model is saying "barely up, lots of down mass")

EX 3 — Strong DOWN, vol-target cap binds
  probs = [0.65, 0.20, 0.15], σ_ewma = 0.02, σ_20d = 0.03, set_size = 1
  s = -0.50, λ_conf = 1.0
  σ_t = max(0.02, 0.015, 1e-3) = 0.02
  vol_mult = min(0.10 / 0.02, 3.0) = 3.0   ← cap binds
  raw = -0.50 * 0.25 * 1.0 * 3.0 = -0.375
  target_position = -0.375  (37.5% short — cap protected against 5× multiplier)

EX 4 — Conformal set 3 (model abstains)
  probs = [0.35, 0.30, 0.35], σ_ewma = anything, set_size = 3
  s = 0.0, BUT also λ_conf = 0.0
  target_position = 0.0  (model is uncertain, stay flat)

EX 5 — Below min_signed_signal
  probs = [0.30, 0.40, 0.32], set_size = 1
  s = 0.02, |s| < 0.05
  target_position = 0.0  (signal too weak)

EX 6 — Conformal set 2 (medium confidence)
  probs = [0.20, 0.30, 0.50], σ_ewma = 0.10, σ_20d = 0.10, set_size = 2
  s = 0.30, λ_conf = 0.5
  vol_mult = min(0.10 / 0.10, 3.0) = 1.0
  raw = 0.30 * 0.25 * 0.5 * 1.0 = 0.0375
  target_position = +0.0375  (medium-confidence half-shrunk)
```

### Hyperparameters

| Param | V1 default | Source | Tuning |
|---|---|---|---|
| `target_vol` | 0.10 | F2F paper uses 0.15 on futures; cash-equity GLD discounted | A/B test 0.15 in V2 if 0.10 underbets |
| `kelly_fraction` | 0.25 (quarter-Kelly) | MacLean-Thorp-Ziemba; Downey simulations on uncertain probabilities | Ramp to 0.5 only after ≥6 months OOS evidence of well-calibrated probabilities |
| `vol_mult_cap` | 3.0 | F2F paper caps notional at ±20%, equivalent here at σ_target=0.10 | Fixed |
| `min_signed_signal` | 0.05 | Selective-classification arXiv:2110.14914 | Sweep ∈ {0.03, 0.05, 0.08} on val fold |
| `position_limit` | 1.0 | Owner constraint (cash account, no margin) | Fixed |
| EWMA λ | 0.94 | RiskMetrics standard | Fixed |
| 20d floor multiplier | 0.5× | Sanity floor against EWMA collapse | Fixed |
| Conformal α | 0.10 (90% coverage) | Standard split conformal | Fixed |

### What's Deleted From doc 07

- **`(max_prob - 0.33)` formula** — replaced by signed score.
- **`confidence_scale = 3`** — replaced by signed score's natural [-1, +1] range.
- **`stage2_sizing_v4_with_conformal`** — replaced by `compute_target_position`.
- **Multinomial-Kelly framing** (Smoczynski-Tomkins reference) — wrong tool for continuous-return ETF; deleted.
- **"30% lower realized decision loss" claim** — verified absent from arXiv:2601.07852. Citation was fabricated.
- **`kelly_lite.py` file** — delete after migration; the new formula has no Kelly-lite component, only single-asset fractional Kelly applied via `kelly_fraction`.

---

## PILLAR 2 — Per-Trade Stop-Loss and Exit Logic

### Why Add A Stop At All

The model decides at bar close. The stop fires intrabar on the high/low touch. On a typical 30-min RTH bar these are equivalent for ~95% of bars. On the 5% containing FOMC / CPI / NFP / geopolitical headlines, the difference is the model watching a -3% drawdown unfold for 28 minutes after entry without being allowed to react until close.

Three pieces of empirical evidence lock the decision:

1. **Kaminski & Lo (2014, J. Financial Markets):** Under momentum and Markov regime-switching processes, stop-loss rules add 50-100 bp/month during stop-out periods. The "stopping premium" is positive whenever the underlying exhibits regime persistence — which gold does (well-documented vol clustering and macro-regime sensitivity).
2. **Han, Zhou, Zhu (2014, SSRN):** A 10% stop on momentum strategies (1926-2013) reduced max monthly loss from -49.79% to -11.36% AND **doubled the Sharpe.** Not a free lunch, a structural improvement.
3. **Forecast-to-Fill (arXiv:2511.08571):** The Sharpe-2.88 / MDD-0.52% gold benchmark uses ATR stops. The published best gold result chose to include them; that is the strongest single data point.

Counter-evidence: Lo & Remorov (2017) find tight stops underperform on individual stocks due to costs. Resolution: use *wide* stops (2× ATR(14) ≈ 50-80 bp on a typical GLD 30-min bar) that fire only on tail moves and preserve the cost budget on normal whipsaw.

### The Three Stops

#### Hard ATR stop

- **Long:** exit at `entry_px - 2.0 × ATR_14_at_entry`
- **Short:** exit at `entry_px + 2.0 × ATR_14_at_entry`
- ATR is **frozen at entry** (computed from the bar that opens the trade) for the lifetime of the trade. This prevents the stop from drifting away from the entry price as ATR expands during a losing trade.
- Fires when the bar's low (long) or high (short) touches the stop price.

#### Trailing ATR stop

- **Long:** trail price = `peak_close_so_far - 1.5 × ATR_14_current`
- **Short:** trail price = `trough_close_so_far + 1.5 × ATR_14_current`
- ATR for the trail is **the live ATR**, not the entry ATR — the trail is supposed to adapt as conditions change.
- The trail price is a ratchet: it can move toward the entry but never away from the favorable direction. Long trail can only rise; short trail can only fall.
- Fires when the bar's low (long) or high (short) touches the trail price.

#### Time stop

- 390 bars maximum hold (30 RTH days × 13 30-min bars). At `bars_in_trade == 390`, force-flat at the close.
- F2F uses 30 trading days on daily bars; same 30-day calendar on 30-min granularity.
- Belt-and-suspenders. Will rarely fire because session-flat at 15:55 ET caps a single trade at 13 bars before the next session, and the model re-decides every bar. Catches the "stuck-on-same-prediction-forever" edge case.

### Re-Entry Gate

After a stop-out, block same-direction re-entry until both:

- **At least 1 bar has elapsed since the stop.** Single-bar cooldown prevents same-bar stop-then-re-enter.
- **One of:** (a) `max_prob ≥ 0.55` AND argmax matches the stopped-out direction, or (b) argmax has flipped to a different side at any point during the cooldown.

Implementation: `ReentryGate` keeps `last_stop_bar` and `last_stop_side`. On each bar, if `bar_index - last_stop_bar < cooldown` or condition above fails, return `False` for that side. Opposite-side entries are unaffected.

### Session Flat

- **Force-flat at 15:55 ET** (5 minutes before close). All open positions exit at the bar's close.
- **Re-eligible at 09:35 ET** (5 minutes after open). No new entries before that.
- This eliminates Sunday-night CME gold spikes, Trump-tweet after-hours moves, and FOMC-after-hours tail risk. At $100 capital, a 3% gap is $3 of P&L — small but a high-variance, undiversified exposure that the model has no signal on (no overnight bars in training).
- Cost: any genuine overnight model edge. Lim-Zohren-Roberts and similar deep-momentum literature is daily and explicitly captures overnight; we deliberately punt that for V1.

### News Blackout

- ±15 minutes around scheduled FOMC, CPI, NFP, GDP, JOLTS, PCE releases. Calendar already wired in doc 04 features (calendar event proximity windows).
- During blackout: no new entries. Existing positions hold. Stops apply normally.
- Rationale: scheduled events show 5-10× normal volatility (J. Banking & Finance 2024 on gold's asymmetric FOMC reaction). A 50-bp stop has zero chance of holding inside a 1000-pip 5-minute FOMC bar.
- We do NOT widen stops or close positions pre-event for V1. Adding entry blackout is the simplest robust mitigation.

### Backtest Implementation (Vectorized Two-Pass)

The doc 06 engine stays vectorized. Pass 1 produces raw model-driven positions; pass 2 corrects for stops, session, blackout, re-entry. Pseudocode:

```python
def apply_exit_overlay(target_positions, bars, atr_14, probs, config):
    # Pass 1 already complete: target_positions = model-driven raw signal.

    out = target_positions.copy()
    n = len(out)
    in_trade = False
    side = 0
    entry_px = entry_atr = peak_px = trough_px = None
    entry_bar = stop_bar = -10_000
    last_stop_side = 0

    for t in range(n):
        # Session and blackout — gate new entries
        if not is_within_session(bars.index[t]):
            out.iloc[t] = 0.0
            continue
        blackout = is_blacked_out(bars.index[t], config.calendar)

        # Force-flat at 15:55 ET
        if must_force_flat(bars.index[t]):
            out.iloc[t] = 0.0
            in_trade = False
            continue

        if in_trade:
            # Update peak/trough
            if side == +1: peak_px = max(peak_px, bars.high.iloc[t])
            else:           trough_px = min(trough_px, bars.low.iloc[t])

            # Hard stop touch
            hard = entry_px - side * config.hard_mult * entry_atr
            hard_hit = (side == +1 and bars.low.iloc[t] <= hard) or \
                       (side == -1 and bars.high.iloc[t] >= hard)

            # Trail stop touch
            if side == +1: trail = peak_px - config.trail_mult * atr_14.iloc[t]
            else:          trail = trough_px + config.trail_mult * atr_14.iloc[t]
            trail_hit = (side == +1 and bars.low.iloc[t] <= trail) or \
                        (side == -1 and bars.high.iloc[t] >= trail)

            # Time stop
            time_hit = (t - entry_bar) >= config.max_bars

            if hard_hit or trail_hit or time_hit:
                out.iloc[t] = 0.0
                in_trade = False
                stop_bar = t
                last_stop_side = side
                continue

            # Still in trade. If model says exit (target = 0 or sign flip), close at close.
            if np.sign(target_positions.iloc[t]) != side:
                out.iloc[t] = target_positions.iloc[t]   # flip or flat
                in_trade = (target_positions.iloc[t] != 0.0)
                if in_trade:
                    side = int(np.sign(target_positions.iloc[t]))
                    entry_px = bars.close.iloc[t]
                    entry_atr = atr_14.iloc[t]
                    peak_px = trough_px = entry_px
                    entry_bar = t
                continue

            # Model agrees with current side. Hold (do not over-write target_positions).
            out.iloc[t] = target_positions.iloc[t]
            continue

        # Not in trade. Check entry.
        if target_positions.iloc[t] == 0.0:
            out.iloc[t] = 0.0
            continue
        if blackout:
            out.iloc[t] = 0.0
            continue

        new_side = int(np.sign(target_positions.iloc[t]))

        # Re-entry gate
        if new_side == last_stop_side:
            cooldown = (t - stop_bar) < config.reentry_cooldown_bars
            argmax_flipped_since = any(
                np.argmax(probs.iloc[k]) != (2 if last_stop_side == +1 else 0)
                for k in range(stop_bar, t)
            )
            if cooldown and not argmax_flipped_since:
                if probs.iloc[t].max() < config.reentry_prob_threshold:
                    out.iloc[t] = 0.0
                    continue

        # Enter
        out.iloc[t] = target_positions.iloc[t]
        in_trade = True
        side = new_side
        entry_px = bars.close.iloc[t]
        entry_atr = atr_14.iloc[t]
        peak_px = trough_px = entry_px
        entry_bar = t

    return out
```

**Bias note:** the overlay assumes worst-case fill at exactly `stop_px` whenever low ≤ stop ≤ high. This is conservative; real fills may be slightly better or worse. The doc 06 cost model already pads 2 bp for slippage surprises which absorbs this.

### Live Implementation (Client-Side Polling, Not Broker Bracket)

**Critical constraint:** Alpaca rejects bracket orders for fractional positions with `"fractional orders must be simple orders (Code = 42210000)"`. At $100 capital with GLD ~$200, every position is fractional. Therefore **stops must be enforced client-side.**

The live cycle in doc 08 already runs every 30 minutes during RTH. The cycle calls `resolve_live_target` at the top, which does this:

```python
def resolve_live_target(...) -> LiveAction:
    # Step 0: session check
    if not is_within_session(now_et) or must_force_flat(now_et):
        if current_position_qty != 0.0:
            return LiveAction.EXIT
        return LiveAction.HOLD

    # Step 1: existing position — check stops
    if current_position_qty != 0.0:
        side = int(np.sign(current_position_qty))
        # Hard stop
        hard_px = current_entry_px - side * config.hard_mult * current_entry_atr
        if (side == +1 and bar_low <= hard_px) or (side == -1 and bar_high >= hard_px):
            return LiveAction.EXIT
        # Trail
        peak = max(current_peak_px, bar_high) if side == +1 else min(current_peak_px, bar_low)
        if side == +1:
            trail_px = peak - config.trail_mult * atr_now
            if bar_low <= trail_px:
                return LiveAction.EXIT
        else:
            trail_px = peak + config.trail_mult * atr_now
            if bar_high >= trail_px:
                return LiveAction.EXIT
        # Time
        if bar_index_in_trade >= config.max_bars:
            return LiveAction.EXIT

    # Step 2: blackout
    if is_blacked_out(now_et, config.calendar):
        if current_position_qty != 0.0:
            return LiveAction.HOLD     # existing position holds; stops still apply via Step 1
        return LiveAction.HOLD          # no new entries

    # Step 3: model-driven sizing
    target_frac = compute_target_position(probs, sigma_ewma, sigma_20d, conformal_set_size, ...)
    target_notional = target_frac * capital
    target_qty = target_notional / bar_close

    # Min notional gate
    if abs(target_qty * bar_close) < config.min_notional:
        target_qty = 0.0

    # Step 4: re-entry gate
    if last_stop_side is not None and int(np.sign(target_qty)) == last_stop_side:
        ... apply gate logic ...

    # Step 5: convert to action
    delta = target_qty - current_position_qty
    if abs(delta * bar_close) < config.min_notional:
        return LiveAction.HOLD
    if current_position_qty == 0.0 and target_qty != 0.0:
        return LiveAction.ENTER(qty=target_qty, side=int(np.sign(target_qty)))
    if current_position_qty != 0.0 and target_qty == 0.0:
        return LiveAction.EXIT
    return LiveAction.REBALANCE(qty_delta=delta)
```

**State persistence (CRITICAL):** `current_entry_px`, `current_entry_atr`, `current_peak_px`, `bar_index_in_trade`, `last_stop_bar`, `last_stop_side` must persist across cycles via doc 08's SQLite state store. If the cycle crashes mid-trade and restarts, recover state from SQLite. If state is missing, reconstruct from Alpaca's `get_orders(filter=closed, after=last_known_bar)` plus current position — but this is best-effort and a recovery alert should fire.

**Why this works at 30-min cadence:** the model gets a free intra-bar look at high/low because by the time the cycle runs (bar close + a few seconds), the bar's high/low is already known from market data. We're not running a tick-by-tick stop, we're running a **bar-completion stop check.** Faster reaction would require sub-bar polling, which the current launchd architecture doesn't support and is unnecessary at this size.

**What about an actual intra-bar move that breaches the stop and reverses before bar close?** The bar's low touches the stop, but the bar closes above it. We will exit anyway at the bar close price — slightly favorable execution vs the stop price. This is conservative (we treated the stop as fired in the model) but accepts a small amount of overrun in exchange for not running a tighter polling loop.

---

## PILLAR 3 — Profit-Taking

### Decision: No Fixed Take-Profit In V1

Empirical case is one-sided once you frame the system correctly:

1. **Forecast-to-Fill (Sharpe 2.88 on gold)** uses NO fixed take-profit. Trailing stop + timeout + regime de-risk handle exits. This is the exact precedent.
2. **Baur & Dimpfl (SSRN 4233700):** "cut your losses, let your profits run" with fixed TP underperforms buy-and-hold across daily/quarterly/annual windows including 2008 and 2020 crises. Folk wisdom does not hold up.
3. **Positive-skew structure of trend / momentum returns:** <7% of trades drive cumulative profit (CFM, Quantica). Truncating the right tail with a fixed TP is asymmetrically destructive.
4. **5-bp cost gate:** TP creates extra round-trips. At 5 bp/RT, ~100 extra TP exits/year = 50 bp drag. To net zero, TP must add ≥50 bp/year of return. No 30-min single-asset literature clears that bar.
5. **The model is the TP.** Sizing = `s × kelly × λ_conf × vol_mult`. As confidence falls, position falls smoothly. As argmax flips, position reverses. The continuous re-rebalance IS the profit-taking mechanism.

**Do not add:** R-multiples, ATR-multiple TP, scale-outs, fixed-percentage TP. Each one fails the 5-bp cost gate or the positive-skew test.

### Optional: Signal-Decay Exit (Gated A/B)

One TP-like rule survives the cost gate: **close the position when current `max_prob` decays below entry `max_prob × floor`**, even if argmax hasn't flipped. This trails the *signal*, not the price. It composes with everything else because it only ever pushes position toward 0; it never adds risk.

```python
# Per trade:
entry_max_prob = max_prob_at_entry

# Each subsequent bar while in trade:
if max_prob[t] < entry_max_prob * config.signal_decay_floor:
    out.iloc[t] = 0.0     # close, override sizing layer
```

**Hyperparameter:** `signal_decay_floor` ∈ {0.5, 0.6, 0.7, 0.8}. Default 0.7 (heuristic midpoint). Tune on val fold.

**Ship gate:** include this rule in the final V1 stack only if val-fold A/B shows ≥30 bp annualized Sharpe improvement at 7-bp cost stress vs the no-signal-decay version. Otherwise drop it; the principle of "ship the simpler model that ties" applies.

### Drawdown Circuit-Breaker (Portfolio Level — Stays From doc 07)

The existing portfolio-level drawdown circuit-breaker is preserved unchanged:

- DD ≤ -5% → halve sizing on next bar
- DD ≤ -10% → quarter sizing
- DD ≤ -15% → halt sizing entirely until DD recovers above -10% OR 65 bars (5 trading days) elapse, whichever first.

Note: at quarter-Kelly (V1 default), hitting -15% portfolio DD means the model is materially broken — that's the right place to halt and investigate.

---

## Integration With Existing Docs

### Changes Required In doc 06 (Backtest)

1. Add `apply_exit_overlay` call between the model-driven `target_positions` and the cost-deducted `strategy_returns`. One-line change in `vectorized_backtest`:

   ```python
   target_positions = model_driven_positions(...)               # existing
   final_positions  = apply_exit_overlay(target_positions, bars, atr_14, probs, exit_config)  # NEW
   gross_returns    = final_positions * next_returns            # existing, but uses final_positions
   ```

2. Add `atr_14` precomputation step — already in doc 04 features so this is just a join.
3. Add `probs` series passthrough — already produced by model inference; just need to keep it around for the overlay.
4. Add ablation runner: 8 variants from `(no_stops, no_signal_decay, fixed_size)` to `(stops, signal_decay, signed_score_size)`.

### Changes Required In doc 07 (Sizing)

1. **Replace** `stage2_sizing` and `stage2_sizing_v4_with_conformal` with `compute_target_position` from this doc.
2. **Modify** `ConformalSizer` to expose `set_size(logits) -> int` instead of `confidence_factor(logits) -> float`. The sizer in this doc owns the mapping from set size to shrinkage.
3. **Delete** `kelly_lite.py` — the new formula has no Kelly-lite component.
4. **Update** "30% lower decision loss" claim — remove fabricated number, replace with "Wright 2026 (arXiv:2601.07852) develops utility-weighted forecasting with cost-aware loss; we apply a quarter-Kelly fractional shrinkage as conservative initialization rather than the paper's full optimization pipeline."

### Changes Required In doc 08 (Live)

1. **Replace** line 198 ("Stop-loss: handled at strategy level via drawdown circuit-breaker") with the per-trade stop logic in this doc.
2. **Add** `resolve_live_target` call at the top of `cycle.py`'s `run_cycle()` function. Replaces direct calls to `stage2_sizing`.
3. **Add** state persistence schema in `state.py` (SQLite):

   ```sql
   CREATE TABLE trade_state (
       trade_id INTEGER PRIMARY KEY AUTOINCREMENT,
       opened_at TEXT NOT NULL,
       side INTEGER NOT NULL,
       entry_px REAL NOT NULL,
       entry_atr REAL NOT NULL,
       peak_px REAL NOT NULL,
       trough_px REAL NOT NULL,
       bars_in_trade INTEGER NOT NULL,
       closed_at TEXT,
       exit_reason TEXT       -- 'hard_stop' | 'trail_stop' | 'time_stop' | 'session_flat' | 'model_flip' | 'signal_decay'
   );

   CREATE TABLE last_stop (
       side INTEGER,
       bar_index INTEGER,
       at TEXT
   );
   ```

4. **Add** `exit_reason` to the wandb log payload per cycle.
5. **Confirm** session-flat at 15:55 ET fires correctly via launchd `StartCalendarInterval` array.
6. **Confirm** that the cycle does NOT submit bracket orders (document this constraint explicitly to prevent regression).

---

## Hyperparameter Table (Consolidated)

| Pillar | Param | V1 default | Tunable? | Source |
|---|---|---|---|---|
| Sizing | `target_vol` | 0.10 | A/B vs 0.15 | F2F arXiv:2511.08571 |
| Sizing | `kelly_fraction` | 0.25 | Ramp to 0.5 after 6mo OOS | MacLean-Thorp-Ziemba |
| Sizing | `vol_mult_cap` | 3.0 | Fixed | F2F leverage cap |
| Sizing | `min_signed_signal` | 0.05 | Sweep {0.03, 0.05, 0.08} | arXiv:2110.14914 |
| Sizing | EWMA λ | 0.94 | Fixed | RiskMetrics |
| Sizing | 20d floor multiplier | 0.5 | Fixed | Sanity |
| Sizing | Conformal α | 0.10 | Fixed | Standard split CP |
| Sizing | `position_limit` | 1.0 | Fixed | Owner constraint (cash account) |
| Sizing | `min_notional` | $1.00 | Fixed | Alpaca minimum |
| Stop | `atr_window` | 14 bars | Fixed | Wilder; F2F |
| Stop | `hard_stop_mult` | 2.0 | Sweep {1.5, 2.0, 2.5} | F2F arXiv:2511.08571 |
| Stop | `trail_stop_mult` | 1.5 | Sweep {1.0, 1.5, 2.0} | F2F |
| Stop | `time_stop_bars` | 390 | Fixed | F2F (30 days × 13 RTH bars) |
| Re-entry | `reentry_prob_threshold` | 0.55 | Sweep {0.50, 0.55, 0.60} | Heuristic |
| Re-entry | `reentry_cooldown_bars` | 1 | Fixed | Heuristic |
| Session | `session_flat_time_et` | 15:55 | Fixed | Gap-risk literature |
| Session | `session_eligible_time_et` | 09:35 | Fixed | Open-auction noise |
| Blackout | `blackout_minutes` | ±15 | Fixed | J. Banking & Finance 2024 |
| TP | `enable_signal_decay` | False (V1 default), enable only if val A/B shows ≥30bp lift | A/B | This doc |
| TP | `signal_decay_floor` | 0.7 | Sweep {0.5, 0.6, 0.7, 0.8} on val fold | Heuristic |
| Drawdown | tier 1 | -5% → halve | Fixed | doc 07 |
| Drawdown | tier 2 | -10% → quarter | Fixed | doc 07 |
| Drawdown | tier 3 | -15% → halt + 65-bar / -10% recovery rule | Fixed | doc 07 |

**Tuning protocol:** all sweeps run on **validation fold only**, never test. Joint sweep over `{hard_stop_mult, trail_stop_mult, reentry_prob_threshold}` = 3 × 3 × 3 = 27 combinations × 4 walk-forward folds. Pick the combination that maximizes Sharpe at **7-bp cost** (the stress case). Lock for OOS test.

---

## Ablation Plan (Mandatory)

Run these 8 variants on the same val fold with bootstrap CIs at the 95% level. Decision rule below the table.

| # | Variant | Sizing | Stops | Signal-decay | Description |
|---|---|---|---|---|---|
| 1 | Stage 1 baseline | fixed ±1 share | none | none | "always full size when not flat" — current doc 07 Stage 1 |
| 2 | + signed score | signed score, no Kelly, no vol-target | none | none | Isolates: does signed-score sizing alone help? |
| 3 | + vol-target | signed × kelly × vol_mult | none | none | Isolates: does vol-target add value? |
| 4 | + conformal | full sizing formula | none | none | Isolates: does conformal shrinkage add value? |
| 5 | + hard stop | full sizing | hard ATR only | none | Isolates: does hard stop alone help? |
| 6 | + trailing | full sizing | hard + trail | none | Isolates: does trailing add over hard alone? |
| 7 | + session/blackout/re-entry | full sizing | hard + trail + time + session + blackout + re-entry | none | The full V1 stack without signal-decay |
| 8 | + signal-decay | full sizing | full stops | enabled | Full V1 stack with optional signal-decay |

**Decision:**

- Ship variant 7 if its bootstrap-CI lower bound on Sharpe > variant 1's CI upper bound at 7 bp cost. Otherwise ship variant 1 (the simpler model that ties wins).
- Ship variant 8 over variant 7 only if variant 8's mean Sharpe ≥ variant 7's mean + 30 bp annualized at 7 bp cost.
- If variant 5 alone matches variant 7, drop trailing + signal-decay. Simplicity wins.
- If variant 7 fails to beat the GLD buy-and-hold Sharpe (~2.4 over 2020-2025), document honestly. Sharpe 2.0 with 8% MDD on a flat-or-bearish 5-year regime would be a different and equally publishable story.

---

## Open Questions For Owner Decision

1. **Quarter-Kelly (0.25) or half-Kelly (0.5) at V1 launch?** Quarter is the ML-uncertainty-literature default. Half is the historical default. Recommend quarter; the cost of being wrong on calibration is fatal at half.
2. **`target_vol = 0.10` or 0.15?** F2F uses 0.15 on gold *futures*. GLD is unleveraged spot, so 0.10 is the conservative analog. Recommend 0.10 V1 with A/B at 0.15 in V2.
3. **Session-flat at 15:55 ET, no overnight positions: confirm.** Saves us from the after-hours tail; gives up genuine overnight model edge. Recommend yes for V1.
4. **Enable signal-decay exit by default, or only ship if A/B wins?** Recommend "only ship if A/B wins by 30 bp." Defaults to disabled.
5. **News blackout entry-only, or also force-flat existing positions before scheduled events?** Recommend entry-only for V1. Force-flat is more conservative but loses any genuine event-driven model edge.
6. **Min-notional handling on rebalance:** if current = 0.05 share and target = 0.06 share at GLD $200 (delta = $2), submit the rebalance? Recommend yes — `min_notional` only gates entry, not rebalance.
7. **Exit price recording in vectorized backtest:** `stop_px` (worst case, conservative) or bar close (model's view)? Recommend `stop_px`. Conservative is the right bias for paper-to-live transition.

---

## Citations

### Sizing
- Forecast-to-Fill — arXiv:2511.08571 (Singha, Aguilera-Toste, Lahiri, Nov 2025) — the gold benchmark; w_t = f_t × σ_target / σ_t, λ_Kelly=0.40, σ*=15%
- Lim, Zohren, Roberts (2019) — arXiv:1904.04912 — Deep Momentum Networks; signed-forecast vol-target framework
- Zhang, Zohren, Roberts (2019) — arXiv:1911.10107 — DRL for trading; same framework
- Wright (2026) — arXiv:2601.07852 — Utility-Weighted Forecasting (no conformal, no "30%" claim — the doc 07 citation was fabricated)
- Kato (2024) — arXiv:2410.16333 — Conformal Predictive Portfolio Selection
- Utility-Directed Conformal Prediction — arXiv:2410.01767 — modified non-conformity score for utility
- Guo et al. (2017) — arXiv:1706.04599 — Temperature scaling for NN calibration
- MacLean, Thorp, Ziemba — Good and Bad Properties of Kelly — fractional Kelly variance/growth tradeoff
- Downey — fractional Kelly under uncertainty simulations
- Chalkidis et al. (2021) — arXiv:2110.14914 — Trading via selective classification

### Stops
- Kaminski & Lo (2014) — J. Financial Markets — when do stop-loss rules stop losses; momentum + regime stopping premium
- Han, Zhou, Zhu (2014) — SSRN:2407199 — taming momentum crashes; 10% stop doubles Sharpe
- Lo & Remorov (2017) — J. Financial Markets — stop-loss with serial correlation, regime, costs
- Wilder (1978) — New Concepts in Technical Trading Systems — ATR
- López de Prado (2018) — Advances in Financial Machine Learning — triple-barrier method (labels, not trading)
- LuxAlgo / Build Alpha / QuantStock practitioner consensus — 2× ATR for intraday
- J. Banking & Finance (2024) — gold's asymmetric FOMC reaction

### Profit-take
- Forecast-to-Fill — arXiv:2511.08571 — no profit target, trailing only
- Baur & Dimpfl — SSRN:4233700 — empirical refutation of fixed exit rules vs buy-and-hold
- CFM — "A Good Time for Trend Following" — positive-skew structure
- Hedge Fund Journal — "Making Fat Right Tails Fatter" — convexity / right-tail
- Man Group — "Trend Following Deep Dive" — empirical Sharpes by horizon

### Alpaca / Live
- Alpaca docs — Fractional Trading — $1 minimum, 9 decimal precision, DAY-only TIF, no bracket
- Alpaca docs — Working with Orders — alpaca-py BRACKET syntax (whole shares only)
- Alpaca docs — Intraday Margin Framework — PDT retired June 4, 2026
- Alpaca forum — fractional + bracket error 42210000

---

## Implementation Day Plan

| Hour | Task |
|---|---|
| 1 | `compute_target_position` + worked-example unit tests |
| 2 | `ATR(14)` Wilder + golden-value test |
| 3 | `StopState` class + hard / trail / time touch tests |
| 4 | `ReentryGate` + tests (stop-then-block, then-allow) |
| 5 | `is_within_session`, `must_force_flat`, `is_blacked_out` + calendar tests |
| 6 | `apply_exit_overlay` (vectorized two-pass) + test on synthetic |
| 7 | `resolve_live_target` (state-aware) + test_backtest_overlay_vs_live |
| 8 | Ablation runner (variants 1-8) + bootstrap CIs |
| 9 | Hyperparameter sweep (27 combos × 4 folds) |
| 10 | Lock V1 stack, write decision into STATUS.md |

Total: 1.5 days for a competent agent. Concurrent with doc 06 changes; sequential with doc 08 changes.

---

## Hand-off Protocol

1. Update STATUS.md with: ablation winner, hyperparameter sweep best-combo, val Sharpe at 7-bp cost.
2. Notify doc 06 owner that `apply_exit_overlay` is ready and stable.
3. Notify doc 08 owner that `resolve_live_target` is ready, state schema is published.
4. If `enable_signal_decay` ships False, document the val A/B numbers that supported the decision in this doc's "DEVIATION FROM SPEC" section.

Now go build.
