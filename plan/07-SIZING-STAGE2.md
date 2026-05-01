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

# Doc 06 (backtest) calls this per-bar:
size = stage2_sizing(
    probs: np.ndarray,                  # (3,) — softmax probs from model
    realized_vol_20d: float,            # annualized
    conformal_factor: float = 1.0,      # 1.0 high-conf, 0.5 medium, 0.0 low (from ConformalSizer)
    target_vol: float = 0.10,
    confidence_scale: float = 3.0,
    kelly_fraction: float = 0.5,
    position_limit: float = 1.0,
) -> float

# Doc 09 (live) imports same interface
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
2. Doc 06 imports your `stage2_sizing` for backtest comparisons
3. Doc 09 imports same function for live trading

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
