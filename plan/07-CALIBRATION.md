
---

# 07 — Confidence & Calibration (Spec)

**Status:** ✅ V1 spec, research-backed (3 parallel calibration agents 2026-05-04 — methods, conformal deep-dive, metrics + drift)
**Last verified:** 2026-05-04

## Why This Doc Exists

The plan covers calibration in fragments across 5 docs:
- **doc 05 / 05** mention label smoothing 0.1 + dropout + stochastic depth (training-time regularization that incidentally affects calibration).
- **doc 09** sketches temperature scaling + split conformal but cites a fabricated "30% lower decision loss" Wright 2026 number and uses naive split-CP.
- **doc 10** uses a 3-bucket discrete conformal shrinkage {1: 1.0, 2: 0.5, 3: 0.0} and signed score for sizing.
- **doc 11** has a single-signal drift detector (entropy z-score > 2 sigma + KL on argmax).

This doc consolidates and upgrades. Three parallel research agents on 2026-05-04 produced ~9000 words of analysis on:
- Which scalar best measures confidence for a 3-class weak-signal financial model (signed for sizing, MSP for gate, entropy for drift, set size for abstention, MC dropout decomposition for aleatoric/epistemic split).
- Which post-hoc calibration method (temperature primary, Dirichlet-ODIR fallback; skip Platt / isotonic / BBQ / matrix / vector / spline / Mix-n-Match).
- Which conformal variant (APS + Mondrian, not naive split-CP, not RAPS, not ACI in V1).
- Which metrics survive class imbalance (classwise AdaECE + macro Brier + NLL, not top-label ECE).
- How to monitor drift before realized PnL collapses (3-tier stack with named thresholds).

## Glossary

- **MSP:** max softmax probability = max(P_down, P_flat, P_up). Used as confidence gate post-temperature.
- **Signed score:** s = P_up - P_down ∈ [-1, +1]. Used as sizing magnitude × direction.
- **APS:** Adaptive Prediction Sets (Romano-Sesia-Candès 2020). Cumulative-sum non-conformity score.
- **Mondrian CP:** class-conditional conformal prediction. Per-class quantile, per-class coverage guarantee.
- **AdaECE:** Adaptive Expected Calibration Error. Equal-mass binning instead of equal-width. Roelofs 2020.
- **Classwise ECE:** ECE evaluated per class (one-vs-rest), then macro-averaged. Kull 2019.
- **Aleatoric uncertainty:** irreducible noise. E[H[p]] over MC samples.
- **Epistemic uncertainty:** model uncertainty. H[E[p]] - E[H[p]] (mutual information / BALD).
- **Snapshot ensemble:** average over the last K EMA checkpoints from a single training run. Huang 2017.

---

## PART A — The Confidence Scalar Layer

There is no single "confidence scalar" because different downstream consumers want different things. We publish a structured object:

```python
@dataclass(frozen=True)
class Confidence:
    signed_score: float         # P_up - P_down in [-1, +1]    — for sizing direction × magnitude
    msp: float                  # max(p) in [0, 1]              — for re-entry gate, abstention gate
    entropy: float              # -sum(p log p)                  — for drift monitoring
    set_size: int               # |C(x)| in {1, 2, 3}            — for abstention with coverage guarantee
    top_pvalue: float           # APS p-value of top class      — for continuous shrinkage
    aleatoric: float            # E[H[p]] from MC dropout       — for "halve size when ambiguous"
    epistemic: float            # mutual info from MC dropout   — for "abstain when out-of-distribution"
```

| Use case | Scalar | Threshold (V1) | Action |
|---|---|---|---|
| Sizing direction | signed_score | min_signed_signal = 0.05 | flat below |
| Sizing magnitude | abs(signed_score) × sigmoid(5 × (top_pvalue - alpha)) | continuous | scale Kelly |
| Re-entry gate | msp | >= 0.55 (post-T) | allow same-side re-entry |
| Abstention (hard gate) | set_size | >= 2 | flat |
| Aleatoric ambiguity | aleatoric z-score | > 1.5 (vs train baseline) | halve size |
| Epistemic OOD | epistemic z-score | > 2.0 (vs train baseline) | flat for this bar, alert |
| Drift monitoring | entropy z-score | > 2.5 warn / > 3.5 page | reduce / halt |

**The hard rule:** if any of set_size >= 2, epistemic_z > 2.0, or (aleatoric_z > 1.5 AND set_size = 2) fires, the bar is treated as abstain (size = 0). Failures are logged with the triggering scalar.

---

## PART B — Training-Time Calibration

### Loss Function (modifies doc 05 / 05)

```python
loss = F.cross_entropy(
    logits,
    labels,
    label_smoothing=0.05,        # was 0.1 — dropped per Müller arXiv:1906.02629
    weight=None,                 # do NOT class-weight — would solve recall, break calibration
    reduction='mean',
)
```

**Rationale:**
- LS=0.1 + Friendly-SAM + EMA 0.999 + dropout 0.2 + stoch depth 0.15 is **5 layers of regularization** on 16K labeled samples. Over-regularized for a weak-edge target.
- LS uniformly squashes the logit distribution (Müller-Kornblith-Hinton arXiv:1906.02629) — the same job temperature scaling does post-hoc, but tunable. Move that mass to T-scaling.
- LS=0.05 is a conservative halfway step. A/B option: LS=0.0. Pick the one that minimizes val-B NLL.
- Class weights would solve a different problem (recall on minority classes) at the cost of probability calibration. Class imbalance is handled by **stratified sampling in val-B** (T-fitter sees balanced classes) and **per-class reporting** (classwise AdaECE).

### Snapshot Ensemble (modifies doc 06)

After training the final model, save the last 3 EMA checkpoints from epochs N-2, N-1, N. At inference, average their softmax outputs. Free calibration improvement (Huang arXiv:1704.00109). No extra training compute.

### Mixup — DROP for V1

doc 06 mentions Manifold Mixup. Mixup IS a calibration improver (Thulasidasan arXiv:1905.11001) — input-space label smoothing. **But:** mixup on time-series financial features can leak future information across mixup pairs (mixing bars from different timestamps). Drop for V1 unless time-safe implementation is verified. Defer to V2 with explicit "no cross-time mixing" constraint.

---

## PART C — Post-Hoc Temperature Scaling

### Algorithm

```python
class TemperatureScaler(nn.Module):
    def __init__(self, init_T: float = 1.0):
        super().__init__()
        self.log_T = nn.Parameter(torch.tensor(float(np.log(init_T))))

    @property
    def T(self) -> torch.Tensor:
        return self.log_T.exp()

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.T

    def calibrated_probs(self, logits: torch.Tensor) -> torch.Tensor:
        return F.softmax(logits / self.T, dim=-1)

    def fit(self, val_logits, val_labels, max_iter: int = 50) -> float:
        opt = torch.optim.LBFGS(
            [self.log_T], lr=0.1, max_iter=max_iter,
            line_search_fn="strong_wolfe",
        )
        def closure():
            opt.zero_grad()
            loss = F.cross_entropy(val_logits / self.T, val_labels)
            loss.backward()
            return loss
        opt.step(closure)
        return float(self.T.item())
```

**Properties:**
- 1 parameter. LBFGS converges in < 50 iterations, < 1 second.
- Preserves argmax (monotone in logits).
- Empirical T values for calibrated transformers cluster in [1.0, 2.5]; T < 1 is rare and indicates underconfidence.
- Acceptance: max_T_per_fold <= 3.0 and min_T_per_fold >= 0.7. Outside that range = degenerate fit, escalate.

### Fitting On val-B (Stratified)

The val-B slice (25% of val month, ~750 bars) is **stratified by class** before T fits. Stratification guarantees the LBFGS sees DOWN, FLAT, UP in roughly equal measure, so T isn't tuned by the FLAT-dominated class.

### Fallback: Dirichlet-ODIR

Trigger only if post-T classwise-AdaECE > 0.10 on UP or DOWN. Kull et al. arXiv:1910.12656.

12 params (W 9 + b 3) with off-diagonal+intercept L2 regularization. Designed by the Dirichlet authors specifically for small calibration sets (~750 stratified bars). Don't use vanilla Dirichlet (overfits) or matrix scaling (worse: argmax flips under tiny perturbations).

---

## PART D — Conformal Prediction Recipe

### APS Score Function

Cumulative softmax mass up to and including the true class, in descending-probability order.

For each test point, sort classes by probability descending. The non-conformity score for class y is the cumulative sum from the top down to y. Calibration takes the (1-alpha)(1+1/n) quantile of calibration scores. At inference, include classes (in descending prob order) until cumulative mass exceeds q_hat.

### Why Mondrian Is Mandatory

With 70/15/15 imbalance, marginal CP at alpha=0.10 typically achieves:
- Marginal coverage: ~90%
- FLAT coverage: ~95% (over-covered)
- UP coverage: ~75% (under-covered)
- DOWN coverage: ~75% (under-covered)

UP and DOWN are the directions we trade. Mondrian uses 3 separate quantiles, each calibrated on its own class, and guarantees >= 1-alpha coverage per class. ~10 LOC for 3 quantiles. ~750 cal points per class on a 3000-bar val fold — well above the 50-point minimum (Ding et al. 2023).

### Why Not RAPS

RAPS regularizes APS to prevent giant sets in low-confidence regimes. Designed for K=1000 ImageNet where APS could include 50+ classes. With K=3, max set size is already 3. RAPS = APS when k_reg >= 1. Save the hyperparameter tuning.

### Why Not ACI in V1

ACI (Gibbs-Candès arXiv:2106.00170) auto-adapts alpha_t per bar. Strong theoretical guarantee. 10 LOC. **But** adds a misconfiguration failure mode: gamma too high → alpha oscillates → coverage thrashes; gamma too low → adapts too slowly. For a $100 account during V1 launch, the safer V1 is static split-CP recalibrated weekly + reactive on Tier-2 triggers. Add ACI in V2 after observing real coverage drift in shadow mode.

### Continuous Confidence Scalar

Set size on K=3 is discrete {1, 2, 3}. For a smoother sizing signal, use the **top-class APS p-value**: the fraction of calibration scores that are MORE non-conforming than the test score for the top class. Maps continuously from 0 (just barely included) to ~(1-alpha) (very confident).

Combined sizing replaces the doc 10 discrete shrinkage {1: 1.0, 2: 0.5, 3: 0.0} with:

```
if set_size == 0 or set_size == 3:
    shrinkage = 0.0
elif set_size == 1:
    shrinkage = 1.0 * sigmoid(5 * (p_top - alpha))
elif set_size == 2:
    shrinkage = 0.5 * sigmoid(5 * (p_top - alpha))
```

The discrete set-size acts as a hard gate; the p-value adds smooth magnitude inside each bucket.

---

## PART E — MC Dropout (Aleatoric vs Epistemic)

### Algorithm Sketch

T=20 forward passes with dropout enabled at inference. Stack T probability vectors (T, B, K). Compute:
- mean_p = mean over T → calibrated point estimate
- H_mean = entropy of mean_p → total predictive uncertainty
- E_H = mean over T of entropy(p_t) → expected entropy = aleatoric
- epistemic = H_mean - E_H → mutual information / BALD

### Cost on Macbook M4 Pro

- Single forward: ~80-150 ms (24-60M params, FP32, MPS).
- T=20: ~1.6-3 s per cycle. Easily fits the 30-min cycle budget.
- T=20 is the convergence elbow from Gal arXiv:1506.02142 §4.3 — variance estimate stable, returns diminish past T=50.

### Decomposition

H[E[p]] = aleatoric + epistemic = E[H[p]] + (H[E[p]] - E[H[p]])

- **Aleatoric high → mean prediction is high-entropy → "data is genuinely ambiguous."** Action: halve size via the conformal shrinkage path.
- **Epistemic high → MC samples disagree → "model is out of training distribution."** Action: abstain (size = 0). Different members predicting different classes is the classical OOD signal.

V1 baselines (calibrated on val of fold 0, replaced after 30 days of paper trading):
- aleatoric_z = (aleatoric - mu_train) / sigma_train > 1.5 → halve
- epistemic_z = (epistemic - mu_train) / sigma_train > 2.0 → abstain
- both fire → abstain + alert

### Combine With Snapshot Ensemble

The snapshot ensemble (3 EMA checkpoints) and MC dropout (T=20) are independent uncertainty estimates. Combine multiplicatively. Cost: 3 snapshot + 20 MC = 23 forward passes ≈ 2-3.5 s. Budget OK.

---

## PART F — Calibration Metrics

### Adaptive ECE (Equal-Mass Bins)

Instead of equal-width bins (which empty out where the softmax piles up), use 15 equal-mass quantile bins. Each bin holds ~N/15 samples. Roelofs 2020.

### Classwise AdaECE

Per-class AdaECE computed one-vs-rest, then macro-averaged. Worst-class also reported as a guardrail. Kull 2019.

### Macro Brier

Brier score = sum over classes of (p_k - 1{y=k})^2, averaged over samples. Random uniform (1/3, 1/3, 1/3) = 0.667 for K=3. Strictly proper scoring rule — cannot be gamed by uninformative outputs. Target macro-Brier < 0.62 (resolution must beat random by > 7%).

### Conformal Coverage

Empirical fraction of bars where the prediction set contains the true label. Target marginal >= 0.90 at alpha=0.10. Track per-class via Mondrian.

### Targets (V1)

| Metric | Offline target (per fold) | Live target (rolling) | Source |
|---|---|---|---|
| Classwise AdaECE | macro < 0.04, worst < 0.10 | — | Roelofs 2020; Kull 2019 |
| Macro Brier | < 0.62 (vs random 0.667) | < 0.66 (50-bar) | Murphy 1973 |
| NLL | per-fold logged, no fixed threshold | — | proper scoring rule |
| Marginal CP coverage | >= 0.88 (target 0.90) | >= 0.85 (200-bar) | Vovk |
| Per-class CP coverage | >= 0.83 each | >= 0.80 each | Mondrian |
| Macro-F1 | > 0.40 (vs random 0.27) | > 0.40 (5-day rolling) | imbalance |
| Set-size mean | < 1.6 | < 1.8 (200-bar) | informative model |
| Singleton rate | 30%-90% | 20%-95% | not collapsed / exploded |

---

## PART G — Drift Detection (3-Tier Stack)

Replaces doc 11's single-signal entropy z. Tiered by label-availability latency.

### Tier 1 — Unlabeled (Fires Same Cycle, No Label Needed)

| Signal | Window | Threshold | Action |
|---|---|---|---|
| Predictive entropy z | 60d baseline | abs(z) > 2.5 warn / > 3.5 page | warn → log; page → halve size |
| Argmax-mix KL | 1 day vs train prior | KL > 0.20 warn / > 0.50 halve | warn / halve size |
| MC-dropout variance z | 60d baseline | z > 3 | abstain (size=0) on this bar |
| Feature KL (post-RevIN) | 1 day vs 60d | KL > 0.30 | warn — covariate shift |
| Singleton rate (rolling 200-bar) | live | > 0.95 or < 0.20 | auto-pause for review |
| MSP distribution KS | 1 day vs train | p < 0.001 | warn |

### Tier 2 — Labeled (Fires Next-Cycle When Prior Bar Realizes)

| Signal | Window | Threshold | Action |
|---|---|---|---|
| Rolling Brier | 50 bars | > 0.66 (random) for 100 bars | recalibrate |
| Classwise reliability gap | 200 bars | max-class gap > 0.10 | recalibrate |
| Conformal marginal coverage | 200 bars | < 0.85 for 2 consecutive days | recalibrate q_hat; persists → halt new entries |
| Conformal per-class coverage | 200 bars | any class < 0.80 | recalibrate Mondrian; persists → halt |
| Mean set size | 200 bars | > 1.8 | model uninformative → halve |

### Tier 3 — Labeled (Fires Weekly Friday EOD)

| Signal | Window | Threshold | Action |
|---|---|---|---|
| Per-class precision | 5 trading days | UP-precision < 0.40 OR DOWN-precision < 0.40 | retrain trigger |
| Macro-F1 | 5 days | < 0.40 | retrain trigger after consecutive |
| AvUC | 5 days | < 0.55 | retrain trigger |
| Signed-score Sharpe | 5 days | < 0.05 (was > 0.10 in val) | retrain trigger |

### False-Positive Control

- Tier-1: require **2 consecutive days** of the same warn-level signal OR **2 different signals same day** before any size reduction. A single-cycle Tier-1 spike is logged-only.
- Tier-2: always reduces size 50% on first trip. Two consecutive trips → halt new entries.
- Tier-3: triggers full retrain queue.

### Lead-Time Ranking (Per Ovadia et al. arXiv:1906.02530)

1. Entropy z, MC-variance z (lead PnL drift by 1-3 days)
2. Argmax KL (leads by 0-1 day)
3. Coverage, rolling Brier (~coincident with PnL)
4. AvUC, realized accuracy (lagging)

Fastest signals get the most-aggressive thresholds because they cost the least to false-fire.

---

## PART H — Recalibration & Retrain Cadence

### Recalibration (Weekly Friday EOD)

Cost: < 10 seconds per fold. Scheduled task:

1. Fetch last 390 RTH bars of (logits, label) pairs from SQLite state.
2. Split 60% / 40% temporally: trailing 234 bars for T fit, leading 156 bars for q_hat fit.
3. Refit T_star via LBFGS on val-B-equivalent slice.
4. Refit q_marginal and q_per_class via APS on val-C-equivalent slice.
5. Persist new T and q values; old values archived for rollback.
6. Run a self-test: classwise AdaECE on the trailing 200 bars must be < 0.10. If fail, alert and KEEP old T/q.

### Reactive Recalibration (Triggered By Tier-2)

Same procedure but on detection of a Tier-2 trigger between scheduled refits.

### Retrain Triggers (Full Transformer Fit, ~12 h on Mac mini)

- **Scheduled:** every 6 months (walk-forward fold rotation).
- **Performance:** macro-F1 < 0.40 for 10 consecutive trading days.
- **Calibration:** 3 consecutive weekly recals fail to bring classwise AdaECE under 0.10 OR Brier under 0.66.
- **Distribution:** feature KL > 0.50 nats sustained 5d; OR argmax KL > 0.80 sustained 3d.
- **Edge:** signed-score correlation with realized return < 0.05 for 15d (was > 0.10 in val).
- **Risk:** drawdown circuit-breaker fires at -8% paper or -5% live (per doc 10).

### Don't Confuse The Two

- **Recalibration handles "model is right but probabilities shifted."** Refit T + q. Cheap. Weekly.
- **Retrain handles "model is wrong."** Refit weights. Expensive. Triggered.

Empirically Guo / Roelofs / Wright all show T-scaling fixes ECE drift up to ~0.15. Beyond that you need new weights.

---

## PART I — Val Fold Split Protocol

Currently val is used for: early stopping + temperature scaling + conformal q_hat + sizing hyperparameter scan. That's 4 things on one slice — overuse risk. The split:

```
Walk-forward window (3y train + 6mo val + 6mo test):
   train    : 3 years    — model weights
   val-A    : first 50%  — early stopping ONLY
   val-B    : next 25%   — temperature scalar T (stratified across classes)
   val-C    : last 25%   — conformal q_hat (Mondrian per-class quantiles)
   test     : 6 months   — reported metrics only, no tuning
```

Rationale:
- **Early stopping is the heaviest user of val** — it gets the largest slice.
- **T-scaling + conformal q_hat** can't share the same slice without inflating each other (Vovk's CP guarantee assumes calibration set is exchangeable with future test, distinct from any earlier tuning).
- **Sizing hyperparameters** (Kelly fraction, target_vol) tune on val-A + val-B *combined*, never test. Two scalars, leakage risk small.

For live trailing-window recalibration: use trailing 390 bars but split 60/40 (T then q_hat) preserving temporal ordering.

---


## PART J — Hyperparameter Table (Consolidated)

| Pillar | Param | V1 default | Source |
|---|---|---|---|
| Loss | label_smoothing | 0.05 | Müller arXiv:1906.02629 |
| Loss | class_weight | None | preserve calibration |
| Snap-ensemble | K | 3 (epochs N-2, N-1, N) | Huang arXiv:1704.00109 |
| MC dropout | T | 20 | Gal arXiv:1506.02142 §4.3 |
| MC dropout | aleatoric_z halve | 1.5 | this doc |
| MC dropout | epistemic_z abstain | 2.0 | this doc |
| Temperature | optimizer | LBFGS, lr=0.1, max_iter=50, strong-wolfe | Guo arXiv:1706.04599 |
| Temperature | sanity bounds | T in [0.7, 3.0] | this doc |
| Conformal | alpha | 0.10 (90% coverage) | this doc |
| Conformal | score | APS (deterministic) | Romano arXiv:2006.02544 |
| Conformal | mondrian | True (per-class) | Vovk 2012; Ding 2023 |
| Conformal | min cal points / class | 50 | Ding 2023 |
| Conformal | set_size = 1 shrinkage | sigmoid(5 * (p_top - alpha)) | this doc |
| Conformal | set_size = 2 shrinkage | 0.5 * sigmoid(5 * (p_top - alpha)) | this doc |
| Conformal | set_size = 3 or 0 shrinkage | 0.0 (abstain) | this doc |
| Metrics offline | classwise-AdaECE bins | 15 (equal-mass) | Roelofs 2020 |
| Metrics offline | classwise-AdaECE macro target | < 0.04 | this doc |
| Metrics offline | classwise-AdaECE worst-class | < 0.10 | this doc |
| Metrics offline | macro Brier target | < 0.62 (vs random 0.667) | Murphy 1973 |
| Metrics offline | macro-F1 floor | > 0.40 | imbalance |
| Metrics live | rolling Brier window | 50 bars | this doc |
| Metrics live | rolling Brier alarm | > 0.66 for 100 bars | random baseline |
| Metrics live | coverage window | 200 bars (~2 weeks) | Vovk |
| Metrics live | coverage floor | 0.85 (target 0.90) | margin |
| Drift T1 | entropy z warn / page | 2.5 / 3.5 | gaussian tail |
| Drift T1 | argmax-KL warn / halve | 0.20 / 0.50 nats | this doc |
| Drift T1 | MC-var z abstain | 3.0 | Krishnan 2020 FP budget |
| Drift T1 | feat-KL warn | 0.30 nats | this doc |
| Drift T2 | per-class coverage | < 0.80 | this doc |
| Drift T3 | per-class precision | < 0.40 | random=0.33 |
| Drift T3 | macro-F1 retrain | < 0.40 for 10d | imbalance |
| Drift T3 | feat-KL retrain | > 0.50 sustained 5d | this doc |
| Drift T3 | argmax-KL retrain | > 0.80 sustained 3d | this doc |
| Drift T3 | signed-Sharpe retrain | < 0 for 15d | this doc |
| Recalibration | cadence | weekly Friday EOD | this doc |
| Recalibration | window | trailing 390 bars (~30d) | Bandi 2008 |
| Recalibration | T/q split within window | 60/40 temporal | leakage budget |
| Retrain | scheduled | 6 months (fold rotation) | walk-forward CV |
| Retrain | recal-fail consecutive | 3 | this doc |
| Val split | val-A early-stop | 50% | this doc |
| Val split | val-B T scaling | 25% | this doc |
| Val split | val-C conformal | 25% | this doc |
| Val split | val-B stratification | balanced 250 / class | imbalance |
| (V2) ACI | gamma | 0.01 | Gibbs-Candès 2021 |
| (V2) Deep ensembles | members | 5 | Lakshminarayanan 2017 |

---

## PART K — Ablation Plan

Run on the same val fold with bootstrap CIs. Decision rules below.

| # | Variant | Loss | Post-hoc | Conformal | Uncertainty | Live drift |
|---|---|---|---|---|---|---|
| 1 | Baseline doc 05/05/07 | CE+LS=0.1 | none | naive split-CP | none | entropy z only |
| 2 | + LS=0.05 | CE+LS=0.05 | none | naive split-CP | none | entropy z only |
| 3 | + Temperature | CE+LS=0.05 | T-scaling | naive split-CP | none | entropy z only |
| 4 | + APS | CE+LS=0.05 | T-scaling | APS marginal | none | entropy z only |
| 5 | + Mondrian | CE+LS=0.05 | T-scaling | APS Mondrian | none | entropy z only |
| 6 | + MC dropout | CE+LS=0.05 | T-scaling | APS Mondrian | MC T=20 | entropy z only |
| 7 | + Snap ensemble | CE+LS=0.05 | T-scaling | APS Mondrian | MC + snap | entropy z only |
| 8 | + 3-tier drift | CE+LS=0.05 | T-scaling | APS Mondrian | MC + snap | 3-tier |
| 9 | + Dirichlet-ODIR fallback | CE+LS=0.05 | T+Dirichlet | APS Mondrian | MC + snap | 3-tier |

Decision:
- Ship variant 8 (full V1 stack without Dirichlet) if classwise-AdaECE macro < 0.04 AND worst < 0.10 on val.
- Escalate to variant 9 only if 8 fails the worst-class threshold on UP or DOWN.
- If variant 5 alone matches variant 8 on all calibration metrics AND on val Sharpe (via doc 10 sizing path), drop MC dropout + snap ensemble + Tier-1 epistemic gate from V1. Simplicity wins.
- If variant 3 alone matches variant 5 on coverage AND val Sharpe, drop Mondrian. (Unlikely given imbalance.)

---

## PART L — Acceptance Gate For V1 Calibration Pipeline

To ship V1 calibration, ALL must pass on every walk-forward fold:

1. ✅ T fits within sanity bounds (T in [0.7, 3.0]).
2. ✅ classwise-AdaECE macro < 0.04, worst-class < 0.10.
3. ✅ Marginal CP coverage on test fold >= 0.88 (target 0.90, 2pp margin).
4. ✅ Per-class CP coverage >= 0.83 each (1pp margin).
5. ✅ Macro Brier < 0.62.
6. ✅ Macro-F1 > 0.40.
7. ✅ MC dropout aleatoric + epistemic decomposition checks (sum to total predictive entropy +/- 1e-5).
8. ✅ End-to-end calibration latency < 100 ms per bar (excluding MC dropout).
9. ✅ MC dropout latency < 5 s per cycle.
10. ✅ Snapshot ensemble adds < 1 s per cycle.
11. ✅ Drift monitor unit tests all pass.
12. ✅ Friday recalibration job runs end-to-end on a synthetic week.

If any fail on any fold, document under `## DEVIATION FROM SPEC: <date> - <issue>` at the top of this doc and AskUserQuestion.

---

## PART M — Open Questions For Owner Decision

1. **LS=0.05 or LS=0.0?** Recommend 0.05 as conservative halfway. A/B on val NLL.
2. **MC dropout T=20 or T=30?** T=20 is the convergence elbow per Gal 2016; T=30 is more conservative but adds 50% latency. Recommend 20.
3. **Snapshot ensemble size: 3 or 5 EMA checkpoints?** 3 captures most of the gain (Huang 2017). 5 = more storage, more inference cost. Recommend 3.
4. **Mondrian or marginal CP for V1?** Recommend Mondrian — minority class coverage matters for trading. ~10 LOC cost.
5. **Recalibration window: 390 bars (30d) or 195 bars (15d) or 780 bars (60d)?** Recommend 390. 30d is the empirical sweet spot for intraday financial models per Bandi/Mykland.
6. **ACI in V1 or V2?** Recommend V2. V1 stays static-recal. Re-evaluate after first 3 months of paper trading.
7. **Dirichlet-ODIR auto-trigger or manual?** Recommend auto-trigger if classwise-AdaECE worst > 0.10 on UP/DOWN after T-scaling.
8. **Retrain trigger thresholds — calibrate now or after first 30 days of paper trading?** Recommend lock the structure now, recalibrate the exact threshold values on day 31 from observed baselines.

---

## Citations

### Calibration (training-time + post-hoc)
- Müller, Kornblith, Hinton — "When Does Label Smoothing Help?" — arXiv:1906.02629
- Mukhoti et al. — "Calibrating Deep Neural Networks using Focal Loss" — arXiv:2002.09437
- Guo, Pleiss, Sun, Weinberger — "On Calibration of Modern Neural Networks" — arXiv:1706.04599
- Kull et al. — "Beyond temperature scaling: Obtaining well-calibrated multi-class probabilities with Dirichlet calibration" — arXiv:1910.12656
- Hendrycks, Gimpel — "A Baseline for Detecting Misclassified and Out-of-Distribution Examples" — arXiv:1610.02136
- Thulasidasan et al. — "On Mixup Training: Improved Calibration and Predictive Uncertainty" — arXiv:1905.11001

### Conformal Prediction
- Romano, Sesia, Candès — "Classification with Valid and Adaptive Coverage" (APS) — arXiv:2006.02544
- Angelopoulos, Bates, Malik, Jordan — "Uncertainty Sets for Image Classifiers using Conformal Prediction" (RAPS) — arXiv:2009.14193
- Gibbs, Candès — "Adaptive Conformal Inference Under Distribution Shift" (ACI) — arXiv:2106.00170
- Bhatnagar, Wang, Xiong, Bai — "Improved Online Conformal Prediction via Strongly Adaptive Online Learning" (SAOCP) — arXiv:2302.07869
- Bhatnagar, Wang — auto-tuned ACI — arXiv:2208.08401
- Angelopoulos, Bates — "A Gentle Introduction to Conformal Prediction" (tutorial) — arXiv:2107.07511
- Vovk — "Conditional validity of inductive conformal predictors" — ACML 2012
- Ding, Angelopoulos, Bates, Jordan, Tibshirani — "Class-Conditional Conformal Prediction with Many Classes"
- Kato — "Conformal Predictive Portfolio Selection" — arXiv:2410.16333
- "A Gentle Introduction to Conformal Time Series Forecasting" — arXiv:2511.13608
- **Wright (2026), arXiv:2601.07852 — confirmed misattributed in plan doc 09. Has zero conformal content.**

### Uncertainty Estimation
- Gal, Ghahramani — "Dropout as a Bayesian Approximation" — arXiv:1506.02142
- Lakshminarayanan, Pritzel, Blundell — "Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles" — arXiv:1612.01474
- Kendall, Gal — "What Uncertainties Do We Need in Bayesian Deep Learning?" — arXiv:1703.04977
- Depeweg et al. — "Decomposition of Uncertainty in Bayesian Deep Learning" — arXiv:1710.07283
- Huang et al. — "Snapshot Ensembles: Train 1, Get M for Free" — arXiv:1704.00109
- Folgoc et al. — "Is MC Dropout Bayesian?" — arXiv:2102.08501

### Metrics + Drift
- Nixon et al. — "Measuring Calibration in Deep Learning" — arXiv:1904.10683
- Roelofs et al. — "Mitigating Bias in Calibration Error Estimation" — arXiv:1909.10155
- Vaicenavicius et al. — "Evaluating Model Calibration in Classification"
- Kumar et al. — "Verified Uncertainty Calibration"
- Krishnan, Tickoo — "Improving model calibration with accuracy versus uncertainty optimization" (AvUC) — arXiv:2007.10546
- Ovadia et al. — "Can You Trust Your Model's Uncertainty?" — arXiv:1906.02530
- Murphy 1973 — "A New Vector Partition of the Probability Score" — Brier decomposition

---

## Implementation Day Plan

| Hour | Task |
|---|---|
| 1 | TemperatureScaler class + test_temperature_argmax_preserved |
| 2 | metrics.py: AdaECE + classwise-AdaECE + macro Brier + NLL + tests |
| 3 | APSConformal (marginal) + test_aps_coverage |
| 4 | APSConformal Mondrian + test_mondrian_per_class_coverage |
| 5 | mc_dropout_predict + test_mc_dropout_decomposition |
| 6 | snapshot_ensemble + test_snapshot_ensemble_avg |
| 7 | DriftMonitor Tier 1 + test_drift_tier1_thresholds |
| 8 | DriftMonitor Tier 2 + Tier 3 + tests |
| 9 | val split protocol + test_val_split_no_leakage |
| 10 | Friday recalibration job + integration test |
| 11 | Live cycle integration patch (cycle.py) |
| 12 | Ablation runner (variants 1-9) + bootstrap CIs |

Total: 1.5 days for a competent agent. Concurrent with doc 10 changes; sequential with doc 11 changes.

---

## Hand-off Protocol

1. Update STATUS.md with: T per fold, classwise-AdaECE per fold, marginal + per-class coverage per fold, MC dropout baseline distributions, Friday recal job status.
2. doc 05 + doc 06: update LS to 0.05 (one-line change in loss config).
3. doc 09: delete Wright 2026 conformal claim (this doc finalizes; doc 10 already removed the fabricated number).
4. doc 10: update sizing layer to read calibrated probs from this pipeline instead of raw softmax. Replace {1:1.0, 2:0.5, 3:0.0} with set_size_gate × sigmoid(5 × (p_top - alpha)).
5. doc 11: replace single-signal drift detection with the 3-tier DriftMonitor; add Friday EOD launchd plist for recalibration.

Now go build.
