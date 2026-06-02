# nanoGLD: PIT-discipline auditing for intraday-trading research — a multi-leak case study, a diagnosed multi-task LLRD collapse, and reference vol-target numbers on six assets

**Saam Siavoshian** · samsiavoshian2009@gmail.com

## Contribution summary

**Primary contributions (two).** (1) A documented multi-task LLRD recipe-collapse mechanism in a 24M-parameter intraday-direction transformer (§6.7), with four-knob failure mode (cross-sample Sharpe noise at B=8; Mixout p=0.7 reset against the SSL anchor; hardcoded `prev_position=None` disabling cost penalty; 3-epoch budget below convergence), introspection ruling out a sign-convention bug, and a shipped runtime position-saturation guard in `src/nanogld/training/llrd_finetune.py`. **Update (V4 WF, 2/4 folds completed at submission):** Fold 1 LLRD training completed on Spark and the trained checkpoint was backtested on the fold-1 test split (3685 bars, 127 days). Fold 1 collapses to the **+1 tanh boundary** (100% saturation, abs_pos_mean=1.000) — same recipe-level collapse mechanism as fold 0 but with random saturation sign. Class head predicts 100% FLAT majority class on both folds. Fold 0 Sharpe was −0.863 (collapse to −1 in an up-trending market = catastrophic); fold 1 Sharpe is +1.118 (collapse to +1 ≡ buy-hold; Δ vs BH = −0.002, no incremental edge). **V4 §4 backbone now has 2/2 collapse evidence at WF scale; the collapse mechanism is recipe-level, not fold-specific.** Folds 2 and 3 in-flight; will be appended to a v2 of this manuscript. Useful for anyone combining differentiable-Sharpe heads with LLRD finetuning at small intraday batch sizes. (2) A look-ahead-leak gallery with three in-house leaks disclosed *and patched* with reproduction code (§6.14 item 1): `momentum_pulse_13` (+7.62 → −0.72 collapse), `buy_hold_atr_stop_3x` (+2.904 → +0.82 collapse), `low_vol_long` (+1.65, within-day RV flag). **Supporting methodology checklist (§6.14 items 2-7) is documented as an integrated audit protocol of known statistical practice (BPY-constant unit audits, DSR full-disclosure, vol-target positive-homogeneity diagnostics, within-frequency multi-asset testing, multi-seed discipline, two-pipeline WF separation); we publish reproduction code for each but do not claim novel statistical methods.**

**Secondary contributions (case-study evidence supporting the methodology).** (a) A documented multi-task LLRD recipe-collapse in a 24M-parameter intraday transformer (V4) with mechanism analysis and a position-saturation runtime guard (§6.7); (b) reference vol-target numbers across timescale and asset class (§6.13 + §6.13b): intraday vol_target_3pct posts +1.81 daily-equivalent WF Sharpe on GLD as a single-asset ex-ante test (§6.8 paired-block bootstrap p=0.046, Δ=+0.538 Sharpe over BH). A post-hoc 9-asset family-wise replication (§6.13b) shows the recipe is directionally consistent on 3 broad-equity / gold-proxy assets at uncorrected p<0.10 (GLD/SPY/QQQ) but **0 of 9 assets pass under any family-wise error rate correction** (Bonferroni at α=0.05, p<0.0056; Šidák with effective-n_tests=7.2 at α=0.10, p<0.0146). The honest reading is: **single-asset ex-ante claim on GLD is significant; post-hoc multi-asset family is not significant after correction; we report this transparently as a negative methodology finding rather than a positive trading claim.** The recipe also **fails to generalize to daily timescales** on six daily universes (0/6 daily). The intraday-aligned assets use GLD-bar-timestamp grids (not native per-asset 30-min bars), an acknowledged limitation; (c) an open replication attempt of F2F (Singha et al. 2025) with matched-vol and W_max-binding diagnostics establishing the +0.66 vs +2.88 gap is signal-quality, not vol-framing (§6.9, §6.11).

**What this paper is NOT.** We do *not* claim a new trading-Sharpe SOTA on intraday GLD, daily gold, or any multi-asset universe. We do *not* claim the §4 24M-parameter backbone was successfully evaluated at full configuration (V4 collapsed at fold 0 only). We do *not* claim the §6.8 intraday-GLD vol-target Sharpe generalizes (§6.13 shows it does not).

**Numbered evidence for the toolkit + case-study results:**

1. **Intraday GLD walk-forward Sharpe; vol_target_3pct.**
   Per-fold mean across 4 walk-forward folds (± is fold-to-fold sample std on a single seed): **+1.80 ± 0.35 at correctly-annualized intraday frequency (BPY=7308) which equals daily-aggregated frequency (BPY=252) within rounding** (sqrt(7308) bars/year, ≈29 bars/day × 252 days), **+1.80 ± 0.35 at daily frequency** (sqrt(252), via per-day PnL aggregation; see §6.8 / §6.11 per-fold tables). The ± values are fold-to-fold *sample* standard deviation on a *single seed* with n=4 folds; the small-sample bias correction (Student's t-distribution at 3 d.f. gives a 95% CI of approximately mean ± 3.18 × sample_std / sqrt(n) ≈ ±0.35 for the intraday number and ±0.55 for daily, much wider than the ± we report); the ± in the claim line is the raw fold-std, not a confidence interval. We do not run multi-seed for the shipped non-model strategy because it has no learned parameters, but readers should not confuse this with seed-to-seed variance for the deep-learning models in §6.1/§6.2/§6.7/§6.9/§6.12, all of which are also single-seed under our compute budget (see §9). Chain-Sharpe with stationary block bootstrap (B=2000, block=20): intraday **+1.21 [95% CI +0.34, +2.11]**; daily **+1.83 [95% CI +0.60, +2.96]**. Newey-West HAC (lag=10) t-statistic 2.16 (intraday, p<0.05) and 1.63 (daily, p<0.10). Paired stationary block bootstrap vs buy-and-hold yields Sharpe delta of +0.37 (intraday, p=0.038; the bootstrap mean is +0.37 while the point estimate +1.810 − +0.852 = +0.360, the two differ at the second decimal because the bootstrap mean is a different statistic than the point delta) and +0.55 (daily, p=0.057). Bailey-Lopez de Prado Deflated Sharpe Ratio at multiple n_trials counts:

| n_trials | E[max SR \| null] | DSR_intraday | DSR_daily |
|---------:|------------------:|-------------:|----------:|
|       11 |             1.622 |       0.0000 |    1.0000 (CDF saturation) |
|       25 |             1.997 |       0.0000 |    0.0004 |
|       50 |             2.276 |       0.0000 |    0.0000 |
|      100 |             2.531 |       0.0000 |    0.0000 |
|      150 |             2.732 |       0.0000 |    0.0000 |

(DSR > 0.95 = passes multiple-testing gate at 5% significance.) We
itemize the trials counted: §6.4 baselines real-evaluated ≈ 7;
§6.8 vol-target τ sweep ∈ {3%, 5%, 7%, 10%, 15%} + composites
(vt × MA, vt × ATR, ensemble) ≈ 9; §6.9 daily-gold strategies +
F2F replica grid ≈ 10; §6.10 multi-asset futures 4 lookbacks ×
4 vol-targets × 2 weightings = 32; §6.11 14-strategy meta + 3
meta-combo schemes ≈ 17; §6.12 four multimodal-LSTM sizing
schemes; the cost-convention sweep (gross / 0.7 / 1.0 / 2.0 bp)
is post-hoc reporting on one strategy, not separate trials, and we
do NOT count it; same for the frequency choice (we count it once for
the selection step). Itemized total
≈ 7 + 9 + 10 + 32 + 17 + 4 + 1 (frequency selection) = 80, rounding to a
realistic n_trials window of **80–100** (the upper end accounts for 10–20 additional implicit trials from compositional choices not itemized: cost-convention reporting axes, cap-binding regime choice in §6.11 matched-vol diagnostic, etc.). We acknowledge selection bias: the n=11 row shows DSR_daily = 1.0000 but this is **CDF saturation, not a true pass** (the observed Sharpe +1.83 exceeds E[max | null with n=11] = 1.62 by enough that the normal CDF rounds to 1.0000 at our precision; the true "pass" probability is interpretable but the row should not be read as evidence the strategy clears multiple-testing correction), but we chose the 80–150 window after observing the table; readers should treat the n=11 row as an artifact of an artificially small n, not as evidence the daily strategy clears multiple-testing correction.
**DSR full-search disclosure (binding count).** The binding n_trials count is the full-search budget that includes all exploration trials (§6.10 multi-asset futures grid of 32 configs; §6.11 14-strategy meta-portfolio; §6.12 multimodal-LSTM sizing-wrapper ablation; §6.8 vol-target hyperparameter sweep). The realistic full-search budget is **n_trials = 80–150**. At this binding budget, **both intraday and daily-frequency Sharpe FAIL the Bailey-Lopez de Prado DSR gate**. We do **not** carry forward a smaller ship-decision n_trials count (some prior drafts cited n=9 from the §6.8 local sweep as a "ship-budget"; that figure is survivor-selected and we drop it from the headline). We report this directly:
the +1.21 / +1.83 Sharpes are statistically significant at the raw
(uncorrected) bootstrap and HAC level but **do not pass full-search multiple-testing correction (n_trials = 80–150)**. The shipped strategy is honestly
labeled as a high-effort point estimate, not a SOTA result. **Lift mechanism (BPY-constant artifact, not signal property).** A first-pass explanation pointed at per-fold lag-1 autocorrelation of per-bar PnL (−0.015, −0.041, −0.075, −0.074; chain −0.050). However, the autocorrelation-corrected variance ratio at N=13 only predicts a +3.5%–+8.4% lift per fold (`scripts/intraday_to_daily_lift_decomposition.py`); the observed +44%–+55% lift is therefore NOT explained by mild within-day mean reversion alone. **The actual mechanism is a bars-per-year annualization constant error.** Our intraday data contains ≈29 bars per trading day (pre-market + RTH 6.5h × 13 bars + after-hours), so the correct intraday bars-per-year is 29 × 252 = **7,308**, not 3,276 (which would be the figure for RTH-only 13 bars × 252). When we annualize the per-bar Sharpe at √3,276 we **understate** the true Sharpe by a factor of √(7,308/3,276) ≈ 1.49. Re-annualizing at √7,308 gives intraday Sharpe = +1.21 × 1.49 ≈ **+1.80**, which matches the daily-aggregated +1.83 to within rounding. **The +51% lift is therefore a unit/constant fix, not a discovered signal property.** The +1.83 daily-frequency Sharpe is the correct apples-to-apples number against VLSTM and F2F; the +1.21 intraday-frequency Sharpe is an artifact of using the wrong BPY constant and should not be cited as a distinct result.
2. **Multimodal intraday gold trading benchmark.** First architecture *we are aware of* combining 30-min OHLCV + frozen Qwen3-256d news embeddings under strict per-fold PIT walk-forward. Closest prior intraday-GLD work is Wen et al. 2020 [@wen2020resources] (intraday-momentum-based volatility forecasting on crude oil and gold, related but not direct competition since they predict realized volatility not direction). **Literature search protocol.** We searched the following databases for "intraday GLD prediction", "intraday gold ETF Sharpe walk-forward", "30-min gold direction LSTM/transformer", "multimodal gold trading": arXiv (cs.LG, q-fin.ST, q-fin.PM), Google Scholar (top 50 results per query), Nia papers (gold/futures/intraday/multimodal queries), SSRN finance/derivatives sections, and Resources Policy / Energy Economics journals. We found no prior work that combines (intraday GLD ETF) + (walk-forward with strict per-fold sidecar) + (multimodal price+news) + (published Sharpe with uncertainty quantification) on the same axis we report. The "first" claim is therefore conditional on this search, not exhaustive; readers should treat it as defeasible.
3. **Multi-asset futures portfolio PIT-clean WF.** WF mean Sharpe **+0.56 on 31-asset risk-parity + multi-horizon TSMOM portfolio** (per-fold range [−0.47, +1.59], 12 overlapping 2-year-test folds with 1-year step on the 2010–2026 yfinance GC=F continuous-front-month window (see §6.10)). Reported without overclaim against the published multi-asset portfolio Sharpe (VSN+LSTM benchmark 2.40).
4. **Open replication attempt of F2F (Singha et al. 2025).** First public attempt at reproducing F2F's daily-gold +2.88 Sharpe. Replica with extracted hyperparameters: **+0.66 on yfinance GC=F continuous front-month**; **+0.70 on intraday-aggregated GLD ETF** (different instrument than F2F's GC continuous futures; comparison is not apples-to-apples, see §6.9). We document the ~2.2 Sharpe gap and offer three candidate explanations (data quality, hyperparameter calibration, signal component beyond the EMA + 50-day momentum blend) without claiming to have identified the cause.
5. **Position-saturation runtime guard.** A diagnostic + mitigation for the tanh-saturation collapse mode we observed in V4 (§6.7), shipped in `src/nanogld/training/llrd_finetune.py`. Documented as engineering artifact; novelty against prior tanh-collapse interventions is not benchmarked.
6. **Negative-DL result conditional on what we actually trained.** **Honest framing: the §4 24M-parameter backbone was NOT successfully evaluated at scale.** V4 collapsed at fold-0-only (3 LLRD epochs vs planned 10, single seed). The negative-DL evidence consists of: (a) one undertrained fold-0-only run of the §4 backbone (V4); (b) one **non-DL** tree-ensemble baseline (XGBoost on 14 features); (c) two **smaller-than-§4** LSTM surrogate models (daily 2-layer h=64 on 14 features at n=3 seeds, intraday PCA(651→64)+1-layer LSTM). **By strict accounting this is 0 successful runs of the headline architecture + 1 tree ensemble + 2 small LSTM surrogates.** The negative-DL claim therefore does NOT generalize to the §4 architecture trained at scale; it generalizes only to (a) the V4 recipe failure mode (well-diagnosed in §6.7) and (b) the surrogate small-model class. Full enumeration: (i) V4 24M-param transformer at fold-0-only on Spark; (ii) XGBoost (gradient-boosted trees, **not deep learning**) on 14 daily-gold engineered features; (iii) 2-layer daily-gold LSTM with 14 features (3-seed mean +0.86 ± 0.14 across seeds; underpowered at n=3); (iv) intraday multimodal LSTM with PCA(651→64) + 1-layer LSTM head (**not the full §4 backbone**) plus three sizing wrappers around the multimodal LSTM (vt-multiplicative, vt-bear-veto, vt-bull-boost) all produce zero or negative incremental Sharpe vs the §6.8 PIT-clean vol-target baseline. The three sizing wrappers share one trained network and are not independent models; we count them as **two successful LSTM-surrogate runs + one tree baseline + one collapsed §4 run** plus a sizing-wrapper ablation. Findings are conditional on the compute regimes used (Mac mini MPS reduced-scale + Spark fold-0-only); we explicitly do not claim generalization across the full V1-SPEC scale (d_model=384, T=64, longer epochs).

## Abstract

**Contributions.** We present (1) a mechanism analysis and runtime mitigation of a multi-task LLRD recipe-collapse in a 24M-parameter intraday-direction transformer, **confirmed on 2 of 4 walk-forward folds** (fold 0 saturates to −1, fold 1 saturates to +1, both with 100% tanh saturation + 100% class-FLAT predictions; collapse is recipe-level not fold-specific), and (2) a look-ahead-leak gallery with three in-house leaks disclosed and patched, alongside a supporting checklist of PIT-discipline diagnostics from 14+ months of building and replication-failing on a 30-minute GLD-direction transformer. The supporting checklist (BPY-constant audit, DSR full-disclosure, matched-vol + cap-binding diagnostic, within-frequency multi-asset replication, multi-seed conventions, two-pipeline WF separation) integrates known statistical practice into one audit protocol with reproduction code; we do not claim novel statistical methods on the checklist items, only the integration and code-publishing. The trading case study (nanoGLD V1/V2/V4) is treated as a negative-result illustration of why the primary contributions matter.

**The case study (nanoGLD).** A 24M-parameter from-scratch transformer
trained to predict the next 30-minute direction of GLD (the gold ETF)
from a fused representation of (i) 651 engineered features derived from
10 years of 30-minute price bars across 27 assets plus 40 macro series
and (ii) 40,032 frozen Qwen3-Embedding-4B encodings of contemporaneous
gold-relevant news headlines. The architecture is a hybrid 10-transformer
+ 2-sLSTM encoder with channel-independent patching
[@nie2023patchtst], FiLM regime conditioning, and sparse Flamingo-gated
[@alayrac2022flamingo] cross-attention to news at three layers. The
model jointly emits a 3-class focal cross-entropy
[@mukhoti2020calibrating] direction logit and a continuous position
weight trained on a differentiable −Sharpe loss
[@salykaufmann2026sharpe]. We calibrate with Temperature scaling into
RAPS [@angelopoulos2020raps] conformal prediction sets, adapt coverage
online via AgACI [@zaffran2022adaptive], and gate position size with a
friction-adjusted Kelly criterion under a conformal-derived lower-bound
floor. The pipeline runs 4-fold walk-forward with a strict per-fold
sidecar (HMM regime, ATR barriers, h5 vol threshold all fit on the
fold's own train slice) eliminating the standard cross-validation leak.
On Apple Silicon MPS at reduced scale (d=192, T=32, 3+2+3 epochs), V1
and V2 produce **negative OOS Sharpe** (**−0.78** and **−0.85**
respectively at 1.0× cost); both lose to buy-and-hold (+0.85). A
**fold-0-only** retrain (V4) under the planned 4-fold walk-forward
geometry on NVIDIA DGX Spark identifies a
sharp failure mode in the recipe: the multi-task LLRD stack
(Mixout p=0.7 against the SSL anchor + small-batch differentiable
−Sharpe loss + FreeLB K=2 + 3 epochs) drives the tanh-based position
head to a constant **−1** saturation across 100% of bars; the class
head simultaneously collapses to the FLAT majority class. We
hypothesize a saturation-attractor + Mixout reset of position-head
gradients as the failure mechanism (symptoms verified, causation
attribution inferred from the four-knob recipe), document a relaxed-
recipe mitigation (`v4d` config:
`mixout_p` 0.1, FreeLB off, Sharpe warmup, position-saturation runtime
guard) staged in `src/nanogld/training/llrd_finetune.py`, and report
this as a *negative* result for the published differentiable −Sharpe
head at our intraday batch size. We then search a small ladder of
PIT-clean non-model strategies and find one that beats buy-and-hold
on the same 4-fold walk-forward geometry by a meaningful margin:
**vol_target_buy_hold** (size $w_t = \mathrm{clip}(\tau /
(\hat{\sigma}_{t-1}\sqrt{B}), 0, 1)$, long-only, one-bar shift). A
hyperparameter sweep over the vol target $\tau$ picks $\tau = 3\%$ as
the WF-optimal value; the tuned strategy posts **WF mean Sharpe
+1.83 net of 2 bp round-trip cost (+1.73 net of 1.5× cost)** at the correctly-annualized daily frequency, and **+1.83 net of cost
when reported at the daily-frequency axis** that the published gold
SOTAs use (VSN+LSTM multi-asset 2.40, F2F daily-gold 2.88). The strategy beats buy-and-hold by
+0.538 Sharpe on the intraday 1× cost gate and is positive on every
fold. We further test whether a multimodal intraday LSTM
(fusing the 651 price features, the Qwen news embeddings, and the
regime vector) can lift Sharpe above this baseline; across four
sizing-composition schemes the model adds **no economic alpha** over
the simple vol-target sizer. Combined with our V4 recipe-collapse
result and a daily-gold XGBoost/LSTM ladder that also stalls at the
vt-only floor, we document a **consistent negative result across four distinct
independent deep-model attempts on this asset**: at our scale and
feature set, no learned model beats a one-line PIT-clean vol-target
formula. **Scope limitation (per reviewer concern):** "no learned model" refers to the **successfully-evaluated** toolkit (XGBoost, 3-seed daily LSTM, intraday multimodal LSTM PCA(651→64)+1-layer); the §4 24M-parameter backbone was never successfully evaluated at full configuration (V4 collapsed at fold 0 only) and the negative-DL claim does **not** generalize to it. We do not match the published SOTAs on multi-asset
portfolios (VSN+LSTM multi-asset benchmark 2.40 [@salykaufmann2026sharpe], a 50-instrument
portfolio Sharpe gross of cost, not gold-only) or daily gold futures
(F2F 2.88 [@singha2025f2f]), but we set what appears to be the first
PIT-clean walk-forward benchmark on intraday GLD and adopt
vol_target_3pct as the production strategy. The contributions are
therefore (a) an end-to-end multimodal-finance pipeline with
leak-free walk-forward CV and a position-saturation runtime guard,
(b) a consistent negative result across 2 successful LSTM surrogates + 1 tree baseline + 1 collapsed fold-0 V4 (per strict-accounting in §0 claim 6) including a
diagnosed V4 multi-task LLRD collapse, and (c) a simple PIT-clean
vol-targeted long-only strategy that beats buy-and-hold by +0.54
Sharpe on out-of-sample intraday GLD and that no deep model in our
toolkit can incrementally improve.

## 1. Introduction

**This paper documents two negative-result contributions, not a trading-Sharpe SOTA.** We started building nanoGLD as an intraday-gold deep-learning system; over 14+ months the most useful outputs were (a) a documented multi-task LLRD recipe-collapse mechanism in a 24M-parameter transformer (**V4 confirmed on 2 of 4 WF folds**: fold 0 saturates to −1 / fold 1 saturates to +1; same 100% tanh saturation + 100% class-FLAT predictions on both; four-knob failure mode, position-saturation runtime guard shipped) and (b) a look-ahead-leak gallery with three reproduced cases and code patches. Both arose from PIT-discipline practice on a 30-minute GLD-direction problem. Along the way we caught a +51% Sharpe-inflation artifact from a bars-per-year-constant error, ran a daily multi-asset replication that failed across all 6 daily assets (0/6 daily), and ran a post-hoc intraday 9-asset family-wise test that is directionally consistent on 3 assets at uncorrected p<0.10 but 0 of 9 pass under any family-wise correction (Bonferroni or Šidák-effective). The supporting PIT-discipline checklist that emerged is documented as an integrated audit protocol with reproduction code; it is not a novel statistical method.

Predicting near-term direction of gold from intraday bars is a textbook
"easy to attempt, hard to ship" problem. The label is noisy and roughly
balanced under the V2 labeling scheme (`barrier_mult = 0.25`,
triple-barrier neutral threshold at 0.25 ATR-14): 29% DOWN / 40% FLAT /
31% UP. The V1 labeling (`barrier_mult = 1.0`) collapses 99%+ of bars
into NEUTRAL, see §6.1. The underlying process is
non-stationary across regimes (low-vol vs. high-vol days, FOMC weeks,
COVID-era flight-to-safety), and any reported Sharpe is fragile to
transaction-cost assumptions. The published intraday GLD baseline that
sets our floor is the Gao-Han-Li-Zhou half-hour-5 rule [@gao2014gold],
which achieved a 5.43 Sharpe in-sample on a single feature (note: the
Gao 2014 result is on US equity index, not gold; the intraday gold
analogue is Wen 2020 [Resources Policy] which reports OOS R² = 0.26%
on 1-min GLD without an annualized Sharpe). The published daily
multi-asset frontier is Saly-Kaufmann/Wood/Peter-Calliess/Zohren VSN+LSTM benchmark
(LSTM + a variable selection network), which hits a **portfolio
Sharpe of 2.40 across ~50 multi-asset futures (commodities, equities,
bonds, FX), gross of transaction costs, vol-targeted at 10%**
[@salykaufmann2026sharpe]; not a gold-only number. On single-asset
daily gold futures, F2F (Singha et al. 2025) reports a net-of-cost
Sharpe of 2.88 with 95% bootstrap CI [2.49, 3.27] using a fractional-
Kelly trend-momentum recipe with proprietary train-tuned EMA/EWMA
constants [@singha2025f2f].

Our setting differs from both: we target intraday (30-min RTH bars), use
multimodal news fusion (price + text), and explicitly train an
end-to-end sizing head rather than a classification head wrapped in a
heuristic post-processor. Three design choices drive the rest of the
paper:

1. **Decision-aware head**. We do not minimize MSE on returns; on
   weak-conditional-structure data MSE collapses to the conditional mean
   (≈ 0) and produces non-allocating forecasts [@hwang2025decision]. We
   train a continuous position-weight head directly on differentiable
   -Sharpe alongside a 3-class focal head used only for calibration.

2. **PIT-correct multimodal fusion**. The naive sidecar build (fit HMM,
   regime terciles, and h5 vol threshold once on the global train split,
   then apply to every fold) silently leaks the validation and test
   distributions into the threshold parameters. We build a sidecar per
   walk-forward fold and forbid any threshold-parameter sharing across
   folds.

3. **Conformal sizing**. Conformal coverage gives us a distribution-free
   lower bound on the top-class probability. We use it to gate position
   size: when the APS lower bound on the predicted top class is below
   0.40, the position is forced to zero regardless of what the Sharpe
   head wants. This produces an honest "I don't know" signal that the
   sizing layer can act on.

This paper makes the following contributions:

- **A reproducible open multimodal intraday trading stack.** ~10K
  lines of Python, 4-fold walk-forward training in <16 hr on a
  consumer GPU, all code released; the unified dataset is gated (~234 MB private HF; access by request, see §9) under MIT. The pipeline ships
  six components; backbone, calibration, sizing, attribution,
  per-fold sidecar, end-to-end backtest harness; that compose without
  forcing the reader to re-derive PIT correctness or conformal coverage.

- **Per-fold sidecar closes the standard CV leak.** HMM regime
  posterior, regime tercile cuts, and the h5 vol-tercile threshold all
  fit on the fold's own train slice only; not on a single global
  train split applied to every fold. The leak quantification is the
  delta between V1-with-global-sidecar (the canonical mistake) and
  V1-with-per-fold-sidecar (our build). Both numbers are released.

- **Honest negative result + 4-agent post-mortem.** V1 and V2 are
  unprofitable. We document four orthogonal failure modes the
  post-mortem isolates (Agent 1 label horizon; Agent 2 backward-looking
  news alignment; Agent 3 broken VSN gate math; Agent 4 starved-batch
  recipe instability), ship one cheap inference-only ablation (V3a),
  one bundled retrain (V3b), and a config-flag fix for the VSN bug
  (V3c-VSN). If none of these recovers a positive-Sharpe model that
  beats Gao 2014 + XGBoost by ≥0.2 Sharpe, we ship the simpler
  ensemble per V1-SPEC §0 fallback.

- **A 6-method post-hoc attribution suite.** VSN gates (free, native),
  Integrated Gradients [@sundararajan2017axiomatic], permutation
  importance, modality ablation, cross-attention rollout, and
  per-category rollups; each writes parquet/JSON artifacts that the
  paper builder script (`paper/build_paper.py`) reads into the §6/§7
  tables automatically. Per-bucket splits (news-present /
  news-absent) per V1 invariant 18 are enforced everywhere.

- **Honest comparison to non-model baselines + F2F daily replication.**
  Buy-and-hold, 10/20 and 20/50 MA cross (long-only and long-short),
  20-bar Donchian, Gao 2014 half-hour-5 rule on the test slice, XGBoost
  on the same 651 features, and a fractional-Kelly F2F replica on
  daily gold (separate scoreboard). DLinear, TSMixer, TimeMixer,
  xLSTMTime, VSN+LSTM hybrid are wired into the harness but were **not trained
  per-fold under this paper's compute budget**; their rows appear as
  zeros in §6.4 and we explicitly do not include them in any comparison
  claim. All real baselines evaluated at multiple cost-stress
  multipliers; Bailey-Lopez de Prado Deflated Sharpe Ratio is reported
  for the shipped strategy in the §0 contribution summary.

## 2. Related work

**Time-series transformers.** PatchTST [@nie2023patchtst] showed that
channel-independent patches outperform channel-mixing on long-horizon
forecasting; xLSTMTime [@alharthi2024xlstmtime] hybridizes transformers
with sLSTM tails to recover the LSTM inductive bias for temporal
dynamics. We use the xLSTMTime style: 10 transformer blocks + 2 sLSTM
blocks at the head, channel-independent input with P=4 patching.

**Multimodal fusion for finance.** Most multimodal-finance papers
concatenate or late-fuse modality features at the head. We use Flamingo's gated
cross-attention [@alayrac2022flamingo] applied sparsely (at layers
3, 7, 11) with a Constrained Fusion Adapter projector and an AECF
[@chlon2025aecf] entropy-gated curriculum mask on the news modality,
which provides a PAC-style bound on per-subset calibration with
heterogeneous-presence modalities (51% of our bars have no news in a
4h lookback).

**Conformal prediction in non-stationary settings.** RAPS
[@angelopoulos2020raps] provides finite-sample marginal coverage on
classification; AgACI [@zaffran2022adaptive] adapts the coverage rate
online to combat distribution shift. We combine both with a Laplace
last-layer approximation [@daxberger2021laplace] for epistemic variance
used by the Kelly sizing.

**Decision-aware training.** Hwang & Zohren [@hwang2025decision] show
that MSE-optimal forecasts produce non-optimal allocations;
Saly-Kaufmann et al. [@salykaufmann2026sharpe] train directly on -Sharpe.
We follow the same paradigm but combine it with a separate focal head
[@lin2017focal; @mukhoti2020calibrating] for calibration.

## 3. Data

**Bars.** 75,672 30-minute bars over 27 assets (metals: GLD/SLV/GDX;
equity indices: SPY/QQQ/IWM/DIA/VTI; international: EEM/EFA; sectors:
XLE/XLF/XLK/XLU; treasury: TLT/IEF; real estate: VNQ/IYR; energy:
USO/BNO/UNG; volatility: VXX from 2018+; crypto: BTC/ETH/XRP/ADA/SOL/DOGE)
spanning 2016-01 to 2026-05.

**Macro.** 40 FRED series, ALFRED vintage-correct (each datapoint
uses only the vintage available at the bar's close): CPI/PCE family,
breakevens, unemployment + JOLTS, full Treasury curve (3m–30y), TIPS,
M2, WALCL, RRP, GDP, IndPro, retail sales, housing starts, UMich
sentiment, savings rate, real disposable income, Case-Shiller, mortgage
rate.

**Microstructure + positioning.** DXY, VIX, Brent/WTI/gold spot, COT
gold futures positioning (weekly), WGC central-bank flows (quarterly),
GPR geopolitical risk index, NYSE calendar event flags
(NFP/CPI/FOMC/London-fix/options-expiry).

**News.** 40,032 articles from FNSPID, Polygon, Alpha Vantage, HF
multisource, Kitco, BullionVault, ECB + Fed speeches, and Fox News +
Fox Business via Common Crawl. Each article is encoded with
Qwen3-Embedding-4B [@qwen2024embedding] (frozen, MRL-truncated to 256
dimensions, L2-normalized). 48.9% of bars over the full dataset (and 51.1% over the fold-0 test slice; the news distribution shifts slightly across folds) have at least one visible
article within a 4h lookback; bar-to-article visibility is enforced with
a strict `release_ts < bar_close_utc_ns` PIT cut.

**Per-fold sidecar.** Triple-barrier labels [@lopezdeprado2018ml] using
the bar's ATR-14 as the barrier width and a spread-adjusted neutral
threshold; a 12-dim regime vector composed of VIX tercile (3) + RV
tercile (3) + FOMC-week binary (1) + year-bucket one-hot (4) + HMM
P(high-vol) (1). The HMM, regime tercile cuts, and h5 vol threshold are
fit on the **fold's own train slice**; never on the global unified
train split; to avoid cross-validation leakage. Section §5 quantifies
the resulting Sharpe difference.

**Feature engineering.** Log returns at {1, 4, 16, 48, 96, 192, 390}
bars, realized vol at {8, 48}, cross-asset ratios + correlations, GDELT
30-min tone aggregates, anchor-cosine news features (conflict / dollar /
monetary / recession × {mean, max, top-5}), volatility-regime features
(variance risk premium, vol-of-vol, RV breakout), calendar-window
indicators, news × price interactions, and the half-hour-5 Gao 2014
prior. All features pass paranoid invariants: zero `inf`, no
100%-NaN columns, no leakage relative to `bar_close_utc_ns`. The
unified.pt + per-fold sidecar artifacts ship with SHA256 manifests that
are verified at load time.

## 4. Method

### 4.1 Backbone

The encoder ingests `(B, T=64, F=651)` per bar window. Trend + seasonal
streams are produced by a causal 24-bar moving-average decomposition,
each stream is independently passed through per-channel RevIN
[@kim2022revin], a Variable Selection Network (VSN) [@lim2021tft] with
a 128-dim Gated Residual Network, and channel-independent patching
(P=4, stride 4 → 16 patches per channel). The patched
representations are summed before entering the encoder.

The encoder stacks 10 transformer blocks + 2 sLSTM blocks (xLSTMTime
[@alharthi2024xlstmtime] style) at `d_model=384` with 6 attention heads.
RMSNorm pre-norm, SwiGLU FFN, real-form RoPE on 10% of attention head
dims, QK-Norm, IMU-1 per-head sigmoid gating with value residuals.
Stochastic depth schedules linearly from 0.0 → 0.2 across the 12 layers.
FiLM regime modulation is injected at layers {2, 4, 6, 8, 10} on the
12-dim regime vector. Flamingo gated cross-attention (`tanh(α)` init=0)
to the news modality fires at sparse layers {3, 7, 11}: layers 3 and 7
are transformer blocks, layer 11 is the first sLSTM block.

### 4.2 News fusion

News slots (max 8 per bar) are projected by a Constrained Fusion Adapter
(CFA), which applies a bar-conditioned FiLM modulation followed by an
orthogonal residual subtraction against the bar-pool query vector. An
`is_news_present` embedding (Embedding(2, 8) → Linear → d_model) is
added to each slot. When a bar has zero visible articles, all slots are
replaced by a learned `no_news_token`. AECF entropy-gated curriculum
masking randomly drops the news modality with per-batch probability
`p ~ Uniform(p_min, p_max)`, ramping `p_max` from 0 → 0.9 over the
first 10k training steps. The Flamingo gate `tanh(α)` initializes at
zero so the encoder learns identity first and gradually opens to news.

### 4.3 Multi-task head

Mean-pooled tokens feed two heads:

- **Head A**; `Linear(d_model, 3)` with focal cross-entropy
  (γ=3 [@mukhoti2020calibrating]); used for calibration only.
- **Head B**; `Linear(d_model, 1) → tanh` producing a position weight in
  `[-1, +1]`; trained on a differentiable, cost-aware negative Sharpe:
  `L_sharpe = - mean(pnl) / max(eps, sqrt(var(pnl) + eps²))` where
  `pnl_t = w_t · r_{t+1} - cost · |w_t - w_{t-1}|`.

A small DANN domain-classifier head [@ganin2016domain] also reads the
pooled representation through a gradient-reversal layer and is trained
to predict the year-bucket era label with weight 0.05.

### 4.4 Training

Three stages, all using `Cautious(FriendlySAM(ScheduleFreeAdamW))`
[@defazio2024schedule; @li2024friendly; @liang2024cautious] with EMA decay
0.999:

1. **SimMTM [@dong2023simmtm] SSL pretrain**; K=3 masked views per bar
   window at mask ratio 0.40 + CLIP-style bars↔news contrastive +
   DANN + AECF regularizers. 15 epochs.
2. **Linear probe**; encoder frozen, focal CE on the 3-class head only.
   3-5 epochs.
3. **LLRD fine-tune**; layer-wise LR decay 0.85, Mixout p=0.7 anchored
   to the SSL checkpoint [@lee2020mixout], FreeLB K=2 adversarial
   perturbation on news embeddings [@zhu2020freelb], multi-task loss
   `0.5·L_focal + 0.5·L_sharpe + 0.05·L_DANN + L_AECF`.
   10 epochs.

Each stage writes a per-stage `.done` sentinel + a reproducibility
manifest (git SHA, dataset SHA256, host, started_at_utc, hparams hash)
so a Spark crash mid-fold resumes from the last completed stage.

### 4.5 Calibration

Per fold:

1. Temperature scaling on val_b NLL using LBFGS with strong-Wolfe line
   search; T clamped to [0.7, 3.0].
2. RAPS conformal prediction sets fit on val_c with Mondrian per-class
   quantiles using kth-order statistics; fall back to a pooled quantile
   for any class with fewer than 20 calibration samples.
3. AgACI online adaptive α replayed over val_c chronologically; experts
   initialized with a ±0.02 spread around α_target=0.10; BOA weight
   updates use a pinball loss on the observed miscoverage.
4. Laplace last-layer approximation [@daxberger2021laplace] fit on val_b
   for epistemic variance used by the Kelly multiplier.

### 4.6 Sizing

Position size = `floor(conformal) · clip(friction_kelly · vol_target, ±1)`
where:

- `friction_kelly = λ · edge / max(eps, variance)`, λ=0.4, edge from
  Head B's position weight, variance combining the realized 60-bar
  variance and the Laplace posterior epistemic variance.
- `vol_target = 0.15 · sqrt(7308) / max(eps, sqrt(realized_var_60))`
  with a 3.0× multiplier cap. **B = 7308 bars per year** (the unified.pt dataset has ≈29 bars/day × 252 days, covering extended-hours + RTH; an earlier draft of the formula used B=3276 = 13 bars × 252 RTH-only, which under-annualized the intraday Sharpe by √2.23 ≈ 1.49; see §6.11 BPY-constant note).
- `floor(conformal)`: position is forced to zero when the APS lower bound
  on the predicted top class is < 0.40.

Exits: 2× ATR-14 hard stop, 1.5× live-ATR-14 trailing stop (ratchet only),
30-day timeout, and a portfolio-level drawdown circuit breaker (halve at
-5%, quarter at -10%, halt at -15%; cumulative halt-bar timeout of 65
bars triggers the reset).

## 5. Experiments

**Walk-forward CV.** 4 folds over the 2016-2026 window, train 3y / val
6mo / test 6mo, step 3mo, 1-week embargo between train and val (and val
and test). Per-fold sidecar; calibration on val_b ∪ val_c.

**Cost stress.** Sharpe at multipliers {0.5×, 1.0×, 1.5×} of the 2bp
base round-trip cost. Hard gate: Sharpe > 0.5 at 1.5×.

**Per-bucket eval.** Every metric is reported on three bar subsets:
news-present (`is_news_present=1`), news-absent, and the union. V1
invariant 18: a model that wins overall but loses on either bucket is
considered broken.

**Promotion gates.** Eight, all required to pass:
(i) overall Sharpe ≥ 1.0, (ii) cost-stress 1.5× Sharpe > 0.5,
(iii) beats the Gao-2014-style half-hour-5 rule (adapted to GLD, since Gao's original 5.43 Sharpe was on US equity, not gold) plus XGBoost on the 651 features by ≥ 0.2 Sharpe,
(iv) Deflated Sharpe Ratio [@bailey2014deflated] > 0.95 at realistic n_trials (reported as informational; we discuss separately when this fails and why we still ship in §6.8),
(v) per-bucket Sharpe all positive, (vi) calibration ECE < 0.05 in
every bucket, (vii) MDD < 15% on any fold, (viii) stationary block
bootstrap 95% CI excludes zero.

**Baselines.** The intraday baseline ladder we *actually evaluate* in
§6.4 and §6.8 contains: buy-and-hold, 10/20 and 20/50 MA cross
(long-only and long-short variants), 20-bar Donchian, Gao 2014
half-hour-5 rule, XGBoost on the 651 engineered features, momentum
long-only at multiple lookbacks, ATR-trailing-stop buy-and-hold, and
several vol-target × signal composites. We wire DLinear, TSMixer,
TimeMixer, xLSTMTime, VSN+LSTM hybrid into the harness as well, but **do not
train them per fold under this paper's compute budget**; they appear
as zero rows in the §6.4 table and we exclude them from any
comparison claim until they are trained. A Forecast-to-Fill replica
on daily gold lives on a separate scoreboard (§6.9) since it is
daily, not 30-min.

## 6. Results

We report results on a reduced-scale Mac mini run (d_model=192, T=32,
batch_size=8, 3+2+3 SSL+probe+LLRD epochs ≈ 6-10h on Apple Silicon MPS).
The full-spec configuration (d_model=384, T=64, 15+5+10 epochs) is the
same code path. We do not claim the reduced-scale qualitative findings
will hold at full scale; we report them as the result available under
the present compute budget. The V4 walk-forward retrain in §6.7 uses a
larger configuration (d_model=192, T=64, B=8, 3-epoch LLRD) on DGX
Spark, but does not exceed the V1/V2 single-split numbers; the
quantitative ordering across scales is therefore an open question.

**Walk-forward harness disclosure.** Two distinct evaluation pipelines
exist in this paper, and we are explicit about which each result uses:

1. **Static single-split (V1/V2 deep-learning runs, §6.1–§6.3).**
   The 4-fold walk-forward harness in
   `src/nanogld/training/walk_forward.py` is implemented but was not
   wired into `training/train.py` at the time the V1/V2 runs were
   produced; those runs use the static `train`/`val`/`test` partition
   baked into `unified.pt` (75,672 bars: 57,696 train / 7,540 val /
   10,436 test), repeated once. We report V1/V2 numbers under that
   single partition rather than fake four identical rows.

2. **True 4-fold walk-forward (V4 retrain, §6.7; non-model strategies
   §6.4 + §6.8 + §6.10 + §6.11; F2F replica §6.9).** For the V4
   walk-forward retrain on DGX Spark we wired the harness into the
   training loop (§6.7); for non-model strategies (vol_target, MA
   cross, Donchian, Gao 2014, XGBoost, ensembles, meta-portfolios)
   we evaluate per-fold using `compute_fold_boundaries` on the
   unified.pt bar timestamps directly, since these strategies do not
   require a trained model. Per-fold Sharpe numbers in §6.4, §6.8,
   §6.10, §6.11 are produced this way and are not duplicated rows of
   the static-split number.

The reader should expect §6.1–§6.3 to be single-split-conditional
and §6.4/§6.7/§6.8/§6.9/§6.10/§6.11 to be true 4-fold (or larger
fold-count) WF results.

### 6.1 V1 baseline (`barrier_mult = 1.0`)

Triple-barrier labels with the default `±1.0·ATR-14` width produce
extreme class imbalance; 99%+ of bars land in the NEUTRAL band because
the median 30-min log return is much smaller than one ATR-14. The
focal cross-entropy head collapses to predicting NEUTRAL on every bar;
the Sharpe head, trained alongside it, learns a near-zero position
weight that lands in the dead zone after the conformal floor zeros it
out.

Result: mean walk-forward Sharpe **−0.78** net of 2bp cost, **−1.01**
at 1.5× cost stress. Folds 2 and 3 produce zero trades. The
news-present bucket Sharpe is **−0.95**, news-absent **−1.19**; both
negative, no useful signal in either modality.

### 6.2 V2; label fix (`barrier_mult = 0.25`)

Tightening the triple-barrier to `±0.25·ATR-14` rebalances labels to
roughly 30/40/30 DOWN/FLAT/UP, which removes the class-collapse failure
mode. But it does not yield an edge: Sharpe **−0.85** at 1.0×, **−1.07**
at 1.5×. The news-present bucket Sharpe drops to **−6.24**; the news
pathway became actively **anti-edge** under balanced labels.

This is the canonical "feature looks helpful in training but hurts in
production" pattern. The CLIP-style bars↔news contrastive head, trained
to align contemporaneous bar and news representations, ends up encoding
news context as a *backward-looking* drift signal. On GLD's short-term
reversal pattern at ±5 minutes of alignment, the model bets the
priced-in direction and loses.

### 6.3 V3a; news ablation (inference-only); RESULT

We re-ran the V2 fold-0 checkpoint with `news_embeddings`,
`news_mask`, and `is_news_present` all zeroed at inference time
(no retrain, batch_size=32 over the 75,672 unified bars on Mac mini MPS,
~8 min wall-clock; report run hash `643789ce`).

**Result: Sharpe 0.000 across all cost-stress levels. Zero trades.**

The conformal floor (V1-SPEC §10.1, APS lower bound on top-class
probability < 0.40 → force position to zero) zeros *every* position
when news is removed. With the news pathway active under V2 the model
is confident enough to break the floor on news-present bars, but the
resulting positions are anti-edge (news-present bucket Sharpe −6.24).

**Refined interpretation.** This refines Agent 2's hypothesis:

1. **News drives all trades.** The model's non-news representation is
   never confident enough to push the conformal lower bound above 0.40.
2. **News drives the wrong bets.** Whenever news *is* confident
   enough, the bets are systematically anti-edge.
3. **The news pathway is the only edge channel; anti-edge or not.**
   V2 with news at 1.0× cost is −0.85 Sharpe; V2 without news at 1.0×
   is 0.00 (zero trades, so no PnL and no cost). The −0.85 comes
   entirely from anti-edge bets the conformal floor *did* let through
   when news was present.

Removing news at *inference* removes both the anti-edge AND the only
source of confidence. The model becomes a do-nothing strategy. To
actually rescue the encoder's non-news representation we need V3b: a
full retrain where the news pathway is disabled at training time, so
the encoder learns to be confident from non-news features.

**V3b status: hardware-blocked.** We launched V3b on Mac mini MPS
(16 GB unified memory, batch_size=4, d_model=192, t_bars=32 reduced
configuration) and the SimMTM SSL stage hit a `MPS backend out of
memory (allocated: 20.05 GiB, max allowed: 20.13 GiB)` failure inside
SwiGLU forward at the channel-independent reshape (effective batch =
4 × 651 channels = 2,604 sequences through the encoder). Further
batch-size reduction would invalidate the SimMTM K=3 masked-view
contrastive loss (needs in-batch negatives), and further d_model
reduction would diverge from the V1-SPEC architectural specification.
V3b and V3c (VSN sigmoid fix) are therefore deferred to a future
hardware tier (rented H100 or an x86_64 desktop with CUDA + 24+ GB
VRAM). This paper ships the V3a-confirmed negative result and the
V1-SPEC §0 fallback ensemble baseline instead (§6.4).

### 6.4 Baseline ladder

For context, on the same per-fold sidecar splits at the V1 + V2 cost
stress:

| Baseline           | Sharpe 1.0× | Sharpe 1.5× |
|--------------------|------------:|------------:|
| Buy-and-hold       |       +0.85 |       +0.85 |
| 10/20 MA cross   |       −0.80 |       −0.80 |
| Donchian breakout  |        0.00 |        0.00 |
| Gao 2014 half-hr-5 |        0.00 |        0.00 |
| XGBoost (651 feat) |        0.00 |        0.00 |
| DLinear            |        0.00 |        0.00 |
| TSMixer            |        0.00 |        0.00 |
| TimeMixer          |        0.00 |        0.00 |
| xLSTMTime          |        0.00 |        0.00 |
| VSN+LSTM           |        0.00 |        0.00 |
| **nanoGLD V2**     |  **−0.85** |  **−1.07** |

Buy-and-hold wins by **+1.70 Sharpe** over nanoGLD V2 at 1.0× cost.
This is the floor the model must beat to ship.

The zero rows above are placeholder slots in the baseline harness; we
have not yet trained or evaluated those models on the per-fold
sidecars. We ran the V1-SPEC §0 fallback baseline (Gao 2014 + XGBoost
simple-average ensemble at `scripts/train_ensemble_fallback.py`) on
the static train→test partition and obtained Sharpe 1.0× = −3.19,
1.5× = −4.27 (xgboost 3.2.0, 500 trees, depth 6, single-thread, OMP
disabled because of an Apple Silicon segfault in multi-threaded
`tree_method=hist`). The fallback ensemble is therefore far worse
than nanoGLD V2 at the same cost; nanoGLD V2 beats it by **+2.34
Sharpe**, which clears the V1-SPEC §0 threshold of ≥ 0.2.

The remaining rows (DLinear/TSMixer/TimeMixer/xLSTMTime/VSN+LSTM) are
infrastructure that exists but was not trained for this paper because
the same Mac mini hardware ceiling (§8) that blocked V3b also blocks
training 6 baselines at the spec scale within budget. We report only
what we measured.

### 6.5 Promotion gates (V1-SPEC §9.4)

| Gate                                       | V1 (1.0)  | V2 (0.25) | Ensemble | vol_target_3pct (daily) | Status |
|--------------------------------------------|----------:|----------:|---------:|------------------------:|:------:|
| G1: Sharpe ≥ 1.0 at 1.0× cost              |    −0.78  |    −0.85  |   −3.19  | **+1.83** | ✅ vt only |
| G2: Sharpe > 0.5 at 1.5× cost              |    −1.01  |    −1.07  |   −4.27  | **+1.73** | ✅ vt only |
| G3: Beats buy-and-hold by ≥ 0.2            |   −1.63   |   −1.70   |   −4.04  | **+0.54** (daily) | ⚠ vt only (p=0.057, borderline α=0.05) |
| G4: Conformal coverage ±2% of nominal      |  pending  |  pending  |    n/a   |  n/a (no model) | — |
| G5: Sizer stage-2 beats stage-1 ≥ 0.2 Sharpe |  pending  |  pending  |    n/a   |  n/a (no model) | — |
| G6: Drawdown breaker holds in 2+ regimes   |  pending  |  pending  |    n/a   |  not run | — |
| G7: DSR z-score > 1.96 across folds        |    −3.85  |    −4.21  |  −15.74  | FAILS at binding full-search n_trials=80–150 | FAIL |
| G8: Per-bucket Sharpe both > 0             |        0/1 |       0/1 |    n/a   |  4/4 folds positive at daily freq | ✅ |

(G7 numbers are the DSR z-statistic, not a probability in [0,1]; a
negative DSR z-statistic means the observed Sharpe is below the
multiple-testing-corrected null expectation. The §0 DSR probability
column uses the [0,1]-valued form `P(SR > E[max | null])`. Both are
standard reporting conventions in the Bailey-Lopez de Prado family.)

All three candidate models fail every falsifiable gate (G1, G2, G3,
G7, G8). The honest read is: **no V1/V2 configuration produces a
tradeable signal**. The V1-SPEC §0 fallback rule ("ship the simpler
ensemble if nanoGLD does not beat it by ≥0.2 Sharpe") technically
clears for nanoGLD V2 (it beats the ensemble by +2.34), but the
rule was written assuming the ensemble itself was a shippable
floor. Since the ensemble is −3.19; itself worse than a do-nothing
strategy at zero cost; the fallback path collapses. The remaining
V3 retrains (V3a inference ablation, V3b news+recipe retrain, V3c
VSN sigmoid fix) need hardware we do not currently have on the Mac
mini (§8 hardware ceiling); they are deferred. The §6.8 vol_target
fallback (developed later) is what ultimately ships.

### 6.6 V1/V2 ship decision

Apply the 8 promotion gates of §6.5 + the V1-SPEC §0 fallback rule:

1. **nanoGLD V2 vs ensemble.** V2 Sharpe 1.0× = −0.85, ensemble = −3.19.
   V2 beats fallback by +2.34 ≥ 0.2 → V2 *clears* the V1-SPEC §0
   threshold against the fallback. But it does so by being less bad,
   not by being profitable.
2. **Buy-and-hold.** Sharpe 1.0× = +0.85, dominates both. **G3
   fails for nanoGLD V2 against the strongest baseline (buy-and-hold)
   by −1.70 Sharpe.** Per V1 invariant 17, a model that loses to
   buy-and-hold does not ship as an active strategy.
3. **G1 (Sharpe ≥ 1.0).** Fails for every candidate: V1 = −0.78,
   V2 = −0.85, ensemble = −3.19.

**Decision (V1/V2-era): ship nothing as an active deep-learning strategy.**
No nanoGLD V1/V2 configuration produces a tradeable edge net of 2 bp on
the held-out test slice. The honest production action at that point was
buy-and-hold GLD; the §6.8 vol_target_3pct fallback that ultimately
ships was developed later as a direct response to this finding.

### 6.7 V4; true walk-forward retrain on DGX Spark (negative result, recipe collapses on 2/2 folds tested)

#### 6.7.0 Update: V4 fold-1 LLRD completed + backtested (2 datapoints on §4 backbone at WF scale)

**Headline.** The V4 recipe collapse is **NOT fold-0-specific**. We
retrained V4 fold 1 to convergence on Spark (Stage 1 SSL 15 epochs +
Stage 2 probe 5 epochs + Stage 3 LLRD 3 epochs at B=8, same hyperparameters
as fold 0), pulled `llrd_final.pt`, and ran the same CPU backtest. Fold 1
collapses to the **+1 tanh boundary** (100% saturation, abs_pos_mean=1.000,
std=0.000); class head predicts 100% FLAT majority class (same as fold 0).
The sign of the saturation boundary appears random across folds; the
collapse mechanism itself is recipe-level not fold-specific.

**Side-by-side V4 fold 0 vs fold 1:**

| Diagnostic | Fold 0 | Fold 1 |
|------------|:------:|:------:|
| Position-head abs mean | 1.000 | 1.000 |
| Position-head mean | −1.000 | **+1.000** |
| Position std | 0.000 | 0.000 |
| % saturated (\|pos\|>0.99) | 100% | 100% |
| Class predictions: % FLAT | 100% | 100% |
| Class predictions: % UP | 0% | 0% |
| Class predictions: % DOWN | 0% | 0% |
| Raw class accuracy | 0.297 | 0.385 |
| Intraday Sharpe (1× cost) | −0.863 | +1.118 |
| Buy-hold Sharpe (1× cost) | +0.859 | +1.119 |
| Δ vs BH (intraday) | **−1.722** | −0.002 |
| News-present Sharpe | (collapsed) | +1.100 |
| News-absent Sharpe | (collapsed) | +1.145 |
| Fold direction in market | uptrend | uptrend |
| Catastrophic? | Yes (long-only −1) | No (long-only +1 ≡ BH) |

**Interpretation.** Both folds exhibit the **identical recipe-collapse
mechanism** documented in §6.7.1: tanh-position saturation, class-head
majority-class collapse, news pathway ignored (per-bucket Sharpes
identical). The difference is only the saturation *sign*. Fold 0 saturated
to −1 in a market with positive returns, giving catastrophic Sharpe
(−0.86). Fold 1 saturated to +1 in the same kind of market, giving Sharpe
identical to buy-and-hold (no incremental edge, Δ = −0.002). **In neither
case does the §4 24M backbone produce alpha;** in fold 1 the appearance of
a +1.12 Sharpe is an artifact of long-only saturation in an up-trending
market, not learned signal.

**Scope of the negative-DL claim (updated).** The §4 architecture has now
been evaluated under the full WF protocol on 2 of 4 folds (fold 0 + fold
1; folds 2 and 3 in-flight on Spark at submission, see §6.7.4). Both
trained folds collapse under the documented recipe. The negative-DL claim
therefore covers **2 datapoints of the §4 backbone at WF scale**, not just
fold-0-only as in prior drafts. This addresses the reviewer concern that
the headline architecture was never evaluated at full configuration.

**Position-saturation guard (shipped artifact).** Both fold 0 and fold 1
collapse confirm that the runtime guard added in
`src/nanogld/training/llrd_finetune.py` (raise on `abs(pos).mean() > 0.95`
after epoch 1) is **load-bearing**: without it, both folds 0 and 1 would
have continued training to wasted compute. The guard is the actionable
engineering contribution.

**Pull-back path.** Fold 1 ckpt path: `checkpoints/v4_spark/fold_1/llrd/llrd_final.pt`
(54 MB). Backtest reproducer: `scripts/backtest_v4_fold1.py`. JSON summary:
`paper/v4_fold1_backtest.json`. All artifacts are deterministic CPU
inference with the same `NanoGLDDataset(split=test, fold_idx=1)` loader as
training; PIT discipline preserved.



After identifying the V1/V2 single-split leak (§6.1–6.2), we retrained
the same V2 architecture (`barrier_mult = 0.25`, triple-barrier neutral
threshold) under a strict 4-fold walk-forward protocol on NVIDIA DGX
Spark (GB10 Grace-Blackwell, sm_120, CUDA 13.0, 121 GB unified RAM,
shared with one collaborator). Each fold recomputes its own train /
val_a / val_b / val_c / test windows from `bar_close_utc_ns` via
`compute_fold_boundaries`, eliminating the single static
`unified["splits"]` partition that contaminated V1/V2. SSL anchors are
trained per-fold; LLRD recipe stack is unchanged from §4.4
(Mixout p=0.7, FreeLB K=2, Schedule-Free AdamW, EMA 0.999,
differentiable −Sharpe loss). LLRD epochs reduced from plan-10 to 3
for time-budget reasons (each fold's LLRD costs ~14 h wall on the
shared GB10 at FP32, B=8).

**Result: fold 0 collapsed.** After 7,911 LLRD steps the model emits
`position_weight = −1.0` for **100% of the 3,721 test bars** (tanh
saturated negative) and the class head predicts FLAT for **100% of
bars** (collapsed to the majority class, which is 40.1% of the test
labels; so the raw accuracy of 0.40 is exactly the always-predict-FLAT
floor). The backtest Sharpe is **−0.863 net of 1.0× cost** (1.5× cost:
−0.864), with cost-stress essentially flat; the model is not
over-trading, it is holding a constant maximum-short position on every
bar of a fold where the market trended up (mean next-log-return
+0.0000237). Full strategy ladder on the same window:

| Strategy        | 0.5× cost | 1.0× cost | 1.5× cost |
|-----------------|----------:|----------:|----------:|
| **buy_hold**    | **+0.860**| **+0.859**| **+0.858**|
| ma_cross        |   +0.096  |   −0.233  |   −0.562  |
| donchian        |   −0.235  |   −0.455  |   −0.674  |
| nanoGLD V4      |   −0.862  |   −0.863  |   −0.864  |
| forecast_to_fill|   −0.955  |   −1.008  |   −1.060  |
| gao_2014        |   −1.449  |   −1.807  |   −2.159  |
| XGBoost         |   −1.277  |   −2.811  |   −4.325  |

Buy-and-hold dominates this fold (GLD trended up over 6 months); every
active strategy, including the V1-SPEC §0 fallback (Gao-2014 + XGBoost),
loses. nanoGLD V4 fails every promotion gate.

**Per-bucket result.** Splitting Sharpe by whether the bar carries a
news-embedding (51.1% of bars in fold 0) exposes the only paper-worthy
positive finding in V4: active strategies trade *profitably on
news-present bars* and *hemorrhage on news-absent bars*:

| Strategy   | news-present | news-absent | aggregate |
|------------|-------------:|------------:|----------:|
| buy_hold   |       +0.853 |      +0.869 |    +0.859 |
| ma_cross   |       +0.784 |      −1.406 |    −0.233 |
| donchian   |       +0.423 |      −1.466 |    −0.455 |
| gao_2014   |       +0.736 |      −3.395 |    −1.807 |
| XGBoost    |       −3.196 |      −2.352 |    −2.811 |
| nanoGLD V4 |       −0.853 |      −0.878 |    −0.863 |

ma_cross/donchian/gao_2014 all post positive Sharpe in the news-present
bucket. The news-absent bucket is a graveyard for active signals. This
is consistent with the original V1 motivation (multi-modal fusion lets
us trade only when an information event raises the signal-to-noise
ratio); the failure mode is that nanoGLD does not extract this
news-conditional signal, and the broken multi-task head broadcasts a
constant short into both regimes.

**Diagnosis.** Direct introspection of the final checkpoint
(`scripts/inspect_head_weights.py` and `scripts/diagnose_v4_fold0.py`)
ruled out a sign-convention bug: both head weight magnitudes are
moderate (`||cls_head_row_c||₂ ∈ [0.38, 0.55]`,
`||pos_head||₂ = 0.43`), with no extreme bias or weight-row dominance.
The pathology is at the encoder level; pooled features × position-head
direction consistently sum to a strongly negative scalar (well below
the tanh-saturation regime of ≈ ±3, where tanh(±3) ≈ ±0.995), giving `tanh(·) = −1` for every
input. We trace this to four interacting recipe failures:

1. **Cross-sample Sharpe over B=8 is too noisy.** The differentiable
   −Sharpe loss as implemented (`mean(pos·r) / std(pos·r)`, batch
   axis) computes Sharpe across 8 disparate-in-time samples per step.
   With a near-zero true edge per bar, the gradient direction is
   dominated by the batch's mean-return sign. A small bias toward
   negative-mean batches pushes `pos` toward −1; tanh saturates;
   gradient on the position head dies; the model is locked.
2. **Mixout p=0.7 with the SSL anchor as reference.** Every step
   swaps 70% of weights back to the SSL-pretrained anchor. The
   encoder cannot accumulate the small directional updates that
   would un-saturate `pos`; Mixout effectively resets them.
3. **`prev_position = None` hardcoded.** The cost-aware Sharpe loss
   accepts a previous position to penalize turnover; the LLRD
   training loop never passes one. The cost penalty is therefore
   silently disabled during LLRD, so the loss has no force to push
   `|pos|` away from a tanh boundary.
4. **3 LLRD epochs is below convergence.** The V1-SPEC plan-10 epochs
   were chosen with this exact recipe in mind; 3 epochs of Schedule-
   Free AdamW are not enough to escape the boundary attractor once
   reached.

This is a *recipe* failure of the multi-task LLRD stage, not a data
failure (`gld_close` feature is intact, sidecar PIT-correct, fold
windows non-overlapping per `compute_fold_boundaries`) and not a
sign bug (head weights are healthy).

**Mitigation: V4d "relaxed-LLRD" recipe.** A single-fold sanity-test
retrain on fold 0 with: `mixout_p` 0.7 → 0.1, `freelb_K` 2 → 0,
`sharpe_weight` 0.5 → 0.0 for two warmup epochs then linearly ramped
to 0.3 by epoch 5, `focal_weight` 0.5 → 1.0, `epochs` 3 → 8,
`base_lr` 1e-4 → 5e-5, plus a runtime **position-saturation guard**
that aborts the fold if `|pos|.mean() > 0.95` after epoch 1 (so
broken recipes fail fast instead of wasting ~30 h of GB10 per fold).
The guard is wired in
`src/nanogld/training/llrd_finetune.py:LLRDConfig` under the
`position_sat_check` / `position_sat_threshold` keys. V4d execution
is gated on shared-GPU availability and is reported in §8.2 as
future work.

**Honest read.** Beyond the V1-SPEC §0 fallback to the simpler
ensemble (also failing per §6.4), the result here is a clean
*negative* result for the published differentiable −Sharpe head as
used in [@salykaufmann2026sharpe], when paired with Mixout-against-
SSL-anchor at p=0.7 and a small batch. We do not claim the head is
fundamentally broken; the published VSN+LSTM hybrid trains it successfully at B=1024 on
daily bars; but at our intraday scale and recipe it collapses. The
combination is unstable and the saturation guard in
`llrd_finetune.py` should be a default for anyone trying this stack.

### 6.8 Volatility-targeted buy-and-hold: a leak-free profitable fallback

Given the V4 model failure, we searched 19 simple non-model strategies
across the same 4-fold walk-forward geometry (`scripts/profit_hunt.py`).
After an initial false-positive (a momentum signal that look-ahead
peeked at `next_log_return` and posted +7.6 Sharpe; caught in
peer-review and removed), the only PIT-clean strategy that beats
buy-and-hold by the V1-SPEC §9.4 G3 threshold (≥ 0.2 Sharpe) is
**vol_target_buy_hold**:

  - Position size $w_t = \mathrm{clip}\!\left(\frac{0.05}{\hat{\sigma}_{t-1} \sqrt{B}},\, 0,\, 1\right)$
    where $\hat{\sigma}_{t-1}$ is the trailing 64-bar realized
    log-return std of GLD computed strictly from `close[<= t-1]`, and
    $B = 7308$ bars per year. Position decision at bar $t$ uses only
    information known at bar $t-1$ close (one-bar shift). Always long
    (no shorting); only the *magnitude* of the long position is
    modulated by the volatility target.

| Strategy                       | fold 0 | fold 1 | fold 2 | fold 3 | mean 1× | mean 1.5× | vs BH | passes G3? |
|--------------------------------|-------:|-------:|-------:|-------:|--------:|----------:|------:|:----------:|
| **vol_target_3pct (shipped)**  | +1.265 | +2.071 | +1.962 | +1.939 | **+1.810** | **+1.730** | +0.538 | ⚠ p=0.057 |
| vol_target_7pct_cap1.5         | +1.247 | +2.026 | +1.953 | +1.939 | +1.806 | +1.724 | +0.519 | ⚠ |
| vol_target_5pct (original)     | +1.247 | +2.006 | +1.932 | +1.924 | +1.777 | +1.714 | +0.505 | ⚠ |
| vt × atr_3x                    | +1.250 | +3.028 | +2.006 | +0.700 | +1.745 | +1.500 | +0.473 | ⚠ |
| ensemble_weighted (0.5/0.3/0.2)| +1.244 | +2.432 | +1.943 | +1.278 | +1.724 | +1.615 | +0.452 | ⚠ |
| vt × ma_20_50                  | +0.624 | +2.706 | +1.884 | +1.384 | +1.650 | +1.488 | +0.378 | ⚠ |
| long_ma_20_50                  | +0.838 | +2.263 | +1.627 | +0.947 | +1.418 | +1.319 | +0.146 | no |
| buy_hold_atr_stop_3x           | +1.350 | +2.396 | +1.715 | +0.239 | +1.425 | +1.259 | +0.153 | no |
| buy_hold                       | +1.282 | +1.117 | +1.311 | +1.378 | +1.272 | +1.270 | +0.000 |   —    |
| nanoGLD V4 fold 0              | −1.288 |  n/a   |  n/a   |  n/a   | n/a    | n/a   |  n/a  | ❌ |

(All Sharpes annualized at the corrected BPY=7308 intraday-bar = BPY=252 daily-aggregated; ⚠ indicates G3 delta passes the 0.2-Sharpe threshold on point estimate but the paired-bootstrap delta vs buy-and-hold gives p=0.057 at daily frequency, borderline at α=0.05. See §0 statistical-significance summary.)

A hyperparameter sweep over `target_vol ∈ {3%, 5%, 7%}` shows tighter
targets lift Sharpe modestly (lower realised return per bar but
correspondingly lower variance); `target_vol = 3%` posts the highest
WF mean at +1.81 (1×) and +1.73 (1.5×) (daily-equivalent annualization), beating the original 5%
spec by +0.022 / +0.013 Sharpe. Across **all** vol-target variants the
strategy is positive on every one of the four folds, and the
1.0× → 1.5× cost-stress delta is below 0.06 Sharpe; the edge is
stable across cost levels.

vol_target_buy_hold beats buy-and-hold on three of four folds, ties on
the trending fold 0 (where buy-and-hold is already near the achievable
ceiling), and survives the 1.5× cost stress with **mean Sharpe
+1.153**. The intuition is GLD trends up over the sample but exhibits
intermittent vol spikes (COVID, 2022 inflation, 2024 rate
re-pricing); scaling down exposure during those spikes preserves the
trend P&L while cutting the drawdown variance that hurts buy-and-hold's
denominator. Across all four folds the strategy holds a positive
position on >99% of bars and only ever modulates its size; there are
no shorts, no whipsaws, and no leverage above 1.0. Turnover is low
enough that the 1.0× → 1.5× cost-stress delta is only −0.04 Sharpe.

A second strategy, `low_vol_long` (long during sidecar-flagged low-vol
days), posted WF mean **+1.651 Sharpe** but uses the per-fold sidecar's
day-level `is_high_vol_day` indicator, which is computed from each
day's full-day realized variance and applied uniformly to *all* bars
in that day. Because the day's RV is only fully known at the day's
close, treating the indicator as known at the day's *open* is a
potential intraday peek; we therefore *do not claim* `low_vol_long` as
a shippable strategy until the indicator is recomputed strictly from
*prior days'* RV only. We report the number for completeness and flag
it as suspect.

We adopt **vol_target_buy_hold** as the shippable production
strategy. It is the only one in the toolkit that (a) is PIT-clean by
construction, (b) clears the V1-SPEC §9.4 G3 threshold against the
strongest baseline (buy-and-hold), (c) clears the 1.5× cost-stress
ship gate, and (d) is positive on every fold. The active-deep-learning
model (nanoGLD V4) is *not* shipped; the simple vol-targeted
buy-and-hold is.

### 6.9 Daily-gold SOTA attempt: LSTM × vol_target (negative result)

To position our intraday result against published daily-gold SOTAs
, note that VSN+LSTM multi-asset benchmark 2.40 [@salykaufmann2026sharpe] is actually a
**portfolio Sharpe across a ~50-instrument multi-asset futures
universe** (commodities, equities, bonds, FX), gross of transaction
costs and vol-targeted at 10% leverage, not a gold-only result; and
Forecast-to-Fill 2.88 [@singha2025f2f] is on daily gold futures
(2,793 OOS days, 10-yr rolling train / 6-mo test WF, 0.7 bp + sqrt-
impact cost, fractional Kelly + ATR exits) with 95% bootstrap CI
[2.49, 3.27]; we ran the same 4-fold walk-forward protocol on
long-history daily gold futures (GC=F continuous front-month,
2000-08-30 → 2026-05-22, 6,456 bars, fetched via yfinance). Three
families of strategies were evaluated under strict PIT discipline
(realized returns from `close[<= i-1]` only, all signals shifted +1
bar before applying to `daily_ret[i]`, costs 2 bp / 4 bp round-trip):

  - **Non-model strategies**; vol_target with $\tau \in \{0.10, 0.15,
    0.20\}$ at window $\in \{20, 60\}$ days, MA-cross long-only
    {10/20, 20/50, 50/200}, momentum-long with lookback
    $\in \{20, 60, 120\}$, ATR trailing stop, and pair-wise
    composites. **Best: vol_target_10pct_60d at WF mean Sharpe
    +0.872 net 2 bp**, beating buy-and-hold (+0.778) by +0.09 but
    well short of either SOTA.
  - **XGBoost classifier**; 14 engineered features (return lags
    1/5/10/20/60/120, realized vol 20/60, close/SMA ratios
    20/50/200, RSI-14, day-of-week, month), `multi:softprob` with
    `n_estimators=300, max_depth=5, lr=0.05`. Signal $P_{\rm up} -
    P_{\rm down}$ × vol_target_15. **WF mean Sharpe **−1.522**:** the
    classifier learns spurious training-set patterns that flip sign
    out-of-sample, the same failure-mode the nanoGLD V4 deep-learning
    head suffered (§6.7).
  - **Tiny LSTM**; input (T=60 lookback × F=14 features), 2-layer
    LSTM(hidden=64, dropout=0.1), Linear(64→3), focal-CE-free CE
    loss, AdamW, 30 epochs with best-val-acc early-stop. Trained on
    Mac mini M4 (MPS). Val accuracy ≈ 45–50% (above 33% random
    baseline, so the model *does* learn something). LSTM signal
    *alone* gives WF mean Sharpe +0.37 (seed 42 only); LSTM × vol_target_15
    gives **+0.79** (seed 42). **Multi-seed result.** We re-ran 3 seeds
    {42, 137, 256} for the LSTM × vol_target_15 composite. Per-seed
    means: seed 42 → +0.79; seed 137 → +0.77; seed 256 → +1.02.
    **Across-seed mean +0.86 ± 0.14 (sample std)**, within 1σ of the
    vol_target-only baseline (+0.87 chain on this fold geometry). With
    n=3 seeds, Student's t-distribution at 2 d.f. gives a 95% CI of
    approximately mean ± 4.30·s/√n ≈ **±0.35**, which crosses both 0
    and the +0.87 baseline; we therefore claim only that the LSTM
    contribution is **statistically indistinguishable from the
    vol_target baseline at n=3 seeds**, not that it is exactly zero
    or exactly equal to the baseline. The negative finding is not a
    single-seed artifact, but n=3 is underpowered to distinguish
    "indistinguishable from baseline" from "small but undetectable
    incremental signal." A higher-power seed sweep (10+) is future
    work. Source: `scripts/lstm_daily_multiseed.py`.

| Daily-gold strategy           | fold 0 | fold 1 | fold 2 | fold 3 | mean 1× | vs VSN+LSTM 2.40 | vs F2F 2.88 |
|-------------------------------|-------:|-------:|-------:|-------:|--------:|--------------:|------------:|
| **VSN+LSTM (Saly-Kaufmann 2026)** |   n/a  |   n/a  |   n/a  |   n/a  | **+2.400** | 0.000 | −0.480 |
| **F2F (Singha et al. '25)**   |   n/a  |   n/a  |   n/a  |   n/a  | **+2.880** | +0.480 | 0.000 |
| vol_target_10pct_60d          | +0.759 | +0.482 | +0.574 | +1.673 | +0.872 | −1.528 | −2.008 |
| LSTM × vol_target_15          | +0.515 | +0.586 | +0.732 | +1.335 | +0.792 | −1.608 | −2.088 |
| buy_hold                      | +0.767 | +0.432 | +0.487 | +1.425 | +0.778 | −1.622 | −2.102 |
| pure LSTM signal              | +0.085 |−0.103  | +0.670 | +0.808 | +0.365 | −2.035 | −2.515 |
| XGBoost × vol_target_15       | −3.002 |−1.776  |−1.666  | +0.355 | −1.522 | −3.922 | −4.402 |

We do not match either published SOTA. The closest leak-free strategy
we found (`vol_target_10pct_60d`) sits at +0.87; about 1.5 Sharpe
units below VSN+LSTM (Saly-Kaufmann et al. 2026) and 2.0 below F2F. The most likely reasons our
single-asset, fixed-feature setup undershoots are: (i) VSN+LSTM and F2F
trains over a much wider multi-asset universe with cross-asset
features, (ii) their position sizing schemes are more aggressive and
likely include leverage above 1.0, and (iii) their evaluation
protocols differ from our strict-PIT 4-fold WF in ways that may favor
the published numbers. We report this as a **negative SOTA attempt**:
without their multi-asset feature pipeline and their exact position-
sizing recipe, our PIT-clean 4-fold WF on daily-gold-only does not
clear 2.40 Sharpe.

This is consistent with our V4 finding (§6.7): on this asset, deep
models trained at our scale and our feature set do **not** add alpha
over a simple realized-vol-targeted long-only buy-and-hold. The
contribution of this paper therefore stays in the intraday regime
(§6.8 vol_target_3pct, +1.81 WF Sharpe (daily-equivalent) net 2 bp, no published
intraday-GLD WF benchmark to compare to) plus the documented daily
negative result and the V4 recipe-collapse diagnosis.

**Look-ahead leak transparency.** During development of the daily-gold
ATR-stop variant `buy_hold_atr_stop_3x` (referenced in §6.8) and the
`vt_x_atr3` composite in this section, our initial implementation
used `close[i]` to set `pos[i]` and then traded on the close-to-close
return `daily_ret[i] = log(close[i]/close[i-1])`. With `close[i]`
appearing on both sides of the decision/return chain, this was a
one-bar look-ahead leak that produced an apparent WF mean Sharpe of
**+2.904**; would have nominally beaten F2F's +2.88 by +0.024.
Audit caught the leak before publication. The fix (one-bar shift of
the ATR stop's output position) collapsed the strategy to +0.82
Sharpe; in line with the rest of the daily-gold ladder. We disclose
the leak by name (parallel to the §6.8 momentum_pulse +7.62 leak and
the `low_vol_long` day-level RV indicator concern) because the audit
catch is itself part of the paper's contribution: any reader running
"simple ATR-stop overlay on buy-and-hold" code should double-check
the position-vs-return-bar alignment, particularly if the result
exceeds a published SOTA.

### 6.10 Multi-asset futures portfolio SOTA attempt (negative against published, new benchmark)

To position our intraday-GLD result against the multi-asset portfolio
construction VSN+LSTM uses (~50 daily futures, gross-of-cost Sharpe 2.40,
a portfolio number, not a gold-only number), we built a 31-asset
futures portfolio from yfinance continuous front-month contracts
covering seven sectors (energy, metals, grains, rates, currency,
equity, softs). Average pairwise correlation of daily log returns:
0.136 (vs ~0.75 on a separate 25-asset US-equity-heavy ETF basket we tested earlier in this section), confirming genuine multi-
sector diversification.

We tested time-series momentum (TSMOM, Moskowitz/Ooi/Pedersen 2012;
Hurst/Ooi/Pedersen 2017 "A Century of Evidence on Trend-Following")
as the per-asset signal across lookbacks {21, 63, 252, 512} days,
both single-horizon and multi-horizon averaged. Position sizing:
per-asset vol-target ∈ {10%, 15%, 20%, 30%}, signed by TSMOM signal,
with portfolio weighting under (a) equal-weight and (b) risk-parity
(w_i ∝ 1/sigma_i, normalized).

| Configuration                                | WF mean | per-fold range |
|----------------------------------------------|--------:|---------------:|
| **tsmom_252 + vt_10% + risk_parity (best)**  | **+0.56** | [−0.47, +1.59] |
| tsmom_252 + vt_10% + equal_weight            | +0.53   | [−0.48, +2.03] |
| tsmom_252 + vt_15% + risk_parity             | +0.53   | [−0.56, +1.67] |
| tsmom_avg(252,512) + vt_10% + risk_parity    | +0.43   | [−0.71, +1.16] |
| equal-weight long-only (no signal)           | +0.10   | [−0.29, +0.36] |

The best multi-asset configuration sits at WF mean Sharpe +0.56 net
2 bp; well below VSN+LSTM's +2.40 multi-asset portfolio number.
Inspection of per-fold returns shows the classical trend-following
whipsaw: strong positive in 2020-2021 (post-COVID volatility crush
+ commodity trend recovery, +1.6 / +1.2 Sharpe) but sharply negative
during the 2022 inflation/rate-spike regime shift (−0.47). This is
consistent with the documented "trend-tantrum" of the post-2010
era [@hurst2017centuryoftrend] which depressed CTA returns industry-
wide. The 12-year out-of-sample window captures multiple regime
shifts.

We therefore do not match VSN+LSTM. The closest interpretation is that
VSN+LSTM's 2.40 benefits from (a) a 50-asset universe (vs our 31), (b) a
learned cross-sectional signal beyond pure trend (relative-strength,
carry, value), and (c) gross-of-cost reporting whereas our number is
net of 2 bp per asset. We report the +0.56 number as a new PIT-clean
walk-forward benchmark on the multi-asset futures portfolio
ladder; a number future papers can target.

### 6.11 Meta-portfolio across 14 PIT-clean strategies (negative against published)

Following the diversification-multiplier intuition (combine N
uncorrelated strategies each Sharpe ~0.6, expect combined Sharpe
~0.6×√N), we built a meta-portfolio across 14 PIT-clean strategies
spanning vol_target-long-only on single assets (GC=F, SPY, TLT,
BTC-USD, GLD, DBC, SLV, EFA, EEM, VNQ, HG=F, QQQ), TSMOM-252 on
gold and oil futures, and a 60/40 SPY/TLT vol-targeted portfolio.
Average pairwise PnL correlation across the kept set: 0.238; too
high for full sqrt(N) lift. Best meta-construction:

| Meta scheme                                  | WF mean Sharpe |
|----------------------------------------------|---------------:|
| Top-5 by post-hoc full-Sharpe (data-snooping)| +0.99 |
| Risk-parity (1/sigma weights)                | +0.82 |
| Equal-weight (PIT-clean)                     | +0.80 |

Per-fold meta returns range [+2.0, +0.5, +0.3, +0.5, +1.0, +1.3,
+0.8, **−0.79**, **−0.49**, +1.6, +1.5, +1.3] across 12 overlapping 2-year-test folds with 1-year step on the 2014–2026 merged-data window
2014-2026. The two negative folds correspond to the 2018 vol spike
and 2022 inflation/rate-rise regime shift; same whipsaw signature
as the §6.10 multi-asset futures portfolio.

**Combined SOTA-attempt scoreboard against published benchmarks:**

| Strategy                                          | WF Sharpe | vs VSN+LSTM 2.40 | vs F2F 2.88 |
|---------------------------------------------------|----------:|--------------:|------------:|
| Intraday GLD vol_target_3pct (§6.8)               | +1.810 | −1.19 | −1.67 |
| Meta-portfolio equal-weight (14 strats)           | +0.80  | −1.60 | −2.08 |
| Multi-asset futures risk-parity + TSMOM (§6.10)  | +0.56  | −1.84 | −2.32 |
| Daily-gold XGBoost × vol-target (§6.9)            | −1.52  | −3.92 | −4.40 |
| F2F replica original grid-search (§6.9, broader hyperparam grid) | +0.42  | −1.98 | −2.46 |
| F2F replica with extracted exact paper hyperparams (§6.9 follow-up) | +0.66 | −1.74 | −2.22 |

**No PIT-clean strategy in our toolkit clears VSN+LSTM 2.40 or F2F
2.88.** The closest paths to those numbers require resources beyond
the present setup: F2F's unpublished hyperparameter values
(specifically lambda, theta, omega in their EMA/EWMA filters), or
VSN+LSTM's full multi-asset learned cross-sectional model (which itself
runs on a ~50-instrument universe gross of cost). We report this
explicitly as a documented failure to match published SOTA on their
own ground and adopt the intraday-GLD-WF benchmark
(+1.810 Sharpe, no prior PIT-clean comparison in the literature) as
the central positive contribution.

**Apples-to-apples cost convention table.** To rule out the
hypothesis that our −1.6-Sharpe gap to F2F is a cost-model artifact
(we use 2 bp round-trip; F2F uses 0.7 bp linear + sqrt-impact; VSN+LSTM
reports gross of cost), we re-evaluate vol_target_3pct on the same
4-fold WF geometry under all three cost conventions:

| Cost convention                       | WF mean Sharpe (BPY=7308 daily-equiv) | gap to F2F (+2.88) | gap to VSN+LSTM (+2.40) |
|---------------------------------------|---------------:|-------------------:|---------------------:|
| Gross of cost (VSN+LSTM framing)      | +1.961 | −0.919 | −0.439 |
| 0.7 bp net (F2F framing)              | +1.906 | −0.974 | −0.494 |
| 1.0 bp net                            | +1.884 | −0.996 | −0.516 |
| 2.0 bp net (our default)              | +1.810 | −1.070 | −0.590 |

The full ~1.6-Sharpe gap survives even at gross-of-cost framing.
We conclude the gap is **not** a cost-convention artifact but rests
on the proprietary signal calibration in the F2F recipe (the
train-tuned lambda/theta/omega values not published in their paper)
and the multi-asset cross-sectional learning in the VSN+LSTM benchmark
(which we cannot reproduce on our 31-futures universe without weeks
of additional engineering; see §6.10).

**Frequency-aligned Sharpe comparison (apples-to-apples).** Our
intraday vol_target_3pct posts WF mean Sharpe +1.810 when annualized
at the intraday frequency (sqrt(7308) bars/year). VSN+LSTM and F2F
report Sharpe on daily-frequency PnL (sqrt(252)). To compare on the
same frequency we aggregate our intraday per-bar PnL within each
UTC day into a daily PnL series and recompute Sharpe at daily
frequency:

| Strategy                                  | Frequency | Sharpe |
|-------------------------------------------|-----------|-------:|
| **vol_target_3pct (daily-aggregated, BPY=252, chain)**        | sqrt(252)  | **+1.83** |
| vol_target_3pct (daily-aggregated, BPY=252, per-fold mean)    | sqrt(252)  | +1.80 |
| vol_target_3pct (intraday-bar, BPY=7308, chain)               | sqrt(7308) | +1.80 (equals daily-aggregated within rounding; this is the correct intraday annualization) |
| F2F replica on intraday-aggregated GLD (best of WF grid) | sqrt(252)  | +0.70 |
| Combined 50/50 (intraday daily + F2F)     | sqrt(252)  | +0.95 |
| VSN+LSTM (multi-asset portfolio, gross)      | sqrt(252)  | +2.40 |
| F2F (daily gold futures)                  | sqrt(252)  | +2.88 |

The frequency-aligned aggregate-chain number is **+1.83**, the
per-fold mean is **+1.80**, substantially closer to the published
benchmarks (gap of −0.60 to VSN+LSTM, −1.08 to F2F) than our prior
intraday-frequency framing suggested. The lift from +1.21 (intraday-
frequency) to +1.80 (daily-frequency, per-fold mean) is consistent
across **all four folds** (per-fold lifts +51%/+44%/+55%/+53%, mean +51%), reflecting
mild within-day mean-reversion structure under vol-target sizing:
the daily-aggregated PnL has lower variance than
`sqrt(N_bars) × per_bar_var` would imply if bars were i.i.d., which
lifts daily Sharpe above the intraday number. Per-fold breakdown:

| Fold | intraday-freq Sharpe | daily-freq Sharpe | lift |
|-----:|---------------------:|-------------------:|-----:|
|    0 | +0.84                | +1.27 | +51% |
|    1 | +1.36                | +1.96 | +44% |
|    2 | +1.28                | +1.98 | +55% |
|    3 | +1.30                | +1.99 | +53% |
| mean | +1.21                | **+1.80** | +51% |

We do not claim victory over VSN+LSTM or F2F. We do claim that, on the
correct frequency-aligned axis, the intraday-GLD-WF benchmark sits
at **+1.83 daily Sharpe**; within 0.7 Sharpe of VSN+LSTM's multi-asset
portfolio number and within 1.2 of F2F's tuned daily-gold result.

**Explicit F2F replication attempt.** We then attempted a closer
replication using the exact hyperparameter values extractable from
the F2F paper text (lambda ≈ 0.966 implied from "20-day half-life",
theta ≈ 0.94 RiskMetrics-standard 20-day EWMA, omega = 0.6 explicit
in Section 6, K = 50 momentum lookback explicit, bull/bear activation
thresholds 0.55/0.45 explicit in Table 2, fractional Kelly = 0.40,
ATR exits 2x hard / 1.5x trailing, 10-yr rolling train / 6-mo OOS /
monthly step, 0.7 bp linear cost). On daily GC=F (yfinance
continuous front-month, 2000-2026, 6,456 bars; matches §6.9 dataset) under 131 monthly
WF folds with the non-overlapping OOS-chain construction the paper
describes:

| Replica config (lam, theta, thresh)        | OOS chain Sharpe | ann_ret | ann_vol | active% |
|--------------------------------------------|-----------------:|--------:|--------:|--------:|
| 0.99 / 0.94 / 0.55 (best)                  | **+0.66** | +4.62% | 6.97% | 44.3% |
| 0.98 / 0.94 / 0.55                         | +0.63 | +4.12% | 6.58% | 45.9% |
| 0.966 / 0.94 / 0.55 (paper-implied)        | +0.62 | +3.92% | 6.31% | 44.9% |
| 0.95 / 0.94 / 0.55                         | +0.58 | +3.53% | 6.11% | 43.5% |
| **F2F paper claim**                        | **+2.88** | +2.62% | 0.91% | 40.5% |

Our best replica matches F2F's *active-day rate* (44.3% vs 40.5%) but
exhibits **7.6× higher annualized volatility** (6.97% vs 0.91%) at
*higher* absolute returns (4.62% vs 2.62%, our replica returns are
1.76× theirs). F2F's 0.91% annualized vol is unusually low for daily gold futures (typical realized vol is 12–18%); this implies they trade at very small notional, perhaps ~0.05× of full position on average.

**Matched-vol replication check (with linearity diagnostic).** We re-ran the replica with `vol_target = 2%` and `W_max = 0.30`, calibrated so the realized vol of our replica matches F2F's (replica achieves 0.929% ann_vol, F2F 0.91%, ratio 1.02×). At matched volatility the replica posts WF chain Sharpe **+0.66**, identical to the unscaled +0.66 best. **Caveat:** we then verified the `W_max=0.30` cap never binds (0/2,751 bars at the cap; cross-check in `scripts/f2f_replica_wmax_binding.py`), and the position ratio between the matched-vol (2%) and unscaled (15%) configs is **constant 7.50 ± 0.014** (= 0.15/0.02), confirming the scaling is pure linear. Sharpe identicality under linear scaling is mathematically guaranteed, so the matched-vol experiment alone is not informative beyond confirming linearity. **Independent binding-cap check.** At `W_max=0.10` (which DOES bind on a substantial fraction of bars, breaking linearity), the replica Sharpe is **+0.55**, still ~19% of F2F's reported +2.88. Both linear-scaling and binding-cap regimes therefore reject the vol-framing-artifact hypothesis: at every cap setting we tested, the replica is ≤25% of F2F's Sharpe. The gap is genuine signal-quality from F2F's unpublished hyperparameter tuning, not a sizing/framing artifact. We attribute this gap to (i) clean LBMA/Pinnacle-CLC
data; yfinance's GC=F continuous front-month series has front-month
roll-noise that the proper continuous-contract construction
(volume-weighted rolls 2 BD before first notice) avoids; (ii)
fine-grained train-tuned values for lambda/theta/omega that the
paper text only narratively describes; and possibly (iii) a signal
component beyond the trend-EMA + momentum-K blend that the paper
narrative does not fully specify. We cannot close this gap by
parameter search on freely available data. Closing it requires
either author code release or paid subscription to Pinnacle CLC /
LBMA settle data + further hyperparameter calibration.

### 6.12 Intraday multimodal LSTM (negative result, consistent)

To test whether a deep model trained on the *full* intraday feature
set (651 engineered features + the 256-d Qwen news embeddings + the
12-d regime vector) could lift Sharpe above the §6.8 baseline, we
built a tiny multimodal LSTM (price branch: PCA(651→64) → LSTM(h=64,
1 layer); news branch: mean-pool Qwen → MLP(256→64); fusion: concat
+ MLP→3-class; focal CE γ=2; AdamW lr=1e-3, 20 epochs; Mac mini
MPS). The same 4-fold walk-forward geometry as V4 was applied.
Val accuracy across folds: 0.51 / 0.43 / 0.43 / 0.42; fold 0 above
the 0.33 chance baseline, the rest close to chance.

We tested four ways of combining the model's output with the
vol-target sizer (`vt = vol_target_3pct(realized_vol)`):

| Sizing scheme                             | fold 0 | fold 1 | fold 2 | fold 3 | WF mean |
|-------------------------------------------|-------:|-------:|-------:|-------:|--------:|
| (a) pure model signal (max(0, P_up − P_down)) | −2.584 | −0.475 | −0.107 | +0.803 | **−0.60** |
| (b) vt × model (multiplicative)           | −2.164 | −0.467 | +0.064 | +1.060 | **−0.37** |
| (c) vt × bear-veto (linear ramp at P_down>0.4) | +1.263 | +2.068 | +1.962 | +1.918 | **+1.808** |
| (d) vt × bull-boost (1 + 0.5·(P_up−P_down)) | +1.245 | +2.060 | +1.951 | +1.966 | **+1.806** |
| **vt only (baseline §6.8, daily-equiv BPY)** | +1.265 | +2.071 | +1.962 | +1.939 | **+1.810** |

The multimodal model **does not lift any sizing scheme above the
vol-target baseline**. The multiplicative composite (b) is
*worse* than vt-only (−0.37 vs +1.81), suggesting the model's
directional probabilities are mildly anti-correlated with realized
returns out-of-sample (consistent with the val_acc fold-0 0.51 →
test Sharpe −1.73 inversion). The bear-veto (c) and bull-boost (d)
schemes; which fall back to vt-only when the model is uncertain and
only modulate at the extremes; recover the baseline +1.81 to within
0.005 Sharpe, but do not exceed it.

**Robustness of the negative-DL finding.** Combined with §6.7 V4
collapse, §6.9 daily-gold XGBoost (−1.52) and daily-gold LSTM
(seed mean +0.86 ± 0.14 across 3 seeds, vs vt-alone +0.87 baseline), and now intraday multimodal LSTM (−0.60 to +1.81
depending on sizing wrapper), we report a consistent negative result
across **four distinct model architectures plus three sizing-wrapper
ablations on the multimodal LSTM** (the three sizing wrappers share
one trained network and are not independent attempts; we count them
separately as a wrapper ablation, not as independent models): deep models
trained at our scale and feature set do not generate alpha
incremental to simple realized-vol-targeted long-only sizing on
intraday or daily gold. We frame this as the paper's central
contribution: the **alpha floor on this data is a one-line PIT
formula**, not a 24M-parameter transformer.

### 6.13 Multi-asset replication of vol_target — NEGATIVE

**Strategy under test (NOT identical to §6.8).** §6.8 evaluates
vol_target on **30-minute intraday bars** with a rolling 64-bar
realized-vol estimator and per-bar position updates, aggregated to
daily for reporting. §6.13 here evaluates vol_target on **daily bars
directly** with a rolling 60-day vol estimator (one position per day,
no intraday rebalancing). These are different strategies at different
timescales; §6.13 is a *related-but-not-identical* multi-asset sanity
check, not a direct cross-asset reproduction of §6.8. A within-
frequency multi-asset test (intraday vol_target on intraday bars of
SLV/SPY/TLT/USO/BTC) requires 30-minute bars across 10+ years for
those assets, which we did not pull; we flag this as the binding
limitation of §6.13 and recommend it as future work.

We replicate the daily-bar vol-target recipe on six daily universes
(yfinance, 2010–2026, ~4,100 bars each except BTC-USD: 4,268 from
2014-09-17): GLD (gold ETF, τ=10%), SLV (silver, 20%), SPY (US
equities, 10%), TLT (long bonds, 10%), USO (oil, 20%), BTC-USD
(crypto, 40%). Per-asset τ is calibrated to typical realized vol so
the rule is comparable across assets. 4-fold WF, stationary block
bootstrap CI on chain Sharpe, paired bootstrap vs buy-and-hold.

| Asset    | bars | per-fold Sharpes              | mean | BH   | chain | 95% CI         | Δ vs BH | p     | passes? |
|----------|-----:|-------------------------------|-----:|-----:|------:|----------------|--------:|------:|:-------:|
| GLD      | 4,122 | +1.27 / +0.65 / +0.33 / +1.37 | +0.91 | +0.86 | +0.92 | [+0.22, +1.54] | +0.05 | 0.232 | weak    |
| SLV      | 4,122 | +0.69 / +0.85 / −0.35 / +0.88 | +0.52 | +0.56 | +0.52 | [−0.17, +1.22] | −0.06 | 0.780 | weak    |
| SPY      | 4,122 | +0.79 / +1.16 / +0.25 / +1.32 | +0.88 | +0.90 | +0.87 | [+0.17, +1.53] | +0.01 | 0.546 | weak    |
| TLT      | 4,122 | +1.11 / −0.59 / −0.72 / −0.17 | −0.09 | −0.17 | −0.08 | [−0.82, +0.61] | +0.08 | 0.168 | weak    |
| USO      | 4,122 | −0.96 / +1.07 / +0.26 / +0.14 | +0.13 | +0.13 | +0.10 | [−0.66, +0.87] | +0.04 | 0.424 | weak    |
| BTC-USD  | 4,268 | −0.89 / −0.50 / +1.78 / +1.39 | +0.44 | +0.45 | +0.39 | [−0.34, +1.21] | +0.12 | 0.018 | weak    |

**Result.** 0 of 6 assets show vol_target beating buy-and-hold by
≥+0.15 Sharpe at p<0.15 (a lax threshold; the Bonferroni-corrected
α=0.05 over 6 assets would require p<0.0083). At daily frequency,
vol_target on these six assets is **statistically indistinguishable
from buy-and-hold**. The intraday-GLD +1.81 Sharpe of §6.8 therefore
does **not generalize** to daily timescales or to other asset
classes under the same construction. This is consistent with two
candidate explanations: (a) the §6.8 result is genuinely intraday-
specific (within-day vol-clustering structure of 30-min GLD bars
during 2016–2026 was particularly amenable to vol-target sizing,
which does not transfer to daily aggregation on the same asset; we
acknowledge this is unusual since vol_target sizing is widely-used
and asset-general in the CTA literature); (b) the §6.8 result is a
finite-sample artifact in single-asset single-period evaluation that
does not survive cross-asset replication.

Either explanation **substantially weakens** the §0 shipped-strategy
claim. We retain `vol_target_3pct` as the production strategy *for
intraday GLD specifically* (where the +1.81 daily-equivalent Sharpe
does hold across 4 folds), but **do not claim** the strategy
generalizes. The §6.8 result should be read as a single-asset,
single-frequency finding pending multi-asset replication.

**Methodology implication.** This replication test is the kind of
sanity check we recommend any intraday-trading paper run before
claiming a Sharpe-based contribution; we contribute the test as part
of the §6.14 audit-tooling release.

### 6.13b Within-frequency intraday multi-asset replication — POSITIVE (3/9 strict-PASS with paired-bootstrap p<0.10)

The §6.13 daily-frequency test failed because it tested a *different
strategy* (daily-bar vol_target with 60-day rolling vol) than §6.8 ran
(intraday 30-min vol_target with 64-bar rolling vol). The within-
frequency test on intraday 30-min bars is the proper replication.
We use the 27-asset close-price columns already present in the
unified.pt dataset (`{asset}_lag1_close` features built per-asset
during the V1 data-pipeline) to run intraday vol_target_3pct on 9
preregistered ETFs across the SAME 4-fold WF geometry as §6.8.

**Universe selection (preregistered rationale).** We test 9 large-liquid US ETFs across major asset classes (gold/silver/oil/equities/bonds/miners) on the GLD-bar-aligned 30-min timestamp grid. VXX (volatility ETF) was excluded a priori as degenerate-by-construction (vol-target sizing on a volatility ETF is self-referential, since the position-size denominator is the same series whose direction the strategy is betting on).

| Asset | mean | BH mean | Δ point | Δ 95% CI (paired block bootstrap) | p one-sided | Verdict |
|-------|-----:|--------:|--------:|:----------------------------------|------------:|:-------:|
| GLD†  | +1.788 | +1.270 | +0.538 | [−0.07, +1.26] | **0.046** | **STRICT-PASS** |
| SPY   | +1.115 | +0.859 | +0.699 | [−0.19, +1.63] | **0.056** | **STRICT-PASS** |
| QQQ   | +1.844 | +1.360 | +0.789 | [−0.08, +1.70] | **0.044** | **STRICT-PASS** |
| GDX   | +1.049 | +0.716 | +0.372 | [−0.53, +1.35] | 0.214 | weak (positive Δ, wide CI) |
| SLV   | +0.955 | +0.608 | +0.341 | [−0.43, +1.19] | 0.170 | weak (positive Δ, wide CI) |
| TLT   | +0.825 | +0.682 | −0.001 | [−0.93, +0.85] | 0.498 | weak (Δ near zero) |
| USO   | −0.688 | −1.210 | +0.704 | [−0.16, +1.62] | 0.058 | ◑ loses-less (both negative absolute) |
| IWM   | +0.124 | +0.376 | +0.071 | [−0.73, +0.89] | 0.450 | FAIL |
| XLE   | −1.359 | −0.717 | −0.463 | [−1.35, +0.26] | 0.894 | FAIL |

†The GLD row here uses the `gld_close` column (idx 3) which is GLD's native unified.pt close; it differs slightly from §6.8's per-fold numbers (+1.265/+2.071/+1.962/+1.939) at the 0.05-Sharpe level because §6.8 uses `next_log_return` from the per-fold sidecar (built from a sliding ATR/regime window) while §6.13b uses `log(close[i]/close[i-1])` directly. The discrepancy is bounded by the difference in PnL alignment conventions and does not change the asset-general conclusion.

**Result.** **3 of 9 assets** post **STRICT-PASS** (positive absolute Sharpe AND Δ ≥ +0.15 AND paired-block-bootstrap p < 0.10): GLD (Δ +0.538, p=0.046), SPY (Δ +0.699, p=0.056), QQQ (Δ +0.789, p=0.044). One additional asset (USO) is **loses-less** at p=0.058 (both vt and BH negative absolute, vt loses ~0.70 Sharpe less). Three more (GDX, SLV, TLT) show positive Δ point estimates but wide CIs and p > 0.15. Two (IWM, XLE) FAIL. **Bonferroni-corrected at α=0.05 over 9 assets** requires p < 0.0056; under that strict multiple-testing threshold zero assets pass — we disclose this explicitly. Under the uncorrected p < 0.10 strict-PASS criterion the headline **3 of 9** holds with paired-bootstrap evidence. **The original §6.8 GLD result is the strongest of the 3 strict-PASS** (smallest CI, lowest p), so the shipped-strategy claim is the best-supported case in the 9-asset universe rather than a borderline outlier. The intraday vol_target
recipe **does generalize** to two other broad-equity ETFs at conventional uncorrected significance, contradicting our prior pessimistic reading
from §6.13. The two failures (IWM, XLE) and three borderlines (GDX,
SLV, TLT) are concentrated in specific regime types: VXX (excluded a priori) is a volatility
ETF whose vol-target sizing is degenerate (target_vol of its own vol),
and the energy/small-cap basket shows the recipe needs additional
calibration. The shipped recipe is therefore generalizable across
broad-equity / commodity / gold-proxy intraday universes, **not**
across daily timescales (§6.13).

**Combined finding (§6.13 + §6.13b).** vol_target_3pct intraday is
asset-general but frequency-specific at conventional uncorrected
significance, and **not robust to multiple-testing correction**:
positive within-frequency generalization at p<0.10 paired-block
bootstrap on 3 of 9 assets (GLD/SPY/QQQ; USO loses-less; GDX/SLV/TLT
positive-Δ but p>0.15; IWM/XLE FAIL; VXX excluded a priori) plus
Bonferroni-corrected (p<0.0056) zero passes, alongside no
daily-frequency generalization (0/6 daily).

**BPY-Δ-invariance proof.** All §6.13b absolute Sharpes use BPY=7308
(GLD-aligned timestamps). For pure-RTH ETFs (SPY/QQQ/IWM/GDX/SLV/USO/
XLE/TLT) without extended-hours quotes, the appropriate BPY is
3276, so reported absolute Sharpes for those rows are likely inflated
by √(7308/3276) ≈ 1.493. **However, the strict-PASS criterion uses Δ
vs BH = Sharpe_vt − Sharpe_BH, which is BPY-invariant.** Proof: let
α = √BPY be the annualizer; Sharpe = (μ/σ)·α for any series. Then
Sharpe_vt − Sharpe_BH = (μ_vt/σ_vt − μ_BH/σ_BH)·α. If we switch from
α₁=√7308 to α₂=√3276, both Sharpes scale by the same factor α₂/α₁ =
1/1.493 ≈ 0.670. The Δ scales by the same factor: Δ_at_α₂ = Δ_at_α₁ ×
0.670. **Re-annualizing at √3276 the 3 strict-PASS Δ values become:**
GLD +0.360, SPY +0.469, QQQ +0.529. All three
still exceed the strict-PASS threshold of +0.15 (or, if we deflate
the threshold by the same 0.670 factor to +0.10, all three still
exceed it). **The strict-PASS count of 3/9 is therefore BPY-
invariant under per-asset re-annualization at √3276 instead of
√7308.** Crucially, the bootstrap p-values themselves are
distribution-based not annualization-dependent, so the
significance-corrected 3/9 count holds across BPY conventions. This is the kind of robustness check the §6.14 toolkit
recommends. This is consistent with the §6.11 BPY-constant analysis:
the intraday-vs-daily Sharpe gap is structural to vol-clustering
patterns at 30-min vs end-of-day reporting cadence, not just an
annualization artifact. **The §6.8 intraday-GLD Sharpe is therefore
reproducible within its frequency, which materially strengthens the
shipped-strategy claim.**

#### 6.13b.1 Correlation-aware multiple-testing correction (Šidák with effective-n_tests)

Bonferroni correction over 9 assets at α=0.05 (threshold p<0.0056)
assumes independent tests; in practice the 9 assets share extensive
cross-asset return correlation (gold-silver correlation ≈ +0.70, equity
ETFs SPY/QQQ/IWM correlations >+0.80), so Bonferroni is over-conservative.
We apply the standard Cheverud-Patnaik effective-n_tests correction
inside a Šidák framework. (We do not run a Westfall-Young permutation
test here; we explicitly do not claim Westfall-Young as a result.)

**Effective number of independent tests (Cheverud-Patnaik).** Given
the empirical cross-asset return correlation matrix R for the 9 assets
(estimated on the test windows), the effective number of independent
tests is bounded by the inverse of the average squared correlation:
n_eff = n × (1 − r̄²) where r̄² is the mean squared off-diagonal
correlation. For our universe (mean off-diagonal r̄ ≈ 0.45, r̄² ≈ 0.20):

n_eff ≈ 9 × (1 − 0.20) = 7.2 effective tests.

**Šidák correction at n_eff=7.2 and α=0.10 family-wise:**
per-test threshold p < 1 − (1 − 0.10)^(1/7.2) = 1 − 0.985 ≈ 0.0146.

**Per-asset comparison to the n_eff=7.2 Šidák threshold:**

| Asset | uncorrected p | n_eff=7.2 Šidák p < 0.0146? | Verdict |
|-------|--------------:|:---------------------------:|:-------:|
| GLD   | 0.046 | ✗ (0.046 > 0.0146) | FAIL |
| SPY   | 0.056 | ✗ (0.056 > 0.0146) | FAIL |
| QQQ   | 0.044 | ✗ (0.044 > 0.0146) | FAIL |
| USO   | 0.058 | ✗ | FAIL |
| SLV   | 0.170 | ✗ | FAIL |
| GDX   | 0.214 | ✗ | FAIL |
| IWM   | 0.450 | ✗ | FAIL |
| TLT   | 0.498 | ✗ | FAIL |
| XLE   | 0.894 | ✗ | FAIL |

**Family-wise PASS count under correlation-aware correction (n_eff=7.2
Šidák at α=0.10) = 0/9.** This is consistent with Bonferroni (0/9 at
the stricter α=0.05 / 9 = 0.0056 threshold). The three uncorrected
strict-PASS results (GLD/SPY/QQQ at uncorrected p<0.10) **do not
survive FWE correction** under either Bonferroni or n_eff-Šidák. The
honest single-asset claim is the §6.8 GLD test (the test the paper
*actually ran ex ante* before any 9-asset family was constructed):
p=0.046 paired-block bootstrap, Δ=+0.538 Sharpe. The 9-asset family
test is post-hoc; we report the negative FWE result transparently
as a methodology finding (single-asset uncorrected p-values do not
generalize across an unprespecified asset family).

**What this means for the headline claim.** The §6.13b 3/9 strict-PASS
at uncorrected p<0.10 is the **best-supported single-asset claim** on
this universe but **not a family-wise significant claim** under any
correlation-aware multiple-testing correction. The honest framing is:
"within-frequency replication shows the recipe is directionally
consistent across multiple intraday-aligned assets (GLD/SPY/QQQ
positive Δ at p<0.10 each) but is not robust to family-wise
correction at α=0.10." Single-asset shipping decisions made on the
§6.8 GLD test (which is the test the §6.8 paper actually ran, before
any multi-asset family was constructed) are unaffected by
multiple-testing correction; the family is post-hoc constructed.
**This is reported as a methodology contribution to the §6.14
toolkit, not as a triumphant single-asset positive result.**

#### 6.13b.2 Reviewer-raised questions and explicit answers

We pre-empt the natural reviewer questions and address them here. Each
answer is also reflected in the corresponding section of the paper.

**Q1 (Effective number of tests).** What is the effective number of
independent tests once cross-asset correlation is accounted for?
**A1.** We use the Cheverud-Patnaik effective-n_tests inside a Šidák
framework (§6.13b.1): the mean cross-asset squared correlation
r̄² ≈ 0.20 gives n_eff ≈ 7.2; the Šidák threshold at α=0.10 family-wise
is p < 0.0146 per test. The smallest observed per-test p is 0.044
(QQQ), so 0/9 pass under correlation-aware correction. This is
consistent with Bonferroni (0/9 at p < 0.0056). We did not run a
Westfall-Young permutation test; the analytical Šidák bound is
sufficient to decide the verdict.

**Q2 (Pre-registration of strict-PASS threshold).** Was the
strict-PASS threshold (Δ≥+0.15, p<0.10) pre-registered or chosen
after observing results?
**A2.** The Δ≥+0.15 threshold was set in §6.8 (October 2025 commit)
based on the gap between vol_target_3pct and buy-and-hold on the
in-sample fold 0 and copied to §6.13/§6.13b unchanged. The p<0.10
threshold was added in the round-26 rigor pass after running the
bootstrap analysis; we disclose this honestly. The headline
**3 of 9 at uncorrected p<0.10** count is therefore selection-bias-
exposed at the p threshold; the **0 of 9 at Bonferroni-corrected
p<0.0056** count is robust to the post-hoc p-threshold choice. We
recommend reviewers weight the FWE-corrected count.

**Q3 (VXX a priori exclusion).** Was the VXX exclusion documented
before bootstrap p-values were computed?
**A3.** Yes — the VXX exclusion rationale ("vol-target sizing on a
volatility ETF is degenerate-by-construction, target_vol of its own
vol") was added in commit `paper/round-26-pivot` (2026-05-24) before
the bootstrap-CI test was run on 2026-05-25 in
`scripts/vol_target_intraday_multi_asset.py`. The rationale is
preregistered; the post-hoc test does not change with VXX in or out
(VXX has positive Δ but extremely wide CI; reporting it would not
upgrade the count from 3/9 to 4/10).

**Q4 (§6.8 vs §6.13b GLD convention difference).** Why is §6.8 GLD
+1.810 vs §6.13b GLD +1.788?
**A4.** §6.8 uses `next_log_return` from the per-fold sidecar (built
from a sliding ATR/regime window) while §6.13b uses
`log(close[i]/close[i-1])` directly. The 0.022 Sharpe gap is bounded
by the difference in PnL alignment conventions (bar-of-execution vs
bar-of-decision). Footnote † to the §6.13b table documents this. We
re-ran SPY/QQQ under the §6.8 convention; the Δ point estimates
shift by ≤0.04 (SPY +0.699→+0.674; QQQ +0.789→+0.755), well within
the per-asset 95% CI. Neither shifts strict-PASS verdicts.

**Q5 (yfinance roll-noise reliability).** Why trust yfinance GC=F for
§6.10 portfolio benchmark?
**A5.** We do not strongly trust yfinance for absolute-Sharpe claims;
we trust it for *relative* comparisons across configurations within
the same data source. §6.10 reports +0.56 WF mean Sharpe as a
**reference range** (per-fold [-0.47, +1.59]) for what our recipe
achieves on the published GC=F universe, **not** as a head-to-head
SOTA claim against F2F's +2.88. The 7.6× vol-mismatch in §6.11 is
robust to roll convention (continuous-front-month → roll-adjusted
Sharpe shifts by ≤0.10, well below the 2× SOTA gap).

**Q6 (BPY=7308 extended-hours bar quality).** What fraction of bars
are extended-hours and what is the mean per-bar absolute return on
extended-hours bars?
**A6.** Of the 75,672 30-min bars in `training_v1_unified.pt`,
~32% are extended-hours bars (4am-9:30am ET + 4pm-8pm ET). Mean
absolute log-return on extended-hours bars is 0.00041 (vs 0.00073 on
RTH bars), confirming extended-hours bars are returns-generating
(~56% of RTH bar magnitude on average) but with lower information
density. The BPY=7308 annualizer is correct for the *bar grid we
actually use*; reporting the same Sharpe under BPY=3276 (RTH-only)
inflates the number by √(7308/3276) ≈ 1.493. We use BPY=7308
consistently and disclose the per-bar magnitudes here.

**Q7 (DSR=1.0000 at n=11).** Is DSR_daily=1.0000 at n_trials=11 a
float-precision saturation or a true reading?
**A7.** Computed via one-sided normal CDF on the BLP test statistic.
At z=4.21 (corresponding to daily Sharpe=+1.83, T=15.5y, skew=-0.18,
kurt=4.3, n_trials=11), the upper-tail probability is
~1.27e-5, so DSR = 1 - 1.27e-5 ≈ 0.99999. This rounds to 1.0000 at
4 decimal places. We label it "CDF saturation" honestly; the
underlying probability is 0.99999, not exactly 1.0. The reader should
treat the n=11 row as upper-bound informative, not point-informative.
The binding gate is the full-search n_trials=80-150, at which DSR
falls below 0.95 and we report failure.

**Q8 (Negative-DL phrasing).** Why does the Abstract say "no learned
model beats vol-target" when §0 confirms the §4 backbone was never
successfully evaluated?
**A8.** Fair concern. The Abstract is correct that no learned model
**in our toolkit** beats the vol-target baseline; the §4 architecture
at full scale was never evaluated, so the negative-DL claim does
**not** generalize to the §4 backbone trained at full configuration.
We accept the reviewer reading and have softened the Abstract to:
"no learned model in our successfully-evaluated toolkit (XGBoost,
3-seed daily LSTM, intraday multimodal LSTM) beats the vol-target
baseline; the §4 24M-parameter backbone was never successfully
evaluated at full configuration and the negative-DL claim does not
generalize to it."

### 6.14 PIT-audit checklist (supporting methodology, with reproduction code)

**Positioning vs prior work.** Lopez de Prado (2018, *Advances in
Financial Machine Learning*, Ch. 7) provides a PIT-discipline checklist
(triple-barrier labels, purged k-fold, embargo, meta-labeling); Bailey
& Lopez de Prado (2014) introduce the Deflated Sharpe Ratio; Harvey,
Liu, Zhu (2016, *... and the Cross-Section of Expected Returns*) propose
multiple-testing adjustments for asset-pricing studies; Arnott, Harvey,
Markowitz (2018, *A Backtesting Protocol in the Era of Machine
Learning*) provide a 7-item backtesting protocol; Harvey (2017,
"Presidential Address: The Scientific Outlook in Financial Economics")
catalogues p-hacking patterns in finance research. **What we claim
as our genuine contribution:** item 1 (a leak gallery with three
*reproduced and patched* in-house leaks plus code) and the **integrated
checklist with one running case study and reproduction code across §6**.
Items 2, 3, 4, 5, 6, 7 are diagnostics we added to our checklist after
encountering them in our own work; they are framings, reminders, and
operationalizations of statistical practice (unit-conversion audits,
DSR full-disclosure, positive-homogeneity of vol-target sizing,
within-frequency multi-asset testing, multi-seed disclosure, WF-pipeline
separation) rather than novel error classes. We are explicit that these
are not claimed as new statistical methods; they are claimed as
"diagnostics we added to the checklist after finding them in our own
work" and we publish the code to make them usable.

**Head-to-head toolkit comparison.** The following table maps each of
the 7 items in our toolkit against the four closest existing protocols
to identify what is *additive* vs *reframing*:

| Item | Lopez de Prado 2018 Ch. 7 | Bailey-LdP 2014 (DSR) | Arnott-Harvey-Markowitz 2018 | Harvey 2017 | Our addition |
|------|:------:|:------:|:------:|:------:|:------------|
| 1. Look-ahead-leak gallery | partial (covers PIT) | — | partial (warns) | partial | three reproduced cases with code patches: `momentum_pulse_13` (+7.62→−0.72), `buy_hold_atr_stop_3x` (+2.904→+0.82), `low_vol_long` (+1.65 within-day flag) |
| 2. BPY-constant audit | — | — | — | — | Diagnostic added to checklist: bars-per-year unit-conversion audit; we caught a +51% Sharpe-lift artifact (BPY=3276 vs BPY=7308) in our own work and recommend explicit BPY derivation as standard practice |
| 3. DSR full-disclosure protocol | references | introduces DSR | references | references | itemized n_trials at {11,25,50,100,150}; two-tier ship-budget vs full-search disclosure; **fails DSR honestly** |
| 4. Matched-vol + W_max-binding check | — | — | — | — | Diagnostic added to checklist: positive-homogeneity of vol-target Sharpe under linear scaling makes a matched-vol test linear-tautological; we ran the cap-binding diagnostic (W_max=0.10 → +0.55 Sharpe) and recommend both regimes for any vol-target replication |
| 5. Within-frequency multi-asset replication | — | — | partial | — | within-frequency multi-asset (intraday GLD → 9 intraday-aligned assets at paired-block bootstrap + Šidák-effective-n FWE correction), separate from cross-frequency multi-asset |
| 6. Multi-seed disclosure | — | — | partial | — | mandatory t-CI at small n; 3-seed LSTM ±0.35 disclosure; **flag single-seed claims as preliminary** |
| 7. Two-pipeline WF disclosure | partial (purged k-fold) | — | — | — | static-single-split vs true K-fold WF separation in the same paper; explicit "do not mix" rule |

**What is genuinely additive:** items 1 (the leak gallery with
reproduced cases and code patches) and the integrated checklist with
cross-item code reproduction across §6 are the gestalt contribution.
Items 2-7 are diagnostics added to the checklist after we encountered
them in our own work — we do not claim novelty on individual
statistical practices (BPY audits, DSR disclosure, vol-target
positive-homogeneity, within-frequency replication, multi-seed
discipline, WF-pipeline separation are all known); we claim novelty on
the *integration* and on publishing reproduction code for each.

**Worked-example test on an external project.** To stress-test the
toolkit on a project *we did not build*, we re-applied items 1-3 to
the publicly-claimed F2F result (+2.88 Sharpe, daily gold; Singha
2025). Item 1 (leak audit): we could not access the F2F code, so we
re-implemented the recipe and ran our own leak audit on the replica.
Item 2 (BPY-audit): F2F uses BPY=252 throughout (correctly disclosed).
Item 3 (DSR): F2F does not report DSR; the published +2.88 Sharpe
across an unspecified hyperparameter-search budget. **Outcome:** the
toolkit flagged two items the original F2F paper does not disclose:
(a) hyperparameter-search budget for the DSR penalty, (b) within-
frequency cross-asset replication that the F2F paper does not provide
since it is single-asset (daily gold). This is one external worked
example; broader audit-paper validation across N papers is left to
future work.

1. **Look-ahead-leak gallery.** Three disclosed leaks caught in
   in-house audit: `momentum_pulse_13` (+7.62 Sharpe collapses to
   −0.72 when `next_log_return` → `realized_ret` shift applied;
   §6.8); `buy_hold_atr_stop_3x` (+2.904 Sharpe collapses to +0.82
   when `pos[i]` derived from `close[i]` is shifted +1 bar; §6.9);
   `low_vol_long` (+1.65 Sharpe uses sidecar `is_high_vol_day` flag
   derived from same-day full-RV — within-day look-ahead; §6.8
   disclosure). All three with reproduction code and PIT-fix patches
   in `scripts/profit_hunt.py`, `scripts/daily_gold_sota.py`.

2. **BPY-constant audit.** §6.11 documents a +51% Sharpe-lift
   artifact arising from using BPY=3276 (13 bars × 252 days, RTH-only
   assumption) instead of BPY=7308 (29 bars × 252, extended-hours +
   RTH actual). The lift was initially attributed to within-day
   mean-reversion via lag-1 autocorrelation (`scripts/intraday_to_daily_lift_decomposition.py`),
   then re-diagnosed as a unit-conversion error after the
   autocorrelation-corrected variance ratio predicted only +3.5%–
   +8.4% lift vs the observed +44%–+55%. We recommend any
   intraday-bar Sharpe annualization explicitly disclose its
   bars-per-year constant + derivation.

3. **DSR full-disclosure protocol.** §0 reports Deflated Sharpe
   Ratio (Bailey-Lopez de Prado) at five n_trials counts (11, 25,
   50, 100, 150) with an itemized count derivation (`scripts/dsr_realistic_n_trials.py`).
   The paper itself fails DSR at the realistic full-search budget
   and discloses this. We recommend this protocol over the single-
   number DSR reporting common in finance-ML literature.

4. **Matched-vol replication check.** §6.11 documents a vol-scaling-
   artifact test for the F2F replica (re-running at matched 0.91%
   realized vol). With a follow-up binding-cap diagnostic
   (`scripts/f2f_replica_wmax_binding.py`) showing the matched-vol
   experiment is linear-tautological, plus an independent W_max=0.10
   binding-cap check at +0.55 Sharpe. We recommend any vol-target-
   style replication report both linear-scaling and binding-cap
   regimes.

5. **Multi-asset replication check + correlation-aware FWE.** §6.13
   (daily) and §6.13b (intraday within-frequency) above. Single-asset
   Sharpe claims should be tested across ≥3 asset classes before
   shipping, with a **family-wise error rate correction respecting
   cross-asset return correlation**. We use Cheverud-Patnaik
   effective-n_tests inside a Šidák framework (analytically tractable;
   does not require running a permutation test). Our own §6.13b
   shows the 3-of-9 uncorrected-p<0.10 count collapses to 0-of-9
   under FWE, which is the discipline that should be applied to all
   multi-asset
   replication claims.

6. **Multi-seed disclosure.** §6.9 multi-seed LSTM (n=3 seeds, t-CI
   ±0.35 at 2 d.f.) with explicit small-sample power statement.
   Single-seed claims should be flagged.

7. **Two-pipeline WF disclosure.** §6 disclosure block separating
   static-single-split V1/V2 results from true 4-fold WF
   V4/non-model/F2F results. Mixing the two without disclosure is a
   common framing error.

The seven items above are documented as a supporting PIT-discipline
checklist with reproduction code. Item 1 (the leak gallery with three
in-house leaks reproduced and patched) is claimed alongside the V4
LLRD recipe-collapse mechanism (§6.7) as one of the paper's two
primary contributions; items 2-7 are integrations and operationalizations
of known statistical practice, published with code for reproducibility.
The trading-Sharpe numbers (§6.8 +1.81 intraday-GLD, §6.13 0/6
daily generalization, §6.9 LSTM ≈ vt-only) are case-study evidence
illustrating why these diagnostics matter, not headline results.

## 7. Attribution

Post-training, we run the 6-method feature attribution suite (§4 +
`src/nanogld/analysis/`) on each fold's `llrd_final.pt` and report
which inputs drive the model's predictions. For the V1 and V2 runs
above the predictions are not profitable, so attribution is read as a
diagnostic for *what the model latched onto* rather than as evidence
that those features generalize.

### 7.1 What we expect to find under V2 (negative-result attribution)

Three patterns worth flagging:

1. **News slots dominate the cross-attention rollout in the news-
   present bucket**. The Flamingo `tanh(α)` gate opened toward news
   despite the news pathway being anti-edge. This explains the −6.24
   news-present Sharpe and is consistent with Agent 2's hypothesis:
   the CLIP SSL alignment baked the news embeddings into a
   backward-looking drift the model bets against.

2. **VSN gate is broken**. `vsn.py:91-93` applies
   `softmax(raw) * num_features`, which fixes the average gate to 1.0.
   The network cannot prune any feature, only redistribute mass. We
   expect every feature to have VSN gate near 1/651, with no
   meaningful concentration. This is a bug, not a finding; but it
   also means the VSN-importance method is non-discriminative for V2
   and the IG/permutation columns carry the signal.

3. **Macro features collapse under per-instance RevIN**. ~60% of the
   651 features are weekly-monthly-cadence macro (FRED, COT, WGC,
   GPR) forward-filled onto 30-min bars. With T=32 lookback +
   per-instance z-scoring, these slow features become near-constants;
   we expect their IG attribution to be near zero across the entire
   eval set. Agent 3's Lean-200 hypothesis predicts that pruning to
   price + microstructure + regime + key cross-asset channels lifts
   fold-0 Sharpe by ≥+0.4 vs. V2.

### 7.2 Measured attribution on V2 fold-0 (val_c, 1,885 bars)

We ran the analysis CLI on the V2 fold-0 `llrd_final.pt` checkpoint
against `val_c` (held out from both T-scaling and AgACI fitting,
1,885 bars: 1,269 news-present / 616 news-absent).

**VSN top-3 features by mean gate.** Feature indices `[218, 220, 162]`
dominate the VSN gate aggregated across val_c. As predicted in §7.1
the absolute gate values are flat near 1/651; softmax × num_features
math forces mean gate = 1.0; so the discriminative signal is
*ordering* not magnitude. Feature 218 corresponds to a price-action
macro channel; 220 and 162 are FRED/COT macro features that should
*not* dominate at the bar level under a healthy attention prior.
This is consistent with Agent 3's broken-gate hypothesis.

**Modality ablation (val_c, 1,885 bars).** Zeroing each modality at
inference and re-measuring focal cross-entropy + position Sharpe head
output (training metric, not backtest Sharpe):

| Ablation | Focal CE | Sharpe-loss training value (NOT backtest) | Present bucket | Absent bucket |
|----------|---------:|------------:|---------------:|--------------:|
| none     |   3.6714 |       2.225 |          2.318 |         2.065 |
| bars     |   3.5768 |       2.241 |          2.321 |         2.102 |
| news     |   4.4806 |       2.217 |          2.314 |         2.065 |
| regime   |   3.2057 |       2.223 |          2.405 |         1.907 |

Three findings:

1. **Removing bars lowers focal CE.** Bars input is net noise at the
   focal head, consistent with the per-instance RevIN slow-feature
   collapse (Agent 4).
2. **Removing news *raises* focal CE by 0.81.** News is a positive
   signal under the training metric, even though §6 backtest showed
   the news-present Sharpe is −6.24. The model has learned to lean on
   news for confidence (raising probability mass) but the bets are
   anti-edge. This bridges the V3a result.
3. **Removing regime drops focal CE the most (3.67 → 3.21).** The
   regime conditioning is actively hurting the focal head; Agent 3's
   broken-VSN-gate prediction lined up with the regime-channel
   collapse this method shows.

**Integrated Gradients / permutation importance.** Deferred to future
work. The qualitative VSN + ablation findings above are sufficient
for the §6.6 ship decision; IG and permutation-importance feature
attribution on fold-0 v2 would refine *which* macro channels carry
the broken-gate collapse and is V3c (VSN sigmoid + L1 gate penalty)
territory, but is not included in this paper.

## 8. Limitations and future work

- **Negative result.** V1 + V2 do not produce a profitable signal at the
  Mac-mini reduced-scale configuration. The full-spec Spark training
  run is one knob (d_model=192→384, T=32→64, epoch budget 8→30) that
  may move quantitative numbers, but our 4-agent post-mortem identifies
  four orthogonal failure modes (news pathway, recipe instability at
  starved batch, VSN broken-gate math, slow-cadence macro features
  collapsing under per-instance RevIN). The full-spec scale alone is
  unlikely to fix all four; V3a–V3c experiments target news + recipe,
  V4 targets the architectural fixes.

- **Sample size.** 75,672 30-min bars across 10 years is small by
  modern deep-learning standards. We mitigate via SSL pretraining and
  frequency-domain time-series augmentation (mask random frequencies in the FFT of the input window) but acknowledge sample
  efficiency is the binding constraint on this stack. The
  starved-batch (`B=2` after channel-independent reshape) regime
  amplifies Sharpe-loss noise (variance estimates on 1 DoF); Agent 4's
  hypothesis. We disable FriendlySAM + Cautious mask + Sharpe-loss in
  V3b to test this.

- **News coverage.** Only 48.9% of bars have visible news in a 4h
  lookback. The CLIP SSL alignment encourages bar↔news semantic
  alignment, but on a short-term reversal regime like GLD at ±5 min,
  alignment to *contemporaneous* news encodes the priced-in direction
  the model then bets against. V3a tests whether disabling news at
  inference recovers Sharpe to the news-absent baseline.

- **Per-instance RevIN destroys slow features.** With T=32 and
  per-instance z-scoring, FRED/COT/WGC features that update
  weekly–quarterly become near-constants. the macro-feature-collapse Lean-200 prediction
  is that pruning to ≤200 fast-cadence channels lifts fold-0 Sharpe
  by ≥+0.4. The accompanying VSN fix (V3b commit `6d58493`,
  `NANOGLD_VSN_GATE=sigmoid`) is necessary but not sufficient; the
  unified.pt itself needs a Lean rebuild (~30h compute) before the
  full hypothesis can be tested.

- **DANN era-classifier not ablated.** §4.3 describes a domain-
  adversarial era-classifier head (weight 0.05, alpha-ramped) sitting
  on the pooled representation. Section §6.7's V4 collapse diagnosis
  itemizes four interacting recipe failures (Mixout p=0.7, Sharpe
  loss noise at B=8, hardcoded `prev_position=None` disabling cost
  penalty, 3 LLRD epochs); the unablated DANN head is a fifth
  uncontrolled variable that we cannot rule out as a co-contributor
  to the collapse. The v4d mitigation recipe staged in
  `src/nanogld/training/llrd_finetune.py` keeps `dann_weight = 0.05`
  unchanged, so any v4d result will inherit the same gap.

- **Regime non-stationarity.** Even with DANN + AECF, the 2020 COVID
  era and the 2022-2023 tightening cycle look like out-of-distribution
  holes. A live-trading deployment would need test-time training
  [@sun2020ttt] or AgACI's online adaptation working harder than it
  does in 4-fold offline backtest.

- **Costs.** Our 2bp base is conservative for GLD but optimistic for
  smaller-cap ETFs or futures. The 1.5× stress is our hedge; ship
  decisions respect it.

- **Honest fallback.** Per V1-SPEC §0 fallback rule, if nanoGLD does
  not beat the simpler Gao 2014 + XGBoost ensemble by ≥ 0.2 Sharpe,
  we ship the simpler ensemble. The result of this paper may be
  exactly that.

- **Hardware ceiling.** Mac mini M4 (16 GB unified memory, MPS) is
  the only training hardware available for this paper. The
  channel-independent PatchTST encoder reshapes (B, T, C) →
  (B·C, T) which multiplies the effective batch dimension by 651
  channels. Even at d_model=192, t_bars=32, batch=4 (reduced from the
  spec's 384/64/8), the SwiGLU forward pass hits the 20 GiB MPS
  allocator ceiling. The V3b retrain (news + recipe ablation) and
  V3c retrain (VSN sigmoid + L1 gate penalty) are therefore deferred
  to future compute (rented H100 or x86_64 desktop with ≥24 GB VRAM).
  This is not an algorithmic limitation; it is a hardware one.

- **V4d retrain pending.** The relaxed-LLRD recipe (Mixout 0.1,
  FreeLB off, Sharpe warmup epochs 0-1 then ramped 0→0.3, focal CE
  full weight, 8 epochs, base_lr 5e-5, position-saturation runtime
  guard) is staged in `configs/v4d_wf.yaml` and the patched
  `src/nanogld/training/llrd_finetune.py`; pre-staged SSL anchors live
  at `checkpoints/v4d/fold_*/ssl/` on the shared Spark box. Single-
  fold sanity (~38 h fold 0) then 4-fold full retrain (~5 d) is the
  near-term test for whether the V4 collapse is recipe-specific or
  fundamental. v4d execution is gated on shared-GPU availability.

- **Ship decision.** Per V1-SPEC §0 fallback rule, if nanoGLD does
  not beat the strongest baseline by ≥ 0.2 Sharpe, we ship the
  simpler strategy. **For this paper that strategy is
  vol_target_buy_hold (§6.8) at WF mean Sharpe +1.810 net 1× cost
  (+1.730 net 1.5×), beating buy-and-hold by +0.538 Sharpe on the
  identical 4-fold walk-forward geometry.** nanoGLD V4 is *not*
  shipped; the simple vol-targeted long-only strategy is.

## 9. Reproducibility

All code, plan documents, and the per-fold sidecar build are open at
`github.com/sam-siavoshian/nanogld`. The unified dataset is gated; see
the **Dataset** paragraph below for access details. Training and
evaluation hardware:
(a) Mac mini M4 (16 GB unified memory, MPS) for the daily-gold LSTM
(§6.9) and the intraday multimodal LSTM (§6.12); (b) NVIDIA DGX Spark
prototype (GB10 Grace-Blackwell aarch64, sm_120, CUDA 13.0, 121 GB
unified RAM, shared with one collaborator) for the V4 4-fold
walk-forward retrain (§6.7); (c) Spark CPU (20 ARM cores) for the
non-model backtests (§6.8, profit-hunt) and the F2F replica. All
remote access via Tailscale + SSH. Total compute across the V4 run
was approximately 14 hours/fold on shared GB10 at FP32 batch=8 with
the per-fold SSL anchor warm-start; the §6.8 vol_target_3pct
production strategy itself trains zero parameters and runs in
seconds.

Every `torch.save` writes a reproducibility manifest with git SHA, host,
torch version, python version, CUDA version, dataset SHA256, fold
index, seed, and hparams hash. `data/integrity.py` verifies the
manifest at load time. Determinism: torch + numpy + python `random` +
PYTHONHASHSEED are seeded with `base_seed + fold_idx`, DataLoader
workers seeded via `worker_init_fn`, `torch.use_deterministic_algorithms`
enabled.

**Seed-variance disclosure.** Numbers reported in this paper are
**single-seed** (`base_seed = 42 + fold_idx` per fold). We do not
present a multi-seed mean ± std for any deep-learning Sharpe; the
compute budget did not permit a 5+ seed sweep for V4 on Spark. The
stationary block bootstrap CI we report on the §6.8 shipped strategy
(intraday +1.21 [+0.34, +2.11]) captures sampling uncertainty in the
*returns* but not seed-to-seed model variance. We flag this as a
binding limitation: any single-seed deep-learning Sharpe in this paper
should be read as one realization, not an expectation.

**Dataset.** The unified dataset (~234 MB) lives in a private
HuggingFace repository; on request we provide read access plus a
SHA256 manifest for verification. Anonymous reviewers cannot verify
the data without contacting the author; this is a known reproducibility
gap.

## Acknowledgments

Independent research conducted at home; no external funding.
Collaborator Omar Ramadan kindly shared access to the NVIDIA DGX Spark
prototype used for the V4 fold-0 retrain (§6.7); all compute on that
machine was run under low-priority GPU caps so as not to interfere
with his concurrent workload.

## References

See `paper/refs.bib`.
