# 06 — Backtest Discipline

## YOU ARE THE QUANT BACKTEST AGENT

You own the backtest engine + baseline ladder + honest reporting. You take trained checkpoints from doc 05 and produce equity curves, Sharpe ratios with confidence intervals, regime breakdowns, and the comparison table that goes in the X thread.

**Read 00-OVERVIEW.md FIRST.**
**Read 05-MODEL-TRAINING-CALIBRATION.md** for baseline architecture stubs.
**Read 05-MODEL-TRAINING-CALIBRATION.md** for checkpoint format.
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
src/nanogld/backtest/
├── __init__.py
├── engine.py               # Vectorized backtest (~150 LOC, hand-rolled per Karpathy mode)
├── metrics.py              # Sharpe, Sortino (canonical formula), Calmar, profit factor, expectancy, DSR
├── bootstrap.py            # Stationary block bootstrap via arch.bootstrap.StationaryBootstrap
├── regime.py               # Stratified breakdowns (vol terciles, FOMC week, news density)
├── baselines/
│   ├── __init__.py
│   ├── buy_hold.py         # +1 always
│   ├── ma_crossover.py     # 50/200 EMA
│   ├── donchian.py         # 20-period breakout
│   ├── dlinear.py          # Single-layer linear (Zeng AAAI 2023)
│   ├── tsmixer.py          # MLP-mixer (Chen Google ICLR 2023)
│   ├── timemixer.py        # Multi-scale MLP (Wang ICLR 2024)
│   ├── xlstm_time.py       # xLSTMTime — won 2026 finance benchmark
│   ├── xgboost_baseline.py # Committed config (n_est=500, max_depth=6, lr=0.05)
│   └── forecast_to_fill.py # Replication of arXiv:2511.08571 (the bar to beat)
├── report.py               # Renders honest reporting template (markdown + PNG plots)
└── cli.py                  # `python -m nanogld.backtest run --checkpoint <path>`

tests/
├── test_backtest_synthetic.py # Long +1 forever should match buy-hold metrics
├── test_cost_model.py         # Cost arithmetic on known position size
├── test_bootstrap_seed.py     # Reproducible CIs with fixed seed
└── test_metrics.py            # Sharpe formula sanity (mean/std × annualization)

reports/
├── v1_<commit_sha>_backtest.md
├── v1_<commit_sha>_equity_curve.png
├── v1_<commit_sha>_drawdown.png
└── v1_<commit_sha>_regime_table.png
```

### Files You DO NOT Touch

- `src/nanogld/model/` — doc 05 (you USE the model classes for baselines, don't modify)
- `src/nanogld/training/` — doc 05
- `src/nanogld/sizing/` — doc 07 (you call `stage2_sizing(...)` from doc 07's interface)
- Anything else outside `backtest/` and `reports/`

### Stable Interface You Publish (V1)

```python
from nanogld.backtest.engine import vectorized_backtest, BacktestConfig

# V1: 1x baseline = 2bps (F2F-anchored). Stress at 0.5x (1bp), 1x (2bps), 1.5x (3bps).
config = BacktestConfig(cost_bps_round_trip=2.0, capital=100.0, bars_per_year=3276)
result = vectorized_backtest(bars=df, positions=position_series, config=config)
# Returns: equity_curve, strategy_returns, positions, trade_returns, metrics dict
```

### Acceptance Criteria

1. ✅ `python -m nanogld.backtest run` produces full report on test fold
2. ✅ V1: All baselines run end-to-end (buy-hold, MA, Donchian, DLinear, TSMixer, TimeMixer, xLSTMTime, VLSTM, XGBoost, Gao 2014 half-hour-5 single-feature rule, Forecast-to-Fill daily replica)
3. ✅ Bootstrap CI on Sharpe with 5K resamples reported per strategy
4. ✅ Regime stratification table (high/mid/low vol × FOMC/non-FOMC × high/low news)
5. ✅ V1 (cost-stress): cost sensitivity sweep reported at **{0.5x, 1.0x, 1.5x}** of base 2bps round-trip (= {1, 2, 3} bps) — replaces V1's {3, 5, 7, 10} bps sweep. Hard gate at 1.5x.
6. ✅ Sortino uses canonical formula (`sqrt(mean(min(0, r)^2))` — NOT std of negative subset)
7. ✅ Annualization uses **3276 bars/year** (NEVER 17500)
8. ✅ Honest reporting template followed (every result has CI, cost assumption, fold IDs, regime breakdown)
9. ✅ V1: DSR (Bailey-Lopez de Prado) reported alongside raw Sharpe — **DSR > 1.0 is a hard gate** across all reported configs
10. ✅ V1: **per-bucket eval (news-present, news-absent, both)** for every reported metric (Sharpe, Sortino, MDD, ECE, MCC, F1, calibration coverage)
11. ✅ If V1 ties or loses to the simpler ensemble (Gao 2014 + XGBoost) by < 0.2 Sharpe OOS net of costs, ship the simpler ensemble — full stop

### Spawn Nia Agents When You Need To

- **xLSTMTime implementation** — code released by Beck/Hochreiter 2025, latest ref impl
- **Forecast-to-Fill replication** — paper has methodology; verify exact ATR exit rules + Kelly sizing constants
- **arch.bootstrap.StationaryBootstrap** — verify API current (PyPI version may have changed)
- **Modern XGBoost defaults** — verify `xgboost==2.x` API for our committed config still works

### V1 Critical Findings (DO NOT REVERT, plus V1 layered on top)

1. **bars_per_year = 3276** (NYSE RTH only). Original 17500 was 24/7 calendar — fatal bug, inflates Sharpe 2.31x.
2. **Sortino formula corrected** to canonical target downside dev.
3. **xLSTMTime + VLSTM as mandatory STRONG baselines** per arXiv:2603.01820 (Saly-Kaufmann/Wood/Zohren 2026: VLSTM 2.40, xLSTM 1.79, generic Transformer underperformed).
4. **V1 reframe**: Forecast-to-Fill (arXiv:2511.08571, Sharpe 2.88) is **daily gold futures, NOT directly comparable** to our 30-min intraday problem. F2F replication is a separate scoreboard. Apples-to-apples GLD-specific bar is Gao 2014 (5.43 Sharpe single-feature half-hour-5 rule). V1 target: 1.0–1.5 OOS Sharpe net of 2bp.
5. **Peer-benchmark discount** required per arXiv:2604.18821 (1,726 strategies analyzed; backtests capture launch regime).
6. **GLD 5y buy-and-hold Sharpe approx 2.4** (2020-2025 was a great gold run). Honest baseline nanoGLD must beat to claim alpha.
7. **V1 hard gates**: per-bucket eval, cost-stress {0.5x, 1.0x, 1.5x}, DSR > 1.0. See V1 deltas at top.

### Empirical Bar (V1 reframed)

V1 target: 1.0 to 1.5 OOS Sharpe net of 2 bps round-trip costs over 4-fold walk-forward. Wright F2F 2.88 is daily gold futures EOD-to-EOD with ~30-day holds — NOT comparable to 30-min intraday GLD direction. Apples-to-apples published intraday GLD record is Gao-Han-Li-Zhou 2014 (5.43 Sharpe single-feature half-hour-5 timing). Daily DL frontier per Saly-Kaufmann/Wood/Zohren 2026 (arXiv:2603.01820): VLSTM 2.40 Sharpe.

```
V1 Tier                   Sharpe(1x)   Sharpe(1.5x)  DSR    Per-bucket   Status
Minimum viable              > 0          —             > 0    —            Mandatory
V1 ship gate              > 1.0        > 0.5         > 1.0  both > 0     Mandatory for X thread
F2F-tier (daily)            > 2.5        —             —      —            Separate scoreboard, not directly comparable
```

If V1 ties or loses to the simpler ensemble (Gao 2014 half-hour-5 rule + XGBoost on the same 681 features) by < 0.2 Sharpe OOS net of costs, **ship the simpler ensemble**, full stop. TLOB lesson: "MLP can match transformer." V1 corollary: a 2014 single-feature rule can match a 30M-param transformer, and if so, the 2014 rule ships.

### Hand-off Protocol

1. Update STATUS.md with: per-fold Sharpe, DSR, comparison table, equity curve PNG path
2. Notify doc 07 (sizing) that backtest engine is ready (Stage 2 sizer needs the same engine)
3. Notify user via STATUS update that report is ready for X thread

Now read the implementation specifics.

---

# 06 — Backtest Discipline

**Status:** V1 redlined, implementation-ready, Nia-verified (V1 frozen 2026-05-04, V1 redlines 2026-05-08)
**Last verified:** 2026-05-08

## V1 DELTAS (what changed since V1)

Authoritative source: `plan-edit/V1-SPEC.md` section 9. Summary of every V1 redline that lands in this doc:

1. **Per-bucket eval (NEW HARD requirement)** — every reported metric (Sharpe, Sortino, MDD, ECE, MCC, F1, calibration coverage) reported separately for {news-present, news-absent, both}. Without this we fly blind on the 51% no-news bars. New section "Per-Bucket Eval" below.
2. **Cost-stress at {0.5x, 1.0x, 1.5x} as HARD gate** — replaces V1's {3, 5, 7, 10} bps sweep. Cost model: half-spread base 0.7 bps (k=0.7bps, F2F paper), sqrt-impact gamma=0.02. 1x = 2bps round-trip, 0.5x = 1bp, 1.5x = 3bps. Must show Sharpe > 0.5 at 1.5x cost to ship (Wright F2F died at 1.5x; ours likely worse).
3. **Deflated Sharpe Ratio (DSR) > 1.0 as HARD gate** — Bailey & López de Prado. Multi-config selection penalty enforced. No cherry-picking across the muP sweep + walk-forward folds.
4. **Promotion gates: V1's 6 gates replaced by V1's 8 gates** (see "V1 Promotion Gates" below).
5. **Updated baseline ladder**:
   - NEW STRONG baselines: VLSTM (Saly-Kaufmann 2026 arXiv:2603.01820, hit 2.40 Sharpe daily futures), xLSTMTime (Korkmaz 2024 arXiv:2407.10240) kept.
   - NEW domain-specific: Gao 2014 single-feature half-hour-5 rule (5.43 Sharpe on GLD specifically). If we lose to this we shipped a worse model than 2014.
   - F2F replication explicitly labeled as **daily, NOT directly comparable** to our 30-min problem.
   - If V1 ties or loses to Gao 2014 + XGBoost ensemble, ship the simpler ensemble.
6. **Reframe target Sharpe: 1.0–1.5 OOS net of 2bp** (was "beat F2F 2.88"). 2.88 is daily gold futures, not intraday.
7. **Walk-forward CV kept**: 4 folds, 1-week embargo, train 3y + val 6mo + test 6mo, step 3mo. NYSE RTH bars_per_year=3276 (NEVER 17500). Embargo must be >= max feature window.
8. **Multiple-testing inflation defense**: lock model BEFORE OOS. DSR penalty for multi-config selection. Walk-forward uses fold separation.
9. **Common kill-shots V1 must avoid** (from V1 spec section 9.9):
   - Embargo too short for multi-day rolling features (verify embargo >= max feature window).
   - Look-ahead via global normalization (must be train-only or rolling causal stats; per-channel RevIN already addresses this in doc 04).
   - Wrong annualization (3276 not 252) — already locked.
   - Survivorship bias / regime cherry-pick.
   - Cost assumption fragility — state assumptions explicitly in every reported result.

V1 content below stays as the implementation reference except where directly overridden by V1 deltas above.

---

## CRITICAL CORRECTIONS APPLIED (Nia verification)

- ❌ `bars_per_year=17500` → ✅ **`3276`** (NYSE RTH: 6.5h × 13 30min bars × 252 days). Original 17500 ≈ 24/7 calendar × 48 bars/day, wrong for ETFs. Inflated all Sharpes by **2.31×** if uncorrected.
- ❌ Sortino used `std()` of negative-only subset → ✅ canonical target downside dev `sqrt(mean(min(0, r-MAR)²))` over full sample
- ❌ Skip Deflated Sharpe Ratio → ✅ **add it** — table-stakes for credibility, ~30 LOC (Bailey-Lopez de Prado)
- ❌ Hand-rolled stationary bootstrap → ✅ use `arch.bootstrap.StationaryBootstrap` (canonical impl, accepts seed)
- ❌ backtrader as second-opinion engine → ✅ archive mode; drop entirely
- ❌ "Hit rate alone" → ✅ also report **profit factor** (sum_wins/|sum_losses|) and **expectancy**
- ❌ XGBoost defaults vague → ✅ commit specific config (see Baseline Ladder below)
- ⚠️ **GLD 5Y buy-and-hold Sharpe ≈ 2.4** (2020-2025 was a great gold run). This is the actual bar to beat. Loud warning.
**Owner:** samsiavoshian
**Implementation effort:** 1 day

## What's Locked (V1)

- Walk-forward backtest on test folds only (never on train/val). 4 folds, 1-week embargo, train 3y + val 6mo + test 6mo, step 3mo.
- V1 cost model: 2bps round-trip baseline (F2F-anchored: k=0.7bps half-spread + sqrt-impact gamma=0.02). Stress at {0.5x, 1.0x, 1.5x} = {1, 2, 3} bps. Hard gate at 1.5x.
- V1 Stage 1 sizing: Head B `tanh` * vol_target. V1 Stage 2: Head B + friction-adjusted Kelly + ATR exits + vol target + 30-day timeout + APS conformal floor. See doc 07.
- Bootstrap CI on Sharpe with stationary block bootstrap
- Baseline ladder for honest comparison
- Regime-stratified performance reporting + V1 per-bucket eval (news-present, news-absent, both)
- DSR > 1.0 hard gate. Per-bucket Sharpe both positive hard gate.

## Baseline Ladder (V1)

All baselines run on identical walk-forward protocol with identical cost model. V1 reorganizes by tier (naive → linear → strong sequence → tree → domain-specific → replica → ours).

### Tier 0 — Naive

1. **Buy-and-hold** — long GLD always, net of costs
2. **Momentum / MA crossover** — 50-period EMA vs 200-period EMA, long when fast > slow
3. **Donchian breakout** — 20-period high/low, long when close > prior 20-bar high

### Tier 1 — Linear / MLP-mixer

3a. **DLinear** (Zeng et al. AAAI 2023, arXiv:2205.13504) — single-layer linear model, ~10K params. Floor baseline. **TLOB paper finding: an MLP can match a transformer on financial data.** If nanoGLD doesn't beat DLinear, transformer is overkill.
3b. **TSMixer** (Chen et al. Google ICLR 2023, arXiv:2303.06053) — pure MLP-mixer, ~2M params. Beats most transformer baselines on M5 (the closest published analog to our problem). MANDATORY baseline.
3c. **TimeMixer** (Wang et al. ICLR 2024, arXiv:2405.14616) — multi-scale MLP, ~5M params. Multi-resolution architecture, often beats transformers at small data scale.

### Tier 2 — Strong sequence (V1 NEW STRONG baselines)

3c-bis. **xLSTMTime** (Alharthi & Mahmood 2024, arXiv:2407.10240) — xLSTM-based time-series, ~10M params. 2026 finance benchmark winner per arXiv:2603.01820 (March 2026): **highest breakeven transaction cost buffer and best downside-adjusted Sharpe of all sequence models** on 2010-2025 daily futures. Pure Transformer underperformed.
3c-ter. **VLSTM** (Saly-Kaufmann/Wood/Zohren 2026, arXiv:2603.01820) — V1 NEW. LSTM + VSN feature gate, hit **2.40 Sharpe** on the daily futures benchmark and best Sharpe of all sequence models tested. STRONG baseline. If V1 ties or loses to VLSTM, ship VLSTM (per V1 ship-the-simpler-model rule).

### Tier 3 — Tree

4. **XGBoost on price+macro features** — same 681 features as our model, no news embeddings. Committed config below.
5. **XGBoost on price+macro+news_anchor_cosines** — adds the cheap semantic features

### Tier 4 — Domain-specific (V1 NEW)

5a. **Gao 2014 half-hour-5 single-feature rule** (Gao-Han-Li-Zhou 2014). Single feature: `r_h5 = log(close_t / close_{t-30min})` for the 5th RTH half-hour (~11:30-12:00 ET) predicts last half-hour with **5.43 Sharpe on GLD specifically**, concentrated on high-vol days. **This is the apples-to-apples GLD-specific bar.** If our model loses to this single-feature rule, we shipped a worse model than 2014. V1 spec rule: if V1 ties or loses to Gao 2014 + XGBoost ensemble by < 0.2 Sharpe, ship the simpler ensemble.

### Tier 5 — Replica (DIFFERENT PROBLEM, separate scoreboard)

3d. **Forecast-to-Fill daily replication** (Wright et al. 2026 arXiv:2511.08571) — trend+momentum + vol-target + friction-adjusted Kelly. Cleanest published **daily** gold result (Sharpe 2.88, MDD 0.52% on 5y walk-forward). **NOTE: this is daily gold futures, NOT directly comparable to our 30-min intraday GLD problem.** Reported on a separate scoreboard. F2F honestly publishes that 1.5x costs collapse Sharpe to -0.03 — V1's 1.5x hard gate is a direct response.

### Tier 6 — Ours

6. **nanoGLD (V1, Stage 1 sizing)** — Head B `tanh` * vol target. See doc 07.
7. **nanoGLD (V1, Stage 2 sizing)** — Head B + friction-adjusted Kelly + ATR exits + vol target + 30-day timeout + conformal floor. See doc 07. Stage 2 must beat Stage 1 by >= 0.2 Sharpe OOS to ship the full pipeline; otherwise ship Stage 1.

### XGBoost Baseline Config (committed)

```python
xgb_params = {
    'objective': 'multi:softprob',
    'num_class': 3,
    'n_estimators': 500,
    'max_depth': 6,
    'learning_rate': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'early_stopping_rounds': 50,
    'eval_metric': 'mlogloss',
    'random_state': 42,
}
```

## Metrics Reported Per Run

```
- Total return
- Annualized Sharpe ratio
- Sortino ratio
- Max drawdown
- Calmar ratio
- Hit rate (fraction of profitable trades)
- Profit factor (sum of winning trades / |sum of losing trades|)
- Expectancy (hit_rate × avg_win - (1-hit_rate) × |avg_loss|)
- **Deflated Sharpe Ratio** (Bailey-Lopez de Prado 2014) — corrects for selection bias across N hyperparameter trials
- Trade count
- Average trade return
- Turnover (annualized)
- Bootstrap 95% CI on Sharpe
```

## What's Now Designed (post-deep-dive, V1)

All the items below have been resolved in this doc:
- ✅ V1 cost model: F2F-anchored 2bps round-trip baseline (k=0.7bps half-spread + sqrt-impact gamma=0.02). Stress at {0.5x, 1.0x, 1.5x} = {1, 2, 3} bps. Hard gate at 1.5x.
- ✅ Stationary block bootstrap with Politis-White via `arch.bootstrap.optimal_block_length`
- ✅ Regime taxonomy: vol terciles, FOMC week binary, news-density top-quartile
- ✅ V1 per-bucket eval: news-present, news-absent, both
- ✅ V1 Deflated Sharpe Ratio > 1.0 hard gate (Bailey-Lopez de Prado)
- ✅ Display: bootstrap CI on Sharpe, $-PnL on $100 capital, equity-curve plot

## V1 Promotion Gates (replaces V1's 6 gates)

Eight HARD gates, all must pass to ship V1 model. Fail any gate, the negative result gets reported and we ship the simpler model. Cherry-picking is fireable.

```
Gate 1   Walk-forward Sharpe > 1.0 net of 1x cost (was 0.8 in V1)
Gate 2   Sharpe > 0.5 net of 1.5x cost (NEW V1 hard)
Gate 3   Beats best baseline by >= 0.2 Sharpe on >= 3 of 4 folds
Gate 4   Conformal coverage within +/-2% of nominal on val + per-bucket
Gate 5   Stage 2 sizer (decision-aware head) beats Stage 1 fallback by >= 0.2 Sharpe OOS
Gate 6   Drawdown circuit breaker tested on >= 2 historical regimes
Gate 7   Deflated Sharpe Ratio > 1.0 (NEW V1 hard)
Gate 8   Per-bucket Sharpe (news-present, news-absent) both positive (NEW V1 hard)
```

If any gate fails: report honestly in the X thread. Do not soften.

## Cost Model (V1)

V1 cost model derives from F2F paper (arXiv:2511.08571) and locks the 0.5x/1.0x/1.5x stress scheme:

- Half-spread base: **0.7 bps** (k=0.7bps from F2F paper).
- sqrt-impact: **gamma=0.02**, `cost_t = gamma * sqrt(|delta_w|) + k_bps * |delta_w|`.
- 1x baseline = **2 bps round-trip** (matches retail Alpaca on GLD: $0 commission, 0.5-1bp spread per side, 0.5-1bp slippage padding).
- 0.5x = 1bp round-trip (best-case, tight spread, no impact).
- 1.5x = 3bps round-trip (stress case; F2F died at 1.5x; ours likely worse).

**Cost stress is a HARD gate**: must show Sharpe > 0.5 at 1.5x cost. Report Sharpe at all three levels for every reported config. State assumptions explicitly. Cost assumption fragility is a known V1 kill-shot.

V1 sweep at {3, 5, 7, 10} bps is replaced by V1's {1, 2, 3} bps. The V1 sweep is anchored to the F2F paper's published cost model, not retail Alpaca worst-case padding.

## Honest Reporting Protocol

Every result reports:
- Confidence interval (not just point estimate)
- Which fold(s) it covers
- Cost assumption
- Random seed (or "seed-averaged across N seeds")
- Whether it's IS, val, or OOS

No cherry-picking the best fold. No "we got lucky on test fold 3" without showing the others.

## Full Backtest Engine (vectorized)

```python
import numpy as np
import pandas as pd
from dataclasses import dataclass

@dataclass
class BacktestConfig:
    # V1: cost_bps_round_trip default = 2.0 (1x baseline, F2F-anchored).
    # Stress at 1.0 (0.5x), 2.0 (1x), 3.0 (1.5x). 1.5x is the hard gate.
    # V1 default was 5.0 (retail-padded). Deprecated.
    cost_bps_round_trip: float = 2.0
    capital: float = 100.0
    bars_per_year: int = 3276    # NYSE RTH 6.5h × 13 30min bars × 252 days. NOT 17500 (24/7 calendar — wrong for ETFs)
    position_limit: float = 1.0


@dataclass
class BacktestResult:
    equity_curve: pd.Series
    strategy_returns: pd.Series
    positions: pd.Series
    trade_returns: pd.Series
    metrics: dict


def vectorized_backtest(bars: pd.DataFrame, positions: pd.Series, config: BacktestConfig) -> BacktestResult:
    next_returns = bars.close.pct_change().shift(-1)
    gross_returns = positions * next_returns
    position_diffs = positions.diff().abs().fillna(positions.iloc[0])
    costs = position_diffs * (config.cost_bps_round_trip / 10_000)
    strategy_returns = (gross_returns - costs).fillna(0)
    equity_curve = (1 + strategy_returns).cumprod()
    trade_returns = strategy_returns * config.capital
    metrics = compute_metrics(strategy_returns, positions, config)
    return BacktestResult(equity_curve, strategy_returns, positions, trade_returns, metrics)


def compute_metrics(strategy_returns: pd.Series, positions: pd.Series, config: BacktestConfig) -> dict:
    annualization = np.sqrt(config.bars_per_year)
    mean = strategy_returns.mean()
    std = strategy_returns.std()
    sharpe = (mean / std * annualization) if std > 0 else 0.0

    # Canonical Sortino: target downside deviation = sqrt(mean(min(0, r-MAR)^2)) over FULL sample
    # NOT std() of only-negative subset (common but technically wrong)
    MAR = 0.0
    downside_dev = np.sqrt(np.mean(np.minimum(strategy_returns - MAR, 0) ** 2))
    sortino = ((mean - MAR) / downside_dev * annualization) if downside_dev > 0 else 0.0

    equity = (1 + strategy_returns).cumprod()
    drawdown = (equity - equity.cummax()) / equity.cummax()
    max_dd = drawdown.min()

    annual_return = (1 + strategy_returns).prod() ** (config.bars_per_year / max(1, len(strategy_returns))) - 1
    calmar = annual_return / abs(max_dd) if max_dd < 0 else 0.0

    nonzero = strategy_returns[positions.abs() > 0] if positions is not None else strategy_returns
    hit_rate = (nonzero > 0).mean() if len(nonzero) > 0 else 0.0
    trade_count = int((positions.diff() != 0).sum()) if positions is not None else 0
    turnover = (positions.diff().abs().sum() * (config.bars_per_year / max(1, len(positions)))) if positions is not None else 0

    return {
        'total_return': (1 + strategy_returns).prod() - 1,
        'annual_return': annual_return,
        'sharpe': sharpe,
        'sortino': sortino,
        'max_drawdown': max_dd,
        'calmar': calmar,
        'hit_rate': hit_rate,
        'trade_count': trade_count,
        'turnover': turnover,
    }
```

## Cost Model Decomposition (V1 — F2F-anchored 2bps base)

V1 used a 5bps round-trip with retail-padding rationale. V1 anchors to F2F paper's k=0.7bps half-spread + sqrt-impact gamma=0.02, lands on **2bps round-trip baseline**, and stresses at 0.5x / 1.0x / 1.5x.

```
V1 cost model (F2F paper, arXiv:2511.08571):
- Half-spread:    k = 0.7 bps per side
- sqrt-impact:    gamma = 0.02, cost_t = gamma*sqrt(|delta_w|) + k*|delta_w|
- Per side:       ~1 bp (mostly half-spread; impact tiny at $100 capital)
- Round-trip 1x:  2 bps
- 0.5x stress:    1 bp  (tight-spread best case)
- 1.5x stress:    3 bps (HARD gate: Sharpe > 0.5 here)
```

V1's old retail-padded {3, 5, 7, 10} bps sweep is deprecated. Strategy must hold up at 1.5x (3 bps) to be considered alpha-bearing.

## Bootstrap CI on Sharpe

```python
def stationary_block_bootstrap_sharpe(
    returns: np.ndarray,
    B: int = 5000,
    block_length: int = None,
    bars_per_year: int = 3276    # NYSE RTH 6.5h × 13 30min bars × 252 days. NOT 17500 (24/7 calendar — wrong for ETFs),
) -> tuple[float, float, float]:
    """Returns (mean_sharpe, ci_lower, ci_upper) at 95%."""
    N = len(returns)
    if block_length is None:
        try:
            from arch.bootstrap import optimal_block_length
            opt = optimal_block_length(returns)
            block_length = int(opt['stationary'].iloc[0])
        except ImportError:
            block_length = max(1, int(N ** (1/3)))  # rule of thumb fallback

    sharpes = []
    annualization = np.sqrt(bars_per_year)

    for _ in range(B):
        sample = np.empty(N)
        i = 0
        while i < N:
            start = np.random.randint(0, N)
            length = np.random.geometric(p=1 / block_length)
            length = min(length, N - i)
            for j in range(length):
                sample[i + j] = returns[(start + j) % N]
            i += length

        s = sample.mean() / sample.std() * annualization if sample.std() > 0 else 0
        sharpes.append(s)

    sharpes = np.array(sharpes)
    return sharpes.mean(), np.percentile(sharpes, 2.5), np.percentile(sharpes, 97.5)
```

## Regime Stratification

```python
def regime_stratified_metrics(strategy_returns: pd.Series, features: pd.DataFrame, config: BacktestConfig) -> pd.DataFrame:
    rows = []

    vol = features['realized_vol_240']
    high_vol = vol > vol.quantile(0.66)
    low_vol = vol < vol.quantile(0.33)
    rows.append(('high_vol', compute_metrics(strategy_returns[high_vol], None, config)))
    rows.append(('mid_vol', compute_metrics(strategy_returns[~high_vol & ~low_vol], None, config)))
    rows.append(('low_vol', compute_metrics(strategy_returns[low_vol], None, config)))

    fomc_week = features['is_FOMC_week'].astype(bool)
    rows.append(('fomc_week', compute_metrics(strategy_returns[fomc_week], None, config)))
    rows.append(('non_fomc', compute_metrics(strategy_returns[~fomc_week], None, config)))

    news_count = features['gdelt_conflict_count']
    high_news = news_count > news_count.quantile(0.75)
    rows.append(('high_news', compute_metrics(strategy_returns[high_news], None, config)))
    rows.append(('low_news', compute_metrics(strategy_returns[~high_news], None, config)))

    return pd.DataFrame([{'regime': r, **m} for r, m in rows]).set_index('regime')
```

## Per-Bucket Eval (V1 NEW HARD requirement)

Every reported metric (Sharpe, Sortino, MDD, Calmar, hit rate, profit factor, expectancy, ECE, MCC, F1, conformal coverage, DSR) is reported separately for three buckets:

- **news-present**: bars where `is_news_present == 1` (news token attached at the encoder).
- **news-absent**: bars where `is_news_present == 0` (NO_NEWS token used). Empirical absence rate is ~51% on our data, so this bucket carries roughly half of the eval set.
- **both**: full eval set (concatenation of news-present and news-absent).

Without this we fly blind on the 51% no-news bars. V1 promotion Gate 8 requires per-bucket Sharpe (news-present, news-absent) **both positive**.

```python
def per_bucket_metrics(strategy_returns: pd.Series, is_news_present: pd.Series, config: BacktestConfig) -> pd.DataFrame:
    """V1: report all metrics for {news-present, news-absent, both}."""
    rows = []
    rows.append(('news_present', compute_metrics(strategy_returns[is_news_present == 1], None, config)))
    rows.append(('news_absent',  compute_metrics(strategy_returns[is_news_present == 0], None, config)))
    rows.append(('both',         compute_metrics(strategy_returns,                       None, config)))
    return pd.DataFrame([{'bucket': b, **m} for b, m in rows]).set_index('bucket')
```

Per-bucket eval also runs alongside per-cost stress (0.5x / 1x / 1.5x), per-fold breakdown, and per-regime stratification. The full reporting matrix is `{3 cost levels} x {3 buckets} x {4 folds} x {7 regime cells}` per metric per strategy. Render the cost x bucket matrix in the headline table; relegate fold/regime to appendix tables.

## Deflated Sharpe Ratio (Nia recommended — table-stakes, not overkill)

```python
from scipy.stats import norm

def deflated_sharpe(sharpe: float, n_trials: int, n_obs: int, skew: float, kurt: float) -> float:
    """
    Bailey & Lopez de Prado (2014) Deflated Sharpe Ratio.
    Corrects for: (1) multiple-testing bias across N trials, (2) non-normality of returns.
    Returns probability that true Sharpe > 0 given observed sharpe.
    """
    # Expected max Sharpe under null (random strategies, N trials)
    emc = 0.5772156649  # Euler-Mascheroni
    sharpe_max_null = (1 - emc) * norm.ppf(1 - 1/n_trials) + emc * norm.ppf(1 - 1/(n_trials * np.e))

    # Variance of Sharpe under non-normality (PSR adjustment)
    sigma_sharpe = np.sqrt(
        (1 - skew * sharpe + ((kurt - 1) / 4) * sharpe**2) / (n_obs - 1)
    )

    # DSR = probability adjusted Sharpe > 0
    z = (sharpe - sharpe_max_null) / sigma_sharpe
    return norm.cdf(z)
```

Reported as: `Sharpe = 0.51 [bootstrap CI: 0.18, 0.84], DSR p = 0.73 (n_trials=12, accounting for selection bias)`.

DSR p < 0.95 = strategy may be a lucky selection from many tried. DSR p > 0.95 = robust to multi-testing.

## V1 — 2026 Empirical Findings (finance-papers agent)

- **Forecast-to-Fill (Sharpe 2.88 on gold) STILL UNREPLICATED in 2026.** No public falsification or replication found Jan-May 2026. F2F is **daily, NOT directly comparable** to our 30-min problem (V1 reframe). The bar stands as a separate scoreboard. Building our own honest 30-min gold benchmark with cost-stress + DSR + per-bucket is publishable (gap in literature).
- **arXiv:2604.18821 (April 2026)**: 1,726 strategies analyzed. Backtests mostly capture launch-period market conditions, not skill. Demands peer-benchmark discount in writeup.
- **arXiv:2604.10996 (April 2026)**: "When Valid Signals Fail: Regime Boundaries Between LLM Features and RL Trading Policies" — during macro shocks, LLM-derived features ADD noise; augmented agent UNDERPERFORMS price-only baseline. Implication: add regime-aware gating; consider falling back to price-only mode during high-uncertainty windows. V1 FiLM regime conditioning + AECF entropy-gated curriculum masking address this directly.
- **arXiv:2603.01820 (March 2026)** Saly-Kaufmann/Wood/Zohren Oxford-MAN large-scale benchmark: **VLSTM 2.40 Sharpe, xLSTM 1.79 with best transaction-cost robustness, iTransformer 0.38, Mamba 0.64**. Generic Transformer does NOT win at portfolio level. Reinforces V1 hybrid (10 transformer + 2 sLSTM) decision and adds VLSTM as a STRONG baseline.
- **arXiv:2407.10240** Alharthi & Mahmood xLSTMTime: channel-independent, sLSTM for small data (<=70K samples = our 75K). V1 baseline (kept).
- **Gao-Han-Li-Zhou 2014**: single-feature half-hour-5 rule, 5.43 Sharpe on GLD specifically. Apples-to-apples bar. V1 NEW baseline.
- **No 2026 paper proves transformer+news beats XGBoost+news on single-asset 30min direction with DSR.** This remains an open empirical question — building this comparison ourselves is publishable.
- **TLOB pattern replicated (arXiv:2506.05764 on crypto LOB)**: feature engineering >> architecture depth.

## V1 — 2026-05-08 redline drivers (9-agent Nia synthesis)

Source: `plan-edit/V1-SPEC.md`. The eight V1 promotion gates and the per-bucket / cost-stress / DSR hard gates are direct responses to:

1. **Cost-model fragility** — F2F's own honest publication that 1.5x cost collapses Sharpe to -0.03. V1 makes 1.5x a hard gate.
2. **Multi-config selection bias** — muP sweep + 8-gate config space + walk-forward folds = many trials. DSR > 1.0 enforces the multi-testing penalty.
3. **News presence bimodality** — 51% of bars have no news. Per-bucket eval is non-negotiable; without it, a strategy that wins on news-present and loses on news-absent looks fine in aggregate.
4. **Apples-to-apples comparison** — Wright F2F is daily, ours is 30-min intraday. Reframing target from "beat F2F 2.88" to "1.0–1.5 OOS net of 2bp" is honesty.
5. **Simpler-model-wins discipline** — Gao 2014 single-feature rule is a 12-year-old GLD-specific baseline that still posts 5.43 Sharpe. If we can't beat it + XGBoost ensemble by >= 0.2 Sharpe, ship the simpler ensemble.

## ⚠️ The Brutal GLD Buy-and-Hold Baseline

**GLD 5-year Sharpe (2020-2025) ≈ 2.4** per YCharts/PortfoliosLab. 5Y CAGR ≈ 20.6%, total return ~158%. **2020-2025 was an exceptional gold run.**

This is the actual bar your strategy must beat. A directional 30min model needs to add value AFTER costs vs a brain-dead "long GLD forever" strategy that already has Sharpe 2.4.

**Honest implication:** if your model achieves Sharpe 1.5 OOS, you are technically WORSE than buy-and-hold for this period. The story changes from "I built a great strategy" to "I built a strategy that loses to buy-and-hold during a gold bull run, but might be more useful in flat/bear regimes." Lead with this honestly in the X thread.

## The Honest Reporting Template (V1)

Every reported result includes:
- Test period (which fold[s])
- V1 cost stress table at {0.5x, 1.0x, 1.5x} of base 2bps round-trip (= {1, 2, 3} bps). Hard gate at 1.5x.
- Sharpe with 95% bootstrap CI (stationary block, B=5000)
- V1 Deflated Sharpe Ratio (Bailey-Lopez de Prado). Hard gate: DSR > 1.0.
- V1 per-bucket breakdown {news-present, news-absent, both} for every metric. Hard gate: per-bucket Sharpe both positive.
- Total return, max drawdown, Calmar, Sortino
- Trade count, turnover (annualized), hit rate, profit factor, expectancy
- Regime breakdown (vol terciles + FOMC week + news density)
- Conformal coverage on val + per-bucket. Hard gate: within +/-2% of nominal.
- One-paragraph honest read identifying limitations
- Explicit cost-model assumption statement (k=0.7bps, gamma=0.02, F2F-anchored)

## Implementation Day Plan (Day 8 of week 1, post-training)

| Hour | Task |
|------|------|
| 1-2 | Backtest engine on synthetic data (verify math) |
| 3-4 | Run baselines (buy-hold, MA crossover, XGBoost) |
| 5 | Run nanoGLD Stage 1 (fixed sizing) |
| 6-7 | Bootstrap CIs, regime breakdown, sensitivity sweep |
| 8 | Equity curve plots, save artifacts for blog post |
