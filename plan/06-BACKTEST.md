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

### Stable Interface You Publish

```python
from nanogld.backtest.engine import vectorized_backtest, BacktestConfig

config = BacktestConfig(cost_bps_round_trip=5.0, capital=100.0, bars_per_year=3276)
result = vectorized_backtest(bars=df, positions=position_series, config=config)
# Returns: equity_curve, strategy_returns, positions, trade_returns, metrics dict
```

### Acceptance Criteria

1. ✅ `python -m nanogld.backtest run` produces full report on test fold
2. ✅ All 9 baselines run end-to-end (buy-hold, MA, Donchian, DLinear, TSMixer, TimeMixer, xLSTMTime, XGBoost, Forecast-to-Fill replica)
3. ✅ Bootstrap CI on Sharpe with 5K resamples reported per strategy
4. ✅ Regime stratification table (high/mid/low vol × FOMC/non-FOMC × high/low news)
5. ✅ Cost sensitivity sweep at {3, 5, 7, 10} bps reported
6. ✅ Sortino uses canonical formula (`sqrt(mean(min(0, r)^2))` — NOT std of negative subset)
7. ✅ Annualization uses **3276 bars/year** (NEVER 17500)
8. ✅ Honest reporting template followed (every result has CI, cost assumption, fold IDs, regime breakdown)
9. ✅ DSR (Bailey-Lopez de Prado) reported alongside raw Sharpe
10. ✅ If nanoGLD doesn't beat ALL baselines by ≥0.2 Sharpe OOS, recommend shipping the simpler model in the report

### Spawn Nia Agents When You Need To

- **xLSTMTime implementation** — code released by Beck/Hochreiter 2025, latest ref impl
- **Forecast-to-Fill replication** — paper has methodology; verify exact ATR exit rules + Kelly sizing constants
- **arch.bootstrap.StationaryBootstrap** — verify API current (PyPI version may have changed)
- **Modern XGBoost defaults** — verify `xgboost==2.x` API for our committed config still works

### V1 Critical Findings (DO NOT REVERT)

1. **bars_per_year = 3276** (NYSE RTH only). Original 17500 was 24/7 calendar — fatal bug, inflates Sharpe 2.31×.
2. **Sortino formula corrected** to canonical target downside dev.
3. **xLSTMTime added as mandatory baseline** per arXiv:2603.01820 (won 2026 finance benchmark).
4. **Forecast-to-Fill (Sharpe 2.88) replication** is the ACTUAL bar to beat. NOT replicated in 2026 — building this is publishable.
5. **Peer-benchmark discount** required per arXiv:2604.18821 (1,726 strategies analyzed; backtests capture launch regime).
6. **GLD 5y buy-and-hold Sharpe ≈ 2.4** (2020-2025 was a great gold run). This is the HONEST baseline nanoGLD must beat to claim alpha.

### Empirical Bar (Nia Agent E synthesized)

```
Tier                        Sharpe       Hit rate   DSR   Status
Minimum viable              > 0          —          > 0   Mandatory
Real claim                  > 1.0        > 52%      > 1.0 Mandatory for X thread
Forecast-to-Fill tier       > 2.5        —          —     Publishable contribution
```

If 24-60M Transformer doesn't beat baselines, **explicitly recommend shipping the simpler model in the report.** TLOB lesson: "MLP can match transformer."

### Hand-off Protocol

1. Update STATUS.md with: per-fold Sharpe, DSR, comparison table, equity curve PNG path
2. Notify doc 07 (sizing) that backtest engine is ready (Stage 2 sizer needs the same engine)
3. Notify user via STATUS update that report is ready for X thread

Now read the implementation specifics.

---

# 06 — Backtest Discipline

**Status:** ✅ Complete, implementation-ready, Nia-verified
**Last verified:** 2026-04-30

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

## What's Locked

- Walk-forward backtest on test folds only (never on train/val)
- Cost model: 5bps round-trip (2bps spread + 1bps commission + 0.5bps slippage per side)
- Equal-weight directional sizing for Stage 1 (1 GLD share when argmax != flat)
- Bootstrap CI on Sharpe with stationary block bootstrap
- Baseline ladder for honest comparison
- Regime-stratified performance reporting

## Baseline Ladder

All baselines run on identical walk-forward protocol with identical cost model:

1. **Buy-and-hold** — long GLD always
2. **Momentum / MA crossover** — 50-period EMA vs 200-period EMA, long when fast > slow
3. **Donchian breakout** (20-period high/low, long when close > prior 20-bar high) — second momentum baseline
3a. **DLinear** (Zeng et al. AAAI 2023, arXiv:2205.13504) — single-layer linear model, ~10K params. Floor baseline. **TLOB paper finding: an MLP can match a transformer on financial data.** If nanoGLD doesn't beat DLinear, transformer is overkill.
3b. **TSMixer** (Chen et al. Google ICLR 2023, arXiv:2303.06053) — pure MLP-mixer, ~2M params. Beats most transformer baselines on M5 (the closest published analog to our problem). **MANDATORY baseline.**
3c. **TimeMixer** (Wang et al. ICLR 2024, arXiv:2405.14616) — multi-scale MLP, ~5M params. Multi-resolution architecture, often beats transformers at small data scale.
3c-bis. **xLSTMTime** (xLSTM-based time-series, ~10M params) — 2026 finance benchmark winner per arXiv:2603.01820 (March 2026): **highest breakeven transaction cost buffer and best downside-adjusted Sharpe of all sequence models** tested on 2010-2025 daily futures (commodities, equities, bonds, FX). Pure Transformer underperformed. **Most nanoGLD-relevant 2026 finding.** If our 24-60M Transformer can't beat xLSTMTime at ~10M, ship xLSTMTime.
3d. **Forecast-to-Fill replication** (arXiv:2511.08571) — trend+momentum + vol-target + Kelly sizing, NOT a transformer. Cleanest published gold result (Sharpe 2.88, MDD 0.52% on 5y walk-forward). **The actual bar to beat.**
4. **XGBoost on price+macro features** — same features as our model, no news embeddings
5. **XGBoost on price+macro+news_anchor_cosines** — adds the cheap semantic features
6. **nanoGLD (our model, Stage 1 fixed sizing)**
7. **nanoGLD (Stage 2: vol-target × Kelly-lite, see doc 07)**
8. **nanoGLD (Stage 3: tiny RL sizer, see doc 07)** — only if Stage 2 leaves room

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

## What's Now Designed (post-deep-dive)

All the items below have been resolved in this doc:
- ✅ Slippage model: fixed bps (5bps round-trip), sensitivity at {3, 5, 7, 10}; market impact skipped at our $100 size
- ✅ Stationary block bootstrap with Politis-White via `arch.bootstrap.optimal_block_length`
- ✅ Regime taxonomy: vol terciles, FOMC week binary, news-density top-quartile
- ✅ Sensitivity analysis: cost sweep + latency sweep
- ✅ Display: bootstrap CI on Sharpe, $-PnL on $100 capital, equity-curve plot

## Cost Model Defense

5bps round-trip on GLD ETF at 30min granularity is realistic for retail Alpaca:
- Alpaca commission: $0 (commission-free)
- GLD bid-ask spread: typically 1-2 cents on a $200 ETF = ~0.5-1bps
- Slippage: at $100 trade size, basically 0 market impact, but the next-tick rule on entry adds ~0.5-1bps
- Total per side: ~2bps
- Round trip: ~5bps (rounded up for honesty)

Sensitivity-test at {3, 5, 10}bps to show robustness.

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
    cost_bps_round_trip: float = 5.0
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

## Cost Model Decomposition (5bps defense)

```
ENTRY:                  exit:                   ROUND-TRIP:
- Spread half: 0.5bps   - Spread half: 0.5bps    Best case: ~3bps
- Commission: 0bps      - Commission: 0bps       
- Slippage: 1.0bps      - Slippage: 1.0bps       Honest padding: +2bps for surprises
- Subtotal: 1.5bps      - Subtotal: 1.5bps       USE: 5bps
```

Sensitivity sweep at {3, 5, 7, 10} bps. Report all 4. Strategy must hold up at 7bps to be considered alpha-bearing; otherwise it's marginal.

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

- **Forecast-to-Fill (Sharpe 2.88 on gold) STILL UNREPLICATED in 2026.** No public falsification or replication found Jan-May 2026. The bar stands. Building our own honest 30min gold benchmark with cost+DSR is **publishable** (gap in literature).
- **arXiv:2604.18821 (April 2026)**: 1,726 strategies analyzed. **Backtests mostly capture launch-period market conditions, not skill.** Demands peer-benchmark discount in writeup.
- **arXiv:2604.10996 (April 2026)**: "When Valid Signals Fail: Regime Boundaries Between LLM Features and RL Trading Policies" — during macro shocks, LLM-derived features ADD noise; augmented agent UNDERPERFORMS price-only baseline. **Implication:** add regime-aware gating; consider falling back to price-only mode during high-uncertainty windows.
- **arXiv:2603.01820 (March 2026)** Oxford-MAN large-scale benchmark: **xLSTM most cost-robust, VSN+LSTM hybrid wins on Sharpe**. Generic Transformer does NOT win at portfolio level. Reinforces the xLSTMTime baseline mandate (added above).
- **No 2026 paper proves transformer+news beats XGBoost+news on single-asset 30min direction with DSR.** This remains an open empirical question — building this comparison ourselves is publishable.
- **TLOB pattern replicated (arXiv:2506.05764 on crypto LOB)**: feature engineering >> architecture depth.

## ⚠️ The Brutal GLD Buy-and-Hold Baseline

**GLD 5-year Sharpe (2020-2025) ≈ 2.4** per YCharts/PortfoliosLab. 5Y CAGR ≈ 20.6%, total return ~158%. **2020-2025 was an exceptional gold run.**

This is the actual bar your strategy must beat. A directional 30min model needs to add value AFTER costs vs a brain-dead "long GLD forever" strategy that already has Sharpe 2.4.

**Honest implication:** if your model achieves Sharpe 1.5 OOS, you are technically WORSE than buy-and-hold for this period. The story changes from "I built a great strategy" to "I built a strategy that loses to buy-and-hold during a gold bull run, but might be more useful in flat/bear regimes." Lead with this honestly in the X thread.

## The Honest Reporting Template

Every reported result includes:
- Test period (which fold[s])
- Cost model + sensitivity table (3/5/7/10 bps)
- Sharpe with 95% bootstrap CI (stationary block, B=5000)
- Total return, max drawdown, Calmar
- Trade count, turnover (annualized), hit rate
- Regime breakdown (vol terciles + FOMC week + news density)
- One-paragraph honest read identifying limitations

## Implementation Day Plan (Day 8 of week 1, post-training)

| Hour | Task |
|------|------|
| 1-2 | Backtest engine on synthetic data (verify math) |
| 3-4 | Run baselines (buy-hold, MA crossover, XGBoost) |
| 5 | Run nanoGLD Stage 1 (fixed sizing) |
| 6-7 | Bootstrap CIs, regime breakdown, sensitivity sweep |
| 8 | Equity curve plots, save artifacts for blog post |
