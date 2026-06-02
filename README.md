# nanoGLD

[![License](https://img.shields.io/badge/license-MIT-green.svg)](./LICENSE)
[![PyTorch](https://img.shields.io/badge/pytorch-2.11-EE4C2C.svg)](https://pytorch.org)
[![Python](https://img.shields.io/badge/python-3.11-3776AB.svg)](https://www.python.org)

## What is it

A 24M-parameter transformer that predicts the direction of GLD (gold ETF) over the next 30 minutes, using 10 years of intraday price bars and frozen LLM-encoded news headlines. From-scratch hybrid encoder. Trained on 4-fold walk-forward, calibrated with conformal prediction, sized with friction-adjusted Kelly.

## Architecture

```
bars (T=64 × F=681)               news (Qwen3-4B FP16, 8 slots/bar)
       │                                     │
   decomp → RevIN + VSN                   CFA + AECF
       │                                     │
   patches P=4 ───► encoder (10 transformer + 2 sLSTM) ◄─── cross-attn at {3,7,11}
                                │
                                ▼
                   focal CE head (3-class)  +  Sharpe head (position weight)
                                │
                                ▼
                  T-scaling → RAPS → AgACI → Kelly + ATR exits + DD breaker
```

- **Backbone**: 10 transformer blocks + 2 sLSTM blocks, d_model=384, channel-independent patching (P=4, 16 patches/channel), FiLM regime conditioning at layers {2,4,6,8,10}, sparse news cross-attention at {3,7,11}.
- **Inputs**: bars normalized via per-channel RevIN, gated through Variable Selection Network, decomposed into trend + seasonal streams (summed after patch-embed).
- **News fusion**: Qwen3-Embedding-4B (frozen, MRL-truncated to 256d) → CFA projector → Flamingo-style gated cross-attention.
- **Heads**: focal CE (γ=3) on 3-class direction + tanh Sharpe head producing position weight in [-1, +1] + DANN era-classifier through gradient reversal.
- **Calibration**: T-scaling on val_b → RAPS conformal prediction sets → AgACI online adaptation → Laplace last-layer epistemic variance.
- **Sizing**: friction-adjusted Kelly using head outputs, ATR-14 stops, 15% annualized vol target, drawdown circuit breaker, conformal floor (zero position when APS lower bound < 0.40).

## Dataset

**75,993 bars × 681 features + 40,032 news articles, 2016-01 → 2026-05.** 30-min bars over RTH for equities + 24/7 for crypto. One unified PyTorch tensor file (234 MB) + per-fold sidecar tensors built from the same source.

**Price bars (27 assets, Alpaca + Bitfinex)**:
- Metals: GLD, SLV, GDX
- US indices: SPY, QQQ, IWM, DIA, VTI
- International: EEM, EFA
- Sectors: XLE, XLF, XLK, XLU
- Treasuries: TLT, IEF
- Real estate: VNQ, IYR
- Energy: USO (WTI), BNO (Brent), UNG (nat gas)
- Volatility: VXX (2018+)
- Crypto: BTC, ETH, XRP, ADA, SOL, DOGE

**Macro (40 FRED series, ALFRED vintage-correct)**: CPI/PCE, breakevens, unemployment + JOLTS, full Treasury curve (3m–30y), TIPS, M2, WALCL, RRP, GDP, IndPro, retail sales, housing starts, UMich sentiment, savings rate, real disposable income, Case-Shiller, mortgage rate.

**Microstructure + positioning**: DXY, VIX, Brent/WTI/gold spot, COT gold futures positioning (weekly), WGC central-bank flows (quarterly), GPR geopolitical risk index, NYSE calendar event flags.

**News (40,032 articles, Qwen3-Embedding-4B 256-dim)**:
- FNSPID, Polygon, Alpha Vantage, HF multisource, Kitco, BullionVault, ECB + Fed speeches, Fox News + Fox Business (Common Crawl).
- 48.9% of bars have visible news within a 4h lookback, strict-< t_visible PIT-correct.

**Engineered features (per bar)**: log returns at 1/4/16/48/96/192/390 bars, realized vol (8/48), cross-asset ratios + correlations, GDELT 30-min tone aggregates, anchor-cosine news features (conflict / dollar / monetary / recession × mean/max/top5), v2 cross-asset interactions (flight-to-safety, digital-gold rotation, real-rate × dollar), volatility regime (VRP, vol-of-vol, RV breakout), calendar windows (NFP/CPI/FOMC/London-fix), momentum extensions, news × price interactions, half-hour-5 Gao 2014 prior. **0 leakage, 0 inf, 0 100%-NaN columns.**

**Per-fold sidecar** (built per walk-forward fold to avoid HMM/regime/h5-threshold leakage): triple-barrier labels (±1.0·ATR-14, spread-adjusted neutral), spread proxy, 12-dim regime vector (VIX tercile + RV tercile + FOMC week + year bucket + HMM P(high-vol)), ATR-14 barriers, era label.

## What we predict and how

**Target**: 3-class direction over the next 30-minute bar — DOWN / FLAT / UP — via triple-barrier labeling against the bar's ATR-14, with neutral threshold widened to ±max(spread, fixed). The same forward pass emits a continuous position weight in [-1, +1].

**Inference path**:

1. Model forward → logits (3-class) + raw position weight.
2. Calibrate: T-scaling → RAPS → APS lower bound on top-class probability.
3. Conformal floor: zero the position when APS lower bound < 0.40.
4. Size: friction-adjusted Kelly on the gated position weight, scaled to 15% annual vol target via realized vol over 60 bars.
5. Exits: 2× ATR-14 hard stop, 1.5× ATR-14 trailing stop, 30-day timeout, cumulative drawdown circuit breaker (halve / quarter / halt at -5% / -10% / -15%).

**Training**: 3 stages on a 4-fold walk-forward (train 3y / val 6mo / test 6mo, step 3mo, 1-week embargo). Stage 1 SimMTM SSL pretrain (mask ratio 0.40, K=3 views, CLIP bars↔news contrastive). Stage 2 linear probe with focal CE. Stage 3 LLRD fine-tune with Mixout p=0.7 anchored to the SSL checkpoint, FreeLB adversarial perturbation on news embeddings, EMA decay 0.999. All three stages use `Cautious(FriendlySAM(ScheduleFreeAdamW))` with AECF curriculum modality dropout and DANN gradient reversal on year-bucket era labels.

**Honest target**: 1.0–1.5 out-of-sample Sharpe net of 2bp round-trip costs over 4-fold walk-forward, beating the Gao 2014 + XGBoost simple ensemble by ≥ 0.2 Sharpe. If it loses to the simpler ensemble, ship the simpler ensemble.

## More

- `plan/V1-SPEC.md` — canonical change list.
- `plan/STATUS.md` — what's built, what's pending, how to train + ship.
- `plan/00-OVERVIEW.md` — architecture rationale.
- `plan/05-MODEL-TRAINING-CALIBRATION.md` — model + training detail.
- `plan/06-BACKTEST.md` — eval harness + 8 promotion gates.
- `plan/07-SIZING-AND-EXITS.md` — sizing layer.

## License

MIT. See [`LICENSE`](./LICENSE).
