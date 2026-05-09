# nanoGLD V1 Spec Sheet

Single source of truth for the V1 redlines. All plan docs (00, 04, 05, 06, 07, README, HANDOFF, STATUS) get updated to reflect this. Agents reading this should treat it as the authoritative change list. Anything not on this list stays as V1.

Date: 2026-05-08
Author: nanoGLD owner + 9-agent Nia research synthesis
Replaces: V1 (frozen 2026-05-04)

---

## 0. Headline reframe

**Old target**: "Beat Wright 2026 Forecast-to-Fill 2.88 Sharpe."

**New target**: 1.0 to 1.5 OOS Sharpe net of 2 bps round-trip costs over 4-fold walk-forward. 2.88 is daily gold futures EOD-to-EOD with ~30-day holding, NOT comparable to 30-min intraday GLD direction. Apples-to-apples published intraday GLD record is Gao-Han-Li-Zhou 2014 (5.43 Sharpe single-feature half-hour-5 timing). Daily futures DL frontier per Saly-Kaufmann/Wood/Zohren 2026 (arXiv:2603.01820): VLSTM 2.40 Sharpe.

**Bar to beat (kept honest)**:
- Buy and hold GLD net of costs.
- 50/200 EMA crossover, Donchian breakout.
- DLinear, TSMixer, TimeMixer.
- xLSTMTime, VLSTM (these are STRONG; if either ties or wins, ship the simpler one per V1 rule 3).
- XGBoost on the same 681 features.
- Forecast-to-Fill replication on daily GLD bars (different problem, separate scoreboard).
- Gao 2014 half-hour-5 single-feature rule (this is the GLD-specific bar; if our model loses, we shipped a worse model than 2014).

If V1 fails to beat the simpler ensemble (Gao 2014 + XGBoost) by >= 0.2 Sharpe OOS net of costs, the ship recommendation is the simpler ensemble, full stop.

---

## 1. Architecture changes (Decision 1B + 2B + 3B)

### 1B. Hybrid encoder backbone

V1: pure encoder-only transformer, 12 identical pre-norm transformer blocks.

V1: 10 transformer blocks (layers 1-10) + 2 sLSTM blocks (layers 11-12) at the head. Reasons:
- Saly-Kaufmann/Wood/Zohren 2026: VLSTM 2.40 Sharpe, xLSTM 1.79 with best transaction-cost robustness, iTransformer 0.38 (channel-as-token), Mamba 0.64. Recurrent gated state beats pure attention on noisy non-stationary data.
- xLSTMTime (Alharthi & Mahmood, arXiv:2407.10240) recipe: decomposition -> linear -> BatchNorm -> xLSTM block -> linear -> InstanceNorm. Channel-independent, sLSTM for small data (<=70K samples = our 75K).
- Final 2 layers carry classification-specific mixing where regime context matters most.

### 2B. Channel-independent + patches

V1: ~14-25 channel-group tokens (iTransformer-lite).

V1: full channel-independent + patching, PatchTST style.
- Patch length P = 4 bars, stride S = 4, lookback T_bars = 64 -> 16 patches per channel.
- 681 channels share one transformer backbone (channel-independent).
- Patch projection: `Linear(P, D)` per patch, no bias.
- Position embedding: standard sinusoidal added to patches (RoPE still on attention Q/K).
- Channel mixing: removed entirely from main backbone. Cross-feature signal recovered via VSN feature gate at input (see 4. Features) + grouped cross-channel mixer at the FiLM regime injection points (see 1.5).

Reason: LPatchTST = 2.31 Sharpe vs iTransformer = 0.38 Sharpe on Saly-Kaufmann benchmark. With 75K samples, cross-channel attention overfits to in-sample correlations that do not survive regime shifts. ArXiv:2502.09683 confirms cross-channel layers help only on chaotic-ODE synthetic data.

### 1.5. FiLM regime conditioning blocks

NEW: every 2 transformer layers (layers {2, 4, 6, 8, 10}), insert a FiLM affine modulation conditioned on a 12-dim regime vector.

Regime vector (12-dim, per-bar):
- VIX-tercile one-hot (3-dim)
- FOMC-week binary (1-dim)
- Year-bucket one-hot {2016-2019, 2020-2021, 2022, 2023, 2024+} (5-dim)
- Realized-vol-quintile one-hot computed over 60 bars (no, scalar 1-dim from quintile rank)
- Wait: quintile is 5-dim one-hot. Total = 3+1+5+5 = 14 dim. Tighten to 12 by collapsing year to 4 buckets.

Final regime vector (12-dim):
- VIX-tercile one-hot (3)
- FOMC-week binary (1)
- Year-bucket one-hot {2016-2019, 2020-2022, 2023-2024, 2025+} (4)
- HMM P(high-vol) scalar from 2-state Gaussian HMM on log-returns + RV (1)
- Realized-vol-quintile one-hot 5-dim no wait, 12 dim total means 3+1+4+1+? = 9 plus one-hot vol-quintile 3 = 12. Use vol-tercile not quintile. 3+1+4+1+3 = 12.

Reorganize regime vector (final, 12-dim):
- VIX-tercile one-hot (3)
- Realized-vol-tercile one-hot computed over last 60 bars (3)
- FOMC-week binary (1)
- Year-bucket one-hot {2016-2019, 2020-2022, 2023-2024, 2025+} (4)
- HMM P(high-vol) scalar (1)

Total 12 dim.

FiLM math: `gamma_l, beta_l = Linear_l(regime_vec)` per layer l. Apply as `x = gamma_l * x + beta_l` between attention and FFN of layers {2, 4, 6, 8, 10}. Each FiLM block adds 2 * D = 768 params per layer * 5 layers = 3840 trainable params. Negligible.

### 3B. Multi-task output head (Sharpe loss + 3-class CE)

V1: single 3-class classification head, post-hoc Kelly sizing.

V1: dual head trained jointly on the SAME backbone.

Head A (3-class direction, kept):
- `nn.Linear(D, 3, bias=False)` on mean-pooled tokens.
- Loss: focal loss gamma=3 (NOT vanilla CE, see 5. Calibration).
- Used for calibration (T-scaling, AgACI, LLLA) and conformal sizing fallback.

Head B (position weight, NEW):
- `nn.Linear(D, 1, bias=False)` on mean-pooled tokens, then `tanh` -> position weight in [-1, +1].
- Loss: differentiable approximation of negative Sharpe over mini-batch.
- `L_sharpe = -mean(w * r_next) / (std(w * r_next) + eps)` where w = tanh output, r_next = next-bar log return.
- Cost-aware variant: `L_sharpe_net = -mean(w * r_next - cost * |w_t - w_{t-1}|) / std(...)`.

Combined Stage 3 loss (canonical, mirrors §8.3): `L = 0.5 * L_focal + 0.5 * L_sharpe_net + 0.05 * L_DANN + L_aecf`. The `L_clip_pretrain` term applies only at Stage 1 SSL (see §7.5), not at Stage 3.

Reason: Hwang & Zohren 2025 (arXiv:2510.03129) "MSE-optimal forecasts produce non-optimal allocations." Saly-Kaufmann 2026 trains directly on -Sharpe to hit 2.40. End-to-end decision-aware head correlates with OOS profit.

Sizing pipeline: Head B output is the primary position weight. Head A conformal interval scales position down when uncertainty is high (lower bound on probability < threshold => zero out position). Both heads ship.

---

## 2. News fusion (multimodal)

### 2.1 Cross-attention insertion: sparse, layers [3, 7, 11] of 12

V1: every layer or every-other.

V1: ONLY at layers {3, 7, 11}. Bottom 2 layers pure-bar (let numerical features form representations first), then 3 sparse cross-attn injections. mPLUG-Owl3 (arXiv:2408.04840) Table 8 ablation: 4 sparse Hyper-Attn blocks beat dense 8 blocks. Sparse insertion cuts ~half cross-attn trainable params, reduces overfit risk.

### 2.2 CFA-style FiLM/orthogonal projector before Flamingo K/V

NEW: between MRL-truncated 256-d Qwen3 token and cross-attn K/V projection, insert CFA filter:
- `text_K = FiLM(gamma_bar, beta_bar) * text_proj(text)` where `gamma_bar, beta_bar = Linear_bar(pooled_bar_features)`.
- Orthogonal residual: `text_K = text_K - <text_K, bar_pool> * bar_pool / ||bar_pool||^2`.
- ~50-100k extra params (low-rank d_text=256 -> bottleneck=64 -> d_model=384).

Reason: Lee et al. 2603.22372 ("CFA: Constrained Fusion Adapter") tested 14 TS backbones x 4 text encoders, ~20K experiments. Filtered fusion > unconstrained fusion at small TS scales. Reference: github.com/seunghan96/cfa, layers/text_fusion.py.

### 2.3 AECF entropy-gated curriculum masking

NEW: replace V1 modality dropout 15% constant with AECF (Chlon et al. arXiv:2505.15417):
- Per-batch mask probability sampled from `p ~ U(0.0, 0.9)` (curriculum 0 -> 0.9 over training, stage-1 SSL only; deployment matches empirical 51% absence rate).
- Entropy regularizer on Flamingo gate distribution: `L_aecf = -lambda(x) * sum_m p_m log p_m` where lambda(x) = MC-dropout entropy estimate.
- ~3K extra params, <1% overhead.
- PAC-bound on calibration across 2^M-1 modality subsets.

Reference: github.com/leochlon/aecf.

### 2.4 is_news_present binary embedding

NEW: concat `nn.Embedding(2, 8)(is_news_present)` to the news token. Explicit gate signal so model does not have to infer presence from zero values (Ma et al. CVPR 2022, arXiv:2204.05454). Trivial cost.

### 2.5 Variable per-batch modality dropout (replaces 15% constant)

V1: 15% constant news dropout.

V1: per-batch sampled `p ~ U(0.1, 0.9)` matching empirical 51% absence. Critical: training distribution must bracket inference distribution. 15% guarantees performance cliff at 51% no-news inference.

### 2.6 Per-bucket eval (non-negotiable diagnostic)

NEW: every metric reported separately for {news-present, news-absent, both}. Without this we fly blind on the 51% no-news bars. Add to backtest doc 06.

### 2.7 NEWS_NOT_PRESENT token: kept

V1's learnable NO_NEWS token kept as fallback. CMPT (Cross-Modal Proxy Tokens) deferred to V2 per Decision 4A.

---

## 3. Backbone block details (preserved from V1, with adjustments)

Preserved from V1 (do not touch):
- RMSNorm (eps=1e-6, no bias).
- SwiGLU FFN (hidden = round(8*D/3, 64) = 1024 at D=384, no bias).
- Real-form RoPE (NEVER view_as_complex on MPS or H100). Theta=10000.0. Partial 10% of head_dim.
- QK-Norm BEFORE RoPE.
- Per-head gating (IMU-1, sigmoid Parameter, multiplied with attn out).
- Value residuals (IMU-1).
- No bias anywhere.
- Stochastic depth p=0.15 (scheduled, see 6. Regularization).
- Init: trunc_normal_(std=0.02). Scaled residual init for *_proj and w_down: std=0.02 / sqrt(2*num_layers).

Adjustments for V1:
- Layers 11-12: replace transformer block with sLSTM block (xLSTMTime style, channel-independent).
- Stochastic depth schedule changes from uniform 0.15 to linear 0.0 -> 0.2 across depth (Touvron 2021).

---

## 4. Features (doc 04 changes)

### 4.1 NEW: Half-hour-5 intraday momentum feature

Add `r_h5 = log(close_t / close_{t-30min})` for the 5th RTH half-hour (approximately 11:30-12:00 ET). Per Gao-Han-Li-Zhou 2014, this single feature predicts last half-hour with Sharpe 5.43 on GLD specifically, concentrated on high-vol days. Also add `r_h5_x_vol_terc = r_h5 * vol_tercile_high_indicator` interaction.

Two columns added to `features`:
- `gld_h5_log_return` (float32, NaN outside RTH-5 window propagates forward to end of day)
- `gld_h5_x_vol_high` (float32, interaction with high-vol-tercile binary)

### 4.2 NEW: VSN feature gate at input

Variable Selection Network (Lim 2021, TFT, arXiv:1912.09363) at the input layer.

Math: `gate_i = softmax_i(GRN(x_i))` per feature `i in {1..681}`, then `x_gated = gate * x`. Output is same 681-dim gated input. Gate Residual Network (GRN) is a 2-layer MLP with ELU + GLU + LayerNorm.

Cost: ~2M extra params (GRN per feature group). Worth it: VSN delta over plain LSTM in Saly-Kaufmann benchmark = +0.92 Sharpe (VLSTM 2.40 vs LSTM 1.48).

### 4.3 NEW: Series decomposition (xLSTMTime requirement)

Pre-VSN, pre-RevIN: split each channel into trend + seasonal via 24-bar moving-average kernel. `trend = MA(x, kernel=24, center=False)`, `seasonal = x - trend`. Sum back after RevIN. xLSTMTime (Alharthi & Mahmood) requires this; also Autoformer-style.

### 4.4 RevIN per-channel (upgrade from per-group)

V1: RevIN per channel group (~14 groups).

V1: RevIN per individual channel (681 instances of learnable affine). Trivial cost, strictly more expressive. Huang & Yang ESWA 2026: per-feature RevIN drops RMSE 50% / MAPE 54% on cross-market stock data.

### 4.5 Triple-barrier labeling (replaces fixed 5-bps threshold)

V1: `label = sign(next_log_return)` thresholded at fixed 5 bps.

V1: triple-barrier method (López de Prado, "Advances in Financial Machine Learning"):
- Compute ATR-14 at bar close.
- Up barrier = +1.0 * ATR-14, down barrier = -1.0 * ATR-14, timeout = 1 bar (30 min).
- Label = +1 if up barrier hit first, -1 if down barrier hit first, 0 if timeout reached.
- Spread-adjusted neutral: even if up barrier touched but `|return| < spread_t`, label = 0. TLOB lesson.

Three columns:
- `label_triple_barrier` (int8 in {-1, 0, +1}, mapped to {0, 1, 2} for CE)
- `barrier_up` (float32 ATR-scaled for backtest cross-check)
- `barrier_down` (float32 ATR-scaled)

### 4.6 Spread feature (NEW for adjustment)

Add `gld_spread_bps_t` (float32) computed as 5-min trailing avg of `(ask - bid) / mid * 10_000`. Used by triple-barrier neutral threshold and by sizing layer.

---

## 5. Calibration stack (doc 05 changes)

### 5.1 Loss: focal loss gamma=3 (REPLACES vanilla CE)

V1: `F.cross_entropy(label_smoothing=0.05)`.

V1: focal loss `(1 - p_t)^gamma * log(p_t)` with gamma=3. Mukhoti 2020 (NeurIPS, arXiv:2002.09437) reduces ECE 30-50% pre-T-scaling, T post-fit converges closer to 1.0.

CRITICAL FIX: Xi 2024 (arXiv:2402.04344) shows T-scaling on CE logits HARMS APS adaptive coverage. Focal-trained logits avoid this conflict.

Reference impl: `torrvision/focal_calibration`.

### 5.2 Conformal: AgACI on top of APS (or RAPS for sharper sets)

V1: T-scaling -> APS (Romano 2020) -> q_hat.

V1: T-scaling -> RAPS (Angelopoulos 2020, arXiv:2009.14193) -> AgACI (Zaffran 2022 ICML, arXiv:2202.07282).

- RAPS: APS with size penalty `lambda * max(0, rank - kreg)`. Sharper sets, prevents pathological full-set predictions on 3-class.
- AgACI: online wrapper updates `alpha_t` based on miscoverage history, aggregates over gamma grid `[0.001, 0.005, 0.01, 0.05, 0.1]` via expert advice (BOA aggregator). Provably maintains target coverage under arbitrary distribution shift.

Reference: `mzaffran/AdaptiveConformalPredictionsTimeSeries` (R, port to PyTorch ~50 LOC). `aangelopoulos/conformal_classification`. Or use `ml-stat-Sustech/TorchCP` (production library, 464 stars Nov 2025 active, has RAPS/SAPS/ConfTS).

### 5.3 Epistemic: Laplace last-layer (REPLACES MC dropout T=20)

V1: MC dropout T=20 forward passes.

V1: Laplace last-layer approximation (Daxberger 2021, arXiv:2106.14806) via `aleximmer/Laplace` (`pip install laplace-torch`). Posterior variance feeds into Kelly multiplier as `min(1, sigma_target / sigma_posterior)`. Faster, better-calibrated epistemic, no 20x inference cost.

Snapshot ensemble (last 3 EMA checkpoints) and EMA 0.999: kept.

Skip deep ensembles (5x compute kills budget; Abe 2022 arXiv:2202.06985 shows snapshot+EMA matches 80-90% of DE accuracy at 1x cost).

### 5.4 T-scaling: bounds [0.7, 3.0] kept; expect T near 1.0 with focal

After focal loss, T should land in [0.9, 1.5]; existing [0.7, 3.0] guard rarely activates (a feature, not bug).

---

## 6. Regularization (doc 05 changes)

### 6.1 Mixout p=0.7 (Stage 3 fine-tune anchor = SSL checkpoint)

NEW: Lee, Cho, Kang ICLR 2020 (arXiv:1909.11299). Bernoulli-mix between current and SSL-anchor weights with p=0.7 during stage 3 LLRD fine-tune. Acts as L2-toward-pretrained constraint with adaptive coefficient. Reference impl: `bloodwass/mixout`.

### 6.2 Stochastic depth schedule: linear 0.0 -> 0.2

V1: uniform 0.15.

V1: linear schedule `p_l = 0.2 * l / num_layers`. Concentrates regularization at higher layers where overfit is worst (Touvron 2021, ViT/DeiT pattern).

### 6.3 SimPSI spectral-preserving augmentation (REPLACES naive jittering)

V1: jittering sigma=0.02 + magnitude warping. Fons 2020 (arXiv:2010.15111) shows naive jittering is NET NEGATIVE on Sharpe.

V1: SimPSI (Ryu et al. AAAI 2024, arXiv:2312.05790) + Wave-Mask (Arabi 2024 arXiv:2408.10951). Reweights aug strength so dominant frequency components preserved (jittering kills trend signal). Wave-Mask masks DWT coefficients. Both PIT-safe (no time shift).

Manifold Mixup alpha=0.2 at hidden states: kept (NEVER raw input).

### 6.4 Cautious update mask on Schedule-Free AdamW

NEW: Liang 2024 (arXiv:2411.16085). 5-line patch:

```python
# Inside SF-AdamW step:
update = update * (sign(update) == sign(grad)).float()
```

Mask updates where momentum disagrees with gradient. 1.47x sample efficiency at zero hparam cost. Drop-in compatible with SF-AdamW. Reference: `kyleliang919/C-Optim`.

### 6.5 muP transfer-tune (small upfront, big budget save)

NEW: spend $5 on a 2-4M-param tiny model muP-parameterized sweep (LR, beta_2, init scale, FSAM rho). Transfer to 30M for the one-shot run. Yang 2022 (arXiv:2203.03466), EleutherAI guide. Reference: `microsoft/mup`.

Saves $30-50 of LR-guess risk on the $60-150 one-shot.

### 6.6 FreeLB on news embeddings (decision: include in V1)

NEW: Zhu et al. ICLR 2020 (arXiv:1909.11764). Adversarial perturbation in 256-d Qwen3 news embedding space only. K=2 inner ascent steps, epsilon=0.5. ~30% wall-clock overhead, addresses regime-shift defense.

Not on bars (PIT correctness risk), only on news embeddings.

Reference: `zhuchen03/FreeLB`.

### 6.7 DANN gradient reversal on era-label

NEW: Feng et al. IJCAI 2019 (arXiv:1810.09936). Domain-adversarial training with gradient reversal layer (Ganin 2016) on era-label = year-bucket {2016-2019, 2020-2022, 2023-2024, 2025+}. Lambda ramps 0 -> 0.1 over training.

`L_DANN = -lambda * CE(domain_classifier(GRL(z)), era_label)`.

+3.11% accuracy lift on stock movement prediction (Feng 2019). ~50K extra params for the small discriminator.

---

## 7. SSL pretraining (doc 05 changes)

### 7.1 Mask ratio 0.20 -> 0.40

V1: 0.20.

V1: 0.40 (PatchTST default). V1's 0.20 was image-MAE inheritance, too easy.

### 7.2 SimMTM-style multi-mask + similarity (REPLACES plain MAE)

V1: plain MAE on masked bars.

V1: SimMTM (Dong NeurIPS 2023 Spotlight, arXiv:2302.00861). K=3 masked views per sample, encode each, reconstruct via similarity-weighted blending of neighbor representations. Beats MAE/TS2Vec/TF-C on UEA classification linear-probe.

Loss: `L_recon + lambda_sim * L_similarity` where lambda_sim=0.5.

Reference: `thuml/SimMTM`. ~30 LOC over current MAE.

### 7.3 CLIP-style bars<->news contrastive head (NEW SSL objective)

NEW. Use the 40K Qwen3-256-d news embeddings as anchors. During SSL, add a contrastive InfoNCE loss between bar-rep (CLS) and news-rep for bars within ±5 min of a news event.

`L_clip = -log(exp(sim(z_bar, z_news+) / tau) / sum_neg exp(sim(z_bar, z_neg) / tau))`.

Tau=0.07. ~5% extra pretrain compute. Gives encoder text-semantic priors for free.

### 7.4 Pretrain config

- Patch length 4 (consistent with main backbone). Decoder depth 2 (light decoder = better encoder, MAE finding).
- Mask ratio 0.40.
- 15-20 epochs pretrain. ~25-30% of total compute budget on SSL, 70-75% on stages 2+3.

### 7.5 Multi-task SSL combined loss

`L_pretrain = L_simmtm + 0.5 * L_clip + 0.05 * L_DANN + L_aecf` (canonical, matches §8.3).

---

## 8. Optimizer + training pipeline (doc 05 changes)

### 8.1 Cautious-Schedule-Free-AdamW (V1 + 5-line patch)

`Cautious-Schedule-Free-AdamW(lr=1e-4_from_muP_transfer, betas=(0.9, 0.95), wd=0.1, warmup=300)`.

Friendly-SAM ρ=0.05 wrap kept.
EMA decay 0.999 kept.
Grad clip max_norm=1.0 kept.
WD groups: bias + LayerNorm + pos_embed = no decay (kept).

### 8.2 Stage 3 LLRD: layer-wise LR decay 0.85, Mixout p=0.7

`lr_l = base_lr * 0.85^(num_layers - l - 1)` per layer l.
Mixout p=0.7 anchored to SSL checkpoint.

### 8.3 Combined training loss

Stage 1 SSL: `L = L_simmtm + 0.5 * L_clip + 0.05 * L_DANN + L_aecf`.
Stage 2 linear probe: `L = L_focal_3class` (classifier head only, encoder frozen).
Stage 3 LLRD fine-tune: `L = 0.5 * L_focal + 0.5 * L_sharpe_net + 0.05 * L_DANN + L_aecf`.

---

## 9. Evaluation and gates (doc 06 changes)

### 9.1 Per-bucket eval (NEW, hard requirement)

Every metric reported for {news-present, news-absent, both}.

### 9.2 Cost-stress as HARD gate

V1: Sharpe net of single cost assumption.

V1: report Sharpe at {0.5x, 1.0x, 1.5x} cost levels. Hard gate: must show Sharpe > 0.5 at 1.5x cost (Wright F2F died at 1.5x; ours likely worse).

Cost model:
- Half-spread = 0.7 bps base (k=0.7bps from F2F paper).
- sqrt-impact: gamma=0.02.
- Total round-trip: ~2 bps base, 1x = 2bps, 0.5x = 1bp, 1.5x = 3bps.

### 9.3 Deflated Sharpe Ratio (DSR) as HARD gate

DSR > 1.0 across all reported configs (Bailey & López de Prado). Multi-config selection penalty enforced. No cherry-picking.

### 9.4 Updated promotion gates (V1)

```
Gate 1   Walk-forward Sharpe > 1.0 net of 1x cost (was 0.8)
Gate 2   Sharpe > 0.5 net of 1.5x cost (NEW hard)
Gate 3   Beats best baseline by >= 0.2 Sharpe on >= 3 of 4 folds
Gate 4   Conformal coverage within ±2% of nominal on val + per-bucket
Gate 5   Stage 2 sizer (decision-aware head) beats Stage 1 fallback by >= 0.2 Sharpe OOS
Gate 6   Drawdown circuit breaker tested on >= 2 historical regimes
Gate 7   Deflated Sharpe Ratio > 1.0 (NEW hard)
Gate 8   Per-bucket Sharpe (news-present, news-absent) both positive (NEW hard)
```

Fail any gate, the negative result gets reported. Cherry-picking is fireable.

### 9.5 Walk-forward CV (kept from V1)

4 folds, 1-week embargo, train 3y + val 6mo + test 6mo, step 3mo. NYSE RTH bars_per_year=3276 (NEVER 17500).

---

## 10. Sizing layer (doc 07 changes)

### 10.1 F2F-style strategy machinery on top of model output

V1: Kelly-lite × vol-target × confidence (specced but light on details).

V1 (locks F2F machinery, replaces V1's sizing spec):
- Position weight = output of Head B (`tanh` -> [-1, +1]).
- Friction-adjusted Kelly: `kelly_size = lambda * edge / variance` with `lambda=0.4` (half-Kelly-ish), `edge` from Head B output, `variance` from Laplace last-layer posterior + rolling 60-bar realized variance.
- Cost-aware adjustment: `effective_size = kelly_size * max(0, 1 - cost / |edge|)` (zero out if cost exceeds edge).
- Vol target: scale to 15% annualized vol (using NYSE RTH bars_per_year=3276).
- ATR-14 hard stop at 2x ATR, trailing 1.5x ATR.
- 30-day timeout (per F2F).
- sqrt-impact cost model: `cost_t = gamma * sqrt(|delta_w|) + k_bps * |delta_w|`, gamma=0.02, k=0.7bps.
- Conformal floor: if APS lower-bound on top-class probability < 0.40, force position to 0. Defensive shutoff.

### 10.2 Sizing gates

Stage 1 (basic): Head B output * vol target.
Stage 2 (full): Head B + friction-adjusted Kelly + ATR exits + vol target + 30-day timeout + conformal floor.
Stage 2 must beat Stage 1 by >= 0.2 Sharpe OOS to ship the full pipeline. Otherwise ship Stage 1 (simpler).

---

## 11. Things explicitly NOT in V1

- MoE / Switch / Mixtral / OLMoE / Soft MoE. 75K samples too small for sparse routing to learn.
- KAN / KASPER. KASPER's Sharpe 12.02 is fraud-tier; Renaissance Medallion is ~2.5 net.
- Mamba / S-Mamba / S5 / Hyena. 0.64 Sharpe on financial benchmark.
- Jamba. 50B+ scale required.
- Test-Time Training (TTT). V2 candidate; inference cost too high for one-shot.
- Online learning / EWC. Walk-forward CV already simulates fresh data.
- Muon optimizer. Speedup inversely proportional to scale; 30M too small.
- Sophia. Stanford benchmark (arXiv:2509.02046) shows 2x claim collapses under fair tuning.
- Deep ensembles. 5x compute kills budget.
- LLaVA-Mini one-token compression. Wrong scale.
- Cross-Modal Proxy Tokens (CMPT). Deferred to V2.
- Bars-only distillation. Deferred to V2.
- Set Transformer over present articles. V2 alternative to anchor pooling.
- Full xLSTMTime swap (encoder-wide). V1 keeps hybrid (10 transformer + 2 sLSTM).
- End-to-end Sharpe-loss-only head (drops classification). V1 keeps multi-task.

These can be revisited after V1 baseline lands.

---

## 12. Critical V1 invariants (preserved from V1)

All 17 hard rules from V1 are KEPT:

1. NEVER MSE on returns.
2. STAY FROM-SCRATCH (encoder + sLSTM head + heads, all trainable; only Qwen3 frozen).
3. SHIP THE SIMPLER MODEL IF IT TIES.
4. Apply peer-benchmark discount.
5. bars_per_year = 3276 (NEVER 17500).
6. Point-in-time correctness everywhere. `t_visible` invariant. CI gate: `test_release_ts_lte_t_visible_all_rows`.
7. gitleaks BEFORE first commit.
8. PyTorch 2.11.0 pinned (SDPA fix #174945).
9. FP32 weights everywhere. No autocast, no torch.compile, no quantization on training run. (H100 has bf16 but for V1 we stay FP32 deterministic.)
10. `.contiguous()` Q/K/V before SDPA (PyTorch #181133).
11. Every feature row carries `t_visible`.
12. ALFRED `get_series_all_releases` for ALL FRED series.
13. pandas-ta KAMA / Ichimoku / KST / DPO / TRIX / Vortex FORBIDDEN.
14. Calendar features = binary windows ONLY.
15. Anchor templates = hand-crafted with NO event provenance.
16. Alpaca News field = `created_at` (NOT `published_at`, NEVER `updated_at`).
17. DFF for daily Fed Funds (NOT FEDFUNDS, monthly).

NEW V1 invariants (18-25):

18. Per-bucket eval (news-present / news-absent / both) is non-negotiable.
19. Cost-stress at {0.5x, 1.0x, 1.5x} on every reported Sharpe.
20. DSR > 1.0 hard gate.
21. SimPSI / Wave-Mask aug only; naive jittering FORBIDDEN (Fons 2020 net-negative on Sharpe).
22. Focal loss gamma=3 (NOT vanilla CE) — required for clean T-scaling/APS interaction.
23. Triple-barrier labels with spread-adjusted neutral threshold.
24. Variable per-batch modality dropout p~U(0.1, 0.9), NOT 15% constant.
25. Decision-aware head (multi-task with Sharpe loss) is V1 ship gate. End-to-end profit metric, not just classification accuracy.

---

## 13. Implementation order (suggested, owner decides)

1. Doc redlines (this V1 spec sheet propagates into 00, 04, 05, 06, 07, README, HANDOFF, STATUS).
2. Build src/nanogld/model/ from scratch following V1 spec.
3. Build src/nanogld/training/ with multi-stage SSL → probe → LLRD pipeline.
4. Build src/nanogld/calibration/ with focal+T-scaling+RAPS+AgACI+LLLA.
5. Build src/nanogld/sizing/ with F2F machinery.
6. Build src/nanogld/backtest/ with cost-stress + DSR + per-bucket.
7. muP sweep on 2-4M tiny model ($5).
8. Run SSL pretrain (~25-30% budget).
9. Run linear probe + LLRD fine-tune on H100 ($60-150 main run).
10. Walk-forward eval, gate check, ship-or-iterate.

---

## 11. Post-train feature attribution (added 2026-05-08)

V1 ships with a six-method interpretability suite under `src/nanogld/analysis/`. Runs once per fold after Stage 3 LLRD, on the held-out `val_c` slice (does not double-dip the calibration set).

Methods (each writes parquet + json artifacts under `reports/analysis/fold_N/`):

1. VSN gate importance. The Variable Selection Network already produces a softmax distribution over the 681 features per bar. Mean-over-eval gives a free importance ranking. Split per news-presence bucket per Inv 18.
2. Integrated Gradients (captum). Path-integral attribution against the `channel_inputs` tensor with the other modalities held fixed. Per-class signed + mean-abs aggregates.
3. Permutation importance. Model-agnostic ground truth. Shuffles each feature column across the batch axis, measures Δfocal-loss + ΔSharpe. Capped to top-N features (by VSN gate) so the suite finishes in ~10 minutes.
4. Modality ablation. Zeroes each input stream (bars / news / regime / bars+news), reports focal + Sharpe drop, split per bucket.
5. Cross-attention rollout. Re-computes the NewsFuser softmax weights to expose which news slots get attention, split per bucket.
6. Feature-group rollups. Categorizes the 681 features into 9 buckets (price / volatility / macro / calendar / regime / news / flow / rates / other) and sums per-feature importance into per-category summaries.

Why not SHAP. DeepSHAP and DeepLIFT need layer-by-layer support; our model has SwiGLU + sLSTM + RoPE + GroupNorm + custom RMSNorm, so captum's `DeepLift` falls back to gradients on unsupported layers, which converges to IG anyway. KernelSHAP is O(n_features × n_perturbations × n_samples) ≈ infeasible at 681 × thousands × hundreds. Permutation importance gives the same model-agnostic ranking signal at much lower cost.

Outputs: a single markdown report `analysis_<run_hash>_<git_sha>.md` plus per-method parquet/json artifacts and a manifest. Atomic write throughout.

CLI:

```
uv run python -m nanogld.analysis run \
    --checkpoint checkpoints/v1/fold_0/llrd/llrd_final.pt \
    --unified data/processed/training_v1_unified.pt \
    --sidecar data/processed/training_v1_sidecar.pt \
    --fold 0 --split val_c --device auto
```

Reproducibility manifest includes git SHA, hostname, run hash (8-char sha256 over hashable cfg fields), python + platform version. Defaults: 256 IG samples × 32 steps, 100 features × 3 perm reps, val_c split.

Spec module path: `src/nanogld/analysis/`. Test path: `tests/analysis/`.

---

## 14. Open V1 questions (defer until baseline)

- A/B SimMTM vs T-JEPA (paper says comparable). Keep SimMTM as primary.
- A/B FreeLB ON vs OFF (some risk of distribution-shift overcorrection).
- Should regime vector include news-density bucket too? Currently no.
- Multi-symbol expansion (SPY/QQQ/IWM as auxiliary heads for transfer).
- Hyperparameter freeze date: post-muP sweep, before main H100 run.

---

End of V1 spec sheet. All redlines reference this document.
