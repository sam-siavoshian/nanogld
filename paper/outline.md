# nanoGLD paper outline

Target: 8 pages main + appendix. ICML/NeurIPS-style.

| Section                                          | Words  | Status |
|--------------------------------------------------|--------|--------|
| Abstract                                         | 150    | draft  |
| 1. Introduction                                  | 800    | draft  |
| 2. Related work                                  | 600    | draft  |
| 3. Data                                          | 500    | draft  |
| 4. Method                                        | 1500   | draft  |
| 4.1 Hybrid encoder (PatchTST + xLSTMTime)        |        |        |
| 4.2 Multimodal fusion (CFA + AECF + Flamingo XA) |        |        |
| 4.3 Multi-task head (focal + Sharpe + DANN)      |        |        |
| 4.4 Three-stage training (SimMTM + probe + LLRD) |        |        |
| 4.5 Calibration (T-scale + RAPS + AgACI + LLLA)  |        |        |
| 4.6 Sizing (friction-Kelly + ATR + DD breaker)   |        |        |
| 5. Experiments                                   | 700    | draft  |
| 5.1 Walk-forward CV + per-fold sidecar           |        |        |
| 5.2 Cost stress + per-bucket eval                |        |        |
| 5.3 Baseline ladder (Gao 2014 + XGBoost + …)     |        |        |
| 6. Results                                       | 1000   | TODO (Spark) |
| 7. Attribution                                   | 600    | TODO (Spark) |
| 8. Limitations + future work                     | 300    | draft  |
| 9. Reproducibility                               | 200    | draft  |
| References                                       |        | bib    |
| Appendix A: Hyperparameters table                | -      | TODO   |
| Appendix B: Per-fold metrics breakdown           | -      | TODO (Spark) |
| Appendix C: Sample news cross-attn visualization | -      | TODO (Spark) |

Contributions to highlight
--------------------------

1. **Decision-aware end-to-end head** — focal CE for calibration + tanh
   Sharpe head for sizing, jointly trained. MSE on returns collapses to
   conditional mean ≈ 0; we train the head that actually drives PnL.

2. **PIT-correct multimodal fusion** — per-fold sidecar avoids the
   common HMM/regime/threshold leak; t_visible enforcement on news.

3. **Conformal sizing layer** — RAPS prediction set → AgACI adaptive α →
   APS lower bound feeds a conformal floor on position. Sized by
   friction-adjusted Kelly with Laplace-derived epistemic variance.

4. **Honest baselines** — Gao 2014 half-hour-5 (single-feature 5.43
   Sharpe in 2014) is the floor; if nanoGLD ties or loses, we ship the
   simpler ensemble.

5. **Open release** — full code, per-fold sidecar build, training
   harness, calibration, sizing, backtest, 6-method feature attribution.
   75K bars × 681 features + 40K Qwen3-encoded news.
