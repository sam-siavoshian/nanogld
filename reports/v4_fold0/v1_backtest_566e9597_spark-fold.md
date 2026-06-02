# nanoGLD V1 Backtest Report — run 566e9597

- git_sha: `spark-fold0-bt`
- host: `spark-0ef7`
- platform: `linux/aarch64`
- python: `3.12.3`
- torch: `2.11.0+cu130`
- started: `2026-05-25T18:59:56.487465+00:00`
- folds: 1
- strategies: ['nanogld_v1', 'buy_hold', 'ma_cross', 'donchian', 'gao_2014', 'xgboost', 'dlinear', 'tsmixer', 'timemixer', 'xlstm_time', 'vlstm', 'forecast_to_fill']
- base cost: 2.0 bps round-trip

## Promotion Gates (V1-SPEC §9.4)

| Gate | Value | Threshold | Pass | Note |
| --- | --- | --- | --- | --- |
| gate_1_sharpe_gt_1_at_1x | -0.863 | 1.0 | FAIL | walk-forward mean Sharpe across folds, net 1x cost |
| gate_2_sharpe_gt_0_5_at_1_5x | -0.864 | 0.5 | FAIL | walk-forward mean Sharpe across folds, net 1.5x cost |
| gate_3_beats_baseline_3of4 | 0 | 3 | FAIL | folds where nanogld_v1 beats best baseline by >= 0.2 Sharpe; baselines considered: ['buy_hold', 'ma_cross', 'donchian', 'gao_2014', 'xgboost', 'dlinear', 'tsmixer', 'timemixer', 'xlstm_time', 'vlstm', 'forecast_to_fill'] |
| gate_4_conformal_coverage | pending | +/- 2% of nominal | pending | pending_verification — needs calibration coverage report |
| gate_5_sizer_stage2_beats_stage1 | pending | 0.2 | pending | pending_verification — needs sizer stage A/B run |
| gate_6_drawdown_breaker_2_regimes | pending | 2 | pending | pending_verification — needs regime-tagged drawdown_breaker test |
| gate_7_dsr_gt_1 | -2.528 | 1.0 | FAIL | min deflated Sharpe across folds |
| gate_8_per_bucket_positive | 0 | 1 | FAIL | per-fold pass count for {present,absent} both > 0 |

## Cost-Stress Mean Sharpe (across folds)

| Strategy | 0.5x | 1.0x | 1.5x |
| --- | --- | --- | --- |
| nanogld_v1 | -0.862 | -0.863 | -0.864 |
| buy_hold | 0.860 | 0.859 | 0.858 |
| ma_cross | -0.709 | -0.710 | -0.711 |
| donchian | 0.000 | 0.000 | 0.000 |
| gao_2014 | 0.000 | 0.000 | 0.000 |
| xgboost | 0.000 | 0.000 | 0.000 |
| dlinear | 0.000 | 0.000 | 0.000 |
| tsmixer | 0.000 | 0.000 | 0.000 |
| timemixer | 0.000 | 0.000 | 0.000 |
| xlstm_time | 0.000 | 0.000 | 0.000 |
| vlstm | 0.000 | 0.000 | 0.000 |
| forecast_to_fill | 0.000 | 0.000 | 0.000 |

## Per-Bucket Mean Sharpe (across folds)

| Strategy | present | absent | both |
| --- | --- | --- | --- |
| nanogld_v1 | -0.853 | -0.878 | -0.863 |
| buy_hold | 0.853 | 0.869 | 0.859 |
| ma_cross | -0.535 | -0.914 | -0.710 |
| donchian | 0.000 | 0.000 | 0.000 |
| gao_2014 | 0.000 | 0.000 | 0.000 |
| xgboost | 0.000 | 0.000 | 0.000 |
| dlinear | 0.000 | 0.000 | 0.000 |
| tsmixer | 0.000 | 0.000 | 0.000 |
| timemixer | 0.000 | 0.000 | 0.000 |
| xlstm_time | 0.000 | 0.000 | 0.000 |
| vlstm | 0.000 | 0.000 | 0.000 |
| forecast_to_fill | 0.000 | 0.000 | 0.000 |

## Per-Fold Breakdown

| Fold | nanogld_v1 | buy_hold | ma_cross | donchian | gao_2014 | xgboost | dlinear | tsmixer | timemixer | xlstm_time | vlstm | forecast_to_fill |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | S=-0.863, DSR=-2.528 | S=0.859, DSR=-0.806 | S=-0.710, DSR=-2.375 | S=0.000, DSR=-1.665 | S=0.000, DSR=-1.665 | S=0.000, DSR=-1.665 | S=0.000, DSR=-1.665 | S=0.000, DSR=-1.665 | S=0.000, DSR=-1.665 | S=0.000, DSR=-1.665 | S=0.000, DSR=-1.665 | S=0.000, DSR=-1.665 |

## Honest Limitations

Reported Sharpe is conditional on the per-fold sidecar refactor (plan/STATUS.md §32) landing. Until then, HMM regime terciles + h5 vol thresholds are fit globally and applied to walk-forward folds, which inflates measured Sharpe. Cost-stress at 1.5x is the hard ship gate per V1-SPEC §9.4 — a 1.0x pass with a 1.5x fail means the edge does not survive realistic friction.
