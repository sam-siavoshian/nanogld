# STATUS — nanoGLD Implementation Tracker

**Project:** nanoGLD
**Owner:** samsiavoshian
**Date last updated:** 2026-05-11 (post Block 5/6/7 finish + Spark hardware swap)
**Phase:** All code-side blockers shipped (Block 5/6/7 done, per-fold sidecar landed, SHA256 verify wired, observability + heartbeat + W&B). Hardware target swapped: **H100 RunPod → owner-owned GTX Spark x86_64 desktop over Tailscale + SSH**. HF Hub round-trip dropped; data flows laptop ↔ Spark via rsync.
**Version:** **V1 — locked 2026-05-08, code finalized 2026-05-11.**

---

## Quick Status

```
Planning:        ██████████████ V1 locked 2026-05-08
Data phase:      ██████████████ unified.pt + per-fold sidecars + MANIFEST.json built
Model code:      ██████████████ all 5 spec gaps closed (DANN, AECF, decomp two-stream, sLSTM-11 XA, Cautious(FSAM(SF)))
Training code:   ██████████████ 3 stages + resume sentinels + manifest + W&B + heartbeat
Calibration:     ██████████████ T-scale + RAPS + AgACI replay + Laplace wired + predict_calibrated orchestrator
Sizing:          ██████████████ Kelly + vol-target + ATR + DD + cost + conformal floor
Backtest:        ██████████████ walk_forward + cli + report + 11 baselines (4 existing + 7 new)
Analysis:        ██████████████ 6-method suite (VSN, IG, perm, ablation, attn, groups)
Bug-hunt loop:   ██████████████ 7 waves, 101 fixes
Sidecar leak fix: █████████████ per-fold sidecars via build_v1_sidecar.py --per-fold (§32)
SHA256 verify:   ██████████████ MANIFEST.json on every artifact dir (§45)
Spark scripts:   ██████████████ spark_setup.sh + spark_sync.sh + spark_train.sh + spark_pull_artifacts.sh
Spark training:  ░░░░░░░░░░░░░░ NOT STARTED — owner runs scripts/spark_sync.sh + scripts/spark_train.sh
```

**Next gate:** owner sets `$NANOGLD_SPARK_USER` + `$NANOGLD_SPARK_HOST` env vars, runs `scripts/spark_sync.sh` then `scripts/spark_train.sh --remote N` per fold 0..3. No paid GPU; runs are on owner-owned hardware over Tailscale.

---

## V1 transition complete — 2026-05-08

9-agent Nia research synthesis on top of V1's 27 verifying agents produced V1 redlines. Owner approved Decisions 1B (hybrid encoder), 2B (channel-independent + patches), 3B (multi-task focal CE + Sharpe head), 4A (CFA + AECF + sparse cross-attn) plus all small wins. `plan/V1-SPEC.md` is the canonical change list.

---

## Implementation phase tracker

| Module | Path | Status | Notes |
|---|---|---|---|
| Model | `src/nanogld/model/` | built | hybrid encoder (10 transformer + 2 sLSTM), FiLM at {2,4,6,8,10}, sparse cross-attn at {3,7} (layer-11 XA pending), CFA, AECF (mask not wired), dual head. ~24M params. |
| Training | `src/nanogld/training/` | built | SSL (SimMTM K=3 + CLIP) + linear probe + LLRD with Mixout + FreeLB + EMA. Cautious(FSAM(SF)) wrap pending; bare AdamWScheduleFree currently. |
| Calibration | `src/nanogld/calibration/` | built | T-scaling (strong_wolfe LBFGS, ≥2-class assert) + RAPS Mondrian (order-statistic + n_c<20 fallback) + AgACI (pinball BOA, spread init). LaplaceLLLA module exists, NOT wired. AgACI replay over val_c pending. |
| Sizing | `src/nanogld/sizing/` | built | Friction-Kelly + vol-target 15% + ATR-14 (live ATR plumb pending) + 30d timeout + sqrt-impact + conformal floor (NaN-safe) + drawdown breaker (cumulative halt). |
| Backtest | `src/nanogld/backtest/` | partial | engine + metrics (Sharpe n<2 guard) + DSR + cost-stress (assert_monotone) + per-bucket. 4 of 11 baselines (buy_hold, ma_cross, donchian, gao_2014). MISSING: report.py + cli.py + walk_forward.py + 7 baselines. |
| Analysis | `src/nanogld/analysis/` | built | 6 attribution methods (VSN gate, Integrated Gradients via captum, permutation, modality ablation, cross-attn rollout, feature-group rollups) + report aggregator + CLI. 14 tests pass. SHIPPED 2026-05-08. |
| Data | `src/nanogld/data/` | built | NanoGLDDataset + sidecar alignment check + last-bar filter. F=681 confirmed. utils.py (ET, raw_dir, get_logger) created. |
| Features | `src/nanogld/features/` | built | h5, spread, ATR, regime (HMM convergence assert), triple_barrier, hmm_regime. Per-fold sidecar refactor pending (CRITICAL leak). |

---

## Bug-hunt loop summary (cron, 2026-05-08)

7 waves over the day. Cron fired every 15 min, dispatched 5-9 background agents per wave covering spec compliance, PIT/leakage, numerical stability, edge cases, CUDA/MPS/dtype, library/safety, doc accuracy, regression hunts on prior fixes, integration smoke, memory profiles, deploy-script idempotency, type hint coverage, anti-patterns. 101 fixes shipped. Cron killed by owner request after suite stabilized.

Selected high-impact fixes (full list in commits — see #40 rsync to remote host + commit):
- `news_fuser.py`: NaN row when all-news-absent (fallback unblocks slot 0 with no_news_token).
- `losses.py focal_loss`: clamp `p_t` for `(1-p_t)^γ` only, preserve precise `log_p_t`.
- `losses.py sharpe_loss`: B<2 guard + sqrt(var+eps²) floor.
- `revin.py`: denorm sign-preserving floor + zero-var skip-divide.
- `llrd_finetune.py`: Mixout snapshot/restore semantics fixed (restore BEFORE step), pre-allocated to fix OOM-at-4-8k allocator churn, atomic save (tmp + os.replace), empty-loader guard, n_steps==0 ckpt refusal, NaN-loss raise.
- `simmtm_pretrain.py`: `return_pooled=True` (was using `logits.mean()` rank-1 garbage), shape match for SimMTM (recon vs target.mean(dim=2)), CLIP `news_proj` to d_model (was d_text mismatch), `l_aecf` NameError fix in log line, atomic save.
- `model.py`: decomposition no-op deleted from forward, `value_residual_proj` added to scaled-residual init.
- `slstm_block.py`: BatchNorm1d → GroupNorm(1, d_model) (B=1 crash fix).
- `encoder.py`: drop_path schedule starts at 0.0.
- `__main__.py`: `--device auto` autodetect + reject CPU silently, `--fold` required, DataLoader pin_memory + worker_init_fn (numpy + torch + random + Generator).
- `data/utils.py`: CREATED (was missing — 7 modules failed at import).
- `configs/v1_main.yaml`: `numeric_dim` 651 → 681 (RevIN broadcast crash on first batch).
- `scripts/runpod_train.sh`: wrong module path corrected.
- Calibration: T-scaling `line_search_fn='strong_wolfe'`, RAPS order-statistic kthvalue + min_class_n=20 pooled fallback, AgACI pinball loss + spread init.
- Sizing/backtest: drawdown_breaker cumulative halt timeout, atr_stop `current_atr` arg, metrics.sharpe `n<2` guard, cost_stress `assert_monotone` helper, conformal_floor NaN handling.
- Determinism: `setup_determinism` adds numpy seed + PYTHONHASHSEED.
- Build: LICENSE added (uv build was failing), `.gitignore` added, CI yaml moved to `.github/workflows/` (was at repo root, never ran).
- Deps: upper bounds on transformers / sentence-transformers / huggingface_hub / datasets.

---

## Open before Spark training run (all critical items shipped 2026-05-11)

### Critical — DONE

- **#32 Per-fold sidecar leak** — RESOLVED via `scripts/build_v1_sidecar.py --per-fold` + `src/nanogld/data/walk_forward_splits.py`. Each fold's HMM + regime tercile + h5 vol threshold fits on that fold's train slice only.
- **#51 RunPod scripts idempotency** — RESOLVED by dropping RunPod entirely. Replaced by `scripts/spark_*.sh` (env-var driven, Tailscale + SSH, no HF Hub round-trip, no paid GPU timer).
- **#52 SSL anchor z-vs-averaged** — `__main__.py:116` snapshots `model.state_dict()` AFTER `pretrain_simmtm` returns (left model in z-train mode); disk file has averaged weights but in-memory anchor is z. Mixout regularizes toward wrong target.
- **#41 Resume + manifest** — RESOLVED. Per-stage `.done` sentinels (Spark crash = resume from sentinel) + full reproducibility manifest in every `torch.save` via `src/nanogld/_manifest.py`.
- **#45 SHA256 manifest verify-on-load** — RESOLVED via `src/nanogld/data/integrity.py`. `MANIFEST.json` written by sidecar build, verified in `NanoGLDDataset.__init__` and by `spark_pull_artifacts.sh` post-rsync.
- **#40 Rsync to remote host + commit** — all 101 fixes are local in `plan-edit/`; sync to `~/Desktop/nanogld` and create atomic commits (no Co-Authored-By per CLAUDE.md).

### Spec-compliance gaps (degrade Sharpe)

- **#33 DANN domain classifier wiring** — `dann_loss` + `grad_reverse` exist but no head + no loss term wired. ~M effort.
- **#34 AECF curriculum mask wiring** — `AECFMask` exists but never instantiated. Spec: `U(0, 0.9)` SSL / `U(0.1, 0.9)` Stage 2/3 modality dropout on news_mask. ~S effort.
- **#35 Decomposition two-stream forward** — currently no-op (deleted). Spec wants trend + seasonal through separate RevIN/VSN/PatchEmbed, summed AFTER patches.
- **#36 Cross-attn at sLSTM layer 11** — spec says `{3, 7, 11}`, only `{3, 7}` fire (encoder iterates transformer_blocks then slstm_blocks; slstm_block doesn't accept news kwargs).
- **#37 Cautious(FSAM(SF)) wrap** — bare `AdamWScheduleFree` currently; spec wants the full nesting with FSAM noise filter (current impl is vanilla SAM).

### Backtest pipeline (Block 7 vaporware)

- **#59** `report.py` + `cli.py` + `walk_forward.py` + 7 baselines (dlinear, tsmixer, timemixer, xlstm_time, vlstm, xgboost, F2F). ~1500 LOC.
- **#39** AgACI replay over val_c before save + cost monotonicity assert wired.
- **#54** ma_cross warmup zero (200 bars) + gao_2014 hold last-bar-only per spec.

### Calibration

- **#38** LaplaceLLLA wired in `calibrate.py` (fits on val_b, saves laplace.pt) + `atr_stop` live ATR call site + sizer iterative friction-Kelly cost.
- **#56** Inference orchestrator `predict_calibrated(model_out, calib) → (probs, set, lower_bound)`. Currently no inference path exists; sizer expects `aps_lower_bound` with no producer.
- **#57** Calibrate hardening: val_b ⊥ val_c assert, full reproducibility manifest, atomic dir-level commit.

### Testing + infra

- **#43** Regression locks for ~30 wave-1-2-3 fixes (12 of 13 currently lack tests).
- **#48** `tests/fixtures/v1_micro_sidecar.pt` + `tests/test_pit.py` golden fixture.
- **#58** `t_visible` column + global PIT regression scan.
- **#42** `huggingface-cli` → `hf` migration (HF Hub 1.x renamed the CLI).
- **#46** `.pre-commit-config.yaml` + `.gitleaks.toml`.
- **#47** W&B init + per-stage log file (`fold_out/<stage>.log`) + heartbeat sentinel.
- **#61** `mypy --strict` step in CI (currently decorative — pyproject sets strict=true but never invoked).
- **#44** VSN GRN width 64 → 128 (+130K params, lands at 24M floor; current ~23.88M).
- **#53** sLSTM block `linear` → `out_proj` rename + drop InstanceNorm (residual-init unscaled currently).
- **#60** V1-SPEC + 00-OVERVIEW doc inconsistencies (mostly fixed today).

### Final gates

- **#27** Block 9 Spark training run (4 folds on owner-owned GTX desktop, $0 marginal). Owner runs `scripts/spark_train.sh --remote N` per fold.
- **#28** Block 10 backtest report + ship-or-iterate decision. Owner runs `python -m nanogld.backtest run --checkpoints ... --sidecars ... --calibration-dirs ... --out reports/`.

---

## Files & paths

- Repo: `/Users/samsiavoshian/Desktop/Coding Stuff/Side Projects/ML-Trading/plan-edit/` (local working copy; canonical repo on remote host at `~/Desktop/nanogld`)
- Data on remote host: `data/processed/training_v1_unified.pt` (234 MB), `training_v1_sidecar.pt` (6.8 MB), `v1_hmm.joblib`
- HF Hub: `sam-siavoshian/nanogld-v1-data` (private)
- Branch: `v1-spec-finalize` on github.com/sam-siavoshian/nanogld

---

End of STATUS. See `plan/HANDOFF.md` for the agent-to-agent handoff (note: H100/RunPod sections in HANDOFF.md predate the Spark swap — read in conjunction with this STATUS.md).
