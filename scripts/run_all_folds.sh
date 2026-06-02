#!/usr/bin/env bash
# Train all 4 folds + cascade analysis after each.
# Idempotent via per-stage .done sentinels in checkpoints/v1/fold_<N>/.
#
# Usage:
#     bash scripts/run_all_folds.sh           # full epoch budget
#     bash scripts/run_all_folds.sh 50        # smoke mode: 50 steps/stage
#
# Env vars passed through to mac_mini_train.sh:
#   NANOGLD_DIR         working dir (default $HOME/Desktop/nanogld)
#   NANOGLD_N_FOLDS     fold count (default 4)
#   NANOGLD_DEVICE      torch device (default mps)
#   NANOGLD_BATCH_SIZE  batch (default 8 on Mac mini)
#   NANOGLD_D_MODEL     model dim (default 192 on Mac mini)
#   NANOGLD_T_BARS      lookback (default 32 on Mac mini)
#   NANOGLD_SSL_EPOCHS  default 3
#   NANOGLD_PROBE_EPOCHS default 2
#   NANOGLD_LLRD_EPOCHS default 3
#   NANOGLD_*_MAX_STEPS per-stage step cap

set -Eeuo pipefail

MAX_STEPS="${1:-}"
N_FOLDS="${NANOGLD_N_FOLDS:-4}"
NANOGLD_DIR="${NANOGLD_DIR:-$HOME/Desktop/nanogld}"

cd "${NANOGLD_DIR}"

START="$(date -u +%FT%TZ)"
echo "[run_all] starting at ${START} for ${N_FOLDS} folds; smoke=${MAX_STEPS:-OFF}"

for (( FOLD=0; FOLD<N_FOLDS; FOLD++ )); do
    OUT_DIR="checkpoints/v1/fold_${FOLD}/fold_${FOLD}"
    LLRD_DONE="${OUT_DIR}/llrd/stage.done"
    if [[ -f "${LLRD_DONE}" ]]; then
        echo "[run_all] fold ${FOLD} llrd already done — skipping training"
    else
        echo "[run_all] training fold ${FOLD} at $(date -u +%FT%TZ)"
        if [[ -n "${MAX_STEPS}" ]]; then
            bash scripts/mac_mini_train.sh "${FOLD}" "${MAX_STEPS}"
        else
            bash scripts/mac_mini_train.sh "${FOLD}"
        fi
    fi

    ANALYSIS_DONE="reports/analysis/fold_${FOLD}/analysis_*.md"
    if compgen -G "${ANALYSIS_DONE}" >/dev/null 2>&1; then
        echo "[run_all] fold ${FOLD} analysis already done — skipping"
    else
        echo "[run_all] post-fold cascade fold ${FOLD} at $(date -u +%FT%TZ)"
        bash scripts/post_fold.sh "${FOLD}" || echo "[run_all] WARN: post_fold failed fold ${FOLD}"
    fi
done

echo "[run_all] all folds done at $(date -u +%FT%TZ); started ${START}"

# Final backtest across all 4 folds.
echo "[run_all] running cross-fold backtest"
export PATH="$HOME/.local/bin:$PATH"
CKPTS=""
SIDECARS=""
CALIBS=""
for (( i=0; i<N_FOLDS; i++ )); do
    CKPTS+="checkpoints/v1/fold_${i}/fold_${i}/llrd/llrd_final.pt,"
    SIDECARS+="data/processed/training_v1_sidecar_fold_${i}.pt,"
    CALIBS+="checkpoints/v1/fold_${i}/fold_${i}/calibration_${i},"
done
CKPTS="${CKPTS%,}"
SIDECARS="${SIDECARS%,}"
CALIBS="${CALIBS%,}"

CFG="checkpoints/v1/fold_0/v1_main_macmini.yaml"
caffeinate -dimsu uv run python -m nanogld.backtest run \
    --config "${CFG}" \
    --checkpoints "${CKPTS}" \
    --sidecars "${SIDECARS}" \
    --calibration-dirs "${CALIBS}" \
    --out reports/backtest \
    --device "${NANOGLD_DEVICE:-mps}"

# Refresh paper figures from the new report.
uv run python paper/build_paper.py

echo "[run_all] DONE at $(date -u +%FT%TZ)"
