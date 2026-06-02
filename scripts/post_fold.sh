#!/usr/bin/env bash
# Post-fold auto-cascade: analysis CLI + manifest write for one fold.
# Called after `scripts/mac_mini_train.sh` (or `spark_train.sh`) finishes.
#
# Usage:
#     bash scripts/post_fold.sh <fold>
#     bash scripts/post_fold.sh 0
#
# Hard-fails on:
#   - fold out of [0, NANOGLD_N_FOLDS)
#   - missing llrd_final.pt for that fold

set -Eeuo pipefail

FOLD="${1:?usage: $0 <fold>}"
N_FOLDS="${NANOGLD_N_FOLDS:-4}"
NANOGLD_DIR="${NANOGLD_DIR:-$HOME/Desktop/nanogld}"
DEVICE="${NANOGLD_DEVICE:-mps}"

if [[ ! "${FOLD}" =~ ^[0-9]+$ ]] || (( FOLD < 0 || FOLD >= N_FOLDS )); then
    echo "[post_fold] fold ${FOLD} out of range [0,${N_FOLDS})" >&2
    exit 2
fi

cd "${NANOGLD_DIR}"
export PATH="$HOME/.local/bin:$PATH"

# Per-fold paths follow the canonical training output layout.
OUT_DIR="checkpoints/v1/fold_${FOLD}/fold_${FOLD}"
CKPT="${OUT_DIR}/llrd/llrd_final.pt"
SIDECAR="data/processed/training_v1_sidecar_fold_${FOLD}.pt"
ANALYSIS_OUT="reports/analysis/fold_${FOLD}"

if [[ ! -f "${CKPT}" ]]; then
    echo "[post_fold] missing ${CKPT}; train fold ${FOLD} first" >&2
    exit 3
fi
if [[ ! -f "${SIDECAR}" ]]; then
    echo "[post_fold] missing ${SIDECAR}" >&2
    exit 4
fi

echo "[post_fold] fold=${FOLD} ckpt=${CKPT}"
echo "[post_fold] writing analysis to ${ANALYSIS_OUT}"

caffeinate -dimsu uv run python -m nanogld.analysis run \
    --checkpoint "${CKPT}" \
    --unified data/processed/training_v1_unified.pt \
    --sidecar "${SIDECAR}" \
    --fold "${FOLD}" \
    --split val_c \
    --output-dir "${ANALYSIS_OUT}" \
    --device "${DEVICE}"

# Re-write the manifest in the LLRD dir so the backtest verify-on-load
# step finds an up-to-date SHA256 next to the freshly-cooked checkpoint.
uv run python - <<PY
from pathlib import Path
from nanogld.data.integrity import write_manifest
write_manifest(Path("${OUT_DIR}/llrd"))
print("[post_fold] manifest OK")
PY

echo "[post_fold] OK fold ${FOLD}"
