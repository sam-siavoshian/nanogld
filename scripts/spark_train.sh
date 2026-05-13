#!/usr/bin/env bash
# Run V1 training for ONE fold on GTX Spark desktop.
# Replaces runpod_train.sh — no HF Hub round-trip, no $5-per-hour timer.
#
# Two invocation modes:
#
#   1. ON the Spark box directly:
#        bash scripts/spark_train.sh 0
#
#   2. Remote-trigger from laptop over Tailscale + SSH:
#        bash scripts/spark_train.sh --remote 0
#      (uses $NANOGLD_SPARK_USER@$NANOGLD_SPARK_HOST)
#
# Pre-reqs:
#   - scripts/spark_setup.sh has been run on the box.
#   - scripts/spark_sync.sh pushed data/processed/training_v1_unified.pt +
#     per-fold sidecars + MANIFEST.json.
#
# Hard-fails:
#   - fold out of range [0, NANOGLD_N_FOLDS)
#   - missing $NANOGLD_DIR
#   - missing unified.pt or per-fold sidecar
#   - missing CUDA at runtime
#   - non-finite loss mid-stage (training/__main__.py already raises)

set -Eeuo pipefail

# ---- remote dispatch ----
if [[ "${1:-}" == "--remote" ]]; then
    shift
    : "${NANOGLD_SPARK_USER:?set NANOGLD_SPARK_USER}"
    : "${NANOGLD_SPARK_HOST:?set NANOGLD_SPARK_HOST}"
    REMOTE_DIR="${NANOGLD_DIR:-nanogld}"
    FOLD="${1:?usage: $0 --remote <fold_idx>}"
    echo "[train] remote-trigger fold=${FOLD} on ${NANOGLD_SPARK_USER}@${NANOGLD_SPARK_HOST}:${REMOTE_DIR}"
    exec ssh -o StrictHostKeyChecking=accept-new \
        "${NANOGLD_SPARK_USER}@${NANOGLD_SPARK_HOST}" \
        "cd ${REMOTE_DIR} && NANOGLD_GIT_SHA=\${NANOGLD_GIT_SHA:-\$(git rev-parse HEAD 2>/dev/null || echo dirty-no-git)} bash scripts/spark_train.sh ${FOLD}"
fi

# ---- local-on-Spark execution ----
FOLD="${1:?usage: $0 <fold_idx 0..NANOGLD_N_FOLDS-1>}"
N_FOLDS="${NANOGLD_N_FOLDS:-4}"
CONFIG="${NANOGLD_CONFIG:-configs/v1_main.yaml}"
NANOGLD_DIR="${NANOGLD_DIR:-$HOME/nanogld}"

if [[ ! "${FOLD}" =~ ^[0-9]+$ ]] || (( FOLD < 0 || FOLD >= N_FOLDS )); then
    echo "[train] fold ${FOLD} out of range [0,${N_FOLDS}); set NANOGLD_N_FOLDS to override" >&2
    exit 2
fi
if [[ ! -d "${NANOGLD_DIR}" ]]; then
    echo "[train] ${NANOGLD_DIR} not found; run scripts/spark_setup.sh first" >&2
    exit 3
fi
cd "${NANOGLD_DIR}"
export PATH="$HOME/.local/bin:$PATH"

# PYTHONHASHSEED must be set BEFORE python starts to affect hash() salt
# (V1-SPEC §47 reproducibility). nanogld.training.setup_determinism() logs
# a warning if absent; we set it here per-fold so dict iteration order
# stays reproducible across runs.
export PYTHONHASHSEED="${NANOGLD_SEED:-42}"

UNIFIED="data/processed/training_v1_unified.pt"
SIDECAR="data/processed/training_v1_sidecar_fold_${FOLD}.pt"
if [[ ! -f "${UNIFIED}" ]]; then
    echo "[train] missing ${UNIFIED}; run spark_sync.sh from laptop first" >&2
    exit 4
fi
if [[ ! -f "${SIDECAR}" ]]; then
    echo "[train] missing ${SIDECAR}; build per-fold sidecars first:" >&2
    echo "  uv run python scripts/build_v1_sidecar.py --per-fold" >&2
    exit 5
fi

# Verify SHA256 against MANIFEST.json before burning GPU time.
echo "[train] verify sha256 ..."
uv run python - <<PY
from pathlib import Path
from nanogld.data.integrity import verify_artifacts
verify_artifacts(Path("data/processed"), require=["training_v1_unified.pt", "training_v1_sidecar_fold_${FOLD}.pt"])
print("verify OK")
PY

# Per-fold output dir + log file (V1-SPEC §47).
OUT_DIR="checkpoints/v1/fold_${FOLD}"
mkdir -p "${OUT_DIR}" logs
LOG_FILE="logs/train_fold_${FOLD}_$(date -u +%Y%m%dT%H%M%SZ).log"
echo "[train] fold=${FOLD} config=${CONFIG} out=${OUT_DIR} log=${LOG_FILE}"

# Drop --device auto in favor of explicit cuda (Spark has GPU; CPU smoke
# is a separate explicit flag in __main__.py).
uv run python -m nanogld.training run \
    --config "${CONFIG}" \
    --fold "${FOLD}" \
    --output-dir "${OUT_DIR}" \
    --device cuda 2>&1 | tee -a "${LOG_FILE}"

# Write MANIFEST.json next to the final checkpoint for the backtest
# loader's verify-on-load path.
echo "[train] writing fold MANIFEST.json ..."
uv run python - <<PY
from pathlib import Path
from nanogld.data.integrity import write_manifest
write_manifest(Path("${OUT_DIR}/fold_${FOLD}/llrd"))
print("manifest OK")
PY

# Stage 4: calibration. Produces calibration_<fold>/ next to ssl/probe/llrd.
# Required for the backtest CLI's conformal floor (V1-SPEC §10.1 cutoff
# aps_lower_bound >= 0.40). Without this, the floor is a no-op.
echo "[train] Stage 4: calibration fold ${FOLD} ..."
uv run python -m nanogld.calibration run \
    --config "${CONFIG}" \
    --fold "${FOLD}" \
    --checkpoint "${OUT_DIR}/fold_${FOLD}/llrd/llrd_final.pt" \
    --output-dir "${OUT_DIR}/fold_${FOLD}" \
    --device cuda 2>&1 | tee -a "${LOG_FILE}"

echo "[train] OK fold ${FOLD}"
