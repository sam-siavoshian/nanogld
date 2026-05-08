#!/usr/bin/env bash
# Run the V1 training pipeline on a RunPod H100 instance.
#
# Pulls data from private HF dataset, runs training for the requested
# fold, pushes checkpoints back to HF.
#
# Pre-req: scripts/runpod_setup.sh has been run.

set -euo pipefail

REPO_ID="${NANOGLD_HF_REPO:-${HF_USER}/nanogld-v1-data}"
FOLD="${1:?usage: $0 <fold_idx 0..3>}"
CONFIG="${NANOGLD_CONFIG:-configs/v1_main.yaml}"
CHECKPOINT_REPO="${NANOGLD_HF_CHECKPOINT_REPO:-${HF_USER}/nanogld-v1-checkpoints}"

cd /workspace/nanogld
export PATH="$HOME/.local/bin:$PATH"

echo "[train] downloading data from ${REPO_ID} ..."
mkdir -p data/processed
uv run huggingface-cli download "${REPO_ID}" \
    training_v1_unified.pt training_v1_sidecar.pt v1_hmm.joblib \
    --local-dir data/processed --repo-type dataset

echo "[train] running fold ${FOLD} ..."
uv run python -m nanogld.training.train run \
    --config "${CONFIG}" \
    --fold "${FOLD}" \
    --output-dir "checkpoints/v1/fold_${FOLD}"

echo "[train] uploading checkpoints to ${CHECKPOINT_REPO} ..."
uv run huggingface-cli upload "${CHECKPOINT_REPO}" \
    "checkpoints/v1/fold_${FOLD}" "fold_${FOLD}" \
    --repo-type model

echo "[train] OK fold ${FOLD}"
