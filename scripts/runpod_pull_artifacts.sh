#!/usr/bin/env bash
# Pull RunPod training artifacts back to the local working host.
#
# After RunPod completes Block 9 main run, this script downloads:
#   - 4 fold checkpoints
#   - W&B run id (for log retrieval)
#   - Calibration tensors
# from the private HF checkpoint repo.

set -euo pipefail

REPO_ID="${NANOGLD_HF_CHECKPOINT_REPO:-${HF_USER}/nanogld-v1-checkpoints}"
OUT_DIR="${NANOGLD_CHECKPOINTS_DIR:-checkpoints/v1}"

cd "${HOME}/Desktop/nanogld" 2>/dev/null || cd .

mkdir -p "${OUT_DIR}"
echo "[pull] downloading from ${REPO_ID} -> ${OUT_DIR} ..."
uv run huggingface-cli download "${REPO_ID}" \
    --local-dir "${OUT_DIR}" \
    --repo-type model

echo "[pull] verifying checksums ..."
shopt -s nullglob
for f in "${OUT_DIR}"/fold_*/llrd_final.pt; do
    SHA=$(sha256sum "$f" | cut -d' ' -f1)
    echo "  ${f}: ${SHA:0:16}..."
done

echo "[pull] OK"
