#!/usr/bin/env bash
# rsync trained checkpoints + reports from GTX Spark back to laptop.
# Replaces runpod_pull_artifacts.sh — no HF Hub download.
#
# Run on LAPTOP. Reads from $NANOGLD_SPARK_USER@$NANOGLD_SPARK_HOST:$NANOGLD_DIR.
#
# Default destination: ${NANOGLD_LOCAL_ROOT:-this repo}/checkpoints/v1/
# and ${NANOGLD_LOCAL_ROOT}/reports/.
#
# After pull, verifies SHA256 against each fold's MANIFEST.json so a
# truncated rsync fails fast at the loader, not deep in backtest.

set -Eeuo pipefail

: "${NANOGLD_SPARK_USER:?set NANOGLD_SPARK_USER}"
: "${NANOGLD_SPARK_HOST:?set NANOGLD_SPARK_HOST}"

REMOTE_DIR="${NANOGLD_DIR:-nanogld}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCAL_ROOT="${NANOGLD_LOCAL_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
N_FOLDS="${NANOGLD_N_FOLDS:-4}"

SSH_OPTS=(-o StrictHostKeyChecking=accept-new)
REMOTE="${NANOGLD_SPARK_USER}@${NANOGLD_SPARK_HOST}"

mkdir -p "${LOCAL_ROOT}/checkpoints/v1" "${LOCAL_ROOT}/reports" "${LOCAL_ROOT}/logs"

echo "[pull] checkpoints from ${REMOTE}:${REMOTE_DIR}/checkpoints/v1/ ..."
rsync -avz \
    -e "ssh ${SSH_OPTS[*]}" \
    "${REMOTE}:${REMOTE_DIR}/checkpoints/v1/" "${LOCAL_ROOT}/checkpoints/v1/"

echo "[pull] reports from ${REMOTE}:${REMOTE_DIR}/reports/ ..."
rsync -avz \
    -e "ssh ${SSH_OPTS[*]}" \
    "${REMOTE}:${REMOTE_DIR}/reports/" "${LOCAL_ROOT}/reports/" || true

echo "[pull] logs from ${REMOTE}:${REMOTE_DIR}/logs/ ..."
rsync -avz \
    -e "ssh ${SSH_OPTS[*]}" \
    "${REMOTE}:${REMOTE_DIR}/logs/" "${LOCAL_ROOT}/logs/" || true

echo "[pull] per-fold sha256 verify ..."
for (( f=0; f<N_FOLDS; f++ )); do
    fold_dir="${LOCAL_ROOT}/checkpoints/v1/fold_${f}/fold_${f}/llrd"
    if [[ ! -d "${fold_dir}" ]]; then
        # Older layout (no double fold_N): checkpoints/v1/fold_N/llrd
        fold_dir="${LOCAL_ROOT}/checkpoints/v1/fold_${f}/llrd"
    fi
    if [[ ! -d "${fold_dir}" ]]; then
        echo "[pull] WARN: fold ${f} dir missing — skipping verify"
        continue
    fi
    if [[ -f "${fold_dir}/MANIFEST.json" ]]; then
        (cd "${LOCAL_ROOT}" && uv run python -c "
from pathlib import Path
from nanogld.data.integrity import verify_artifacts
verify_artifacts(Path('${fold_dir}'), require=['llrd_final.pt'])
print('fold ${f}: verify OK')
")
    else
        echo "[pull] WARN: ${fold_dir}/MANIFEST.json missing — bare sha256:"
        for ckpt in "${fold_dir}"/llrd_final.pt; do
            [[ -f "${ckpt}" ]] || continue
            SHA=$(shasum -a 256 "${ckpt}" 2>/dev/null | cut -d' ' -f1 || \
                  sha256sum "${ckpt}" | cut -d' ' -f1)
            echo "  ${ckpt}: ${SHA:0:16}..."
        done
    fi
done

echo "[pull] OK"
