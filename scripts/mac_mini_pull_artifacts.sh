#!/usr/bin/env bash
# rsync trained checkpoints + reports + logs from Mac mini back to laptop.
# Counterpart of spark_pull_artifacts.sh for the Mac mini MPS path.
#
# Run on LAPTOP. Reads from root1@100.83.86.5:~/Desktop/nanogld.
# Per-fold sha256 verify against MANIFEST.json after pull.
#
# Env vars:
#   NANOGLD_MAC_USER  default: root1
#   NANOGLD_MAC_HOST  default: 100.83.86.5 (Tailscale IP from CLAUDE.md memory)
#   NANOGLD_DIR       default: Desktop/nanogld
#   NANOGLD_N_FOLDS   default: 4

set -Eeuo pipefail

MAC_USER="${NANOGLD_MAC_USER:-root1}"
MAC_HOST="${NANOGLD_MAC_HOST:-100.83.86.5}"
REMOTE_DIR="${NANOGLD_DIR:-Desktop/nanogld}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCAL_ROOT="${NANOGLD_LOCAL_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
N_FOLDS="${NANOGLD_N_FOLDS:-4}"

SSH_OPTS=(-o StrictHostKeyChecking=no)
REMOTE="${MAC_USER}@${MAC_HOST}"

mkdir -p "${LOCAL_ROOT}/checkpoints/v1" "${LOCAL_ROOT}/reports" "${LOCAL_ROOT}/logs"

echo "[pull] checkpoints from ${REMOTE}:${REMOTE_DIR}/checkpoints/v1/ ..."
rsync -avz -e "ssh ${SSH_OPTS[*]}" \
    "${REMOTE}:${REMOTE_DIR}/checkpoints/v1/" "${LOCAL_ROOT}/checkpoints/v1/"

echo "[pull] reports from ${REMOTE}:${REMOTE_DIR}/reports/ ..."
rsync -avz -e "ssh ${SSH_OPTS[*]}" \
    "${REMOTE}:${REMOTE_DIR}/reports/" "${LOCAL_ROOT}/reports/" || true

echo "[pull] logs from ${REMOTE}:${REMOTE_DIR}/logs/ ..."
rsync -avz -e "ssh ${SSH_OPTS[*]}" \
    "${REMOTE}:${REMOTE_DIR}/logs/" "${LOCAL_ROOT}/logs/" || true

echo "[pull] per-fold sha256 verify ..."
for (( f=0; f<N_FOLDS; f++ )); do
    fold_root="${LOCAL_ROOT}/checkpoints/v1/fold_${f}"
    llrd_dir="${fold_root}/llrd"
    cal_dir="${fold_root}/calibration_${f}"
    [[ -d "${llrd_dir}" ]] || llrd_dir="${fold_root}/fold_${f}/llrd"
    [[ -d "${cal_dir}" ]] || cal_dir="${fold_root}/fold_${f}/calibration_${f}"

    if [[ -f "${llrd_dir}/MANIFEST.json" ]]; then
        (cd "${LOCAL_ROOT}" && uv run python -c "
from pathlib import Path
from nanogld.data.integrity import verify_artifacts
verify_artifacts(Path('${llrd_dir}'), require=['llrd_final.pt'])
print('fold ${f} llrd: verify OK')
")
    else
        echo "[pull] WARN: ${llrd_dir}/MANIFEST.json missing"
    fi
    if [[ -d "${cal_dir}" && -f "${cal_dir}/MANIFEST.json" ]]; then
        (cd "${LOCAL_ROOT}" && uv run python -c "
from pathlib import Path
from nanogld.data.integrity import verify_artifacts
verify_artifacts(Path('${cal_dir}'))
print('fold ${f} calibration: verify OK')
")
    fi
done

echo "[pull] OK"
