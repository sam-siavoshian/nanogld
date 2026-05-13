#!/usr/bin/env bash
# rsync code + data from laptop to GTX Spark over Tailscale.
#
# Run on LAPTOP, not on Spark. Pushes src/ + tests/ + scripts/ + configs/
# + plan/ + pyproject + lockfile + data/processed/* to $NANOGLD_DIR
# on the Spark box.
#
# Env vars (set in shell config or per-invocation):
#     NANOGLD_SPARK_USER   ssh user (e.g. sam)
#     NANOGLD_SPARK_HOST   tailnet IP or hostname (e.g. 100.xx.xx.xx)
#     NANOGLD_DIR          remote dir (default ~/nanogld)
#     NANOGLD_LOCAL_ROOT   local source dir (default this repo)
#
# Usage:
#     bash scripts/spark_sync.sh
#     bash scripts/spark_sync.sh --data-only      # skip code, push only data/
#     bash scripts/spark_sync.sh --code-only      # skip data, push only code

set -Eeuo pipefail

: "${NANOGLD_SPARK_USER:?set NANOGLD_SPARK_USER (e.g. export NANOGLD_SPARK_USER=sam)}"
: "${NANOGLD_SPARK_HOST:?set NANOGLD_SPARK_HOST (e.g. export NANOGLD_SPARK_HOST=100.xx.xx.xx)}"

REMOTE_DIR="${NANOGLD_DIR:-nanogld}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCAL_ROOT="${NANOGLD_LOCAL_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"

MODE="all"
if [[ "${1:-}" == "--data-only" ]]; then
    MODE="data"
elif [[ "${1:-}" == "--code-only" ]]; then
    MODE="code"
fi

SSH_OPTS=(-o StrictHostKeyChecking=accept-new)
REMOTE="${NANOGLD_SPARK_USER}@${NANOGLD_SPARK_HOST}"

echo "[sync] target: ${REMOTE}:${REMOTE_DIR}"
ssh "${SSH_OPTS[@]}" "${REMOTE}" "mkdir -p ${REMOTE_DIR}"

if [[ "$MODE" != "data" ]]; then
    echo "[sync] code (src/ tests/ scripts/ configs/ plan/ pyproject + lock) ..."
    rsync -avz --delete \
        --include='src/***' \
        --include='tests/***' \
        --include='scripts/***' \
        --include='configs/***' \
        --include='plan/***' \
        --include='.github/***' \
        --include='pyproject.toml' \
        --include='uv.lock' \
        --include='.python-version' \
        --include='.pre-commit-config.yaml' \
        --include='.gitleaks.toml' \
        --include='LICENSE' \
        --include='README.md' \
        --exclude='*' \
        -e "ssh ${SSH_OPTS[*]}" \
        "${LOCAL_ROOT}/" "${REMOTE}:${REMOTE_DIR}/"
fi

if [[ "$MODE" != "code" ]]; then
    if [[ -d "${LOCAL_ROOT}/data/processed" ]]; then
        echo "[sync] data/processed (unified.pt + sidecars + MANIFEST.json) ..."
        ssh "${SSH_OPTS[@]}" "${REMOTE}" "mkdir -p ${REMOTE_DIR}/data/processed"
        rsync -avz \
            -e "ssh ${SSH_OPTS[*]}" \
            "${LOCAL_ROOT}/data/processed/" "${REMOTE}:${REMOTE_DIR}/data/processed/"
    else
        echo "[sync] no local data/processed/; skipping"
    fi
fi

echo "[sync] OK"
