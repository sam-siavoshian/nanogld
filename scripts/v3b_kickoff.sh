#!/usr/bin/env bash
# V3b retrain — Agent 2 (news kill) × Agent 5 (drop FSAM + Cautious +
# sharpe_weight) bundled ablation.
#
# Defaults in mac_mini_train.sh already set:
#   NANOGLD_NEWS_KILL=1
#   NANOGLD_LLRD_USE_FSAM=0
#   NANOGLD_LLRD_USE_CAUTIOUS=0
#   NANOGLD_LLRD_SHARPE_WEIGHT=0.0
#
# This wrapper just clears any prior V3b sentinels and launches the
# 4-fold sweep via run_all_folds.sh in tmux. Owner can `tmux attach
# -t v3b` to watch live.
#
# Usage:
#     bash scripts/v3b_kickoff.sh
#     bash scripts/v3b_kickoff.sh --smoke 100   # 100 steps/stage for sanity

set -Eeuo pipefail

NANOGLD_DIR="${NANOGLD_DIR:-$HOME/Desktop/nanogld}"
cd "${NANOGLD_DIR}"

SMOKE=""
if [[ "${1:-}" == "--smoke" ]]; then
    SMOKE="${2:-100}"
    shift 2 || true
fi

echo "[v3b] news_kill=${NANOGLD_NEWS_KILL:-1}"
echo "[v3b] use_fsam=${NANOGLD_LLRD_USE_FSAM:-0}"
echo "[v3b] use_cautious=${NANOGLD_LLRD_USE_CAUTIOUS:-0}"
echo "[v3b] sharpe_weight=${NANOGLD_LLRD_SHARPE_WEIGHT:-0.0}"
echo "[v3b] smoke=${SMOKE:-OFF}"

# Clear all .done sentinels so we actually retrain.
echo "[v3b] clearing stage.done sentinels"
find checkpoints/v1 -name "stage.done" -delete 2>/dev/null || true

# Backup existing V2 checkpoints before V3b overwrites.
TS="$(date -u +%Y%m%dT%H%M%SZ)"
BACKUP="checkpoints/v2_backup_${TS}"
if [[ ! -d "${BACKUP}" ]]; then
    echo "[v3b] backing up V2 checkpoints to ${BACKUP}"
    cp -R checkpoints/v1 "${BACKUP}"
fi

# Launch under nohup + caffeinate so an SSH drop or lid close doesn't kill
# the run. Per the Mac mini long-running job pattern (CLAUDE.md).
mkdir -p logs
LOG="logs/v3b_${TS}.log"
PIDFILE="logs/v3b_${TS}.pid"

if [[ -f "${PIDFILE}" ]] && kill -0 "$(cat "${PIDFILE}")" 2>/dev/null; then
    echo "[v3b] already running (PID $(cat "${PIDFILE}")); tail: tail -f ${LOG}"
    exit 0
fi

CMD="bash scripts/run_all_folds.sh ${SMOKE}"
echo "[v3b] launching detached: ${CMD}"
nohup caffeinate -dimsu bash -c "${CMD}" > "${LOG}" 2>&1 &
echo $! > "${PIDFILE}"
disown

echo "[v3b] launched PID $(cat "${PIDFILE}"). Watch with:"
echo "   tail -f ${LOG}"
echo "[v3b] kill with:"
echo "   kill \$(cat ${PIDFILE})"
