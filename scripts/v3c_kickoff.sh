#!/usr/bin/env bash
# V3c retrain — Agent 4 VSN-prune-fix on top of V3b (Agent 2 + 5).
#
# Waits for V3b to finish (all 4 folds' llrd/stage.done sentinels), then
# clears sentinels and retrains with NANOGLD_VSN_GATE=sigmoid so the VSN
# can actually prune features rather than just redistribute mass.
#
# Usage:
#     bash scripts/v3c_kickoff.sh                    # wait then full retrain
#     bash scripts/v3c_kickoff.sh --skip-wait        # launch immediately
#     bash scripts/v3c_kickoff.sh --smoke 100        # smoke mode

set -Eeuo pipefail

NANOGLD_DIR="${NANOGLD_DIR:-$HOME/Desktop/nanogld}"
cd "${NANOGLD_DIR}"

SKIP_WAIT=0
SMOKE=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-wait) SKIP_WAIT=1; shift ;;
        --smoke) SMOKE="${2:-100}"; shift 2 ;;
        *) echo "[v3c] unknown arg: $1" >&2; exit 1 ;;
    esac
done

if (( ! SKIP_WAIT )); then
    echo "[v3c] waiting for V3b — all 4 folds' llrd/stage.done sentinels"
    while true; do
        DONE_COUNT=$(find checkpoints/v1 -path "*llrd/stage.done" 2>/dev/null | wc -l | tr -d ' ')
        if (( DONE_COUNT >= 4 )); then
            break
        fi
        echo "[v3c] $(date -u +%FT%TZ) — ${DONE_COUNT}/4 V3b folds done; sleeping 5 min"
        sleep 300
    done
    echo "[v3c] V3b complete; pausing 30 s to let V3b backtest finish"
    sleep 30
fi

# Tag V3b checkpoints before V3c overwrites.
TS="$(date -u +%Y%m%dT%H%M%SZ)"
BACKUP="checkpoints/v3b_backup_${TS}"
if [[ ! -d "${BACKUP}" ]]; then
    echo "[v3c] backing up V3b checkpoints to ${BACKUP}"
    cp -R checkpoints/v1 "${BACKUP}"
fi

# Clear sentinels for full retrain (sigmoid VSN changes encoder math
# enough that SSL anchors should be redone too).
echo "[v3c] clearing stage.done sentinels"
find checkpoints/v1 -name "stage.done" -delete 2>/dev/null || true

# V3c env: VSN sigmoid + keep V3b's other ablations.
export NANOGLD_VSN_GATE=sigmoid
export NANOGLD_NEWS_KILL="${NANOGLD_NEWS_KILL:-1}"
export NANOGLD_LLRD_USE_FSAM="${NANOGLD_LLRD_USE_FSAM:-0}"
export NANOGLD_LLRD_USE_CAUTIOUS="${NANOGLD_LLRD_USE_CAUTIOUS:-0}"
export NANOGLD_LLRD_SHARPE_WEIGHT="${NANOGLD_LLRD_SHARPE_WEIGHT:-0.0}"
export NANOGLD_BATCH_SIZE="${NANOGLD_BATCH_SIZE:-4}"

mkdir -p logs
LOG="logs/v3c_${TS}.log"
PIDFILE="logs/v3c_${TS}.pid"
CMD="bash scripts/run_all_folds.sh ${SMOKE}"
echo "[v3c] env: VSN_GATE=sigmoid NEWS_KILL=${NANOGLD_NEWS_KILL} FSAM=${NANOGLD_LLRD_USE_FSAM} Cautious=${NANOGLD_LLRD_USE_CAUTIOUS} SHARPE_W=${NANOGLD_LLRD_SHARPE_WEIGHT} BS=${NANOGLD_BATCH_SIZE}"
echo "[v3c] launching detached: ${CMD}"
nohup caffeinate -dimsu bash -c "${CMD}" > "${LOG}" 2>&1 &
echo $! > "${PIDFILE}"
disown

echo "[v3c] launched PID $(cat "${PIDFILE}"). Watch with:"
echo "   tail -f ${LOG}"
echo "[v3c] kill with:"
echo "   kill \$(cat ${PIDFILE})"
