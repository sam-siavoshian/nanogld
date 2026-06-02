#!/usr/bin/env bash
# fold_orchestrator.sh — automatically advances through all 4 folds.
#
# Polls every minute. When the current fold's tmux session exits AND
# the fold's checkpoints/<run>/fold_N/llrd/stage.done sentinel exists,
# kicks off the next fold. Stops when all folds complete OR the heartbeat
# file goes stale (>5 min old) which means the training stalled.
#
# Usage:
#   fold_orchestrator.sh <run_name> <config_name> [start_fold=0] [end_fold=3]
#
# Example:
#   fold_orchestrator.sh v4_wf v4_wf.yaml 0 3
#
# Designed to run inside its own long-lived tmux session so the
# orchestrator survives SSH disconnect:
#   tmux new -d -s saam-v4-orchestrator "bin/fold_orchestrator.sh v4_wf v4_wf.yaml 0 3 |& tee logs/orchestrator_v4_wf.log"

set -euo pipefail

RUN="${1:-}"
CONFIG="${2:-}"
START="${3:-0}"
END="${4:-3}"

if [[ -z "$RUN" || -z "$CONFIG" ]]; then
    echo "usage: $0 <run_name> <config_name> [start_fold=0] [end_fold=3]"
    exit 2
fi

WORKSPACE="${SAAM_NANOGLD_WORKSPACE:-$HOME/saam-nanogld}"
HEARTBEAT="$WORKSPACE/.heartbeat"
STALL_S=300              # 5 min stall → relaunch
MAX_RELAUNCH_PER_FOLD=12 # ~12 × 5 min = 1 h of stall time before giving up
INTER_LAUNCH_WAIT_S=60   # gap between launches to let system recover

# tmux session name follows the same rule as bin/safe_run.sh:
# "saam-<config_basename>-fold<N>". Config basename strips the .yaml.
CONFIG_BASENAME="${CONFIG%.yaml}"

log() {
    echo "[$(date -u +%FT%TZ)] $*"
}

check_heartbeat() {
    if [[ -f "$HEARTBEAT" ]]; then
        local age=$(( $(date -u +%s) - $(stat -c %Y "$HEARTBEAT" 2>/dev/null || echo 0) ))
        echo "$age"
    else
        echo "999999"
    fi
}

fold_done() {
    local fold=$1
    [[ -f "$WORKSPACE/checkpoints/$RUN/fold_$fold/llrd/stage.done" ]]
}

session_alive() {
    local session=$1
    tmux has-session -t "$session" 2>/dev/null
}

launch_fold() {
    local fold=$1
    log "launching fold $fold INLINE via bin/run_fold_inline.sh (blocks until exit)"
    cd "$WORKSPACE"
    # Inline run: blocks the orchestrator on the python process. When
    # python exits (success or crash), control returns. No tmux-in-tmux
    # nesting. Cgroup MemoryMax still enforced via systemd-run inside.
    bin/run_fold_inline.sh "$fold" "$CONFIG" 8 || true
}

log "orchestrator starting: run=$RUN config=$CONFIG folds=$START..$END"
log "workspace=$WORKSPACE heartbeat=$HEARTBEAT stall_s=${STALL_S}s max_relaunch_per_fold=${MAX_RELAUNCH_PER_FOLD}"

current=$START
relaunch_count=0

while (( current <= END )); do
    log "==== fold $current (relaunch=$relaunch_count) ===="

    if fold_done "$current"; then
        log "fold $current already done; advancing"
        current=$((current + 1))
        relaunch_count=0
        continue
    fi

    # Inline blocking call: orch waits here until python exits.
    launch_fold "$current"

    if fold_done "$current"; then
        log "fold $current COMPLETE"
        current=$((current + 1))
        relaunch_count=0
        continue
    fi

    relaunch_count=$((relaunch_count + 1))
    if (( relaunch_count > MAX_RELAUNCH_PER_FOLD )); then
        log "ERROR: fold $current relaunched $relaunch_count times — aborting"
        exit 6
    fi

    log "fold $current python exited without sentinel; relaunching after ${INTER_LAUNCH_WAIT_S}s"
    sleep "$INTER_LAUNCH_WAIT_S"
done

log "all folds done ($START..$END)"
