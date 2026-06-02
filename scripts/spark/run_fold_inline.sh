#!/usr/bin/env bash
# run_fold_inline.sh — run one fold inline (NO tmux), under systemd-run scope.
#
# Used when called from inside an existing tmux session (e.g. the
# orchestrator's session). Avoids tmux-in-tmux issues by skipping the
# `tmux new-session` wrap. Heartbeat + log + cgroup MemoryMax kept.
#
# Usage:
#   run_fold_inline.sh <fold_idx> <config_name> [mem_max_gb=8]

set -euo pipefail

FOLD="${1:-}"
CONFIG="${2:-v4_wf.yaml}"
MEM_MAX_GB="${3:-8}"

if [[ -z "$FOLD" ]]; then
    echo "usage: $0 <fold_idx> <config_name> [mem_max_gb=8]"
    exit 2
fi

WORKSPACE="${SAAM_NANOGLD_WORKSPACE:-$HOME/saam-nanogld}"
VENV_PY="$WORKSPACE/.venv/bin/python"
CONFIG_PATH="$WORKSPACE/configs/$CONFIG"
CFG_BASENAME=$(basename "$CONFIG" .yaml)

if [[ ! -x "$VENV_PY" ]]; then
    echo "error: venv python not found at $VENV_PY"
    exit 3
fi
if [[ ! -f "$CONFIG_PATH" ]]; then
    echo "error: config not found at $CONFIG_PATH"
    exit 3
fi

OUTPUT_DIR=$(grep -E "^\s*output_dir:" "$CONFIG_PATH" | tail -1 | awk '{print $2}')
OUTPUT_DIR="${OUTPUT_DIR:-checkpoints/v4}"

UTC=$(date -u +%Y%m%dT%H%M%SZ)
LOG_FILE="$WORKSPACE/logs/inline-${CFG_BASENAME}-fold${FOLD}_${UTC}.log"
HEARTBEAT="$WORKSPACE/.heartbeat"
mkdir -p "$WORKSPACE/logs"

PROTECT_PIDS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null \
    | tr -d ' ' | grep -v '^$' | sort -u | tr '\n' ',' | sed 's/,$//')

GIT_SHA="${SAAM_NANOGLD_GIT_SHA:-unknown-spark-$(date -u +%Y%m%dT%H%M%SZ)}"
SCOPE_NAME="saam-nanogld-${CFG_BASENAME}-fold${FOLD}-${UTC}"
MEM_BYTES=$((MEM_MAX_GB * 1024 * 1024 * 1024))

# Heartbeat updater: background loop, killed on script exit.
(
    while :; do
        date -u +%s > "$HEARTBEAT.tmp" && mv "$HEARTBEAT.tmp" "$HEARTBEAT"
        sleep 30
    done
) &
HB_PID=$!
trap "kill $HB_PID 2>/dev/null || true" EXIT

echo "===== run_fold_inline ====="
echo "fold:         $FOLD"
echo "config:       $CONFIG_PATH"
echo "scope:        $SCOPE_NAME"
echo "MemoryMax:    ${MEM_MAX_GB}G"
echo "log_file:     $LOG_FILE"
echo "protect_pids: $PROTECT_PIDS"
echo "==========================="

# Set our own shell oom_score_adj BEFORE delegating, defense-in-depth.
echo 1000 > /proc/self/oom_score_adj 2>/dev/null || true

# Run python under cgroup scope. systemd-run --user --scope blocks until
# the command exits, so we tee its stdout to both the log file + our
# console. The scope enforces MemoryMax + OOMPolicy=kill on us alone.
systemd-run --user --scope --quiet \
    --unit="$SCOPE_NAME" \
    -p "MemoryMax=$MEM_BYTES" \
    -p "MemorySwapMax=0" \
    -p "OOMPolicy=kill" \
    -p "TasksMax=512" \
    env \
        "PYTHONPATH=$WORKSPACE/src:$WORKSPACE/bin" \
        "SAAM_NANOGLD_PROTECT_PIDS=$PROTECT_PIDS" \
        "NANOGLD_GIT_SHA=$GIT_SHA" \
        "NANOGLD_ALLOW_DIRTY_MANIFEST=1" \
        "$VENV_PY" "$WORKSPACE/bin/spark_train.py" \
        --config "$CONFIG_PATH" \
        --fold "$FOLD" \
        --output-dir "$WORKSPACE/$OUTPUT_DIR" \
        --device cuda 2>&1 | tee -a "$LOG_FILE"
