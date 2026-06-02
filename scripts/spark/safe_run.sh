#!/usr/bin/env bash
# safe_run.sh — launch a Spark Python job inside a memory-capped cgroup scope.
#
# Mission: NEVER kill or starve Omar's processes.
#
# Defenses layered here:
#   1. systemd-run --user --scope wraps the command in its own cgroup.
#      If we exceed -p MemoryMax / MemorySwapMax, the kernel kills only
#      processes inside the scope, leaving the rest of the system alone.
#   2. tmux session so it survives SSH disconnects and Saam can attach.
#   3. heartbeat file ~/saam-nanogld/.heartbeat updated every 30s by a
#      background loop so Mac mini cron can detect stalls.
#   4. all stdout+stderr tee'd to logs/<session>_<utc>.log for forensics.
#   5. on exit (any reason) tear down the heartbeat loop.
#
# Usage:
#   safe_run.sh <tmux_session> <mem_max_gb> <swap_max_gb> -- <command...>
#
# Example:
#   safe_run.sh saam-nanogld-v3b 25 0 -- \
#     /home/omarramadan/saam-nanogld/.venv/bin/python -m nanogld.training run --config configs/v3b.yaml --fold 0

set -euo pipefail

if [[ $# -lt 4 ]]; then
    echo "usage: $0 <tmux_session> <mem_max_gb> <swap_max_gb> -- <command...>"
    exit 2
fi

SESSION="$1"
MEM_MAX_GB="$2"
SWAP_MAX_GB="$3"
shift 3
if [[ "${1:-}" != "--" ]]; then
    echo "error: expected -- separator before command, got '${1:-<missing>}'"
    exit 2
fi
shift 1

if [[ $# -lt 1 ]]; then
    echo "error: no command provided"
    exit 2
fi

# Validate numeric inputs.
if ! [[ "$MEM_MAX_GB" =~ ^[0-9]+$ ]] || ! [[ "$SWAP_MAX_GB" =~ ^[0-9]+$ ]]; then
    echo "error: mem_max_gb and swap_max_gb must be positive integers"
    exit 2
fi

WORKSPACE="${SAAM_NANOGLD_WORKSPACE:-$HOME/saam-nanogld}"
LOG_DIR="$WORKSPACE/logs"
HEARTBEAT="$WORKSPACE/.heartbeat"
SENTINEL_DIR="$WORKSPACE/.sentinels"

mkdir -p "$LOG_DIR" "$SENTINEL_DIR"

UTC=$(date -u +%Y%m%dT%H%M%SZ)
LOG_FILE="$LOG_DIR/${SESSION}_${UTC}.log"

echo "===== safe_run.sh launching ====="
echo "session:      $SESSION"
echo "workspace:    $WORKSPACE"
echo "MemoryMax:    ${MEM_MAX_GB}G"
echo "MemSwapMax:   ${SWAP_MAX_GB}G"
echo "log_file:     $LOG_FILE"
echo "heartbeat:    $HEARTBEAT"
echo "command:      $*"
echo "================================="

# Refuse to launch if a tmux session with the same name is already alive.
if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "error: tmux session $SESSION already exists; refusing to clobber"
    echo "       run 'tmux attach -t $SESSION' to inspect, or kill it first"
    exit 3
fi

# Inner script run inside tmux + systemd-run scope.
INNER=$(mktemp -t safe_run_inner.XXXXXX.sh)
trap 'rm -f "$INNER"' EXIT
cat > "$INNER" <<INNER_EOF
#!/usr/bin/env bash
set -e
cd "$WORKSPACE"

# Heartbeat loop. Background, killed on parent exit.
(
    while :; do
        date -u +%s > "$HEARTBEAT.tmp" && mv "$HEARTBEAT.tmp" "$HEARTBEAT"
        sleep 30
    done
) &
HEARTBEAT_PID=\$!
trap "kill \$HEARTBEAT_PID 2>/dev/null || true; rm -f $HEARTBEAT" EXIT

echo "===== inner started at \$(date -u +%FT%TZ) =====" | tee -a "$LOG_FILE"
echo "user: \$(whoami)" | tee -a "$LOG_FILE"
echo "host: \$(hostname)" | tee -a "$LOG_FILE"
echo "pid:  \$\$" | tee -a "$LOG_FILE"
echo "cgroup: \$(cat /proc/self/cgroup)" | tee -a "$LOG_FILE"

# Make this whole shell the kernel's preferred OOM victim BEFORE the
# Python guard runs (defense in depth: scope OOM > shell OOM > Python guard).
echo 1000 > /proc/self/oom_score_adj 2>/dev/null || true

# Run the user command.
exec $(printf '%q ' "$@") 2>&1 | tee -a "$LOG_FILE"
INNER_EOF
chmod +x "$INNER"

# We have to keep INNER alive after this script exits — so copy it.
INNER_PERSIST="$WORKSPACE/.last_inner_$SESSION.sh"
cp "$INNER" "$INNER_PERSIST"

# systemd-run --user --scope spawns a transient scope in the user manager.
# -p MemoryMax=<bytes>      hard cap on RAM for everything in the scope
# -p MemorySwapMax=0        no swap (Omar's swap is full anyway)
# -p OOMPolicy=kill         kernel kills the WHOLE scope, not just one PID
SCOPE_NAME="saam-nanogld-${SESSION}-${UTC}"
MEM_BYTES=$((MEM_MAX_GB * 1024 * 1024 * 1024))
SWAP_BYTES=$((SWAP_MAX_GB * 1024 * 1024 * 1024))

CMD=(
    systemd-run --user --scope --quiet
    --unit="$SCOPE_NAME"
    -p "MemoryMax=$MEM_BYTES"
    -p "MemorySwapMax=$SWAP_BYTES"
    -p "OOMPolicy=kill"
    -p "TasksMax=512"
    bash "$INNER_PERSIST"
)

tmux new-session -d -s "$SESSION" "${CMD[*]}; exec bash"
echo "tmux session '$SESSION' started under scope '$SCOPE_NAME'"
echo "attach: tmux attach -t $SESSION"
echo "tail:   tail -f $LOG_FILE"
