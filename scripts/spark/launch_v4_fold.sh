#!/usr/bin/env bash
# launch_v4_fold.sh — kick off one V4 walk-forward fold on Spark.
#
# Defenses (delegated to safe_run.sh + oom_guard.py + the trainer's
# existing per-stage sentinel resume):
#   - systemd-run --user --scope MemoryMax=25G MemorySwapMax=0
#   - oom_score_adj=+1000 on the python pid
#   - torch.cuda.set_per_process_memory_fraction(0.20)
#   - watchdog kills self if Omar's RAM spikes or system free RAM drops
#   - tmux session for SSH-survival
#   - heartbeat file for stall detection
#   - per-stage stage.done sentinels (existing trainer behavior)
#
# Usage:
#   launch_v4_fold.sh <fold_idx> [config_name]
#
# Default config is v4_wf.yaml. Output goes to checkpoints/v4/fold_N/.

set -euo pipefail

FOLD="${1:-}"
CONFIG="${2:-v4_wf.yaml}"

if [[ -z "$FOLD" ]]; then
    echo "usage: $0 <fold_idx> [config_name=v4_wf.yaml]"
    exit 2
fi

if ! [[ "$FOLD" =~ ^[0-9]+$ ]]; then
    echo "error: fold_idx must be a non-negative integer"
    exit 2
fi

WORKSPACE="${SAAM_NANOGLD_WORKSPACE:-$HOME/saam-nanogld}"
VENV_PY="$WORKSPACE/.venv/bin/python"
CONFIG_PATH="$WORKSPACE/configs/$CONFIG"

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
CFG_BASENAME=$(basename "$CONFIG" .yaml)
SESSION="saam-${CFG_BASENAME}-fold${FOLD}"

# Protected PIDs: Omar's current GPU/CPU users — refreshed at launch time.
# `nvidia-smi --query-compute-apps=pid` lists active GPU processes; we
# filter our own pid out at runtime.
PROTECT_PIDS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' ' | grep -v '^$' | sort -u | tr '\n' ',' | sed 's/,$//')
echo "protect_pids = $PROTECT_PIDS"

# Compose the python command. PYTHONPATH must be set because the package
# is not pip-installed (torchsort C extension blocks editable install on
# this aarch64 host).
# Allow the spark workspace (not a git repo) to skip git_sha resolution.
# Spark workspace is rsync'd from local laptop, so the git SHA needs to
# be passed via env from the syncing side (or accepted as dirty).
GIT_SHA="${SAAM_NANOGLD_GIT_SHA:-unknown-spark-$(date -u +%Y%m%dT%H%M%SZ)}"

PYCMD=(
    env
    "PYTHONPATH=$WORKSPACE/src:$WORKSPACE/bin"
    "SAAM_NANOGLD_PROTECT_PIDS=$PROTECT_PIDS"
    "NANOGLD_GIT_SHA=$GIT_SHA"
    "NANOGLD_ALLOW_DIRTY_MANIFEST=1"
    "$VENV_PY"
    "$WORKSPACE/bin/spark_train.py"
    --config "$CONFIG_PATH"
    --fold "$FOLD"
    --output-dir "$WORKSPACE/$OUTPUT_DIR"
    --device cuda
)

cd "$WORKSPACE"

# safe_run.sh cgroup MemoryMax. Our actual training RSS is small (~3 GB
# for the python process + dataset). 8 GB is a generous cap that
# leaves enough headroom for transient dataloader spikes without
# starving Omar's GPU jobs. The kernel cgroup OOMPolicy=kill ensures
# Spark NEVER kills Omar's processes — only our scope dies if we exceed.
exec "$WORKSPACE/bin/safe_run.sh" "$SESSION" 8 0 -- "${PYCMD[@]}"
