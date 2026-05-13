#!/usr/bin/env bash
# Single-command tmux launcher: runs all 4 folds back-to-back on Spark.
# V1-SPEC §47 + plan §G8.
#
# Run on LAPTOP. Opens a detached tmux session over ssh that:
#   1. cds to $NANOGLD_DIR (default ~/nanogld)
#   2. Loops folds 0..N-1
#   3. Calls scripts/spark_train.sh for each fold
#   4. Breaks the loop if any fold returns non-zero (NaN, OOM, etc.)
#   5. Echoes TRAINING_DONE and sleeps 24h so owner can attach + inspect
#
# Attach to watch progress:
#   ssh "$NANOGLD_SPARK_USER@$NANOGLD_SPARK_HOST" -t tmux attach -t nanogld
# Detach without killing the session: Ctrl-b d
# List sessions: ssh "$NANOGLD_SPARK_USER@$NANOGLD_SPARK_HOST" tmux ls
#
# Env vars:
#   NANOGLD_SPARK_USER     ssh user on Spark (required)
#   NANOGLD_SPARK_HOST     tailscale IP/hostname (required)
#   NANOGLD_DIR            remote repo dir (default ~/nanogld)
#   NANOGLD_N_FOLDS        number of folds (default 4)
#   NANOGLD_SEED           PYTHONHASHSEED for the run (default 42)
#   NANOGLD_SESSION        tmux session name (default nanogld)

set -Eeuo pipefail

: "${NANOGLD_SPARK_USER:?set NANOGLD_SPARK_USER}"
: "${NANOGLD_SPARK_HOST:?set NANOGLD_SPARK_HOST}"

REMOTE_DIR="${NANOGLD_DIR:-nanogld}"
N_FOLDS="${NANOGLD_N_FOLDS:-4}"
SESSION="${NANOGLD_SESSION:-nanogld}"
SEED="${NANOGLD_SEED:-42}"

REMOTE="${NANOGLD_SPARK_USER}@${NANOGLD_SPARK_HOST}"
SSH_OPTS=(-o StrictHostKeyChecking=accept-new)

# Single bash command that runs on Spark inside the tmux session.
# Notes:
#   - export PYTHONHASHSEED before any uv run so hash() salt is correct
#     (data/utils.py:setup_determinism warns if missing)
#   - record git_sha into NANOGLD_GIT_SHA so manifest builder picks it up
#   - emit a clearly-greppable TRAINING_DONE marker so callers can detect
#   - sleep at the end so owner can attach + read final logs
LAST_FOLD=$((N_FOLDS - 1))
REMOTE_CMD=$(cat <<EOF
set -Eeuo pipefail
cd ${REMOTE_DIR}
export PYTHONHASHSEED=${SEED}
export NANOGLD_GIT_SHA=\${NANOGLD_GIT_SHA:-\$(git rev-parse HEAD 2>/dev/null || echo dirty-no-git)}
echo "[tmux] launched at \$(date -u +%FT%TZ); git_sha=\$NANOGLD_GIT_SHA"
for f in \$(seq 0 ${LAST_FOLD}); do
    echo "[tmux] ==== starting fold \$f ===="
    if ! bash scripts/spark_train.sh "\$f"; then
        rc=\$?
        echo "[tmux] fold \$f failed (rc=\$rc); stopping loop"
        echo "TRAINING_FAILED fold=\$f rc=\$rc"
        sleep 86400
        exit "\$rc"
    fi
done
echo "TRAINING_DONE all=${N_FOLDS}"
sleep 86400
EOF
)

ssh "${SSH_OPTS[@]}" "${REMOTE}" "tmux has-session -t ${SESSION} 2>/dev/null" \
    && { echo "[launch] session '${SESSION}' already running on ${REMOTE}; attach with:"; \
         echo "  ssh ${REMOTE} -t tmux attach -t ${SESSION}"; \
         exit 0; } || true

ssh "${SSH_OPTS[@]}" "${REMOTE}" "tmux new-session -d -s ${SESSION} 'bash -lc \"${REMOTE_CMD//\"/\\\"}\"' && tmux list-sessions"

echo "[launch] tmux session '${SESSION}' started on ${REMOTE}"
echo "[launch] attach with:"
echo "  ssh ${REMOTE} -t tmux attach -t ${SESSION}"
echo "[launch] watch the heartbeat from laptop:"
echo "  watch -n 30 ssh ${REMOTE} 'stat -c \"%y %n\" ~/nanogld/checkpoints/v1/fold_*/.heartbeat 2>/dev/null'"
