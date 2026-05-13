#!/usr/bin/env bash
# Sequential 4-fold launcher for Mac mini (replaces tmux pattern).
#
# Run ON Mac mini (or remote-trigger from laptop). Detaches via
# 'nohup ... & disown' so ssh disconnect doesn't kill the run.
#
# Usage on box:
#     bash scripts/mac_mini_launch.sh
# From laptop:
#     ssh root1@100.83.86.5 'bash -lc "cd ~/Desktop/nanogld && nohup bash scripts/mac_mini_launch.sh > logs/launch.log 2>&1 & disown"'
#
# Logs land in:
#     logs/launch.log                          full driver log
#     logs/train_fold_N_<ts>.log               per-fold detail
#
# Breaks loop on first non-zero fold exit. Writes TRAINING_DONE / TRAINING_FAILED
# marker to logs/launch.log so monitors can grep it.

set -Eeuo pipefail

NANOGLD_DIR="${NANOGLD_DIR:-$HOME/Desktop/nanogld}"
N_FOLDS="${NANOGLD_N_FOLDS:-4}"
SEED="${NANOGLD_SEED:-42}"
START_FOLD="${1:-0}"

cd "${NANOGLD_DIR}"
export PATH="$HOME/.local/bin:$PATH"
export PYTHONHASHSEED="${SEED}"
export NANOGLD_GIT_SHA="${NANOGLD_GIT_SHA:-$(git rev-parse HEAD 2>/dev/null || echo dirty-no-git)}"

echo "[launch] started $(date -u +%FT%TZ) git_sha=${NANOGLD_GIT_SHA} start_fold=${START_FOLD}"

LAST_FOLD=$((N_FOLDS - 1))
for (( f=START_FOLD; f<=LAST_FOLD; f++ )); do
    echo "[launch] ==== starting fold ${f} at $(date -u +%FT%TZ) ===="
    if ! bash scripts/mac_mini_train.sh "${f}"; then
        rc=$?
        echo "[launch] fold ${f} failed (rc=${rc})"
        echo "TRAINING_FAILED fold=${f} rc=${rc}"
        exit "${rc}"
    fi
    echo "[launch] ==== fold ${f} done at $(date -u +%FT%TZ) ===="
done

echo "TRAINING_DONE all=${N_FOLDS} at $(date -u +%FT%TZ)"
