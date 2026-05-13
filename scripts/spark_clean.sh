#!/usr/bin/env bash
# Sentinel + heartbeat cleanup for retries.
#
# Run on SPARK (or remote via ssh). Clears the sentinel files that the
# orchestrator uses to skip stages so a retry actually re-runs the
# failed stage. Refuses to delete checkpoints/* by default — those are
# expensive to recompute and the orchestrator already skips completed
# stages safely.
#
# Modes:
#   spark_clean.sh fold N                clear .heartbeat + stage.oom for fold N
#   spark_clean.sh fold N --all          also clear all stage.done (forces full retrain)
#   spark_clean.sh fold N --stage ssl    clear stage.done for one stage only
#   spark_clean.sh all                   .heartbeat + stage.oom for every fold
#   spark_clean.sh dry-run fold N        print what would be deleted (no action)
#
# Env vars:
#   NANOGLD_DIR     repo root on box (default ~/nanogld)
#   NANOGLD_N_FOLDS default 4

set -Eeuo pipefail

NANOGLD_DIR="${NANOGLD_DIR:-$HOME/nanogld}"
N_FOLDS="${NANOGLD_N_FOLDS:-4}"
CKPT_ROOT="${NANOGLD_DIR}/checkpoints/v1"

dry=0
target="${1:?usage: $0 {fold N|all|dry-run fold N} [--all|--stage <ssl|probe|llrd>]}"
if [[ "${target}" == "dry-run" ]]; then
    dry=1
    shift
    target="${1:?usage after dry-run: $0 dry-run fold N}"
fi

_remove() {
    local path="$1"
    if [[ -e "${path}" ]]; then
        if (( dry )); then
            echo "[dry-run] would rm ${path}"
        else
            rm -f "${path}"
            echo "rm ${path}"
        fi
    fi
}

clean_fold() {
    local fold="$1"
    local mode="${2:-default}"      # default | all | <stage>
    local fold_dir="${CKPT_ROOT}/fold_${fold}"
    if [[ ! -d "${fold_dir}" ]]; then
        echo "[clean] fold ${fold} dir missing: ${fold_dir}" >&2
        return 0
    fi
    _remove "${fold_dir}/.heartbeat"
    for stage in ssl probe llrd; do
        _remove "${fold_dir}/${stage}/stage.oom"
    done
    if [[ "${mode}" == "all" ]]; then
        for stage in ssl probe llrd; do
            _remove "${fold_dir}/${stage}/stage.done"
        done
        _remove "${fold_dir}/calibration_${fold}/MANIFEST.json"
    elif [[ "${mode}" =~ ^(ssl|probe|llrd)$ ]]; then
        _remove "${fold_dir}/${mode}/stage.done"
    fi
}

if [[ "${target}" == "all" ]]; then
    for (( f=0; f<N_FOLDS; f++ )); do
        clean_fold "${f}"
    done
    exit 0
fi

if [[ "${target}" == "fold" ]]; then
    fold="${2:?usage: $0 fold <N>}"
    shift 2
    mode="default"
    while (( $# > 0 )); do
        case "$1" in
            --all) mode="all"; shift ;;
            --stage) mode="${2:?--stage needs value}"; shift 2 ;;
            *) echo "unknown arg: $1" >&2; exit 2 ;;
        esac
    done
    clean_fold "${fold}" "${mode}"
    exit 0
fi

echo "[clean] unknown target: ${target}" >&2
exit 2
