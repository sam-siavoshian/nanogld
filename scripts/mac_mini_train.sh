#!/usr/bin/env bash
# Run V1 training for ONE fold on Mac mini M4 (MPS).
# Mac-mini counterpart of spark_train.sh.
#
# Key differences vs spark_train.sh:
#   - --device mps (no CUDA on Apple Silicon)
#   - Caffeinate wrap to prevent sleep on long runs
#   - No tmux; caller uses nohup or runs in foreground
#   - Batch size override via NANOGLD_BATCH_SIZE env (Mac mini 16 GB
#     unified means default config batch_size of 16+ will OOM; we
#     pass batch_size_override config to the trainer)
#
# Usage:
#     bash scripts/mac_mini_train.sh <fold> [max_steps]
# e.g.
#     bash scripts/mac_mini_train.sh 0 10        # smoke test
#     bash scripts/mac_mini_train.sh 0           # real fold 0
#
# Hard-fails on:
#   - fold out of [0, NANOGLD_N_FOLDS)
#   - missing NANOGLD_DIR
#   - missing unified.pt or per-fold sidecar
#   - missing MPS at runtime

set -Eeuo pipefail

FOLD="${1:?usage: $0 <fold_idx 0..N-1> [max_steps]}"
MAX_STEPS="${2:-}"
N_FOLDS="${NANOGLD_N_FOLDS:-4}"
CONFIG="${NANOGLD_CONFIG:-configs/v1_main.yaml}"
NANOGLD_DIR="${NANOGLD_DIR:-$HOME/Desktop/nanogld}"
BATCH_SIZE="${NANOGLD_BATCH_SIZE:-8}"
DEVICE="${NANOGLD_DEVICE:-mps}"

if [[ ! "${FOLD}" =~ ^[0-9]+$ ]] || (( FOLD < 0 || FOLD >= N_FOLDS )); then
    echo "[train] fold ${FOLD} out of range [0,${N_FOLDS})" >&2
    exit 2
fi
if [[ ! -d "${NANOGLD_DIR}" ]]; then
    echo "[train] ${NANOGLD_DIR} missing; run mac_mini_setup.sh first" >&2
    exit 3
fi
cd "${NANOGLD_DIR}"
export PATH="$HOME/.local/bin:$PATH"

# PYTHONHASHSEED must be set BEFORE python starts.
export PYTHONHASHSEED="${NANOGLD_SEED:-42}"

UNIFIED="data/processed/training_v1_unified.pt"
SIDECAR="data/processed/training_v1_sidecar_fold_${FOLD}.pt"
if [[ ! -f "${UNIFIED}" ]]; then
    echo "[train] missing ${UNIFIED}" >&2
    exit 4
fi
if [[ ! -f "${SIDECAR}" ]]; then
    echo "[train] missing ${SIDECAR}; run 'uv run python scripts/build_v1_sidecar.py --per-fold' first" >&2
    exit 5
fi

# Per-fold output dir + timestamped log.
OUT_DIR="checkpoints/v1/fold_${FOLD}"
mkdir -p "${OUT_DIR}" logs
TS="$(date -u +%Y%m%dT%H%M%SZ)"
LOG_FILE="logs/train_fold_${FOLD}_${TS}.log"
echo "[train] fold=${FOLD} device=${DEVICE} batch_size=${BATCH_SIZE} config=${CONFIG}"
echo "[train] log=${LOG_FILE}"

# Verify SHA256 before training (no point burning GPU if data tampered).
if [[ -f data/processed/MANIFEST.json ]]; then
    uv run python - <<PY
from pathlib import Path
from nanogld.data.integrity import verify_artifacts
verify_artifacts(Path("data/processed"), require=["training_v1_unified.pt", "training_v1_sidecar_fold_${FOLD}.pt"])
print("verify OK")
PY
else
    echo "[train] WARN: no MANIFEST.json; skipping sha256 verify"
fi

# Build a small override config so we don't mutate v1_main.yaml.
# Overrides:
#   - batch_size to NANOGLD_BATCH_SIZE (16 GB RAM ceiling on Mac mini)
#   - num_workers=0 (worker fork copies OOM fast on Apple Silicon)
#   - numeric_dim auto-detected from unified.pt (different builds of
#     the unified dataset have shipped with 651 vs 681 features; read
#     the actual shape so RevIN's affine_weight matches the input).
OVERRIDE_CFG="${OUT_DIR}/v1_main_macmini.yaml"
# Mac mini 16 GB unified memory + channel-independent encoder (B*F=651
# sequences per batch) + FSAM 2-pass param snapshots + EMA + Mixout
# anchor exceeds the MPS recommendedMaxWorkingSetSize at d_model=384
# even with batch_size=4. Defaults below shrink the architecture to
# fit the box. Override via NANOGLD_D_MODEL / NANOGLD_T_BARS env vars.
D_MODEL="${NANOGLD_D_MODEL:-192}"
T_BARS="${NANOGLD_T_BARS:-32}"
NUM_HEADS="${NANOGLD_NUM_HEADS:-4}"  # must divide d_model
# Epoch budget for Mac mini. Full V1-SPEC is 15+5+10 epochs ≈ 5 days/fold
# on MPS at d_model=192 t_bars=32. Default 3+2+3 ≈ 6-10 h/fold so all 4
# folds fit in a day. Bump to spec values via env if wall-clock allows.
SSL_EPOCHS="${NANOGLD_SSL_EPOCHS:-3}"
PROBE_EPOCHS="${NANOGLD_PROBE_EPOCHS:-2}"
LLRD_EPOCHS="${NANOGLD_LLRD_EPOCHS:-3}"

uv run python - <<PY
import torch, yaml
from pathlib import Path
cfg = yaml.safe_load(Path("${CONFIG}").read_text())
unified = torch.load("${UNIFIED}", map_location="cpu", weights_only=False)
numeric_dim = int(unified["features"].shape[1])
cfg.setdefault("model", {})["numeric_dim"] = numeric_dim
cfg["model"]["d_model"] = ${D_MODEL}
cfg["model"]["num_heads"] = ${NUM_HEADS}
cfg["model"]["t_bars"] = ${T_BARS}
cfg["model"]["patch_len"] = min(int(cfg["model"].get("patch_len", 4)), ${T_BARS})
cfg["model"]["patch_stride"] = min(int(cfg["model"].get("patch_stride", 4)), ${T_BARS})
cfg.setdefault("dataloader", {})["batch_size"] = ${BATCH_SIZE}
cfg["dataloader"]["lookback_T"] = ${T_BARS}
cfg["dataloader"]["num_workers"] = 0
cfg.setdefault("ssl", {})["epochs"] = ${SSL_EPOCHS}
cfg.setdefault("probe", {})["epochs"] = ${PROBE_EPOCHS}
cfg.setdefault("llrd", {})["epochs"] = ${LLRD_EPOCHS}
Path("${OVERRIDE_CFG}").write_text(yaml.safe_dump(cfg))
print(f"override: numeric_dim={numeric_dim} d_model=${D_MODEL} t_bars=${T_BARS} batch=${BATCH_SIZE} -> ${OVERRIDE_CFG}")
PY

# Smoke-mode: bail after N steps per stage via NANOGLD_MAX_STEPS env var.
# Each stage's batch loop checks and breaks.
if [[ -n "${MAX_STEPS}" ]]; then
    export NANOGLD_MAX_STEPS="${MAX_STEPS}"
    echo "[train] SMOKE mode: NANOGLD_MAX_STEPS=${MAX_STEPS} (each stage breaks after ${MAX_STEPS} steps)"
fi

# Caffeinate to prevent sleep on long runs (Mac mini lid-closed default).
echo "[train] starting Python at $(date -u +%FT%TZ)"
caffeinate -dimsu uv run python -m nanogld.training run \
    --config "${OVERRIDE_CFG}" \
    --fold "${FOLD}" \
    --output-dir "${OUT_DIR}" \
    --device "${DEVICE}" 2>&1 | tee -a "${LOG_FILE}"

# Write per-fold MANIFEST.json for backtest verify-on-load.
uv run python - <<PY
from pathlib import Path
from nanogld.data.integrity import write_manifest
write_manifest(Path("${OUT_DIR}/fold_${FOLD}/llrd"))
print("manifest OK")
PY

# Stage 4: calibration. Skip for smoke (max_steps).
if [[ -z "${MAX_STEPS}" ]]; then
    echo "[train] Stage 4 calibration fold ${FOLD} ..."
    caffeinate -dimsu uv run python -m nanogld.calibration run \
        --config "${OVERRIDE_CFG}" \
        --fold "${FOLD}" \
        --checkpoint "${OUT_DIR}/fold_${FOLD}/llrd/llrd_final.pt" \
        --output-dir "${OUT_DIR}/fold_${FOLD}" \
        --device "${DEVICE}" 2>&1 | tee -a "${LOG_FILE}"
fi

echo "[train] OK fold ${FOLD} at $(date -u +%FT%TZ)"
