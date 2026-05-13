#!/usr/bin/env bash
# One-time setup on GTX Spark desktop (replaces runpod_setup.sh).
#
# Run ON the Spark box (after SSH-ing in via Tailscale):
#     bash scripts/spark_setup.sh
#
# Or remote-trigger from laptop:
#     ssh "$NANOGLD_SPARK_USER@$NANOGLD_SPARK_HOST" 'bash ~/nanogld/scripts/spark_setup.sh'
#
# Idempotent: safe to re-run.
# Hard-fails on missing CUDA. Logs to /var/log/nanogld_setup.log if writable,
# else ~/nanogld_setup.log.

set -Eeuo pipefail

LOG_FILE="${NANOGLD_SETUP_LOG:-$HOME/nanogld_setup.log}"
mkdir -p "$(dirname "$LOG_FILE")"
exec > >(tee -a "$LOG_FILE") 2>&1

NANOGLD_DIR="${NANOGLD_DIR:-$HOME/nanogld}"

if ! command -v uv >/dev/null 2>&1; then
    echo "[setup] installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"

if [[ ! -d "$NANOGLD_DIR" ]]; then
    echo "[setup] $NANOGLD_DIR missing — rsync from laptop first:" >&2
    echo "  rsync -avz <laptop>/plan-edit/ $NANOGLD_SPARK_USER@$NANOGLD_SPARK_HOST:$NANOGLD_DIR/" >&2
    exit 2
fi
cd "$NANOGLD_DIR"

echo "[setup] uv sync --frozen ..."
uv sync --frozen

echo "[setup] python version:"
uv run python --version

echo "[setup] CUDA + torch check:"
uv run python - <<'PY'
import sys
import torch
print(f"torch: {torch.__version__}")
print(f"cuda available: {torch.cuda.is_available()}")
if not torch.cuda.is_available():
    print("ERROR: no CUDA visible on Spark box", file=sys.stderr)
    sys.exit(1)
print(f"cuda device: {torch.cuda.get_device_name(0)}")
print(f"cuda compute capability: {torch.cuda.get_device_capability(0)}")
free, total = torch.cuda.mem_get_info()
print(f"vram: free={free / 1e9:.1f} GB / total={total / 1e9:.1f} GB")
PY

echo "[setup] disk free:"
df -h "$NANOGLD_DIR" | tail -1

# Hard-fail when free disk is below the threshold needed for 4-fold run
# (rough budget: 4 × {ssl_anchor.pt ~1GB + probe.pt ~1GB + llrd_final.pt ~3GB
# + calibration ~200MB + logs ~500MB} = ~24 GB plus headroom for crash dumps).
MIN_GB="${NANOGLD_MIN_FREE_GB:-50}"
FREE_GB=$(df -BG "$NANOGLD_DIR" | awk 'NR==2 {gsub("G",""); print $4}')
if [[ -z "$FREE_GB" || ! "$FREE_GB" =~ ^[0-9]+$ ]]; then
    echo "[setup] WARN: could not parse df output; skipping free-disk gate"
elif (( FREE_GB < MIN_GB )); then
    echo "[setup] free disk ${FREE_GB} GB < required ${MIN_GB} GB" >&2
    echo "[setup] clean up or set NANOGLD_MIN_FREE_GB to override" >&2
    exit 6
else
    echo "[setup] free-disk gate OK (${FREE_GB} GB >= ${MIN_GB} GB)"
fi

echo "[setup] nvidia-smi:"
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,driver_version,memory.total,temperature.gpu --format=csv,noheader
else
    echo "[setup] WARN: nvidia-smi not in PATH; relying on torch.cuda probe only"
fi

echo "[setup] data dir:"
mkdir -p "$NANOGLD_DIR/data/processed"
ls -la "$NANOGLD_DIR/data/processed" || true

if command -v gitleaks >/dev/null 2>&1 && [[ -f "$NANOGLD_DIR/.gitleaks.toml" ]]; then
    echo "[setup] gitleaks scan ..."
    (cd "$NANOGLD_DIR" && gitleaks detect --no-banner --config .gitleaks.toml) || \
        echo "[setup] WARN: gitleaks reported findings"
fi

echo "[setup] OK"
