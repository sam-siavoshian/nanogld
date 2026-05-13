#!/usr/bin/env bash
# One-time setup on Mac mini M4 (replaces spark_setup.sh for the MPS path).
#
# Run ON Mac mini (after ssh-ing in via Tailscale):
#     bash scripts/mac_mini_setup.sh
# Or remote-trigger from laptop:
#     ssh root1@100.83.86.5 'bash ~/Desktop/nanogld/scripts/mac_mini_setup.sh'
#
# Idempotent. Hard-fails on missing MPS. Logs to ~/nanogld_mac_mini_setup.log.

set -Eeuo pipefail

LOG_FILE="${NANOGLD_SETUP_LOG:-$HOME/nanogld_mac_mini_setup.log}"
mkdir -p "$(dirname "$LOG_FILE")"
exec > >(tee -a "$LOG_FILE") 2>&1

NANOGLD_DIR="${NANOGLD_DIR:-$HOME/Desktop/nanogld}"

echo "[setup] starting at $(date -u +%FT%TZ) on $(hostname)"

# Install uv if missing — Mac mini didn't have it before.
if ! command -v uv >/dev/null 2>&1; then
    echo "[setup] installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"
uv --version

if [[ ! -d "$NANOGLD_DIR" ]]; then
    echo "[setup] $NANOGLD_DIR missing; clone or rsync from laptop first" >&2
    exit 2
fi
cd "$NANOGLD_DIR"

echo "[setup] git pull origin main ..."
git pull --ff-only origin main || {
    echo "[setup] WARN: git pull failed; continuing with current HEAD"
}
echo "[setup] HEAD: $(git rev-parse HEAD)"

echo "[setup] uv sync --frozen ..."
uv sync --frozen

echo "[setup] python version:"
uv run python --version

echo "[setup] MPS + torch check (Mac mini = no CUDA; MPS is the device):"
uv run python - <<'PY'
import sys
import torch
print(f"torch: {torch.__version__}")
mps_ok = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
print(f"mps available: {mps_ok}")
print(f"mps built: {getattr(torch.backends.mps, 'is_built', lambda: False)()}")
if not mps_ok:
    print("ERROR: MPS not available on Mac mini", file=sys.stderr)
    sys.exit(1)
# Quick MPS allocation smoke
x = torch.randn(8, 384, device="mps")
y = (x @ x.t()).mean()
print(f"mps smoke OK: y={float(y):.4f}")
PY

echo "[setup] free disk:"
df -h "$NANOGLD_DIR" | tail -1
MIN_GB="${NANOGLD_MIN_FREE_GB:-30}"
FREE_GB=$(df -g "$NANOGLD_DIR" | awk 'NR==2 {print $4}')
if [[ "$FREE_GB" =~ ^[0-9]+$ ]] && (( FREE_GB < MIN_GB )); then
    echo "[setup] free disk ${FREE_GB} GB < required ${MIN_GB} GB (override via NANOGLD_MIN_FREE_GB)" >&2
    exit 6
fi
echo "[setup] free-disk gate OK (${FREE_GB} GB >= ${MIN_GB} GB)"

echo "[setup] memory pressure:"
vm_stat | head -5
# Mac mini ceiling: 16 GB unified. Warn if available < 6 GB (training needs ~4-6 GB headroom for batch_size=8).
PAGES_FREE=$(vm_stat | awk '/Pages free/ {gsub(/\./,""); print $3}')
PAGES_INACTIVE=$(vm_stat | awk '/Pages inactive/ {gsub(/\./,""); print $3}')
PAGE_SIZE=$(vm_stat | head -1 | awk '{print $8}')
if [[ -n "$PAGES_FREE" && -n "$PAGES_INACTIVE" && -n "$PAGE_SIZE" ]]; then
    AVAIL_BYTES=$(( (PAGES_FREE + PAGES_INACTIVE) * PAGE_SIZE ))
    AVAIL_GB=$(( AVAIL_BYTES / 1024 / 1024 / 1024 ))
    echo "[setup] available memory ~${AVAIL_GB} GB"
fi

echo "[setup] sleep prevention check:"
# Per CLAUDE.md memory: pmset SleepDisabled=1 already set on Mac mini.
pmset -g | grep -E "sleep|SleepDisabled" | head -5 || true
echo "[setup] tip: run training under 'caffeinate -dimsu' to keep the box awake."

echo "[setup] data dir:"
mkdir -p "$NANOGLD_DIR/data/processed"
ls -lh "$NANOGLD_DIR/data/processed" | head -10

echo "[setup] OK at $(date -u +%FT%TZ)"
