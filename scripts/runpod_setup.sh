#!/usr/bin/env bash
# RunPod H100 one-time setup.
#
# Pre-req on the RunPod side:
#   - SSH access to the pod with /workspace mounted.
#   - HF_TOKEN env var set (or `huggingface-cli login` already run).
#
# Idempotent: safe to re-run.

set -euo pipefail

cd /workspace

if ! command -v uv >/dev/null 2>&1; then
    echo "[setup] installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

if [[ ! -d nanogld ]]; then
    echo "[setup] cloning nanogld repo..."
    git clone "${NANOGLD_REPO:-https://github.com/sam-siavoshian/nanogld.git}" nanogld
fi
cd nanogld

echo "[setup] uv sync --frozen ..."
uv sync --frozen

echo "[setup] python version:"
uv run python --version

echo "[setup] torch + CUDA check:"
uv run python -c "
import torch
print(f'torch: {torch.__version__}')
print(f'cuda available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'cuda device: {torch.cuda.get_device_name(0)}')
    print(f'cuda compute capability: {torch.cuda.get_device_capability(0)}')
"

echo "[setup] HF auth status:"
if [[ -n "${HF_TOKEN:-}" ]]; then
    echo "[setup] HF_TOKEN env present"
    uv run huggingface-cli whoami || true
else
    echo "[setup] HF_TOKEN missing — run 'huggingface-cli login' or set HF_TOKEN env"
fi

echo "[setup] OK"
