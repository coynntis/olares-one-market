#!/usr/bin/env bash
set -euo pipefail

export PIP_BREAK_SYSTEM_PACKAGES=1
export PYTHONPATH="/opt/gsplat:/app:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="/usr/local/lib:${LD_LIBRARY_PATH:-}"

if command -v colmap >/dev/null 2>&1; then
  colmap -h >/dev/null || {
    echo "ERROR: colmap failed — is NVIDIA driver mounted?" >&2
    exit 1
  }
fi

python - <<'PY'
import fastapi
import gsplat
import torch

print("runtime ok", torch.__version__, "cuda_available", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("CUDA not available — SplatLab requires GPU")
PY

DATA_ROOT="${SPLATLAB_DATA_ROOT:-/data/splatlab}"
mkdir -p "${DATA_ROOT}/datasets" "${DATA_ROOT}/jobs" "${DATA_ROOT}/exports" "${DATA_ROOT}/cache"

if [[ -f /app/requirements.txt ]]; then
  python -m pip install --break-system-packages --no-cache-dir -q -r /app/requirements.txt || true
fi

exec python -m uvicorn main:app --host 0.0.0.0 --port 7860 --proxy-headers
