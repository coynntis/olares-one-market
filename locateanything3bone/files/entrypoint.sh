#!/usr/bin/env bash
# PyTorch images: build uses /opt/conda/bin/python; Olares may run /usr/bin/python first.
set -euo pipefail
# Blackwell (RTX 5090M): v1.1.1 defaults to ffa (sm90); force fa4 + flash_attn_cute.
export MAGI_ATTENTION_PREBUILD_FFA="${MAGI_ATTENTION_PREBUILD_FFA:-0}"
if [[ -z "${MAGI_ATTENTION_KERNEL_BACKEND:-}" && -z "${MAGI_ATTENTION_FA4_BACKEND:-}" ]]; then
  export MAGI_ATTENTION_KERNEL_BACKEND=fa4
fi
if [[ -x /opt/conda/bin/python ]]; then
  export PATH="/opt/conda/bin:${PATH}"
fi
PY="$(command -v python)"
mkdir -p "${GRADIO_TEMP_DIR:-/output/gradio}"
if ! "${PY}" -c "import uvicorn" 2>/dev/null; then
  echo "[entrypoint] uvicorn missing on ${PY}; installing /app/requirements.txt"
  "${PY}" -m pip install --break-system-packages -r /app/requirements.txt
fi
exec "${PY}" -m uvicorn app:app --host 0.0.0.0 --port "${SERVER_PORT:-7860}" --app-dir /app
