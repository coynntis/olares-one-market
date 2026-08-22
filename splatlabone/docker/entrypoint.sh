#!/usr/bin/env bash
set -euo pipefail

# Prefer directory containing this script (ConfigMap overlay uses /tmp/splatlab-app)
APP_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$APP_ROOT"

export PIP_BREAK_SYSTEM_PACKAGES=1
export PATH="/opt/extra/bin:/opt/colmap/bin:/usr/local/bin:${PATH}"
# Bundled COLMAP/MKL + ffmpeg libs first, then CUDA (image + host NVIDIA inject).
export LD_LIBRARY_PATH="/opt/extra/lib:/opt/colmap/lib:/opt/runtime-libs:/usr/local/cuda/lib64:/usr/local/nvidia/lib64:/usr/local/nvidia/lib:/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"

# Installed wheels first. Do NOT put /opt/gsplat ahead of site-packages — source tree
# shadows the CUDA-built wheel and breaks `import gsplat`.
PY_PATH_PREPEND=""
for d in \
  /opt/py-site \
  /opt/conda/lib/python3.12/site-packages \
  /opt/conda/lib/python3.11/site-packages \
  /usr/local/lib/python3.12/dist-packages \
  /usr/local/lib/python3.11/dist-packages \
  /opt/vggt-omega \
  /opt/da3/src \
  /opt/lingbot-map \
  /opt/instantsplat \
  /opt/hloc \
  /opt/lightglue \
  /opt/fastmap \
  /opt/gluemap \
  /opt/dense-sfm \
  "${APP_ROOT}"
do
  if [ -d "$d" ]; then
    if [ -n "${PY_PATH_PREPEND}" ]; then
      PY_PATH_PREPEND="${PY_PATH_PREPEND}:${d}"
    else
      PY_PATH_PREPEND="${d}"
    fi
  fi
done
export PYTHONPATH="${PY_PATH_PREPEND}${PYTHONPATH:+:${PYTHONPATH}}"

if [ -x /opt/colmap/bin/colmap ]; then
  if colmap -h >/dev/null 2>&1; then
    echo "colmap ok: /opt/colmap/bin/colmap"
  else
    echo "WARN: colmap present but failed to start (shared libs / GPU). Feed-forward presets still work." >&2
    ldd /opt/colmap/bin/colmap 2>&1 | grep 'not found' || true
  fi
else
  echo "WARN: /opt/colmap/bin/colmap missing — rebuild image ≥1.1.5 (COLMAP bundle)." >&2
fi
if command -v glomap >/dev/null 2>&1; then
  echo "glomap ok: $(command -v glomap)"
else
  echo "WARN: glomap not on PATH" >&2
fi
if [ -d /opt/dense-sfm/dense_sfm ]; then
  echo "dense-sfm ok: /opt/dense-sfm"
else
  echo "WARN: /opt/dense-sfm missing" >&2
fi
if command -v ffmpeg >/dev/null 2>&1; then
  echo "ffmpeg ok: $(command -v ffmpeg)"
else
  echo "WARN: ffmpeg not on PATH" >&2
fi
if command -v da3-cli >/dev/null 2>&1; then
  echo "da3-cli ok: $(command -v da3-cli)"
elif [ -x /opt/depth-anything-cpp/build/examples/cli/da3-cli ]; then
  echo "da3-cli ok: /opt/depth-anything-cpp/build/examples/cli/da3-cli"
else
  echo "WARN: da3-cli missing" >&2
fi

python - <<'PY'
import sys
print("python", sys.executable)
print("sys.path[0:5]", sys.path[:5])
import fastapi
import torch
try:
    import gsplat
except ModuleNotFoundError as e:
    print("FATAL: gsplat not importable.", e, file=sys.stderr)
    print("Looked in:", file=sys.stderr)
    for p in sys.path:
        print(" ", p, file=sys.stderr)
    raise SystemExit(
        "gsplat missing — image needs gsplat in dist-packages (rebuild splatlabone ≥1.1.14). "
        "Chart overlay cannot install CUDA wheels."
    )
print("runtime ok", torch.__version__, "cuda_available", torch.cuda.is_available(), "gsplat", gsplat.__file__)
if not torch.cuda.is_available():
    raise SystemExit("CUDA not available — SplatLab requires GPU")
PY

DATA_ROOT="${SPLATLAB_DATA_ROOT:-/data/splatlab}"
mkdir -p "${DATA_ROOT}/datasets" "${DATA_ROOT}/jobs" "${DATA_ROOT}/exports" "${DATA_ROOT}/cache"

if [[ -f "${APP_ROOT}/requirements.txt" ]]; then
  python -m pip install --break-system-packages --no-cache-dir -q -r "${APP_ROOT}/requirements.txt" || true
fi

exec python -m uvicorn main:app --host 0.0.0.0 --port 7860 --proxy-headers
