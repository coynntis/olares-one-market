#!/usr/bin/env bash
# One-time pip venv + optional MagiAttention on Olares One (init container).
set -euxo pipefail

DEPS_DIR="${DEPS_DIR:-/workspace/deps}"
VENV="${VENV_PATH:-${DEPS_DIR}/venv}"
MARKER="${DEPS_MARKER:-${DEPS_DIR}/.bootstrap-v1.ok}"
APP_SRC="${APP_SRC:-/app-src}"

if [ -f "${MARKER}" ]; then
  echo "[deps] skip — marker ${MARKER} exists"
  exit 0
fi

mkdir -p "${DEPS_DIR}"
if [ ! -x "${VENV}/bin/python" ]; then
  python3 -m venv "${VENV}"
fi
# shellcheck disable=SC1091
source "${VENV}/bin/activate"

python -m pip install --upgrade pip setuptools wheel
python -m pip install "numpy>=1.26.0,<3"
python -m pip install -r "${APP_SRC}/requirements.txt"

if [ "${INSTALL_MAGIATTENTION:-1}" = "1" ]; then
  bash "${APP_SRC}/install_magiattention.sh"
  # MagiAttention pip can remove uvicorn/fastapi; restore app stack.
  python -m pip install -r "${APP_SRC}/requirements.txt"
fi

python -m uvicorn --version

touch "${MARKER}"
echo "[deps] bootstrap done → ${VENV}"
