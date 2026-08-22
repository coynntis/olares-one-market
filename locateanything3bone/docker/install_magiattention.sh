#!/usr/bin/env bash
# MagiAttention — NVIDIA model card (Hopper FFA / Blackwell FA4 via flash_attn_cute).
set -euxo pipefail

if [ "${INSTALL_MAGIATTENTION:-0}" != "1" ]; then
  echo "[magi] skip INSTALL_MAGIATTENTION!=1"
  exit 0
fi

# Olares One (Blackwell sm_120): default pip install prebuilds hundreds of Hopper sm_90
# flex_flash_attn kernels — very RAM-heavy and not used on 5090M. See MagiAttention install docs.
export MAGI_ATTENTION_PREBUILD_FFA="${MAGI_ATTENTION_PREBUILD_FFA:-0}"
# Single-GPU / no InfiniBand: skip magi_attn_comm (needs infiniband/mlx5dv.h + IBGDA).
export MAGI_ATTENTION_SKIP_MAGI_ATTN_COMM_BUILD="${MAGI_ATTENTION_SKIP_MAGI_ATTN_COMM_BUILD:-1}"
_magi_jobs="${MAGI_MAX_JOBS:-4}"
export MAX_JOBS="${_magi_jobs}"
export CMAKE_BUILD_PARALLEL_LEVEL="${_magi_jobs}"
export NINJAFLAGS="${NINJAFLAGS:--j${_magi_jobs}}"
echo "[magi] config: PREBUILD_FFA=${MAGI_ATTENTION_PREBUILD_FFA} SKIP_COMM=${MAGI_ATTENTION_SKIP_MAGI_ATTN_COMM_BUILD} MAX_JOBS=${MAX_JOBS}"
_upgrade_git() {
  # Kaniko / minimal layers may ship a stub git; install Debian git before submodules.
  if ! command -v apt-get >/dev/null 2>&1; then
    return 0
  fi
  export DEBIAN_FRONTEND=noninteractive
  apt-get update -qq
  apt-get install -y --no-install-recommends git ca-certificates
  apt-get install -y --only-upgrade git 2>/dev/null || true
  hash -r
}

_upgrade_git
export PATH="/usr/local/cuda/bin:/usr/bin:/usr/sbin:/usr/local/sbin:/usr/local/bin:/sbin:/bin:${PATH:-}"
free -h 2>/dev/null || true
for _need in git bash python ninja; do
  if ! command -v "$_need" >/dev/null 2>&1; then
    echo "[magi] FATAL: required command missing: $_need (PATH=$PATH)"
    exit 127
  fi
done
echo "[magi] git: $(command -v git) — $(git --version)"

_pip() {
  if [ "${PIP_BREAK_SYSTEM_PACKAGES:-0}" = "1" ]; then
    python -m pip install --break-system-packages "$@"
  else
    python -m pip install "$@"
  fi
}

_script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_magi_build_req="${MAGI_BUILD_REQ:-}"
if [ -z "${_magi_build_req}" ] && [ -n "${APP_SRC:-}" ]; then
  _magi_build_req="${APP_SRC}/magi_build_requirements.txt"
fi
if [ -z "${_magi_build_req}" ] && [ -f "${_script_dir}/magi_build_requirements.txt" ]; then
  _magi_build_req="${_script_dir}/magi_build_requirements.txt"
fi
if [ -z "${_magi_build_req}" ] && [ -f /tmp/magi_build_requirements.txt ]; then
  _magi_build_req=/tmp/magi_build_requirements.txt
fi

_magi_tag="${MAGI_ATTENTION_VERSION:-v1.1.1}"

rm -rf /tmp/MagiAttention
echo "[magi] stage: clone repo (${_magi_tag})"
git clone https://github.com/SandAI-org/MagiAttention.git /tmp/MagiAttention
cd /tmp/MagiAttention
echo "[magi] stage: checkout ${_magi_tag}"
git checkout "${_magi_tag}"
echo "[magi] stage: init submodules"
git submodule update --init --recursive
echo "[magi] stage: install upstream requirements"
_pip -r requirements.txt
# Blackwell (RTX 5090M): install FA4 backend; skip for Hopper-only builds (ARCHS=sm90).
if [ "${MAGI_INSTALL_FLASH_ATTN_CUTE:-1}" = "1" ]; then
  _fa4_archs="${MAGI_FLASH_ATTN_CUTE_ARCHS:-sm100}"
  _fa_dir="magi_attention/functional/flash-attention"
  _fa_install="${_fa_dir}/install.sh"
  if [ -f scripts/install_flash_attn_cute.sh ] && [ -f "${_fa_install}" ]; then
    echo "[magi] stage: install flash_attn_cute (${_fa4_archs}) for Blackwell FA4 backend"
    /bin/bash scripts/install_flash_attn_cute.sh "${_fa4_archs}"
  else
    echo "[magi] warn: flash-attention tree not in MagiAttention v1.0.5 (no ${_fa_install}); skipping FA4 cute pre-install"
    echo "[magi] warn: rely on MAGI_ATTENTION_FA4_BACKEND=1 at runtime; or set MAGI_INSTALL_FLASH_ATTN_CUTE=0"
  fi
fi
if [ -n "${_magi_build_req}" ] && [ -f "${_magi_build_req}" ]; then
  echo "[magi] build-time deps from ${_magi_build_req}"
  _pip -r "${_magi_build_req}"
else
  echo "[magi] warn: magi_build_requirements.txt not found; installing minimal build deps"
  _pip debugpy einops jinja2 filelock
fi
echo "[magi] stage: build/install MagiAttention (verbose)"
_pip -v --no-build-isolation .
echo "[magi] stage: import test"
python -c "import magi_attention; print('[magi] import ok', magi_attention.__file__)"
python - <<'PY'
import importlib.util
import os

os.environ["MAGI_ATTENTION_KERNEL_BACKEND"] = "fa4"
from magi_attention.common.enum import MagiAttentionKernelBackend
from magi_attention.env import general as magi_env

if not importlib.util.find_spec("flash_attn_cute"):
    raise SystemExit(
        "[magi] FATAL: flash_attn_cute missing — install scripts/install_flash_attn_cute.sh sm100"
    )
if magi_env.kernel_backend() != MagiAttentionKernelBackend.FA4:
    raise SystemExit("[magi] FATAL: kernel backend is not fa4 (check MAGI_ATTENTION_* env)")
print("[magi] flash_attn_cute ok, kernel_backend=fa4")
PY
if [ -f /app/requirements.txt ]; then
  echo "[magi] stage: restore app requirements (uvicorn/fastapi/gradio)"
  _pip -r /app/requirements.txt
fi
echo "[magi] runtime: set MAGI_ATTENTION_FA4_BACKEND=1 when using on Blackwell"
