# syntax=docker/dockerfile:1
#
# SplatLab One — COLMAP 4.x, PyTorch cu128, gsplat, geometry backends, Viser viewer
#
# Kaniko has NO NVIDIA driver (libcuda.so.1) and NO GPU.
# - CUDA *compile* (gsplat, FastMap): use libcuda stubs + TORCH_CUDA_ARCH_LIST — no device needed.
# - CUDA *import/run* of extensions: may need real GPU; verify .so at build, run on Olares.
# - colmap-src: COPY-only (no RUN) — colmap image /bin/sh needs libcuda under Kaniko.
# - python-builder: COPY context stubs/libcuda.so.1 FIRST (not COPY --from devel), then RUN.
# - Final stage: COPY only — zero RUN.
# - Shell RUNs use `set -eu` (NOT -x): Kaniko always streams build stdout; -x floods UI.

# Official COLMAP image — source of binary + MKL/Qt/etc. NEVER RUN here under Kaniko.
FROM colmap/colmap:latest AS colmap-src

FROM pytorch/pytorch:2.11.0-cuda12.8-cudnn9-devel AS python-builder

ARG CACHEBUST=24

# Kaniko: ANY RUN invokes /bin/sh → needs libcuda.so.1 before first RUN.
# Do NOT COPY --from=…cuda-devel (Kaniko re-unpacks multi-GB image, 30–60+ min).
# Tiny stub in build context unblocks /bin/sh; first RUN swaps in real CUDA toolkit stub.
COPY stubs/libcuda.so.1 /usr/local/lib/libcuda.so.1
COPY stubs/libcuda.so.1 /usr/local/cuda/lib64/stubs/libcuda.so.1

# Olares One = RTX 5090M Blackwell sm_120. Extra arches explode nvcc RAM.
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_BREAK_SYSTEM_PACKAGES=1 \
    CUDA_HOME=/usr/local/cuda \
    TORCH_CUDA_ARCH_LIST="12.0+PTX" \
    MAX_JOBS=2 \
    CMAKE_BUILD_PARALLEL_LEVEL=2 \
    NINJAFLAGS="-j2" \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64/stubs:/usr/local/cuda/lib/stubs:/usr/local/lib

WORKDIR /build

# git + GlueMap C++ deps + ffmpeg; replace mini-stub with real CUDA stubs for nvcc link
RUN set -eu; \
    rm -f /usr/lib/python3.12/EXTERNALLY-MANAGED /usr/lib/python*/EXTERNALLY-MANAGED 2>/dev/null || true; \
    stub=""; \
    for d in /usr/local/cuda/lib64/stubs /usr/local/cuda/lib/stubs; do \
      if [ -f "${d}/libcuda.so" ]; then stub="${d}/libcuda.so"; break; fi; \
    done; \
    if [ -n "${stub}" ]; then \
      ln -sf "${stub}" "$(dirname "${stub}")/libcuda.so.1"; \
      ln -sf "${stub}" /usr/local/lib/libcuda.so.1; \
      echo "swapped in real CUDA stub: ${stub}"; \
    else \
      echo "WARN: no toolkit libcuda.so stub; keeping context mini-stub"; \
    fi; \
    test -e /usr/local/lib/libcuda.so.1; \
    apt-get update; \
    apt-get install -y --no-install-recommends \
      git ca-certificates \
      cmake build-essential \
      libeigen3-dev libceres-dev libmetis-dev libboost-dev \
      libgomp1 pybind11-dev \
      ffmpeg; \
    rm -rf /var/lib/apt/lists/*; \
    git --version; \
    cmake --version; \
    ffmpeg -version | head -1

# --- COLMAP bundle (shell works here because stubs exist) ---
COPY --from=colmap-src /usr/local/bin/ /tmp/colmap-bins/
COPY --from=colmap-src /usr/local/lib/ /opt/colmap-bundle/lib/
COPY --from=colmap-src /usr/local/share/colmap /opt/colmap-bundle/share/colmap
COPY --from=colmap-src /usr/lib/x86_64-linux-gnu/ /tmp/colmap-usr-gnu/
COPY --from=colmap-src /lib/x86_64-linux-gnu/ /tmp/colmap-lib-gnu/
COPY glomap-shim.sh /tmp/glomap-shim.sh

RUN set -eu; \
    mkdir -p /opt/colmap-bundle/bin /opt/colmap-bundle/lib; \
    cp -a /tmp/colmap-bins/colmap /opt/colmap-bundle/bin/colmap; \
    if [ -x /tmp/colmap-bins/glomap ]; then \
      cp -a /tmp/colmap-bins/glomap /opt/colmap-bundle/bin/glomap; \
      echo "glomap: native binary from colmap image"; \
    else \
      cp -a /tmp/glomap-shim.sh /opt/colmap-bundle/bin/glomap; \
      chmod +x /opt/colmap-bundle/bin/glomap; \
      echo "glomap: installed COLMAP 4 global_mapper shim"; \
    fi; \
    rm -rf /tmp/colmap-bins /tmp/glomap-shim.sh; \
    find /tmp/colmap-usr-gnu /tmp/colmap-lib-gnu -type f \( \
        -name 'libmkl*.so*' -o -name 'libomp*.so*' -o -name 'libiomp5.so*' \
        -o -name 'libGLEW*.so*' -o -name 'libboost_program_options*.so*' \
        -o -name 'libglog*.so*' -o -name 'libcurl*.so*' -o -name 'libnghttp2*.so*' \
        -o -name 'librtmp*.so*' -o -name 'libssh*.so*' -o -name 'libpsl*.so*' \
        -o -name 'libOpenImageIO*.so*' -o -name 'libQt6*.so*' \
        -o -name 'libOpenGL*.so*' -o -name 'libGLX*.so*' -o -name 'libGLdispatch*.so*' \
        -o -name 'libGL.so*' -o -name 'liblapack*.so*' -o -name 'libblas*.so*' \
        -o -name 'libspqr*.so*' -o -name 'libcholmod*.so*' -o -name 'libamd*.so*' \
        -o -name 'libcamd*.so*' -o -name 'libccolamd*.so*' -o -name 'libcolamd*.so*' \
        -o -name 'libsuitesparseconfig*.so*' -o -name 'libceres*.so*' \
        -o -name 'libmetis*.so*' -o -name 'libgomp*.so*' \
        -o -name 'libfreeimage*.so*' -o -name 'libraw*.so*' -o -name 'libOpenEXR*.so*' \
        -o -name 'libImath*.so*' -o -name 'libIex*.so*' -o -name 'libIlmThread*.so*' \
        -o -name 'libjpeg*.so*' -o -name 'libpng*.so*' -o -name 'libtiff*.so*' \
        -o -name 'libwebp*.so*' -o -name 'libxcb*.so*' -o -name 'libX11*.so*' \
        -o -name 'libXau*.so*' -o -name 'libXdmcp*.so*' -o -name 'libicu*.so*' \
        -o -name 'libdouble-conversion*.so*' -o -name 'libpcre2*.so*' \
        -o -name 'libzstd*.so*' -o -name 'liblzma*.so*' -o -name 'libbz2*.so*' \
        -o -name 'libbrotli*.so*' -o -name 'libssl*.so*' -o -name 'libcrypto*.so*' \
        -o -name 'libgflags*.so*' -o -name 'libtbb*.so*' -o -name 'libhwloc*.so*' \
      \) -exec cp -a {} /opt/colmap-bundle/lib/ \; ; \
    export LD_LIBRARY_PATH="/opt/colmap-bundle/lib:/tmp/colmap-usr-gnu:/tmp/colmap-lib-gnu:${LD_LIBRARY_PATH:-}"; \
    for _ in 1 2 3 4 5; do \
      find /opt/colmap-bundle/bin /opt/colmap-bundle/lib -type f \( -name 'colmap' -o -name 'glomap' -o -name '*.so*' \) \
        | while read -r f; do \
            ldd "$f" 2>/dev/null | awk '/=> \// {print $3} /^\// && !/=>/ {print $1}' || true; \
          done \
        | sort -u \
        | while read -r so; do \
            [ -f "$so" ] || continue; \
            case "$so" in \
              /tmp/colmap-*|/opt/colmap-bundle/lib/*) cp -a "$so" /opt/colmap-bundle/lib/ || true ;; \
              /lib/*|/usr/lib/*|/usr/local/lib/*) cp -a "$so" /opt/colmap-bundle/lib/ || true ;; \
            esac; \
          done; \
    done; \
    rm -rf /tmp/colmap-usr-gnu /tmp/colmap-lib-gnu; \
    test -x /opt/colmap-bundle/bin/colmap; \
    ls /opt/colmap-bundle/lib/libmkl_intel_lp64.so* ; \
    echo "colmap bundle libs: $(ls /opt/colmap-bundle/lib | wc -l)"

COPY requirements.txt requirements-geometry.txt fetch_backends.py /build/

# Keep setuptools in torch's range (<82); pin numpy so backends cannot thrash 1.x↔2.x
RUN set -eu; \
    echo "CACHEBUST=${CACHEBUST}"; \
    python -m pip install --break-system-packages "pip>=24" "setuptools>=70,<82" wheel; \
    python -m pip install --break-system-packages ninja "numpy>=2.0,<2.2" \
      "scikit-build-core>=0.10" "pybind11>=3.0" cmake

# Peak RAM: gsplat CUDA compile — alone in this layer
RUN set -eu; \
    python -m pip install --break-system-packages --no-build-isolation gsplat; \
    python -c "import gsplat; print('gsplat', gsplat.__file__)"

RUN python -m pip install --break-system-packages -r /build/requirements.txt

RUN python -m pip install --break-system-packages -r /build/requirements-geometry.txt

# Network fetch (tarballs) — retries + git fallback; GlueMap excluded (submodules)
RUN python /build/fetch_backends.py

# GlueMap + depth-anything.cpp: recursive clone (tarballs omit submodules)
RUN set -eu; \
    rm -rf /opt/gluemap /opt/depth-anything-cpp /opt/dense-sfm-refine; \
    git clone --depth 1 https://github.com/colmap/gluemap.git /opt/gluemap; \
    cd /opt/gluemap; \
    git submodule update --init --recursive --depth 1; \
    test -d /opt/gluemap/thirdparty; \
    git clone --depth 1 https://github.com/mudler/depth-anything.cpp.git /opt/depth-anything-cpp; \
    cd /opt/depth-anything-cpp; \
    git submodule update --init --recursive --depth 1; \
    test -d /opt/depth-anything-cpp/third_party/ggml; \
    test -f /opt/depth-anything-cpp/third_party/ggml/CMakeLists.txt; \
    echo "depth-anything.cpp ggml submodule ok"; \
    git clone --depth 1 https://github.com/IceTea-CV/DenseSfM-Refine.git /opt/dense-sfm-refine; \
    cd /opt/dense-sfm-refine; \
    git submodule update --init --recursive --depth 1 || true; \
    echo "dense-sfm-refine ok"

# SplatLab Dense-SfM adapter (hloc LoFTR → COLMAP DB) at /opt/dense-sfm
COPY dense-sfm-overlay/ /opt/dense-sfm/
RUN test -f /opt/dense-sfm/dense_sfm/__init__.py && test -f /opt/dense-sfm/run_matching.py

# Editable installs: --no-deps to avoid numpy thrash / huge optional deps
RUN set -eu; \
    python -m pip install --break-system-packages --no-deps -e /opt/vggt-omega || true; \
    python -m pip install --break-system-packages --no-deps -e /opt/da3 || true; \
    python -m pip install --break-system-packages --no-deps -e /opt/lingbot-map || true; \
    python -m pip install --break-system-packages --no-deps -e /opt/lightglue || true; \
    python -m pip install --break-system-packages --no-deps -e /opt/hloc || true

# Peak RAM: FastMap CUDA ext — alone, MAX_JOBS=1.
# Kaniko has NO GPU; nvcc links against libcuda stubs (same as gsplat). Compile OK without device.
# Do NOT require `import fastmap.cuda` here — stub may fail cudaGetDeviceCount; check .so instead.
RUN set -eu; \
    python -m pip install --break-system-packages --no-deps \
      trimesh "pyglet<2" pyyaml dacite loguru prettytable psutil; \
    cd /opt/fastmap; \
    MAX_JOBS=1 CMAKE_BUILD_PARALLEL_LEVEL=1 \
      python setup.py build_ext --inplace; \
    python -m pip install --break-system-packages --no-deps --no-build-isolation -e /opt/fastmap; \
    SO="$(find /opt/fastmap -name 'cuda*.so' -o -name '_C*.so' | head -1)"; \
    test -n "$SO" && test -f "$SO"; \
    echo "fastmap CUDA ext ok: $SO"; \
    python -c "import fastmap; print('fastmap py ok', fastmap.__file__)"

# da3-cli (CPU ggml) — must succeed (fast_dacpp / fast_hybrid)
RUN set -eu; \
    cd /opt/depth-anything-cpp; \
    test -d third_party/ggml; \
    cmake -B build -DDA_BUILD_CLI=ON -DDA_GGML_CUDA=OFF -DDA_BUILD_TESTS=OFF; \
    cmake --build build -j1; \
    test -x build/examples/cli/da3-cli; \
    mkdir -p /opt/bin-export /opt/extra/lib; \
    cp -a build/examples/cli/da3-cli /opt/bin-export/da3-cli; \
    ldd /opt/bin-export/da3-cli | awk '/=> \// {print $3} /^\// && !/=>/ {print $1}' | sort -u \
      | while read -r so; do \
          [ -f "$so" ] || continue; \
          case "$so" in /lib/*|/usr/lib/*|/opt/depth-anything-cpp/*) cp -a "$so" /opt/extra/lib/ || true ;; esac; \
        done; \
    /opt/bin-export/da3-cli --help >/dev/null 2>&1 || /opt/bin-export/da3-cli 2>&1 | head -5 || true; \
    echo "da3-cli ok"

# ffmpeg + ffprobe + libs → /opt/extra (final stage has no apt); hard-require
RUN set -eu; \
    mkdir -p /opt/extra/bin /opt/extra/lib; \
    cp -a "$(command -v ffmpeg)" /opt/extra/bin/ffmpeg; \
    if command -v ffprobe >/dev/null 2>&1; then cp -a "$(command -v ffprobe)" /opt/extra/bin/ffprobe; fi; \
    for _ in 1 2 3; do \
      find /opt/extra/bin -type f \
        | while read -r f; do \
            ldd "$f" 2>/dev/null | awk '/=> \// {print $3} /^\// && !/=>/ {print $1}' || true; \
          done \
        | sort -u \
        | while read -r so; do \
            [ -f "$so" ] || continue; \
            case "$so" in /lib/*|/usr/lib/*) cp -a "$so" /opt/extra/lib/ || true ;; esac; \
          done; \
    done; \
    test -x /opt/extra/bin/ffmpeg; \
    /opt/extra/bin/ffmpeg -version | head -1; \
    echo "ffmpeg bundle libs: $(ls /opt/extra/lib | wc -l)"

# GlueMap C++ (pygluemap) — needs pyceres (gluemap --no-deps skips it)
RUN set -eu; \
    python -c "import scikit_build_core.build, pybind11; print('pybind11', pybind11.__version__)"; \
    python -m pip install --break-system-packages "pyceres==2.6"; \
    python -c "import pyceres; print('pyceres ok', getattr(pyceres, '__version__', '?'))"; \
    export CMAKE_PREFIX_PATH="/usr:${CMAKE_PREFIX_PATH:-}"; \
    export CMAKE_BUILD_PARALLEL_LEVEL=1 MAX_JOBS=1 NINJAFLAGS="-j1"; \
    python -m pip install --break-system-packages --no-deps --no-build-isolation -e /opt/gluemap; \
    python -c "import gluemap; import pygluemap; print('pygluemap ok', pygluemap.__file__)"; \
    python -m pip install --break-system-packages --force-reinstall --no-deps "numpy>=2.0,<2.2"; \
    test -f /opt/gsplat/examples/simple_viewer.py; \
    test -f /opt/gsplat/examples/simple_trainer.py; \
    test -d /opt/hloc; \
    test -d /opt/fastmap; \
    test -d /opt/gluemap; \
    test -d /opt/depth-anything-cpp/third_party/ggml; \
    python -c "import numpy; print('numpy', numpy.__version__)"; \
    mkdir -p /opt/bin-export /opt/runtime-libs; \
    for f in /usr/local/bin/gluemap*; do \
      [ -e "$f" ] && cp -a "$f" /opt/bin-export/; \
    done; \
    touch /opt/bin-export/.keep; \
    for pat in 'libceres.so*' 'libgomp.so*' 'libmetis.so*' 'libcholmod.so*' 'libamd.so*' 'libcamd.so*' 'libccolamd.so*' 'libcolamd.so*' 'libsuitesparseconfig.so*'; do \
      find /usr/lib /lib -name "$pat" -exec cp -a {} /opt/runtime-libs/ \; 2>/dev/null || true; \
    done; \
    touch /opt/runtime-libs/.keep; \
    # Do NOT full-copy site-packages → /opt/py-site here (doubles layer size; Kaniko
    # snapshot after this RUN OOMs dockerbuilder). Runtime uses dist-packages COPY.
    mkdir -p /opt/py-site; \
    python - <<'PY'
import glob, os

import gsplat
print("gsplat ok", gsplat.__file__)
# fastmap.cuda may fail under libcuda stub — check .so, not import
import fastmap
print("fastmap ok", fastmap.__file__)

# Smoke imports (no fastmap.cuda — needs real GPU at runtime on Olares)
mods = [
    "torch", "gsplat", "fastapi", "uvicorn", "cv2", "scipy", "viser",
    "kornia", "h5py", "pycolmap", "einops", "omegaconf", "plyfile",
    "fastmap", "pyceres", "gluemap", "pygluemap",
]
missing = []
for m in mods:
    try:
        __import__(m)
        print("import ok", m)
    except Exception as e:
        missing.append(f"{m}: {e}")
        print("import FAIL", m, e)
if missing:
    raise SystemExit("smoke import failed:\n  " + "\n  ".join(missing))

sos = glob.glob("/opt/fastmap/**/cuda*.so", recursive=True)
sos += glob.glob("/opt/fastmap/**/*.so", recursive=True)
sos = [s for s in sos if "cuda" in os.path.basename(s).lower()]
print("fastmap cuda so", sos[:5])
if not sos:
    raise SystemExit("fastmap CUDA .so missing after build_ext")

for path, kind in [
    ("/opt/colmap-bundle/bin/colmap", "x"),
    ("/opt/colmap-bundle/bin/glomap", "x"),
    ("/opt/dense-sfm/dense_sfm", "d"),
    ("/opt/extra/bin/ffmpeg", "x"),
    ("/opt/bin-export/da3-cli", "x"),
]:
    ok = os.path.isdir(path) if kind == "d" else os.path.isfile(path) and os.access(path, os.X_OK)
    if not ok:
        raise SystemExit(f"missing required artifact: {path}")
    print("artifact ok", path)

print("=== PYTHON/GSPLAT/BACKENDS BUILD DONE ===")
PY

# Final stage: COPY / ENV / WORKDIR / ENTRYPOINT only — NO RUN.
FROM pytorch/pytorch:2.11.0-cuda12.8-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_BREAK_SYSTEM_PACKAGES=1 \
    PYTHONPATH=/opt/gsplat:/opt/dense-sfm:/opt/vggt-omega:/opt/da3/src:/opt/lingbot-map:/opt/instantsplat:/opt/hloc:/opt/lightglue:/opt/fastmap:/opt/gluemap:/app \
    SPLATLAB_VISER_BASE_PORT=8780 \
    SPLATLAB_MODEL_CACHE=/data/splatlab/cache/models \
    MODELSCOPE_CACHE=/data/splatlab/cache/modelscope \
    PATH="/opt/extra/bin:/opt/colmap/bin:${PATH}" \
    LD_LIBRARY_PATH=/opt/extra/lib:/opt/colmap/lib:/opt/runtime-libs:/usr/local/cuda/lib64:/usr/local/nvidia/lib64:/usr/local/nvidia/lib \
    GLUEMAP_ROOT=/opt/gluemap

WORKDIR /app

COPY --from=python-builder /opt/colmap-bundle/ /opt/colmap/
COPY --from=python-builder /opt/extra/ /opt/extra/
COPY --from=python-builder /usr/local/lib/python3.12/dist-packages /usr/local/lib/python3.12/dist-packages
COPY --from=python-builder /opt/bin-export/ /usr/local/bin/
COPY --from=python-builder /opt/runtime-libs/ /opt/runtime-libs/
COPY --from=python-builder /opt/gsplat /opt/gsplat
COPY --from=python-builder /opt/vggt-omega /opt/vggt-omega
COPY --from=python-builder /opt/da3 /opt/da3
COPY --from=python-builder /opt/lingbot-map /opt/lingbot-map
COPY --from=python-builder /opt/instantsplat /opt/instantsplat
COPY --from=python-builder /opt/hloc /opt/hloc
COPY --from=python-builder /opt/lightglue /opt/lightglue
COPY --from=python-builder /opt/fastmap /opt/fastmap
COPY --from=python-builder /opt/gluemap /opt/gluemap
COPY --from=python-builder /opt/dense-sfm /opt/dense-sfm
COPY --from=python-builder /opt/dense-sfm-refine /opt/dense-sfm-refine

COPY entrypoint.sh /app/entrypoint.sh
COPY . /app/

EXPOSE 7860

# bash wrapper: Kaniko often drops COPY --chmod execute bit
ENTRYPOINT ["/bin/bash", "/app/entrypoint.sh"]
