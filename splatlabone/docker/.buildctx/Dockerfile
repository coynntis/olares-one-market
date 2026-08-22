# syntax=docker/dockerfile:1
#
# SplatLab One — COLMAP 4.x (global_mapper), PyTorch cu128, gsplat
# Zero apt-get (Kaniko snapshot/OOM on nvidia devel layers).
#
# python-builder = pytorch devel (nvcc + system Python 3.12, PEP 668)
# colmap-tools   = official COLMAP image (binaries only)
# final          = pytorch runtime + COPY colmap + COPY pip site-packages

FROM pytorch/pytorch:2.11.0-cuda12.8-cudnn9-devel AS python-builder

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_BREAK_SYSTEM_PACKAGES=1

WORKDIR /build

# Ubuntu 24.04 marks system Python externally managed (PEP 668).
RUN rm -f /usr/lib/python3.12/EXTERNALLY-MANAGED /usr/lib/python*/EXTERNALLY-MANAGED 2>/dev/null || true

COPY requirements.txt /build/requirements.txt

RUN set -eux; \
    python -m pip install --break-system-packages --upgrade pip setuptools wheel; \
    python -m pip install --break-system-packages ninja numpy; \
    python -m pip install --break-system-packages --no-build-isolation gsplat; \
    python -m pip install --break-system-packages -r /build/requirements.txt; \
    python -c "import io, tarfile, urllib.request; url='https://github.com/nerfstudio-project/gsplat/archive/refs/heads/main.tar.gz'; data=urllib.request.urlopen(url, timeout=300).read(); tarfile.open(fileobj=io.BytesIO(data)).extractall('/opt'); import os; os.rename('/opt/gsplat-main', '/opt/gsplat')"; \
    python -c "import importlib.util; assert importlib.util.find_spec('gsplat'); print('gsplat package ok')"; \
    echo "=== PYTHON/GSPLAT BUILD DONE ==="

FROM colmap/colmap:latest AS colmap-tools

FROM pytorch/pytorch:2.11.0-cuda12.8-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_BREAK_SYSTEM_PACKAGES=1 \
    PYTHONPATH=/opt/gsplat:/app \
    LD_LIBRARY_PATH=/usr/local/lib:${LD_LIBRARY_PATH:-}

WORKDIR /app

# COLMAP CLI + shared libs (GLOMAP path uses colmap global_mapper when glomap absent).
COPY --from=colmap-tools /usr/local/bin/colmap /usr/local/bin/colmap
COPY --from=colmap-tools /usr/local/lib/ /usr/local/lib/
COPY --from=colmap-tools /usr/local/share/colmap /usr/local/share/colmap

# gsplat + FastAPI stack compiled in devel stage (system site-packages).
COPY --from=python-builder /usr/local/lib/python3.12/dist-packages /usr/local/lib/python3.12/dist-packages
COPY --from=python-builder /opt/gsplat /opt/gsplat

COPY entrypoint.sh /app/entrypoint.sh
COPY . /app/

# Kaniko has no NVIDIA driver — libcuda.so.1 only exists at runtime on GPU nodes.
RUN chmod +x /app/entrypoint.sh && \
    test -x /usr/local/bin/colmap && \
    python -c "import importlib.util; assert importlib.util.find_spec('torch'); assert importlib.util.find_spec('gsplat'); assert importlib.util.find_spec('fastapi'); print('packages ok')"

EXPOSE 7860

ENTRYPOINT ["/app/entrypoint.sh"]
