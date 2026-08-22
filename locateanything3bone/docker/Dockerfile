# Native linux/amd64 build on Olares One (or any GPU host). Push to ghcr.io — see scripts/build-and-push-ghcr.sh
ARG BASE_IMAGE=docker.io/pytorch/pytorch:2.12.0-cuda13.0-cudnn9-devel
FROM ${BASE_IMAGE}

ARG INSTALL_MAGIATTENTION=1
ARG MAGI_ATTENTION_VERSION=v1.1.0
# Bump to invalidate Kaniko layer cache after dependency fixes.
ARG CACHEBUST=3

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV PIP_BREAK_SYSTEM_PACKAGES=1
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ENV PYTHONPATH=/app
ENV PATH="/opt/conda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
ENV MAGI_ATTENTION_PREBUILD_FFA=0
ENV MAGI_ATTENTION_FA4_BACKEND=1
ENV MAGI_ATTENTION_SKIP_MAGI_ATTN_COMM_BUILD=1

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      ca-certificates build-essential ninja-build && \
    apt-get install -y --no-install-recommends git && \
    apt-get install -y --only-upgrade git && \
    git --version && \
    rm -rf /var/lib/apt/lists/*

RUN python -m pip install --break-system-packages --upgrade pip setuptools wheel && \
    python -m pip install --break-system-packages "numpy>=1.26.0,<3"

COPY files/requirements.txt /app/requirements.txt

COPY files/install_magiattention.sh files/magi_build_requirements.txt /tmp/
RUN chmod +x /tmp/install_magiattention.sh && \
    INSTALL_MAGIATTENTION=${INSTALL_MAGIATTENTION} \
    MAGI_ATTENTION_VERSION=${MAGI_ATTENTION_VERSION} \
    MAGI_BUILD_REQ=/tmp/magi_build_requirements.txt \
    MAGI_INSTALL_FLASH_ATTN_CUTE=1 \
    MAGI_FLASH_ATTN_CUTE_ARCHS=sm100 \
    /bin/bash /tmp/install_magiattention.sh && \
    rm -rf /tmp/MagiAttention /tmp/install_magiattention.sh /tmp/magi_build_requirements.txt

# Install runtime deps on every python on PATH (conda + /usr/bin/python).
RUN set -eux; \
    for py in /opt/conda/bin/python python /usr/bin/python; do \
      if command -v "${py}" >/dev/null 2>&1; then \
        echo "==> pip install app deps for ${py}"; \
        "${py}" -m pip install --break-system-packages -r /app/requirements.txt; \
        "${py}" -m uvicorn --version; \
      fi; \
    done

COPY files/worker.py files/app.py files/entrypoint.sh /app/
RUN chmod +x /app/entrypoint.sh

EXPOSE 7860

ENTRYPOINT ["/app/entrypoint.sh"]
