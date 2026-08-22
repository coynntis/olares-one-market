# Pre-baked CosyVoice env (recommended for Olares One production).
#
# Official CosyVoice stack (FunAudioLLM/CosyVoice):
#   Python 3.10, conda env, torch==2.3.1, numpy==1.26.4, gradio==5.4.0
#   Base: nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04 (see docker/Dockerfile upstream)
#
# Build (linux/amd64, GPU host with docker/nerdctl):
#   docker build -t ghcr.io/<user>/cosyvoice2yueone:1.0.0 -f cosyvoice2yueone/docker/Dockerfile .
# Or: ./cosyvoice2yueone/docker/build-and-push-ghcr.sh
#
# Chart still runs server.py from deployment; this image only bakes deps under /opt/src/CosyVoice.

FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ARG VENV_NAME=cosyvoice
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    HF_HOME=/models/huggingface \
    PYTHONPATH=/opt/src/CosyVoice:/opt/src/CosyVoice/third_party/Matcha-TTS

RUN apt-get update && apt-get install -y --no-install-recommends \
    git build-essential curl wget ffmpeg unzip git-lfs sox libsox-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/* \
    && git lfs install

RUN curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o /tmp/miniconda.sh \
    && bash /tmp/miniconda.sh -b -p /opt/conda \
    && rm /tmp/miniconda.sh \
    && /opt/conda/bin/conda create -y -n ${VENV_NAME} python=3.10

ENV PATH=/opt/conda/envs/${VENV_NAME}/bin:/opt/conda/bin:$PATH

WORKDIR /opt/src

RUN git clone --recursive https://github.com/FunAudioLLM/CosyVoice CosyVoice

RUN conda install -y -c conda-forge pynini==2.1.5

# Inference-slimmed: skip train/tensorrt/deepspeed; whisper installed separately.
RUN grep -vE '^(deepspeed|tensorrt|openai-whisper|onnx|onnxruntime-gpu|wandb|pytest|pytest-cov)==|tensorrt-cu12' CosyVoice/requirements.txt \
    | grep -v '^--extra-index-url' > /tmp/cosyvoice-req.txt \
    && pip install --no-cache-dir --upgrade pip setuptools wheel \
    && pip install --no-cache-dir -r /tmp/cosyvoice-req.txt \
    && pip uninstall -y onnxruntime-gpu onnxruntime 2>/dev/null || true \
    && pip install --no-cache-dir 'onnx>=1.16.0' 'onnxruntime>=1.18.0' \
    && pip install --no-cache-dir 'numpy==1.26.4' cython einops scipy \
    && pip install --no-cache-dir --no-deps --no-build-isolation -e CosyVoice/third_party/Matcha-TTS \
    && pip install --no-cache-dir --no-deps --no-build-isolation 'openai-whisper==20231117' \
    && pip install --no-cache-dir more-itertools numba tiktoken tqdm

WORKDIR /opt/src/CosyVoice

RUN python -c "import torch, numpy, gradio, transformers, whisper; print('torch', torch.__version__, 'whisper ok')"

CMD ["python", "-c", "print('cosyvoice2yueone image ready — mount chart server or override CMD')"]
