# Audio8 TTS Preview 0.6B — CPU ONNX INT4 OpenAI-compatible server.
# Model downloaded at container start into ARKTTS_MODEL_DIR (~1 GiB).
FROM python:3.11-slim-bookworm

RUN apt-get update \
  && apt-get install -y --no-install-recommends libsndfile1 ca-certificates \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt \
  && pip install --no-cache-dir --no-deps "tokenizers==0.22.2" \
  && pip install --no-cache-dir "huggingface_hub[cli]>=0.26.0"

COPY arktts_runtime /app/arktts_runtime
COPY download_model.py bootstrap_voices.py entrypoint.sh /app/
COPY seed /seed

RUN chmod +x /app/entrypoint.sh \
  && useradd -m -u 1000 -s /bin/bash audio8 \
  && mkdir -p /models /voices \
  && chown -R audio8:audio8 /app /seed /models /voices

USER 1000
ENV HOST=0.0.0.0 \
    PORT=8024 \
    ARKTTS_MODEL_DIR=/models \
    ARKTTS_VOICES_DIR=/voices \
    ARKTTS_SEED_DIR=/seed \
    ARKTTS_PRECISION=int4 \
    ARKTTS_CODEC_PRECISION=fp16 \
    ARKTTS_THREADS=16 \
    OMP_NUM_THREADS=16 \
    MKL_NUM_THREADS=16

EXPOSE 8024
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=5 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8024/api/health')"

ENTRYPOINT ["/app/entrypoint.sh"]
