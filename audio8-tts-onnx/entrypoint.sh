#!/usr/bin/env bash
set -euo pipefail

export ARKTTS_MODEL_DIR="${ARKTTS_MODEL_DIR:-/models}"
export ARKTTS_VOICES_DIR="${ARKTTS_VOICES_DIR:-/voices}"
export ARKTTS_REGISTRATION_DIR="${ARKTTS_REGISTRATION_DIR:-$ARKTTS_MODEL_DIR/registration}"
export ARKTTS_SEED_DIR="${ARKTTS_SEED_DIR:-/seed}"
export ARKTTS_PRECISION="${ARKTTS_PRECISION:-int4}"
export ARKTTS_CODEC_PRECISION="${ARKTTS_CODEC_PRECISION:-fp16}"
export ARKTTS_THREADS="${ARKTTS_THREADS:-16}"
export HOST="${HOST:-0.0.0.0}"
export PORT="${PORT:-8024}"

mkdir -p "$ARKTTS_MODEL_DIR" "$ARKTTS_VOICES_DIR"

echo "ensuring Audio8 ONNX model…"
python /app/download_model.py

echo "bootstrapping seed voices…"
python /app/bootstrap_voices.py

echo "starting Audio8 TTS on ${HOST}:${PORT} threads=${ARKTTS_THREADS}"
exec python -m uvicorn arktts_runtime.service:app \
  --app-dir /app \
  --host "$HOST" \
  --port "$PORT"
