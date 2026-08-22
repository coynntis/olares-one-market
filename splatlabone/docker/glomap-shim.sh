#!/usr/bin/env bash
# Compatibility shim: standalone GLOMAP CLI → COLMAP 4.x global_mapper pipeline.
# Used when /usr/local/bin/glomap is absent from the colmap image (merged upstream).
set -euo pipefail

COLMAP_BIN="${COLMAP_BIN:-colmap}"

if [[ "${1:-}" != "mapper" ]]; then
  echo "glomap shim: only 'mapper' is supported (got: ${1:-})" >&2
  echo "hint: GLOMAP lives in COLMAP 4 as: colmap global_mapper" >&2
  exit 1
fi
shift

IMAGE_PATH=""
OUTPUT_PATH=""
DATABASE_PATH=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --image_path) IMAGE_PATH="${2:-}"; shift 2 ;;
    --output_path) OUTPUT_PATH="${2:-}"; shift 2 ;;
    --database_path) DATABASE_PATH="${2:-}"; shift 2 ;;
    *) shift ;;
  esac
done

if [[ -z "$IMAGE_PATH" || -z "$OUTPUT_PATH" ]]; then
  echo "glomap shim: need --image_path and --output_path" >&2
  exit 1
fi

mkdir -p "$OUTPUT_PATH"
WS="$(cd "$(dirname "$OUTPUT_PATH")" && pwd)"
DB="${DATABASE_PATH:-${WS}/database.db}"

if [[ ! -s "$DB" ]]; then
  "$COLMAP_BIN" feature_extractor --database_path "$DB" --image_path "$IMAGE_PATH"
  "$COLMAP_BIN" exhaustive_matcher --database_path "$DB"
fi

"$COLMAP_BIN" global_mapper \
  --database_path "$DB" \
  --image_path "$IMAGE_PATH" \
  --output_path "$OUTPUT_PATH"
