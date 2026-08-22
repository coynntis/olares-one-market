#!/usr/bin/env bash
# Build linux/amd64 runtime (no dsh baked in) and push to ghcr.io.
#
#   export GHCR_TOKEN=ghp_...   # PAT with write:packages
#   ./dshone/docker/build-and-push-ghcr.sh
set -euo pipefail

CHART_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
IMAGE="${IMAGE:-ghcr.io/coynntis/dsh-runtime:22.19-1}"
GHCR_USER="${GHCR_USER:-coynntis}"

if command -v docker >/dev/null 2>&1; then
  CTR=docker
elif command -v nerdctl >/dev/null 2>&1; then
  CTR=nerdctl
else
  echo "error: need docker or nerdctl" >&2
  exit 1
fi

if [ -z "${GHCR_TOKEN:-}" ]; then
  echo "error: set GHCR_TOKEN (GitHub PAT with write:packages)" >&2
  exit 1
fi

echo "==> login ghcr.io as ${GHCR_USER}"
echo "${GHCR_TOKEN}" | "${CTR}" login ghcr.io -u "${GHCR_USER}" --password-stdin

echo "==> build+push ${IMAGE} linux/amd64"
if "${CTR}" buildx version >/dev/null 2>&1; then
  "${CTR}" buildx build \
    --platform linux/amd64 \
    -f "${CHART_ROOT}/docker/Dockerfile" \
    -t "${IMAGE}" \
    --push \
    "${CHART_ROOT}/docker"
else
  "${CTR}" build \
    --platform linux/amd64 \
    -f "${CHART_ROOT}/docker/Dockerfile" \
    -t "${IMAGE}" \
    "${CHART_ROOT}/docker"
  "${CTR}" push "${IMAGE}"
fi

echo "Done. Chart expects:"
echo "  image.repository: ghcr.io/coynntis/dsh-runtime"
echo "  image.tag: 22.19-1"
echo "Make package public at https://github.com/users/${GHCR_USER}/packages"
