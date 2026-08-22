#!/usr/bin/env bash
# Build CosyVoice pre-baked image on Olares One (linux/amd64) and push to GHCR.
#
# Prerequisites:
#   docker or nerdctl
#   GHCR_TOKEN — GitHub PAT with write:packages
#
# Usage:
#   export GHCR_TOKEN=ghp_...
#   ./cosyvoice2yueone/docker/build-and-push-ghcr.sh
#
# Optional:
#   IMAGE=ghcr.io/coynntis/cosyvoice2yueone:1.0.0
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
IMAGE="${IMAGE:-ghcr.io/coynntis/cosyvoice2yueone:1.0.0}"
GHCR_USER="${GHCR_USER:-coynntis}"

if command -v docker >/dev/null 2>&1; then
  CTR=docker
elif command -v nerdctl >/dev/null 2>&1; then
  CTR=nerdctl
else
  echo "error: need docker or nerdctl on this host" >&2
  exit 1
fi

if [ -z "${GHCR_TOKEN:-}" ]; then
  echo "error: set GHCR_TOKEN (GitHub PAT with write:packages)" >&2
  exit 1
fi

echo "Building ${IMAGE} (context: ${REPO_ROOT})..."
$CTR build \
  --platform linux/amd64 \
  -t "${IMAGE}" \
  -f "${REPO_ROOT}/cosyvoice2yueone/docker/Dockerfile" \
  "${REPO_ROOT}"

echo "Logging in to ghcr.io..."
echo "${GHCR_TOKEN}" | $CTR login ghcr.io -u "${GHCR_USER}" --password-stdin

echo "Pushing ${IMAGE}..."
$CTR push "${IMAGE}"

echo "Done. Set chart image to ${IMAGE} and skip pip bootstrap when wired."
