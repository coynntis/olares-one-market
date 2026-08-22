#!/usr/bin/env bash
# Build on Olares One (native linux/amd64 + CUDA) and push to GitHub Container Registry.
#
# Prerequisites on the build host:
#   - docker or nerdctl
#   - git clone of olares-one-market (or copy locateanything3bone/ dir)
#   - GHCR_TOKEN: GitHub PAT with write:packages
#
# Usage (on Olares One shell or SSH):
#   export GHCR_TOKEN=ghp_...
#   ./locateanything3bone/scripts/build-and-push-ghcr.sh
#
# Optional:
#   IMAGE=ghcr.io/coynntis/locate-anything:0.1.0
#   INSTALL_MAGIATTENTION=0   # skip MTP compile (faster)
#   GHCR_USER=coynntis
set -euo pipefail

CHART_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REPO_ROOT="$(cd "${CHART_ROOT}/.." && pwd)"
IMAGE="${IMAGE:-ghcr.io/coynntis/locate-anything:0.1.2}"
INSTALL_MAGI="${INSTALL_MAGIATTENTION:-1}"
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

echo "==> login ghcr.io as ${GHCR_USER}"
echo "${GHCR_TOKEN}" | "${CTR}" login ghcr.io -u "${GHCR_USER}" --password-stdin

echo "==> build ${IMAGE} (INSTALL_MAGIATTENTION=${INSTALL_MAGI})"
"${CTR}" build \
  -f "${CHART_ROOT}/docker/Dockerfile" \
  --build-arg "INSTALL_MAGIATTENTION=${INSTALL_MAGI}" \
  -t "${IMAGE}" \
  "${CHART_ROOT}"

echo "==> push ${IMAGE}"
"${CTR}" push "${IMAGE}"

echo ""
echo "Done. Chart defaults expect this image with deps.bootstrapOnDevice: false"
echo "  image.repository: ghcr.io/coynntis/locate-anything"
echo "  image.tag: $(echo "${IMAGE}" | awk -F: '{print $NF}')"
echo ""
echo "Make package public at https://github.com/users/${GHCR_USER}/packages or add imagePullSecrets on Olares."
