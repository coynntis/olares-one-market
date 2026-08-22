#!/usr/bin/env bash
# Build splatlabone-docker.zip for dockerbuilderone upload.
# Dockerfile MUST be at zip root (first entry) — dockerbuilder find_dockerfile checks root.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DOCKER="$ROOT/docker"
STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT

rsync -a \
  --exclude '__pycache__' --exclude '*.pyc' --exclude '.venv' \
  "$ROOT/app/" "$STAGE/"

# Overlay docker build inputs AFTER rsync (never deleted by app sync)
cp "$DOCKER/Dockerfile" "$DOCKER/.dockerignore" "$DOCKER/entrypoint.sh" \
   "$DOCKER/requirements.txt" "$DOCKER/requirements-geometry.txt" \
   "$DOCKER/fetch_backends.py" "$DOCKER/glomap-shim.sh" "$STAGE/"
rm -rf "$STAGE/dense-sfm-overlay" "$STAGE/stubs"
cp -a "$DOCKER/dense-sfm-overlay" "$STAGE/dense-sfm-overlay"
cp -a "$DOCKER/stubs" "$STAGE/stubs"

for f in Dockerfile .dockerignore entrypoint.sh requirements.txt requirements-geometry.txt fetch_backends.py \
         glomap-shim.sh stubs/libcuda.so.1 \
         dense-sfm-overlay/run_matching.py dense-sfm-overlay/dense_sfm/__init__.py \
         main.py pipeline/stages.py pipeline/geometry/da3.py api/viewer.py scripts/download_models.py; do
  test -e "$STAGE/$f" || { echo "missing in stage: $f" >&2; exit 1; }
done

OUT="$DOCKER/splatlabone-docker.zip"
rm -f "$OUT" /tmp/splatlabone-docker.zip

# Python zip — Dockerfile first, no macOS junk, deterministic
python3 - <<PY
import zipfile
from pathlib import Path

stage = Path("$STAGE")
out = Path("$OUT")
# Dockerfile + build files first so naive UIs / extractors see them immediately
first = [
    "Dockerfile",
    ".dockerignore",
    "entrypoint.sh",
    "requirements.txt",
    "requirements-geometry.txt",
    "fetch_backends.py",
    "glomap-shim.sh",
]
skip_suffix = {".md", ".pyc"}
skip_parts = {"__pycache__", ".venv"}

def want(rel: Path) -> bool:
    if any(p in skip_parts for p in rel.parts):
        return False
    if rel.suffix in skip_suffix:
        return False
    return True

with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as zf:
    seen = set()
    for name in first:
        p = stage / name
        if not p.is_file():
            raise SystemExit(f"FATAL: {name} missing before zip")
        zf.write(p, name)
        seen.add(name)
    for p in sorted(stage.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(stage).as_posix()
        if rel in seen or not want(Path(rel)):
            continue
        zf.write(p, rel)
        seen.add(rel)

with zipfile.ZipFile(out) as zf:
    names = zf.namelist()
    assert names[0] == "Dockerfile", f"Dockerfile not first: {names[:5]}"
    assert "Dockerfile" in names
    assert zf.getinfo("Dockerfile").file_size > 100
    print(f"OK {len(names)} files; first={names[:6]}")
    print(f"Dockerfile size={zf.getinfo('Dockerfile').file_size}")
PY

cp "$OUT" /tmp/splatlabone-docker.zip
if [[ -d "$HOME/Desktop" ]]; then
  cp "$OUT" "$HOME/Desktop/splatlabone-docker.zip" 2>/dev/null || true
fi
ls -lh "$OUT" /tmp/splatlabone-docker.zip
