#!/usr/bin/env python3
"""Download Audio8 ONNX INT4 weights into ARKTTS_MODEL_DIR if missing."""

from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ID = os.environ.get(
    "AUDIO8_HF_REPO", "Audio8/Audio8-TTS-Preview-0.6B-ONNX-INT4"
)
MODEL_DIR = Path(os.environ.get("ARKTTS_MODEL_DIR", "/models")).resolve()


def main() -> int:
    marker = MODEL_DIR / "runtime_manifest.json"
    if marker.is_file():
        print(f"model already present: {MODEL_DIR}")
        return 0
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    token = (os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or "").strip() or None
    endpoint = (os.environ.get("HF_ENDPOINT") or "").strip() or None
    if endpoint:
        os.environ["HF_ENDPOINT"] = endpoint
    print(f"downloading {REPO_ID} → {MODEL_DIR}")
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id=REPO_ID,
        local_dir=str(MODEL_DIR),
        token=token,
    )
    if not marker.is_file():
        print("download finished but runtime_manifest.json missing", file=sys.stderr)
        return 1
    print("download complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
