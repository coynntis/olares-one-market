#!/usr/bin/env python3
"""Register seed voices once when VOICES_DIR is empty / missing names."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from arktts_runtime.registration import VoiceRegistration

MODEL_DIR = Path(os.environ.get("ARKTTS_MODEL_DIR", "/models")).resolve()
VOICES_DIR = Path(os.environ.get("ARKTTS_VOICES_DIR", "/voices")).resolve()
REGISTRATION_DIR = Path(
    os.environ.get("ARKTTS_REGISTRATION_DIR", str(MODEL_DIR / "registration"))
).resolve()
SEED_DIR = Path(os.environ.get("ARKTTS_SEED_DIR", "/seed")).resolve()

SEED_VOICES = ("en_default", "zh_default", "yue_default")


def main() -> int:
    manifest_path = MODEL_DIR / "runtime_manifest.json"
    if not manifest_path.is_file():
        print(f"model not ready: missing {manifest_path}", file=sys.stderr)
        return 1
    fingerprint = json.loads(manifest_path.read_text())["model_fingerprint"]
    reg = VoiceRegistration(REGISTRATION_DIR, VOICES_DIR, str(fingerprint))
    status = reg.status()
    if not status["available"]:
        print(f"registration unavailable: {status.get('reason')}", file=sys.stderr)
        return 1

    VOICES_DIR.mkdir(parents=True, exist_ok=True)
    for name in SEED_VOICES:
        target = VOICES_DIR / name
        if target.is_dir() and (target / "meta.json").is_file():
            print(f"voice already present: {name}")
            continue
        wav = SEED_DIR / f"{name}.wav"
        txt = SEED_DIR / f"{name}.txt"
        if not wav.is_file() or not txt.is_file():
            print(f"skip {name}: missing seed files", file=sys.stderr)
            continue
        text = txt.read_text(encoding="utf-8").strip()
        data = wav.read_bytes()
        print(f"registering seed voice: {name}")
        meta = reg.register(data, wav.name, text, name, overwrite=True)
        print(f"  ok frames={meta.get('shape')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
