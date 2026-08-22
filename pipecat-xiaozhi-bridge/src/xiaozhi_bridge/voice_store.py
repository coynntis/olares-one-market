"""Persist voice clone profiles (reference WAV + transcript) on app data volume."""

from __future__ import annotations

import base64
import binascii
import json
import logging
import os
import re
import threading
import time
import uuid
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_LOCK = threading.Lock()
_DATA_URL_RE = re.compile(r"^data:(audio/[a-zA-Z0-9.+-]+|application/octet-stream);base64,(.+)$", re.DOTALL)
_MAX_AUDIO_BYTES = 12 * 1024 * 1024


def voices_dir() -> Path:
    raw = os.environ.get("VOICES_DIR", "/data/voices")
    return Path(raw)


def _now() -> float:
    return time.time()


def _meta_path(voice_id: str) -> Path:
    return voices_dir() / f"{voice_id}.json"


def _audio_path(voice_id: str) -> Path:
    return voices_dir() / f"{voice_id}.wav"


def _decode_audio_blob(audio: str) -> bytes:
    raw = (audio or "").strip()
    if not raw:
        raise ValueError("audio required")
    if raw.startswith("data:"):
        m = _DATA_URL_RE.match(raw)
        if not m:
            raise ValueError("invalid audio data URL")
        b64 = m.group(2)
    else:
        b64 = raw
    try:
        data = base64.b64decode(b64, validate=True)
    except (binascii.Error, ValueError) as e:
        raise ValueError("invalid base64 audio") from e
    if len(data) > _MAX_AUDIO_BYTES:
        raise ValueError("audio too large (max 12MB)")
    if len(data) < 44:
        raise ValueError("audio too short")
    return data


def list_voices() -> list[dict[str, Any]]:
    root = voices_dir()
    root.mkdir(parents=True, exist_ok=True)
    out: list[dict[str, Any]] = []
    for path in sorted(root.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            meta = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(meta, dict):
            continue
        vid = str(meta.get("id") or path.stem)
        if not _audio_path(vid).is_file():
            continue
        out.append(
            {
                "id": vid,
                "name": str(meta.get("name") or vid),
                "ref_text": str(meta.get("ref_text") or ""),
                "language_id": str(meta.get("language_id") or ""),
                "instruct": str(meta.get("instruct") or ""),
                "created_at": int(float(meta.get("created_at") or 0) * 1000),
                "audio_url": f"/api/voices/{vid}/audio",
            }
        )
    return out


def get_voice(voice_id: str) -> dict[str, Any] | None:
    path = _meta_path(voice_id)
    if not path.is_file() or not _audio_path(voice_id).is_file():
        return None
    try:
        meta = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(meta, dict):
        return None
    return {
        "id": voice_id,
        "name": str(meta.get("name") or voice_id),
        "ref_text": str(meta.get("ref_text") or ""),
        "language_id": str(meta.get("language_id") or ""),
        "instruct": str(meta.get("instruct") or ""),
        "created_at": int(float(meta.get("created_at") or 0) * 1000),
        "audio_url": f"/api/voices/{voice_id}/audio",
    }


def read_voice_audio(voice_id: str) -> bytes | None:
    path = _audio_path(voice_id)
    if not path.is_file():
        return None
    return path.read_bytes()


def read_voice_audio_b64(voice_id: str) -> str | None:
    data = read_voice_audio(voice_id)
    if not data:
        return None
    return base64.b64encode(data).decode("ascii")


def create_voice(
    *,
    name: str,
    ref_text: str,
    audio_data: str,
    language_id: str = "",
    instruct: str = "",
) -> dict[str, Any]:
    name = (name or "").strip() or "Voice clone"
    ref_text = (ref_text or "").strip()
    if not ref_text:
        raise ValueError("ref_text required — transcript of the reference recording")
    wav = _decode_audio_blob(audio_data)
    vid = str(uuid.uuid4())
    root = voices_dir()
    root.mkdir(parents=True, exist_ok=True)
    ts = _now()
    meta = {
        "id": vid,
        "name": name,
        "ref_text": ref_text,
        "language_id": language_id.strip(),
        "instruct": instruct.strip(),
        "created_at": ts,
    }
    with _LOCK:
        _audio_path(vid).write_bytes(wav)
        _meta_path(vid).write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    logger.info("voice profile saved id=%s name=%r bytes=%d", vid, name, len(wav))
    return get_voice(vid) or meta


def delete_voice(voice_id: str) -> bool:
    with _LOCK:
        meta = _meta_path(voice_id)
        audio = _audio_path(voice_id)
        existed = meta.is_file() or audio.is_file()
        meta.unlink(missing_ok=True)
        audio.unlink(missing_ok=True)
    return existed
