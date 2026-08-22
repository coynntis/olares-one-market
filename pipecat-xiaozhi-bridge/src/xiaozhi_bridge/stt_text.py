"""Normalize STT transcripts (SenseVoice tags, FunASR empty segments)."""

from __future__ import annotations

import ast
import json
import re
from typing import Any

from xiaozhi_bridge.sensevoice_text import parse_sensevoice_transcript

_FUNASR_BLOB_RE = re.compile(
    r"""^\s*[\[{].*['"]key['"]\s*:.*['"]timestamp['"]\s*:.*[\]}]\s*$""",
    re.DOTALL,
)

# Minimum WAV payload (bytes after header) worth sending to STT.
MIN_STT_WAV_BYTES = 1600


def _text_from_funasr_obj(obj: Any) -> str:
    if isinstance(obj, dict):
        for key in ("text", "value", "sentence", "transcript"):
            val = obj.get(key)
            if isinstance(val, str) and val.strip():
                return val.strip()
        return ""
    if isinstance(obj, list):
        parts: list[str] = []
        for item in obj:
            piece = _text_from_funasr_obj(item)
            if piece:
                parts.append(piece)
        return " ".join(parts).strip()
    if isinstance(obj, str):
        return obj.strip()
    return ""


def _parse_funasr_blob(raw: str) -> str:
    text = raw.strip()
    if not text:
        return ""
    for loader in (json.loads, ast.literal_eval):
        try:
            return _text_from_funasr_obj(loader(text))
        except (json.JSONDecodeError, SyntaxError, ValueError, TypeError):
            continue
    return ""


def is_funasr_metadata_blob(raw: str) -> bool:
    text = (raw or "").strip()
    if not text:
        return False
    if _FUNASR_BLOB_RE.match(text):
        return True
    return ("'key'" in text or '"key"' in text) and (
        "'timestamp'" in text or '"timestamp"' in text
    )


def normalize_stt_transcript(raw: str) -> tuple[str, dict[str, Any] | None]:
    """Return display text + optional SenseVoice meta; empty text if no speech."""
    text = (raw or "").strip()
    if not text:
        return "", None

    if is_funasr_metadata_blob(text):
        inner = _parse_funasr_blob(text)
        return inner, None

    clean, meta = parse_sensevoice_transcript(text)
    if is_funasr_metadata_blob(clean):
        clean = _parse_funasr_blob(clean)

    # nospeech / empty event tags with no words
    if meta.get("event") == "nospeech" or meta.get("language") == "nospeech":
        return "", meta or None

    clean = clean.strip()
    if not clean or is_funasr_metadata_blob(clean):
        return "", meta or None
    return clean, meta or None
