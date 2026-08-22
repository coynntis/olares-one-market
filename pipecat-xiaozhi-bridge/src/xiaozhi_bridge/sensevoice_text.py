"""Parse FunAudioLLM SenseVoice rich transcript prefixes."""

from __future__ import annotations

import re
from typing import Any

_TAG_RE = re.compile(r"<\|([^|]+)\|>")

_LANGUAGES = frozenset({"zh", "en", "yue", "ja", "ko", "nospeech"})
_EVENTS = frozenset(
    {
        "Speech",
        "BGM",
        "Laughter",
        "Applause",
        "Cough",
        "Sneeze",
        "Cry",
        "Breath",
    }
)


def parse_sensevoice_transcript(raw: str) -> tuple[str, dict[str, Any]]:
    """Strip <|tag|> prefixes and return display text + metadata."""
    text = raw or ""
    if "<|" not in text:
        return text.strip(), {}

    tags = _TAG_RE.findall(text)
    clean = _TAG_RE.sub("", text).strip()
    if not tags:
        return clean or text.strip(), {}

    meta: dict[str, Any] = {}
    for tag in tags:
        if tag in _LANGUAGES:
            meta["language"] = tag
        elif tag.startswith("EMO_"):
            meta["emotion"] = tag[4:].lower().replace("_", " ")
        elif tag in ("withitn", "woitn"):
            meta["itn"] = tag == "withitn"
        elif tag in _EVENTS:
            meta["event"] = tag.lower()
        else:
            meta.setdefault("tags", []).append(tag)

    return clean or text.strip(), meta


def clean_sensevoice_transcript(raw: str) -> str:
    return parse_sensevoice_transcript(raw)[0]
