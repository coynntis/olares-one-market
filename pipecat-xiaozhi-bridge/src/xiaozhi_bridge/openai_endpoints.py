"""Resolve OpenAI-compatible base URLs and API keys per service."""

from __future__ import annotations

from xiaozhi_bridge.config import load_settings

# OpenAI Python SDK requires a non-empty api_key even when the upstream ignores it
# (e.g. Olares shared entrances with internal auth).
OPENAI_SDK_PLACEHOLDER_KEY = "ollama"


def client_api_key(resolved: str | None) -> str:
    return resolved if resolved else OPENAI_SDK_PLACEHOLDER_KEY


def auth_headers(api_key: str | None) -> dict[str, str]:
    if api_key:
        return {"Authorization": f"Bearer {api_key}"}
    return {}


def stt_base_url() -> str | None:
    return load_settings().stt_url()


def stt_api_key() -> str | None:
    return load_settings().stt_key()


def tts_base_url() -> str | None:
    return load_settings().tts_url()


def tts_api_key() -> str | None:
    return load_settings().tts_key()


def llm_base_url() -> str | None:
    return load_settings().llm_url()


def llm_api_key() -> str | None:
    return load_settings().llm_key()
