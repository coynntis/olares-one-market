"""Persisted pipeline settings (JSON file) with optional env bootstrap."""

from __future__ import annotations

import json
import logging
import os
import threading
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from xiaozhi_bridge.builtin_tools import DEFAULT_BUILTIN_TOOLS as _DEFAULT_BUILTIN_TOOLS, normalize_builtin_tools

logger = logging.getLogger(__name__)

DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful voice assistant. Keep replies concise for speech (1-3 sentences)."
)

_LOCK = threading.Lock()
_CACHE: "BridgeSettings | None" = None


@dataclass
class BridgeSettings:
    openai_base_url: str = ""
    openai_api_key: str = ""
    stt_base_url: str = ""
    tts_base_url: str = ""
    llm_base_url: str = ""
    stt_api_key: str = ""
    tts_api_key: str = ""
    llm_api_key: str = ""
    stt_model: str = "sensevoice"
    stt_language: str = "yue"
    # melo|sherpa|kokoro = in-cluster CPU; omnivoice = GPU HQ/clone/yue
    tts_provider: str = "sherpa"
    tts_model: str = "kokoro-int8-multi-lang-v1_1"
    tts_response_format: str = "wav"
    tts_language_id: str = "en"
    tts_instruct: str = ""
    tts_voice: str = "bm_lewis"
    tts_voice_mode: str = "instruct"
    tts_active_voice_id: str = ""
    tts_ref_text: str = ""
    tts_num_step: int = 16
    tts_speed: float = 1.0
    # Overlap TTS while LLM still streams. Default OFF — on shared GPU OmniVoice
    # fights llama. Melo (CPU) can overlap safely if desired.
    tts_overlap_llm: bool = False
    # Discard TTS after TTFT. Pays cold tax on silence — does NOT reduce first_audio.
    tts_warmup: bool = False
    tts_warmup_text: str = "嗯"
    # First TTS chunk only — short so first Opus arrives ASAP after LLM handoff.
    tts_first_chunk_chars: int = 12
    # Later chunks — play N while synth N+1.
    tts_max_chunk_chars: int = 40
    tts_min_segment_chars: int = 8
    tts_segment_pad_ms: int = 40
    # After LLM frees GPU, brief settle before OmniVoice. Unused for Melo (CPU).
    tts_post_llm_delay_ms: int = 0
    llm_model: str = ""
    llm_temperature: float = 0.7
    llm_top_p: float = 0.8
    llm_top_k: int = 20
    # Voice: keep replies short so LLM frees GPU fast → OmniVoice RTF stays ~0.1–0.2.
    # Longer reasoning = another chat turn, not one huge max_tokens.
    llm_max_tokens: int = 128
    llm_think_mode: str = "auto"
    downlink_sample_rate: int = 24000
    system_prompt: str = field(default_factory=lambda: DEFAULT_SYSTEM_PROMPT)
    http_timeout: float = 120.0
    llm_profiles: list[dict[str, Any]] = field(default_factory=list)
    active_llm_profile_id: str = ""
    mcp_servers: list[dict[str, Any]] = field(default_factory=list)
    builtin_tools: dict[str, bool] = field(default_factory=lambda: dict(_DEFAULT_BUILTIN_TOOLS))

    def stt_url(self) -> str | None:
        return _nonempty(self.stt_base_url) or _nonempty(self.openai_base_url)

    def tts_url(self) -> str | None:
        return _nonempty(self.tts_base_url) or _nonempty(self.openai_base_url)

    def llm_url(self) -> str | None:
        return _nonempty(self.llm_base_url) or _nonempty(self.openai_base_url)

    def stt_key(self) -> str | None:
        return _nonempty(self.stt_api_key) or _nonempty(self.openai_api_key)

    def tts_key(self) -> str | None:
        return _nonempty(self.tts_api_key) or _nonempty(self.openai_api_key)

    def llm_key(self) -> str | None:
        return _nonempty(self.llm_api_key) or _nonempty(self.openai_api_key)


def _nonempty(value: str | None) -> str | None:
    if value is None:
        return None
    s = str(value).strip()
    return s if s else None


def config_path() -> Path:
    raw = os.environ.get("CONFIG_PATH", "/data/config.json")
    return Path(raw)


def _env_bootstrap() -> dict[str, Any]:
    """One-time env overlay for dev/helm power users; empty strings ignored."""
    mapping = {
        "openai_base_url": "OPENAI_BASE_URL",
        "openai_api_key": "OPENAI_API_KEY",
        "stt_base_url": "STT_BASE_URL",
        "tts_base_url": "TTS_BASE_URL",
        "llm_base_url": "LLM_BASE_URL",
        "stt_api_key": "STT_API_KEY",
        "tts_api_key": "TTS_API_KEY",
        "llm_api_key": "LLM_API_KEY",
        "stt_model": "STT_MODEL",
        "stt_language": "STT_LANGUAGE",
        "tts_model": "TTS_MODEL",
        "tts_response_format": "TTS_RESPONSE_FORMAT",
        "tts_language_id": "TTS_LANGUAGE_ID",
        "tts_instruct": "TTS_INSTRUCT",
        "tts_voice": "TTS_VOICE",
        "tts_voice_mode": "TTS_VOICE_MODE",
        "tts_active_voice_id": "TTS_ACTIVE_VOICE_ID",
        "tts_ref_text": "TTS_REF_TEXT",
        "tts_warmup_text": "TTS_WARMUP_TEXT",
        "tts_provider": "TTS_PROVIDER",
        "llm_model": "LLM_MODEL",
        "llm_temperature": "LLM_TEMPERATURE",
        "llm_top_p": "LLM_TOP_P",
        "llm_top_k": "LLM_TOP_K",
        "llm_max_tokens": "LLM_MAX_TOKENS",
        "llm_think_mode": "LLM_THINK_MODE",
        "system_prompt": "SYSTEM_PROMPT",
    }
    out: dict[str, Any] = {}
    string_fields = (
        "openai_base_url",
        "openai_api_key",
        "stt_base_url",
        "tts_base_url",
        "llm_base_url",
        "stt_api_key",
        "tts_api_key",
        "llm_api_key",
        "stt_model",
        "stt_language",
        "tts_model",
        "tts_response_format",
        "tts_language_id",
        "tts_instruct",
        "tts_voice",
        "tts_voice_mode",
        "tts_active_voice_id",
        "tts_ref_text",
        "tts_warmup_text",
        "tts_provider",
        "llm_model",
        "llm_think_mode",
        "system_prompt",
    )
    for field_name in string_fields:
        env_name = mapping[field_name]
        val = os.environ.get(env_name)
        if val is not None and str(val).strip():
            out[field_name] = str(val).strip()
    warm = os.environ.get("TTS_WARMUP", "").strip()
    if warm:
        out["tts_warmup"] = warm.lower() in ("1", "true", "yes", "on")
    for float_field in ("llm_temperature", "llm_top_p", "http_timeout", "tts_speed"):
        val = os.environ.get(
            {
                "llm_temperature": "LLM_TEMPERATURE",
                "llm_top_p": "LLM_TOP_P",
                "http_timeout": "HTTP_TIMEOUT",
                "tts_speed": "TTS_SPEED",
            }[float_field],
            "",
        ).strip()
        if val:
            try:
                out[float_field] = float(val)
            except ValueError:
                pass
    for int_field, env_name in (
        ("llm_top_k", "LLM_TOP_K"),
        ("llm_max_tokens", "LLM_MAX_TOKENS"),
        ("downlink_sample_rate", "DOWNLINK_SAMPLE_RATE"),
        ("tts_num_step", "TTS_NUM_STEP"),
        ("tts_max_chunk_chars", "TTS_MAX_CHUNK_CHARS"),
        ("tts_first_chunk_chars", "TTS_FIRST_CHUNK_CHARS"),
        ("tts_min_segment_chars", "TTS_MIN_SEGMENT_CHARS"),
        ("tts_segment_pad_ms", "TTS_SEGMENT_PAD_MS"),
        ("tts_post_llm_delay_ms", "TTS_POST_LLM_DELAY_MS"),
    ):
        val = os.environ.get(env_name, "").strip()
        if val:
            try:
                out[int_field] = int(val)
            except ValueError:
                pass
    return out


def _load_file(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError) as e:
        logger.warning("could not read config %s: %s", path, e)
        return {}


def _merge_dict(base: BridgeSettings, patch: dict[str, Any]) -> BridgeSettings:
    current = asdict(base)
    for key, value in patch.items():
        if key not in current:
            continue
        if value is None:
            continue
        if isinstance(current[key], bool):
            if isinstance(value, bool):
                current[key] = value
            else:
                current[key] = str(value).strip().lower() in ("1", "true", "yes", "on")
        elif isinstance(current[key], int):
            try:
                current[key] = int(value)
            except (TypeError, ValueError):
                continue
        elif isinstance(current[key], float):
            try:
                current[key] = float(value)
            except (TypeError, ValueError):
                continue
        elif key in ("system_prompt", "llm_think_mode", "active_llm_profile_id"):
            current[key] = str(value)
        elif key == "mcp_servers" and isinstance(value, list):
            current[key] = value
        elif key == "llm_profiles" and isinstance(value, list):
            current[key] = value
        elif key == "builtin_tools" and isinstance(value, dict):
            current[key] = normalize_builtin_tools({**current.get("builtin_tools", {}), **value})
        else:
            current[key] = str(value).strip()
    return BridgeSettings(**current)


def load_settings(*, force: bool = False) -> BridgeSettings:
    global _CACHE
    with _LOCK:
        if _CACHE is not None and not force:
            return _CACHE

        path = config_path()
        settings = BridgeSettings()
        settings = _merge_dict(settings, _env_bootstrap())
        file_data = _load_file(path)
        if file_data:
            settings = _merge_dict(settings, file_data)
        else:
            # First boot: persist env bootstrap so file exists after install
            if any(_env_bootstrap().values()):
                save_settings(settings, write_if_missing_only=True)

        _CACHE = settings
        return settings


def save_settings(settings: BridgeSettings, *, write_if_missing_only: bool = False) -> None:
    global _CACHE
    path = config_path()
    with _LOCK:
        if write_if_missing_only and path.is_file():
            _CACHE = settings
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(settings), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        _CACHE = settings
        logger.info("saved config to %s", path)


def apply_patch(patch: dict[str, Any]) -> BridgeSettings:
    """Merge partial update; empty api_key fields keep existing secrets."""
    current = load_settings()
    data = asdict(current)
    secret_fields = ("openai_api_key", "stt_api_key", "tts_api_key", "llm_api_key")
    for key in secret_fields:
        if key in patch and not str(patch.get(key, "")).strip():
            patch.pop(key, None)
    updated = _merge_dict(current, patch)
    save_settings(updated)
    return updated


def settings_for_api(settings: BridgeSettings | None = None) -> dict[str, Any]:
    s = settings or load_settings()
    data = asdict(s)
    for key in ("openai_api_key", "stt_api_key", "tts_api_key", "llm_api_key"):
        data[f"{key}_set"] = bool(_nonempty(data.pop(key)))
    return data


def settings_public_summary() -> dict[str, Any]:
    s = load_settings()
    return {
        "configured": bool(s.stt_url() and s.tts_url() and s.llm_url() and s.llm_model),
        "stt_base_url": s.stt_base_url or s.openai_base_url,
        "tts_base_url": s.tts_base_url or s.openai_base_url,
        "llm_base_url": s.llm_base_url or s.openai_base_url,
        "llm_model": s.llm_model,
        "stt_language": s.stt_language,
    }
