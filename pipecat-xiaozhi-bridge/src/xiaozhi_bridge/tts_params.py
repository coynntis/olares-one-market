"""TTS request builders — Melo / Sherpa / Kokoro / OmniVoice."""

from __future__ import annotations

import logging
import re
from typing import Any

from xiaozhi_bridge import voice_store
from xiaozhi_bridge.config import BridgeSettings, load_settings

logger = logging.getLogger(__name__)

MELO_CLUSTER_URL = "http://pipecatxiaozhimelo:8000/v1"
SHERPA_CLUSTER_URL = "http://pipecatxiaozhisherpa:10500/v1"
KOKORO_CLUSTER_URL = "http://pipecatxiaozhikokoro:8880/v1"
AUDIO8_CLUSTER_URL = "http://pipecatxiaozhiaudio8:8024/v1"

# Sherpa HA addon OpenAI shim only accepts voice=speakerN (see ptbsare/sherpa-onnx-tts-stt api.py).
# Actual speaker map for kokoro-int8-multi-lang-v1_1 (Kokoro v1.1-zh, 103 speakers) —
# NOT classic Kokoro af_heart/bm_lewis (those are Kokoro-FastAPI / hexgrad 82M).
SHERPA_KOKORO_VOICE_SID: dict[str, int] = {
    "af_maple": 0,
    "af_sol": 1,
    "bf_vale": 2,
    "zf_001": 3,
    "zf_002": 4,
    "zf_003": 5,
    "zf_004": 6,
    "zf_005": 7,
    "zf_006": 8,
    "zf_007": 9,
    "zf_008": 10,
    "zf_017": 11,
    "zf_018": 12,
    "zf_019": 13,
    "zf_021": 14,
    "zf_022": 15,
    "zf_023": 16,
    "zf_024": 17,
    "zf_026": 18,
    "zf_027": 19,
    "zf_028": 20,
    "zf_032": 21,
    "zf_036": 22,
    "zf_038": 23,
    "zf_039": 24,
    "zf_040": 25,
    "zf_042": 26,
    "zf_043": 27,
    "zf_044": 28,
    "zf_046": 29,
    "zf_047": 30,
    "zf_048": 31,
    "zf_049": 32,
    "zf_051": 33,
    "zf_059": 34,
    "zf_060": 35,
    "zf_067": 36,
    "zf_070": 37,
    "zf_071": 38,
    "zf_072": 39,
    "zf_073": 40,
    "zf_074": 41,
    "zf_075": 42,
    "zf_076": 43,
    "zf_077": 44,
    "zf_078": 45,
    "zf_079": 46,
    "zf_083": 47,
    "zf_084": 48,
    "zf_085": 49,
    "zf_086": 50,
    "zf_087": 51,
    "zf_088": 52,
    "zf_090": 53,
    "zf_092": 54,
    "zf_093": 55,
    "zf_094": 56,
    "zf_099": 57,
    "zm_009": 58,
    "zm_010": 59,
    "zm_011": 60,
    "zm_012": 61,
    "zm_013": 62,
    "zm_014": 63,
    "zm_015": 64,
    "zm_016": 65,
    "zm_020": 66,
    "zm_025": 67,
    "zm_029": 68,
    "zm_030": 69,
    "zm_031": 70,
    "zm_033": 71,
    "zm_034": 72,
    "zm_035": 73,
    "zm_037": 74,
    "zm_041": 75,
    "zm_045": 76,
    "zm_050": 77,
    "zm_052": 78,
    "zm_053": 79,
    "zm_054": 80,
    "zm_055": 81,
    "zm_056": 82,
    "zm_057": 83,
    "zm_058": 84,
    "zm_061": 85,
    "zm_062": 86,
    "zm_063": 87,
    "zm_064": 88,
    "zm_065": 89,
    "zm_066": 90,
    "zm_068": 91,
    "zm_069": 92,
    "zm_080": 93,
    "zm_081": 94,
    "zm_082": 95,
    "zm_089": 96,
    "zm_091": 97,
    "zm_095": 98,
    "zm_096": 99,
    "zm_097": 100,
    "zm_098": 101,
    "zm_100": 102,
}

# Favorites shown first in Sherpa UI (EN named voices + common ZH).
_SHERPA_VOICE_FAVORITES: list[tuple[str, str]] = [
    ("bf_vale", "bf_vale — British female (best EN in Sherpa pack)"),
    ("af_maple", "af_maple — American female"),
    ("af_sol", "af_sol — American female"),
    ("zm_009", "zm_009 — Mandarin male"),
    ("zm_010", "zm_010 — Mandarin male"),
    ("zf_001", "zf_001 — Mandarin female"),
    ("zf_002", "zf_002 — Mandarin female"),
]


def sherpa_voice_choices() -> list[tuple[str, str]]:
    """Favorites first, then remaining v1.1-zh speakers by sid."""
    fav_keys = {k for k, _ in _SHERPA_VOICE_FAVORITES}
    out = list(_SHERPA_VOICE_FAVORITES)
    for name, sid in sorted(SHERPA_KOKORO_VOICE_SID.items(), key=lambda x: x[1]):
        if name in fav_keys:
            continue
        prefix = "Mandarin female" if name.startswith("zf_") else (
            "Mandarin male" if name.startswith("zm_") else "speaker"
        )
        out.append((name, f"{name} — {prefix} (sid {sid})"))
    return out


SHERPA_VOICE_CHOICES = sherpa_voice_choices()

# Classic Kokoro-82M names — Kokoro-FastAPI only (not Sherpa v1.1-zh).
KOKORO_FASTAPI_VOICE_CHOICES: list[tuple[str, str]] = [
    ("bm_lewis", "bm_lewis — Jarvis-like British male"),
    ("bm_george", "bm_george — British male"),
    ("bm_daniel", "bm_daniel — British male"),
    ("am_michael", "am_michael — American male"),
    ("am_adam", "am_adam — American male"),
    ("af_heart", "af_heart — American female (default)"),
    ("af_bella", "af_bella — American female"),
    ("af_sky", "af_sky — American female"),
    ("bf_emma", "bf_emma — British female"),
    ("zf_xiaoxiao", "zf_xiaoxiao — Mandarin female"),
    ("zf_xiaoyi", "zf_xiaoyi — Mandarin female"),
    ("zm_yunxi", "zm_yunxi — Mandarin male"),
    ("zm_yunyang", "zm_yunyang — Mandarin male"),
]

# Back-compat alias used by older imports/tests.
KOKORO_VOICE_CHOICES = KOKORO_FASTAPI_VOICE_CHOICES


_CPU_PROVIDERS = frozenset({"melo", "sherpa", "kokoro", "audio8"})
_CLONE_PROVIDERS = frozenset({"omnivoice"})
_SPEED_PROVIDERS = frozenset({"melo", "sherpa", "kokoro", "omnivoice"})


AUDIO8_VOICE_CHOICES = [
    ("en_default", "en_default — English demo ref"),
    ("zh_default", "zh_default — Mandarin-style (EN ref)"),
    ("yue_default", "yue_default — Cantonese demo ref"),
]


def normalize_tts_language(lang: str) -> str:
    """Map STT/Olares aliases to language codes."""
    code = (lang or "").strip().lower().replace("_", "-")
    if code in ("yue", "cantonese", "hk", "zh-hk", "yue-hant", "zh-yue"):
        return "yue"
    if code in ("zh", "zh-cn", "mandarin", "cmn"):
        return "zh"
    if code in ("en", "en-us", "en-gb", "english"):
        return "en"
    return code


def effective_tts_language(cfg: BridgeSettings) -> str:
    """Prefer explicit TTS language; align with STT when TTS unset."""
    tts = normalize_tts_language(cfg.tts_language_id)
    stt = normalize_tts_language(cfg.stt_language)
    if tts:
        return tts
    return stt or "en"


def effective_tts_provider(cfg: BridgeSettings) -> str:
    """melo | sherpa | kokoro | audio8 | omnivoice."""
    raw = (cfg.tts_provider or "sherpa").strip().lower()
    if raw not in ("melo", "sherpa", "kokoro", "audio8", "omnivoice"):
        raw = "sherpa"
    return raw


def provider_supports_speed(provider: str) -> bool:
    return (provider or "").strip().lower() in _SPEED_PROVIDERS


def is_cpu_tts_provider(cfg: BridgeSettings) -> bool:
    return effective_tts_provider(cfg) in _CPU_PROVIDERS


def provider_supports_clone(provider: str) -> bool:
    return (provider or "").strip().lower() in _CLONE_PROVIDERS


def default_cluster_tts_url(provider: str) -> str:
    if provider == "sherpa":
        return SHERPA_CLUSTER_URL
    if provider == "kokoro":
        return KOKORO_CLUSTER_URL
    if provider == "audio8":
        return AUDIO8_CLUSTER_URL
    if provider == "melo":
        return MELO_CLUSTER_URL
    return MELO_CLUSTER_URL


def chunk_limits_for_provider(cfg: BridgeSettings) -> tuple[int, int]:
    """(first_chars, max_chars).

    CPU TTS (Melo/Sherpa/Kokoro): effectively no bridge-side split — one utterance.
    Those engines already sentence-split internally; tiny bridge chunks sound broken.
    OmniVoice: keep short first chunk for GPU warm.
    """
    max_chars = max(8, int(cfg.tts_max_chunk_chars or 40))
    first_chars = max(4, int(cfg.tts_first_chunk_chars or 12))
    if is_cpu_tts_provider(cfg):
        # User can still force splits via Settings if they set huge explicit values;
        # defaults / small leftovers → single chunk.
        if max_chars < 2000:
            max_chars = 100_000
        if first_chars < 2000:
            first_chars = max_chars
    return first_chars, max_chars


def cpu_tts_single_utterance(cfg: BridgeSettings) -> bool:
    """True when bridge should send the whole reply as one TTS POST."""
    if not is_cpu_tts_provider(cfg):
        return False
    # Explicit tiny max_chars in settings still allows experimental splitting.
    return int(cfg.tts_max_chunk_chars or 0) < 2000


def melo_voice_for_language(cfg: BridgeSettings) -> str:
    """Map language → Melo voice id (EN-US / ZH). Yue falls back to ZH with warning."""
    preset = (cfg.tts_voice or "").strip()
    allowed = {
        "EN-Default",
        "EN-US",
        "EN-BR",
        "EN_INDIA",
        "EN-AU",
        "ZH",
        "ES",
        "FR",
        "JP",
        "KR",
    }
    if preset in allowed:
        return preset
    if preset.upper() == "EN-DEFAULT":
        return "EN-Default"
    lang = effective_tts_language(cfg)
    if lang == "yue":
        logger.warning(
            "MeloTTS has no Cantonese — using ZH; switch TTS provider to OmniVoice for yue"
        )
        return "ZH"
    if lang == "zh":
        return "ZH"
    return "EN-US"


def sherpa_voice_field(cfg: BridgeSettings) -> str:
    """Map UI voice name → sherpa OpenAI voice=speakerN (v1.1-zh sid map)."""
    raw = (cfg.tts_voice or "").strip()
    if not raw or raw.lower() in ("default", "auto"):
        lang = effective_tts_language(cfg)
        if lang == "zh":
            return "speaker58"  # zm_009
        return "speaker2"  # bf_vale — best EN-named voice in Sherpa pack
    m = re.match(r"(?i)^speaker(\d+)$", raw)
    if m:
        return f"speaker{int(m.group(1))}"
    if raw.isdigit():
        return f"speaker{int(raw)}"
    key = raw.lower()
    # Legacy classic Kokoro names → closest Sherpa sid (avoid silent wrong voice).
    legacy = {
        "bm_lewis": 2,
        "bm_george": 2,
        "bm_daniel": 2,
        "bf_emma": 2,
        "af_heart": 0,
        "af_bella": 1,
        "af_sky": 1,
        "am_michael": 58,
        "am_adam": 58,
        "zf_xiaoxiao": 3,
        "zf_xiaoyi": 4,
        "zm_yunxi": 58,
        "zm_yunyang": 59,
    }
    if key in SHERPA_KOKORO_VOICE_SID:
        return f"speaker{SHERPA_KOKORO_VOICE_SID[key]}"
    if key in legacy:
        logger.warning(
            "Sherpa voice %r is classic Kokoro name — mapped to speaker%d (use Kokoro-FastAPI for real %s)",
            raw,
            legacy[key],
            key,
        )
        return f"speaker{legacy[key]}"
    logger.warning("Unknown sherpa voice %r — using speaker2 (bf_vale)", raw)
    return "speaker2"


def kokoro_voice_for_language(cfg: BridgeSettings) -> str:
    preset = (cfg.tts_voice or "").strip()
    if preset and preset.lower() not in ("default", "auto"):
        return preset
    lang = effective_tts_language(cfg)
    if lang == "zh":
        return "zm_yunxi"
    return "bm_lewis"


def resolve_clone_ref(cfg: BridgeSettings, *, voice_id: str | None = None) -> tuple[str, str, str] | None:
    """Return (ref_audio_b64, ref_text, language_id) when a saved profile exists."""
    vid = (voice_id or cfg.tts_active_voice_id or "").strip()
    if not vid:
        return None
    profile = voice_store.get_voice(vid)
    if not profile:
        return None
    ref_b64 = voice_store.read_voice_audio_b64(vid)
    ref_text = str(profile.get("ref_text") or "").strip()
    if not ref_b64 or not ref_text:
        return None
    lang = str(profile.get("language_id") or effective_tts_language(cfg) or "").strip()
    lang = normalize_tts_language(lang)
    return ref_b64, ref_text, lang


def _speech_voice_field(cfg: BridgeSettings) -> str:
    mode = (cfg.tts_voice_mode or "instruct").strip().lower()
    preset = cfg.tts_voice.strip()
    instruct = cfg.tts_instruct.strip()
    if mode == "default":
        return preset or "auto"
    if mode == "instruct":
        return instruct or preset or "auto"
    return instruct or preset or "auto"


def build_melo_speech_json(text: str, cfg: BridgeSettings) -> dict[str, Any]:
    """POST /v1/audio/speech for MeloTTS-FastAPI."""
    body: dict[str, Any] = {
        "input": text,
        "model": cfg.tts_model or "tts",
        "response_format": cfg.tts_response_format.strip().lower() or "wav",
        "voice": melo_voice_for_language(cfg),
    }
    if cfg.tts_speed and cfg.tts_speed > 0:
        body["speed"] = float(cfg.tts_speed)
    return body


def build_sherpa_speech_json(text: str, cfg: BridgeSettings) -> dict[str, Any]:
    """POST /v1/audio/speech for sherpa-onnx OpenAI-compatible port."""
    body: dict[str, Any] = {
        "input": text,
        "model": cfg.tts_model or "kokoro-int8-multi-lang-v1_1",
        "response_format": cfg.tts_response_format.strip().lower() or "wav",
        "voice": sherpa_voice_field(cfg),
    }
    if cfg.tts_speed and cfg.tts_speed > 0:
        body["speed"] = float(cfg.tts_speed)
    return body


def build_kokoro_speech_json(text: str, cfg: BridgeSettings) -> dict[str, Any]:
    """POST /v1/audio/speech for Kokoro-FastAPI."""
    body: dict[str, Any] = {
        "input": text,
        "model": cfg.tts_model or "kokoro",
        "response_format": cfg.tts_response_format.strip().lower() or "wav",
        "voice": kokoro_voice_for_language(cfg),
    }
    if cfg.tts_speed and cfg.tts_speed > 0:
        body["speed"] = float(cfg.tts_speed)
    return body


def audio8_voice_field(cfg: BridgeSettings) -> str:
    """Registered voice name for Audio8 ONNX (seed: en_default / zh_default / yue_default)."""
    raw = (cfg.tts_voice or "").strip() or "en_default"
    allowed = {v for v, _ in AUDIO8_VOICE_CHOICES}
    if raw in allowed:
        return raw
    # Allow user-registered custom names (one path component).
    if raw and raw not in {".", ".."} and "/" not in raw and len(raw) <= 64:
        return raw
    return "en_default"


def build_audio8_speech_json(text: str, cfg: BridgeSettings) -> dict[str, Any]:
    """POST /v1/audio/speech for Audio8 ONNX INT4 (no speed field)."""
    model = (cfg.tts_model or "").strip()
    if model not in {"arktts", "tts-1"}:
        model = "arktts"
    fmt = (cfg.tts_response_format or "wav").strip().lower() or "wav"
    if fmt not in {"wav", "pcm"}:
        fmt = "wav"
    return {
        "input": text,
        "model": model,
        "response_format": fmt,
        "voice": audio8_voice_field(cfg),
    }


def build_speech_json(text: str, cfg: BridgeSettings) -> dict[str, Any]:
    """POST /v1/audio/speech — OmniVoice `voice` field is preset or instruct string."""
    body: dict[str, Any] = {
        "input": text,
        "model": cfg.tts_model or "omnivoice",
        "response_format": cfg.tts_response_format.strip().lower() or "wav",
        "voice": _speech_voice_field(cfg),
        "num_step": max(8, int(cfg.tts_num_step or 16)),
        "class_temperature": 0.0,
        "position_temperature": 0.0,
    }
    lang = effective_tts_language(cfg)
    if lang:
        body["language_id"] = lang
    if cfg.tts_speed and cfg.tts_speed > 0:
        body["speed"] = float(cfg.tts_speed)
    return body


def build_clone_form(text: str, cfg: BridgeSettings, ref_b64: str, ref_text: str, lang: str) -> dict[str, str]:
    """POST /v1/audio/clone — multipart form; field is ref_audio_base64 (not ref_audio_b64)."""
    form: dict[str, str] = {
        "text": text,
        "ref_audio_base64": ref_b64,
        "ref_text": ref_text,
        "num_step": str(max(8, int(cfg.tts_num_step or 16))),
        "response_format": cfg.tts_response_format.strip().lower() or "wav",
        "class_temperature": "0",
        "position_temperature": "0",
        "preprocess_prompt": "true",
        "postprocess_output": "true",
    }
    if lang:
        form["language_id"] = lang
    if cfg.tts_speed and cfg.tts_speed > 0:
        form["speed"] = str(float(cfg.tts_speed))
    return form


def segmenter_min_chars(cfg: BridgeSettings | None = None) -> int:
    s = cfg or load_settings()
    base = max(4, int(s.tts_min_segment_chars or 12))
    if is_cpu_tts_provider(s) and base < 24:
        return 24
    return base


def build_tts_request_body(text: str, cfg: BridgeSettings | None = None) -> dict:
    s = cfg or load_settings()
    provider = effective_tts_provider(s)
    if provider == "melo":
        return {"endpoint": "speech", **build_melo_speech_json(text, s)}
    if provider == "sherpa":
        return {"endpoint": "speech", **build_sherpa_speech_json(text, s)}
    if provider == "kokoro":
        return {"endpoint": "speech", **build_kokoro_speech_json(text, s)}
    if provider == "audio8":
        return {"endpoint": "speech", **build_audio8_speech_json(text, s)}
    clone = resolve_clone_ref(s)
    if clone and (s.tts_voice_mode or "").strip().lower() == "clone":
        ref_b64, ref_text, lang = clone
        return {"endpoint": "clone", **build_clone_form(text, s, ref_b64, ref_text, lang)}
    return build_speech_json(text, s)
