"""Probe STT / TTS / LLM shared entrances from the bridge pod."""

from __future__ import annotations

import io
import logging
import time
import wave
from typing import Any

import httpx

from xiaozhi_bridge.audio import speech_bytes_to_pcm
from xiaozhi_bridge.config import BridgeSettings, load_settings
from xiaozhi_bridge.openai_endpoints import auth_headers, llm_api_key, stt_api_key, tts_api_key
from xiaozhi_bridge.tts_params import (
    build_audio8_speech_json,
    build_clone_form,
    build_kokoro_speech_json,
    build_melo_speech_json,
    build_sherpa_speech_json,
    build_speech_json,
    default_cluster_tts_url,
    effective_tts_provider,
    resolve_clone_ref,
)

logger = logging.getLogger(__name__)

# Longer than "connection test" so RTF isn't dominated by fixed HTTP overhead.
_TTS_PROBE_TEXT = (
    "This is a connection and real-time factor test for the text to speech pipeline."
)
_TTS_PROBE_TEXT_ZH = "这是一次语音合成连接与实时率测试，用于测量合成延迟。"


def _silent_wav_bytes(duration_ms: int = 200, sample_rate: int = 16000) -> bytes:
    n = int(sample_rate * duration_ms / 1000)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(b"\x00\x00" * n)
    return buf.getvalue()


def readiness(s: BridgeSettings | None = None) -> dict[str, Any]:
    cfg = s or load_settings()
    missing: list[str] = []
    if not cfg.stt_url():
        missing.append("stt_base_url")
    if not cfg.tts_url():
        missing.append("tts_base_url")
    if not cfg.llm_url():
        missing.append("llm_base_url")
    if not cfg.llm_model.strip():
        missing.append("llm_model")
    return {
        "ready": len(missing) == 0,
        "missing": missing,
        "stt_base_url": cfg.stt_base_url or cfg.openai_base_url,
        "tts_base_url": cfg.tts_base_url or cfg.openai_base_url,
        "llm_base_url": cfg.llm_base_url or cfg.openai_base_url,
        "llm_model": cfg.llm_model,
        "stt_language": cfg.stt_language,
    }


async def _probe_get(url: str, headers: dict[str, str], timeout: float) -> tuple[int, str]:
    async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
        resp = await client.get(url, headers=headers)
        snippet = (resp.text or "")[:240]
        return resp.status_code, snippet


async def test_stt(cfg: BridgeSettings | None = None) -> dict[str, Any]:
    s = cfg or load_settings()
    base = s.stt_url()
    if not base:
        return {"ok": False, "service": "stt", "error": "STT base URL not set"}
    base = base.rstrip("/")
    timeout = s.http_timeout
    headers = auth_headers(stt_api_key())

    models_url = f"{base}/models"
    try:
        code, body = await _probe_get(models_url, headers, min(timeout, 30.0))
        if code < 400:
            return {"ok": True, "service": "stt", "url": models_url, "status": code, "detail": body}
    except Exception as e:
        logger.debug("stt models probe failed: %s", e)

    transcribe_url = f"{base}/audio/transcriptions"
    data = {"model": s.stt_model, "language": s.stt_language}
    wav = _silent_wav_bytes()
    try:
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
            resp = await client.post(
                transcribe_url,
                data=data,
                files={"file": ("test.wav", wav, "audio/wav")},
                headers=headers,
            )
        ok = resp.status_code < 400
        detail = (resp.text or "")[:240]
        return {
            "ok": ok,
            "service": "stt",
            "url": transcribe_url,
            "status": resp.status_code,
            "detail": detail,
            "error": None if ok else detail,
        }
    except Exception as e:
        return {"ok": False, "service": "stt", "url": transcribe_url, "error": str(e)}


async def test_tts(cfg: BridgeSettings | None = None) -> dict[str, Any]:
    s = cfg or load_settings()
    base = s.tts_url()
    provider = effective_tts_provider(s)
    if not base and provider in ("melo", "sherpa", "kokoro", "audio8"):
        base = default_cluster_tts_url(provider)
    if not base:
        return {"ok": False, "service": "tts", "error": "TTS base URL not set"}
    url_speech = f"{base.rstrip('/')}/audio/speech"
    clone = resolve_clone_ref(s)
    use_clone = (
        provider == "omnivoice"
        and clone
        and (s.tts_voice_mode or "").strip().lower() == "clone"
    )
    url = url_speech
    lang = (s.tts_language_id or "").strip().lower()
    probe = _TTS_PROBE_TEXT_ZH if lang.startswith("zh") or lang == "yue" else _TTS_PROBE_TEXT
    speed = float(s.tts_speed or 1.0)
    fmt = (s.tts_response_format or "wav").strip().lower() or "wav"
    try:
        http_started = time.perf_counter()
        async with httpx.AsyncClient(timeout=s.http_timeout, follow_redirects=True) as client:
            if use_clone and clone:
                ref_b64, ref_text, clone_lang = clone
                url = f"{base.rstrip('/')}/audio/clone"
                form = build_clone_form(probe, s, ref_b64, ref_text, clone_lang)
                resp = await client.post(url, data=form, headers=auth_headers(tts_api_key()))
            elif provider == "sherpa":
                resp = await client.post(
                    url_speech,
                    json=build_sherpa_speech_json(probe, s),
                    headers=auth_headers(tts_api_key()),
                )
            elif provider == "kokoro":
                resp = await client.post(
                    url_speech,
                    json=build_kokoro_speech_json(probe, s),
                    headers=auth_headers(tts_api_key()),
                )
            elif provider == "audio8":
                resp = await client.post(
                    url_speech,
                    json=build_audio8_speech_json(probe, s),
                    headers=auth_headers(tts_api_key()),
                )
            elif provider == "melo":
                resp = await client.post(
                    url_speech,
                    json=build_melo_speech_json(probe, s),
                    headers=auth_headers(tts_api_key()),
                )
            else:
                resp = await client.post(
                    url_speech,
                    json=build_speech_json(probe, s),
                    headers=auth_headers(tts_api_key()),
                )
        http_ms = int((time.perf_counter() - http_started) * 1000)
        raw = resp.content
        ok = resp.status_code < 400 and len(raw) > 100
        audio_ms = 0
        rtf = 0.0
        if ok:
            try:
                pcm, rate = speech_bytes_to_pcm(raw, response_format=fmt, target_rate=None)
                if rate > 0 and pcm:
                    audio_ms = int(len(pcm) / 2 / rate * 1000)
                    if audio_ms > 0:
                        rtf = round(http_ms / audio_ms, 4)
            except Exception as decode_err:
                logger.warning("tts probe decode failed: %s", decode_err)
        detail = (
            f"{len(raw)} bytes | http {http_ms}ms | audio {audio_ms}ms | RTF {rtf:.3f} | speed {speed:g}"
            if ok
            else (resp.text or "")[:240]
        )
        return {
            "ok": ok,
            "service": "tts",
            "url": url,
            "status": resp.status_code,
            "bytes": len(raw),
            "provider": provider,
            "speed": speed,
            "http_ms": http_ms,
            "audio_ms": audio_ms,
            "rtf": rtf,
            "detail": detail,
            "error": None if ok else (resp.text or "")[:240],
        }
    except Exception as e:
        return {"ok": False, "service": "tts", "url": url, "error": str(e)}


async def test_llm(cfg: BridgeSettings | None = None) -> dict[str, Any]:
    s = cfg or load_settings()
    base = s.llm_url()
    if not base:
        return {"ok": False, "service": "llm", "error": "LLM base URL not set"}
    if not s.llm_model.strip():
        return {"ok": False, "service": "llm", "error": "LLM model name not set"}
    url = f"{base.rstrip('/')}/models"
    try:
        code, body = await _probe_get(url, auth_headers(llm_api_key()), min(s.http_timeout, 30.0))
        ok = code < 400
        return {
            "ok": ok,
            "service": "llm",
            "url": url,
            "status": code,
            "model": s.llm_model,
            "detail": body,
            "error": None if ok else body,
        }
    except Exception as e:
        return {"ok": False, "service": "llm", "url": url, "error": str(e)}


async def test_all(cfg: BridgeSettings | None = None) -> dict[str, Any]:
    s = cfg or load_settings()
    results = {
        "stt": await test_stt(s),
        "tts": await test_tts(s),
        "llm": await test_llm(s),
    }
    ok = all(r.get("ok") for r in results.values())
    return {"ok": ok, "results": results}
