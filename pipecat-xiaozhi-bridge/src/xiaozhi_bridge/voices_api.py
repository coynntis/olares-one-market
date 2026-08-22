"""REST API for voice clone profiles."""

from __future__ import annotations

import asyncio
import base64
import io
import logging
import time
import wave
from typing import Any

from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from xiaozhi_bridge import voice_store
from xiaozhi_bridge.config import apply_patch, load_settings
from xiaozhi_bridge.tts_client import synthesize_speech_pcm

logger = logging.getLogger(__name__)
PREVIEW_TIMEOUT = 90.0


async def list_voices_handler(_: Request) -> Response:
    voices = await asyncio.to_thread(voice_store.list_voices)
    cfg = load_settings()
    return JSONResponse(
        {
            "voices": voices,
            "active_voice_id": cfg.tts_active_voice_id,
            "voice_mode": cfg.tts_voice_mode,
        }
    )


async def create_voice_handler(request: Request) -> Response:
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "invalid json"}, status_code=400)
    if not isinstance(body, dict):
        return JSONResponse({"error": "expected object"}, status_code=400)

    name = str(body.get("name") or "").strip()
    ref_text = str(body.get("ref_text") or "").strip()
    audio = body.get("audio")
    if not isinstance(audio, str) or not audio.strip():
        return JSONResponse({"error": "audio required (base64 or data URL)"}, status_code=400)
    language_id = str(body.get("language_id") or "").strip()
    instruct = str(body.get("instruct") or "").strip()
    set_active = bool(body.get("set_active"))

    try:
        voice = await asyncio.to_thread(
            voice_store.create_voice,
            name=name,
            ref_text=ref_text,
            audio_data=audio,
            language_id=language_id,
            instruct=instruct,
        )
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=400)

    if set_active:
        apply_patch({"tts_active_voice_id": voice["id"], "tts_voice_mode": "clone"})

    return JSONResponse({"voice": voice}, status_code=201)


async def delete_voice_handler(request: Request) -> Response:
    voice_id = str(request.path_params.get("voice_id") or "").strip()
    if not voice_id:
        return JSONResponse({"error": "voice_id required"}, status_code=400)
    deleted = await asyncio.to_thread(voice_store.delete_voice, voice_id)
    if not deleted:
        return JSONResponse({"error": "not found"}, status_code=404)
    cfg = load_settings()
    if cfg.tts_active_voice_id == voice_id:
        apply_patch({"tts_active_voice_id": ""})
    return JSONResponse({"ok": True})


async def get_voice_audio_handler(request: Request) -> Response:
    voice_id = str(request.path_params.get("voice_id") or "").strip()
    data = await asyncio.to_thread(voice_store.read_voice_audio, voice_id)
    if not data:
        return JSONResponse({"error": "not found"}, status_code=404)
    return Response(data, media_type="audio/wav")


async def activate_voice_handler(request: Request) -> Response:
    voice_id = str(request.path_params.get("voice_id") or "").strip()
    if not voice_id:
        return JSONResponse({"error": "voice_id required"}, status_code=400)
    voice = await asyncio.to_thread(voice_store.get_voice, voice_id)
    if not voice:
        return JSONResponse({"error": "not found"}, status_code=404)
    apply_patch({"tts_active_voice_id": voice_id})
    return JSONResponse({"ok": True, "active_voice_id": voice_id})


async def preview_voice_handler(request: Request) -> Response:
    voice_id = str(request.path_params.get("voice_id") or "").strip()
    voice = await asyncio.to_thread(voice_store.get_voice, voice_id)
    if not voice:
        return JSONResponse({"error": "not found"}, status_code=404)

    try:
        body = await request.json()
    except Exception:
        body = {}
    sample = str(voice.get("ref_text") or "").strip() or "你好，我是 Agent R。"
    if isinstance(body, dict) and str(body.get("text") or "").strip():
        sample = str(body["text"]).strip()

    cfg = load_settings()
    started = time.perf_counter()
    try:
        pcm = await asyncio.wait_for(
            synthesize_speech_pcm(
                sample,
                output_sample_rate=cfg.downlink_sample_rate,
                voice_id=voice_id,
            ),
            timeout=PREVIEW_TIMEOUT,
        )
    except Exception as e:
        logger.exception("voice preview failed voice_id=%s", voice_id)
        return JSONResponse({"error": f"TTS preview failed: {e}"}, status_code=502)

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(cfg.downlink_sample_rate)
        wf.writeframes(pcm)
    elapsed_ms = int((time.perf_counter() - started) * 1000)
    return JSONResponse(
        {
            "audio_wav_b64": base64.b64encode(buf.getvalue()).decode("ascii"),
            "text": sample,
            "tts_ms": elapsed_ms,
        }
    )
