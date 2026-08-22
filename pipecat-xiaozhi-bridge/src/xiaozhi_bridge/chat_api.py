"""REST chat for web UI — text, optional vision image, optional TTS."""

from __future__ import annotations

import asyncio
import base64
import binascii
import io
import json
import logging
import re
import time
import wave
from typing import Any

from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from xiaozhi_bridge import chat_store
from xiaozhi_bridge.config import load_settings
from xiaozhi_bridge.llm_capabilities import model_likely_supports_vision
from xiaozhi_bridge.pipecat_llm import MAX_TOOL_ROUNDS, complete_agent_chat
from xiaozhi_bridge.mcp_tools import tools_configured
from xiaozhi_bridge.server import synthesize_speech_pcm

MAX_IMAGE_BYTES = 8 * 1024 * 1024
DATA_URL_RE = re.compile(r"^data:(image/[a-zA-Z0-9.+-]+);base64,(.+)$", re.DOTALL)
CHAT_TTS_TIMEOUT_SECONDS = 15.0

logger = logging.getLogger(__name__)


def _chat_llm_timeout() -> float:
    cfg = load_settings()
    return max(float(cfg.http_timeout or 120.0), 30.0)


def _parse_image(image: str | None) -> tuple[str, str] | None:
    if not image or not str(image).strip():
        return None
    raw = str(image).strip()
    if raw.startswith("data:"):
        m = DATA_URL_RE.match(raw)
        if not m:
            return None
        mime, b64 = m.group(1), m.group(2)
    else:
        mime, b64 = "image/jpeg", raw
    try:
        data = base64.b64decode(b64, validate=True)
    except (binascii.Error, ValueError):
        return None
    if len(data) > MAX_IMAGE_BYTES:
        raise ValueError("image too large (max 8MB)")
    return mime, b64


def _build_user_content(text: str, image: str | None) -> str | list[dict[str, Any]]:
    parsed = _parse_image(image)
    if not parsed:
        return text
    mime, b64 = parsed
    parts: list[dict[str, Any]] = [
        {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
        {"type": "text", "text": text or "Describe this image."},
    ]
    return parts


async def post_chat(request: Request) -> Response:
    try:
        body = await request.json()
    except json.JSONDecodeError:
        return JSONResponse({"error": "invalid json"}, status_code=400)
    if not isinstance(body, dict):
        return JSONResponse({"error": "expected object"}, status_code=400)

    text = str(body.get("text") or "").strip()
    image = body.get("image")
    speak = bool(body.get("speak"))
    conversation_id = str(body.get("conversation_id") or "").strip() or None
    device_id = str(body.get("device_id") or "").strip() or None

    if not text and not image:
        return JSONResponse({"error": "text or image required"}, status_code=400)

    if not device_id:
        return JSONResponse({"error": "device_id required"}, status_code=400)

    cfg = load_settings()
    if not cfg.llm_url() or not cfg.llm_model.strip():
        return JSONResponse(
            {"error": "LLM not configured — set base URL and model in Settings"},
            status_code=503,
        )

    try:
        _parse_image(image if isinstance(image, str) else None)
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=400)

    try:
        conv = await asyncio.to_thread(
            chat_store.resolve_device_conversation, device_id, conversation_id
        )
    except PermissionError as e:
        return JSONResponse({"error": str(e)}, status_code=403)
    except LookupError as e:
        return JSONResponse({"error": str(e)}, status_code=404)
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=400)
    cid = conv["id"]
    parsed_image = _parse_image(image if isinstance(image, str) else None)
    vision_warning: str | None = None
    if parsed_image and not model_likely_supports_vision(cfg.llm_model):
        vision_warning = (
            f"Model {cfg.llm_model!r} may not support images. "
            "Use a vision-capable LLM (e.g. Qwen-VL, GPT-4o, Gemma vision) to analyze photos."
        )
        logger.warning("chat image attached but model may not support vision: %s", cfg.llm_model)

    user_msg = await asyncio.to_thread(
        chat_store.append_message,
        cid,
        role="user",
        text=text or "(image)",
        image_data_url=image if isinstance(image, str) else None,
        source="text",
    )

    messages = await asyncio.to_thread(chat_store.llm_messages_for_conversation, cid, cfg.system_prompt)

    llm_started = time.perf_counter()
    has_mcp = tools_configured()
    llm_timeout = _chat_llm_timeout() * (min(MAX_TOOL_ROUNDS, 4) if has_mcp else 1)
    try:
        llm_result = await asyncio.wait_for(
            complete_agent_chat(messages, device_id=device_id),
            timeout=llm_timeout,
        )
    except asyncio.TimeoutError:
        return JSONResponse(
            {"error": f"LLM/tools timed out after {llm_timeout:.0f}s — try again or shorten the request"},
            status_code=504,
        )
    except Exception as e:
        return JSONResponse({"error": f"LLM failed: {e}"}, status_code=502)
    llm_elapsed = time.perf_counter() - llm_started
    reply = llm_result.text

    if llm_result.tool_trace or llm_result.generated_images:
        image_msgs = await asyncio.to_thread(
            chat_store.append_agent_trace,
            cid,
            llm_result.tool_trace or [],
            source="text",
            generated_images=llm_result.generated_images or [],
        )
        msgs = await asyncio.to_thread(chat_store.list_messages, cid, limit=50)
        # Prefer final text reply (not the generated-image bubble).
        assistant_row = None
        for m in reversed(msgs):
            if m.get("role") != "assistant":
                continue
            meta = m.get("meta") if isinstance(m.get("meta"), dict) else {}
            if meta.get("generated_image") or meta.get("tool_calls"):
                continue
            if str(m.get("text") or "").strip():
                assistant_row = m
                break
        assistant_msg = assistant_row or {
            "id": "",
            "role": "assistant",
            "text": reply,
            "conversation_id": cid,
            "created_at": int(time.time() * 1000),
        }
    else:
        image_msgs = []
        assistant_msg = await asyncio.to_thread(
            chat_store.append_message,
            cid,
            role="assistant",
            text=reply,
            source="text",
        )

    out: dict[str, Any] = {
        "text": reply,
        "conversation_id": cid,
        "user_message": user_msg,
        "assistant_message": assistant_msg,
        "stats": {
            "llm_ms": llm_result.elapsed_ms or int(llm_elapsed * 1000),
            "prompt_tokens": llm_result.usage.prompt_tokens,
            "completion_tokens": llm_result.usage.completion_tokens,
            "tokens_per_sec": llm_result.tokens_per_sec,
            "backend": llm_result.backend,
            "tool_rounds": llm_result.tool_rounds,
        },
        "agent_steps": [s.to_dict() for s in llm_result.agent_steps],
        "generated_images": image_msgs,
    }
    if image_msgs:
        logger.info(
            "chat generated_images count=%d urls=%s",
            len(image_msgs),
            [m.get("image_url") for m in image_msgs],
        )
    if vision_warning:
        out["vision_warning"] = vision_warning
    if speak and reply.strip():
        if not cfg.tts_url():
            out["tts_error"] = "TTS not configured"
        else:
            try:
                tts_started = time.perf_counter()
                pcm = await asyncio.wait_for(
                    synthesize_speech_pcm(reply, output_sample_rate=cfg.downlink_sample_rate),
                    timeout=min(CHAT_TTS_TIMEOUT_SECONDS, float(cfg.http_timeout or CHAT_TTS_TIMEOUT_SECONDS)),
                )
                buf = io.BytesIO()
                with wave.open(buf, "wb") as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(cfg.downlink_sample_rate)
                    wf.writeframes(pcm)
                out["audio_wav_b64"] = base64.b64encode(buf.getvalue()).decode("ascii")
                out["stats"]["tts_ms"] = int((time.perf_counter() - tts_started) * 1000)
            except asyncio.TimeoutError:
                out["tts_error"] = f"TTS timeout after {CHAT_TTS_TIMEOUT_SECONDS:.0f}s"
            except Exception as e:
                out["tts_error"] = str(e)
    out["stats"]["total_ms"] = int(
        (out["stats"].get("llm_ms") or 0) + (out["stats"].get("tts_ms") or 0)
    )
    logger.info(
        "chat complete conv=%s speak=%s stats=%s tts=%s",
        cid,
        speak,
        out.get("stats"),
        "ok" if "audio_wav_b64" in out else out.get("tts_error", "skip"),
    )

    return JSONResponse(out)
