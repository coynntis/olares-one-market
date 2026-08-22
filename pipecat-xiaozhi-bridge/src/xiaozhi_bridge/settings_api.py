"""REST API for bridge pipeline configuration."""

from __future__ import annotations

import json
import os
from typing import Any

from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from xiaozhi_bridge.config import apply_patch, load_settings, settings_for_api

ALLOWED_PATCH_KEYS = frozenset(
    {
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
        "tts_num_step",
        "tts_speed",
        "tts_provider",
        "tts_overlap_llm",
        "tts_warmup",
        "tts_warmup_text",
        "tts_first_chunk_chars",
        "tts_max_chunk_chars",
        "tts_min_segment_chars",
        "tts_segment_pad_ms",
        "tts_post_llm_delay_ms",
        "llm_model",
        "llm_temperature",
        "llm_top_p",
        "llm_top_k",
        "llm_max_tokens",
        "llm_think_mode",
        "downlink_sample_rate",
        "system_prompt",
        "http_timeout",
        "mcp_servers",
        "builtin_tools",
        "llm_profiles",
        "active_llm_profile_id",
    }
)


def _settings_token() -> str | None:
    return os.environ.get("SETTINGS_TOKEN", "").strip() or None


def _authorized(request: Request) -> bool:
    token = _settings_token()
    if not token:
        return True
    auth = request.headers.get("authorization", "")
    if auth.startswith("Bearer "):
        return auth[7:].strip() == token
    return request.query_params.get("token", "").strip() == token


async def get_config(_: Request) -> Response:
    return JSONResponse(settings_for_api())


async def put_config(request: Request) -> Response:
    if not _authorized(request):
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    try:
        body = await request.json()
    except json.JSONDecodeError:
        return JSONResponse({"error": "invalid json"}, status_code=400)
    if not isinstance(body, dict):
        return JSONResponse({"error": "expected object"}, status_code=400)

    patch: dict[str, Any] = {k: body[k] for k in ALLOWED_PATCH_KEYS if k in body}
    if not patch:
        return JSONResponse({"error": "no recognized fields"}, status_code=400)

    updated = apply_patch(patch)
    from xiaozhi_bridge.pipecat_llm import invalidate_capabilities_cache
    from xiaozhi_bridge.mcp_tools import invalidate_mcp_tools_cache

    invalidate_capabilities_cache()
    if "mcp_servers" in patch or "builtin_tools" in patch:
        invalidate_mcp_tools_cache()
    return JSONResponse({"ok": True, "settings": settings_for_api(updated)})


async def config_status(_: Request) -> Response:
    from xiaozhi_bridge.connection_test import readiness

    return JSONResponse(readiness())


async def test_config_connection(request: Request) -> Response:
    import logging

    from xiaozhi_bridge.connection_test import test_all, test_llm, test_stt, test_tts
    from xiaozhi_bridge.llm_capabilities import fetch_llm_capabilities
    from xiaozhi_bridge.pipecat_llm import invalidate_capabilities_cache

    logger = logging.getLogger(__name__)
    invalidate_capabilities_cache()
    try:
        body = await request.json()
    except json.JSONDecodeError:
        body = {}
    if not isinstance(body, dict):
        body = {}
    service = str(body.get("service") or "all").strip().lower()
    try:
        if service == "stt":
            result = await test_stt()
        elif service == "tts":
            result = await test_tts()
        elif service == "llm":
            result = await test_llm()
            caps = await fetch_llm_capabilities()
            if isinstance(result, dict) and caps.get("ok"):
                result["capabilities"] = caps
        else:
            result = await test_all()
        return JSONResponse(result)
    except Exception as e:
        logger.exception("config test failed service=%s", service)
        return JSONResponse({"ok": False, "error": str(e), "service": service}, status_code=200)


async def llm_capabilities(_: Request) -> Response:
    from xiaozhi_bridge.llm_capabilities import fetch_llm_capabilities
    from xiaozhi_bridge.pipecat_llm import invalidate_capabilities_cache

    invalidate_capabilities_cache()
    return JSONResponse(await fetch_llm_capabilities())
