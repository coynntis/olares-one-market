"""REST API for named LLM profiles."""

from __future__ import annotations

import asyncio
import json
from typing import Any

from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from xiaozhi_bridge import llm_profile_store
from xiaozhi_bridge.config import settings_for_api
from xiaozhi_bridge.settings_api import _authorized


async def list_llm_profiles_handler(_: Request) -> Response:
    profiles, active_id = await asyncio.to_thread(llm_profile_store.list_profiles)
    return JSONResponse({"profiles": profiles, "active_profile_id": active_id})


async def create_llm_profile_handler(request: Request) -> Response:
    if not _authorized(request):
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    try:
        body = await request.json()
    except json.JSONDecodeError:
        return JSONResponse({"error": "invalid json"}, status_code=400)
    if not isinstance(body, dict):
        return JSONResponse({"error": "expected object"}, status_code=400)

    from_current = bool(body.get("from_current"))
    try:
        if from_current:
            profile = await asyncio.to_thread(
                llm_profile_store.save_current_as_profile,
                str(body.get("name") or "").strip(),
            )
        else:
            profile = await asyncio.to_thread(
                llm_profile_store.create_profile,
                name=str(body.get("name") or ""),
                llm_base_url=str(body.get("llm_base_url") or ""),
                llm_model=str(body.get("llm_model") or ""),
                system_prompt=str(body.get("system_prompt") or ""),
                set_active=bool(body.get("set_active")),
            )
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=400)

    from xiaozhi_bridge.pipecat_llm import invalidate_capabilities_cache

    invalidate_capabilities_cache()
    return JSONResponse(
        {"profile": profile, "settings": settings_for_api()},
        status_code=201,
    )


async def update_llm_profile_handler(request: Request) -> Response:
    if not _authorized(request):
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    profile_id = str(request.path_params.get("profile_id") or "").strip()
    try:
        body = await request.json()
    except json.JSONDecodeError:
        return JSONResponse({"error": "invalid json"}, status_code=400)
    if not isinstance(body, dict):
        return JSONResponse({"error": "expected object"}, status_code=400)

    patch: dict[str, Any] = {}
    for key in ("name", "llm_base_url", "llm_model", "system_prompt"):
        if key in body:
            patch[key] = body[key]
    try:
        profile = await asyncio.to_thread(llm_profile_store.update_profile, profile_id, **patch)
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=400)
    except LookupError:
        return JSONResponse({"error": "not found"}, status_code=404)

    from xiaozhi_bridge.pipecat_llm import invalidate_capabilities_cache

    invalidate_capabilities_cache()
    return JSONResponse({"profile": profile, "settings": settings_for_api()})


async def delete_llm_profile_handler(request: Request) -> Response:
    if not _authorized(request):
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    profile_id = str(request.path_params.get("profile_id") or "").strip()
    deleted = await asyncio.to_thread(llm_profile_store.delete_profile, profile_id)
    if not deleted:
        return JSONResponse({"error": "not found"}, status_code=404)
    return JSONResponse({"ok": True, "settings": settings_for_api()})


async def activate_llm_profile_handler(request: Request) -> Response:
    if not _authorized(request):
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    profile_id = str(request.path_params.get("profile_id") or "").strip()
    try:
        profile = await asyncio.to_thread(llm_profile_store.activate_profile, profile_id)
    except LookupError:
        return JSONResponse({"error": "not found"}, status_code=404)

    from xiaozhi_bridge.pipecat_llm import invalidate_capabilities_cache

    invalidate_capabilities_cache()
    return JSONResponse({"profile": profile, "settings": settings_for_api()})
