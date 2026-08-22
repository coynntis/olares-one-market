"""REST API for persisted chat conversations (scoped by device-id)."""

from __future__ import annotations

import asyncio
import json

from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.responses import Response as StarletteResponse

from xiaozhi_bridge import chat_store


async def _run(fn, *args, **kwargs):
    return await asyncio.to_thread(fn, *args, **kwargs)


def _device_id_from_request(request: Request, body: dict | None = None) -> str | None:
    q = (request.query_params.get("device_id") or "").strip()
    if q:
        return q
    if body and body.get("device_id"):
        return str(body["device_id"]).strip()
    return None


async def list_conversations_handler(request: Request) -> Response:
    did = _device_id_from_request(request)
    if not did:
        return JSONResponse({"error": "device_id required"}, status_code=400)
    items = await _run(chat_store.list_conversations_for_device, did)
    current = await _run(chat_store.get_device_current_conversation_id, did)
    return JSONResponse({"conversations": items, "current_conversation_id": current})


async def get_active_conversation(request: Request) -> Response:
    did = _device_id_from_request(request)
    if not did:
        return JSONResponse({"error": "device_id required"}, status_code=400)
    conv_id = request.query_params.get("id", "").strip() or None
    try:
        conv = await _run(chat_store.resolve_device_conversation, did, conv_id)
    except PermissionError as e:
        return JSONResponse({"error": str(e)}, status_code=403)
    except LookupError as e:
        return JSONResponse({"error": str(e)}, status_code=404)
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=400)
    messages = await _run(chat_store.list_messages, conv["id"])
    current = await _run(chat_store.get_device_current_conversation_id, did)
    return JSONResponse(
        {
            "conversation": conv,
            "messages": messages,
            "current_conversation_id": current,
            "device_id": did,
        }
    )


async def create_conversation_handler(request: Request) -> Response:
    title = "Chat"
    device_id: str | None = None
    try:
        body = await request.json()
        if isinstance(body, dict):
            if body.get("title"):
                title = str(body["title"])
            device_id = _device_id_from_request(request, body)
    except json.JSONDecodeError:
        device_id = _device_id_from_request(request)
    if not device_id:
        return JSONResponse({"error": "device_id required"}, status_code=400)
    try:
        conv = await _run(chat_store.create_conversation, title, device_id)
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=400)
    return JSONResponse(conv)


async def activate_conversation_handler(request: Request) -> Response:
    cid = request.path_params["conversation_id"]
    did = _device_id_from_request(request)
    if not did:
        try:
            body = await request.json()
            if isinstance(body, dict):
                did = _device_id_from_request(request, body)
        except json.JSONDecodeError:
            pass
    if not did:
        return JSONResponse({"error": "device_id required"}, status_code=400)
    try:
        conv = await _run(chat_store.activate_device_conversation, did, cid)
    except PermissionError as e:
        return JSONResponse({"error": str(e)}, status_code=403)
    except LookupError as e:
        return JSONResponse({"error": str(e)}, status_code=404)
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=400)
    messages = await _run(chat_store.list_messages, cid)
    return JSONResponse(
        {
            "conversation": conv,
            "messages": messages,
            "current_conversation_id": cid,
            "device_id": did,
        }
    )


async def get_conversation_messages(request: Request) -> Response:
    cid = request.path_params["conversation_id"]
    did = _device_id_from_request(request)
    conv = await _run(chat_store.get_conversation, cid)
    if not conv:
        return JSONResponse({"error": "conversation not found"}, status_code=404)
    if did and conv.get("device_id") and conv["device_id"] != did:
        return JSONResponse({"error": "conversation belongs to another device"}, status_code=403)
    messages = await _run(chat_store.list_messages, cid)
    return JSONResponse({"conversation": conv, "messages": messages})


async def delete_conversation_handler(request: Request) -> Response:
    cid = request.path_params["conversation_id"]
    did = _device_id_from_request(request)
    try:
        ok = await _run(chat_store.delete_conversation, cid, device_id=did)
    except PermissionError as e:
        return JSONResponse({"error": str(e)}, status_code=403)
    if not ok:
        return JSONResponse({"error": "conversation not found"}, status_code=404)
    return JSONResponse({"ok": True})


async def clear_conversation_handler(request: Request) -> Response:
    cid = request.path_params["conversation_id"]
    did = _device_id_from_request(request)
    conv = await _run(chat_store.get_conversation, cid)
    if not conv:
        return JSONResponse({"error": "conversation not found"}, status_code=404)
    if did and conv.get("device_id") and conv["device_id"] != did:
        return JSONResponse({"error": "conversation belongs to another device"}, status_code=403)
    await _run(chat_store.clear_conversation_messages, cid)
    return JSONResponse({"ok": True})


async def get_message_image(request: Request) -> StarletteResponse:
    mid = request.path_params["message_id"]
    result = await _run(chat_store.read_message_image, mid)
    if not result:
        return JSONResponse({"error": "image not found"}, status_code=404)
    data, mime = result
    return StarletteResponse(content=data, media_type=mime)
