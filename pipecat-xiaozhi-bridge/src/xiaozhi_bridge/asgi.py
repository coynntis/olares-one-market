"""
ASGI app: web UI + REST config/chat + xiaozhi WebSocket on one port.
"""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path

from contextlib import asynccontextmanager

from starlette.applications import Starlette
from starlette.responses import PlainTextResponse
from starlette.routing import Mount, Route, WebSocketRoute
from starlette.staticfiles import StaticFiles
from starlette.websockets import WebSocket, WebSocketDisconnect

from xiaozhi_bridge.chat_api import post_chat
from xiaozhi_bridge.history_api import (
    activate_conversation_handler,
    clear_conversation_handler,
    create_conversation_handler,
    delete_conversation_handler,
    get_active_conversation,
    get_conversation_messages,
    get_message_image,
    list_conversations_handler,
)
from xiaozhi_bridge.server import run_xiaozhi_session
from xiaozhi_bridge import session_registry
from xiaozhi_bridge import device_registry
from xiaozhi_bridge.udp_audio import start_udp_hub, stop_udp_hub
from xiaozhi_bridge.mcp_api import (
    mcp_add_from_suggestion,
    mcp_list_servers,
    mcp_put_servers,
    mcp_suggestions,
    mcp_test,
)
from xiaozhi_bridge.settings_api import (
    config_status,
    get_config,
    llm_capabilities,
    put_config,
    test_config_connection,
)
from xiaozhi_bridge.voices_api import (
    activate_voice_handler,
    create_voice_handler,
    delete_voice_handler,
    get_voice_audio_handler,
    list_voices_handler,
    preview_voice_handler,
)
from xiaozhi_bridge.llm_profiles_api import (
    activate_llm_profile_handler,
    create_llm_profile_handler,
    delete_llm_profile_handler,
    list_llm_profiles_handler,
    update_llm_profile_handler,
)

logger = logging.getLogger("xiaozhi_bridge.asgi")


def _static_dir() -> Path:
    env = os.environ.get("STATIC_DIR")
    if env:
        return Path(env)
    return Path(__file__).resolve().parent.parent.parent / "client" / "dist"


class _StarletteWsPort:
    __slots__ = ("_ws",)

    def __init__(self, ws: WebSocket) -> None:
        self._ws = ws

    async def send_text(self, data: str) -> None:
        await self._ws.send_text(data)

    async def send_bytes(self, data: bytes) -> None:
        await self._ws.send_bytes(data)

    async def close(self, code: int = 1000, reason: str = "") -> None:
        try:
            await self._ws.close(code=code, reason=reason)
        except Exception:
            pass


async def health(_: object) -> PlainTextResponse:
    return PlainTextResponse("OK\n")


async def xiaozhi_websocket(websocket: WebSocket) -> None:
    await websocket.accept()
    scope = websocket.scope
    path = scope.get("path") or ""
    qs = scope.get("query_string", b"")
    if isinstance(qs, bytes):
        qs = qs.decode("latin-1")
    req_path = path + ("?" + qs if qs else "")

    def header_get(name: str) -> str | None:
        return websocket.headers.get(name)

    parsed_qs = {}
    if qs:
        from urllib.parse import parse_qs

        parsed_qs = parse_qs(qs)
    device_header = header_get("device-id")
    if device_header is None and "device-id" in parsed_qs:
        device_header = parsed_qs["device-id"][0]
    if not device_header:
        await websocket.send_text("端口正常，如需测试连接，请使用test_page.html")
        await websocket.close()
        return

    replaced = asyncio.Event()
    port: XiaozhiWsPort = _StarletteWsPort(websocket)

    async def close_replaced() -> None:
        replaced.set()
        await port.close(1000, "replaced by new connection")

    claim_token = await device_registry.claim(device_header, close_replaced)

    async def messages():
        try:
            while True:
                if replaced.is_set():
                    break
                m = await websocket.receive()
                if m["type"] == "websocket.disconnect":
                    break
                if "text" in m:
                    yield m["text"]
                elif "bytes" in m:
                    yield m["bytes"]
        except WebSocketDisconnect:
            logger.debug("websocket disconnect device=%s", device_header)

    try:
        from xiaozhi_bridge.browser_tool_bridge import register_device_port, unregister_device_port

        register_device_port(device_header, port)
        await run_xiaozhi_session(req_path, header_get, messages(), port)
    finally:
        from xiaozhi_bridge.browser_tool_bridge import unregister_device_port

        unregister_device_port(device_header)
        await device_registry.release(device_header, claim_token)


@asynccontextmanager
async def _lifespan(_: Starlette):
    hub = await start_udp_hub()
    if hub:
        hub.set_handler(session_registry.dispatch_udp_audio)
    yield
    await stop_udp_hub()


def build_app() -> Starlette:
    static = _static_dir()
    routes: list[Route | Mount | WebSocketRoute] = [
        WebSocketRoute("/xiaozhi/v1", xiaozhi_websocket),
        Route("/health", health, methods=["GET"]),
        Route("/api/config", get_config, methods=["GET"]),
        Route("/api/config", put_config, methods=["PUT"]),
        Route("/api/config/status", config_status, methods=["GET"]),
        Route("/api/config/test", test_config_connection, methods=["POST"]),
        Route("/api/config/llm-capabilities", llm_capabilities, methods=["GET"]),
        Route("/api/conversations", list_conversations_handler, methods=["GET"]),
        Route("/api/conversations", create_conversation_handler, methods=["POST"]),
        Route("/api/conversations/active", get_active_conversation, methods=["GET"]),
        Route("/api/conversations/{conversation_id}", delete_conversation_handler, methods=["DELETE"]),
        Route("/api/conversations/{conversation_id}/activate", activate_conversation_handler, methods=["POST"]),
        Route("/api/conversations/{conversation_id}/messages", get_conversation_messages, methods=["GET"]),
        Route("/api/conversations/{conversation_id}/clear", clear_conversation_handler, methods=["POST"]),
        Route("/api/messages/{message_id}/image", get_message_image, methods=["GET"]),
        Route("/api/chat", post_chat, methods=["POST"]),
        Route("/api/mcp/suggestions", mcp_suggestions, methods=["GET"]),
        Route("/api/mcp/servers", mcp_list_servers, methods=["GET"]),
        Route("/api/mcp/servers", mcp_put_servers, methods=["PUT"]),
        Route("/api/mcp/servers/from-suggestion", mcp_add_from_suggestion, methods=["POST"]),
        Route("/api/mcp/test", mcp_test, methods=["POST"]),
        Route("/api/voices", list_voices_handler, methods=["GET"]),
        Route("/api/voices", create_voice_handler, methods=["POST"]),
        Route("/api/voices/{voice_id}", delete_voice_handler, methods=["DELETE"]),
        Route("/api/voices/{voice_id}/audio", get_voice_audio_handler, methods=["GET"]),
        Route("/api/voices/{voice_id}/activate", activate_voice_handler, methods=["POST"]),
        Route("/api/voices/{voice_id}/preview", preview_voice_handler, methods=["POST"]),
        Route("/api/llm-profiles", list_llm_profiles_handler, methods=["GET"]),
        Route("/api/llm-profiles", create_llm_profile_handler, methods=["POST"]),
        Route("/api/llm-profiles/{profile_id}", update_llm_profile_handler, methods=["PUT"]),
        Route("/api/llm-profiles/{profile_id}", delete_llm_profile_handler, methods=["DELETE"]),
        Route("/api/llm-profiles/{profile_id}/activate", activate_llm_profile_handler, methods=["POST"]),
    ]
    if static.is_dir():
        routes.append(
            Mount(
                "/",
                app=StaticFiles(directory=str(static), html=True),
                name="static",
            )
        )
    else:
        logger.warning("Static dir missing (%s) — API + /xiaozhi/v1 only", static)

    return Starlette(routes=routes, lifespan=_lifespan)


app = build_app()
