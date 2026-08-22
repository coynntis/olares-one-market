"""REST API for MCP server configuration and capability probes."""

from __future__ import annotations

import json
from typing import Any

from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from xiaozhi_bridge.config import apply_patch, load_settings
from xiaozhi_bridge.mcp_catalog import list_suggestions
from xiaozhi_bridge.mcp_tools import invalidate_mcp_tools_cache
from xiaozhi_bridge.mcp_probe import (
    normalize_server,
    normalize_servers,
    server_from_suggestion,
    test_all_enabled,
    test_server,
    test_server_by_id,
)
from xiaozhi_bridge.settings_api import _authorized


async def mcp_suggestions(_: Request) -> Response:
    return JSONResponse({"suggestions": list_suggestions()})


async def mcp_list_servers(_: Request) -> Response:
    cfg = load_settings()
    return JSONResponse({"servers": cfg.mcp_servers})


async def mcp_put_servers(request: Request) -> Response:
    if not _authorized(request):
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    try:
        body = await request.json()
    except json.JSONDecodeError:
        return JSONResponse({"error": "invalid json"}, status_code=400)
    if not isinstance(body, dict):
        return JSONResponse({"error": "expected object"}, status_code=400)
    servers_raw = body.get("servers")
    try:
        servers = normalize_servers(servers_raw if isinstance(servers_raw, list) else [])
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=400)
    updated = apply_patch({"mcp_servers": servers})
    invalidate_mcp_tools_cache()
    return JSONResponse({"ok": True, "servers": updated.mcp_servers})


async def mcp_add_from_suggestion(request: Request) -> Response:
    if not _authorized(request):
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    try:
        body = await request.json()
    except json.JSONDecodeError:
        return JSONResponse({"error": "invalid json"}, status_code=400)
    if not isinstance(body, dict):
        return JSONResponse({"error": "expected object"}, status_code=400)
    catalog_id = str(body.get("catalog_id") or "").strip()
    if not catalog_id:
        return JSONResponse({"error": "catalog_id required"}, status_code=400)
    shared_base_url = str(body.get("shared_base_url") or body.get("url") or "").strip()
    name_override = str(body.get("name") or "").strip()
    try:
        draft = server_from_suggestion(
            catalog_id,
            shared_base_url=shared_base_url,
            name_override=name_override,
        )
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=400)

    cfg = load_settings()
    servers = list(cfg.mcp_servers)
    servers.append(draft)
    try:
        servers = normalize_servers(servers)
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=400)
    updated = apply_patch({"mcp_servers": servers})
    invalidate_mcp_tools_cache()
    return JSONResponse({"ok": True, "server": draft, "servers": updated.mcp_servers})


async def mcp_test(request: Request) -> Response:
    try:
        body = await request.json()
    except json.JSONDecodeError:
        body = {}
    if not isinstance(body, dict):
        body = {}

    service = str(body.get("service") or "one").strip().lower()
    if service == "all":
        result = await test_all_enabled()
        return JSONResponse(result)

    server_id = str(body.get("server_id") or "").strip()
    if server_id:
        return JSONResponse(await test_server_by_id(server_id))

    inline = body.get("server")
    if isinstance(inline, dict):
        try:
            norm = normalize_server(inline)
        except ValueError as e:
            return JSONResponse({"error": str(e)}, status_code=400)
        return JSONResponse(await test_server(norm))

    return JSONResponse({"error": "server_id or server object required"}, status_code=400)
