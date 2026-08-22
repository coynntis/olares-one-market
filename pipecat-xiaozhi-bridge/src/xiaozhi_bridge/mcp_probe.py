"""MCP server validation, probe (list tools), and HTTP health checks."""

from __future__ import annotations

import logging
import time
import uuid
from typing import Any

import httpx

from xiaozhi_bridge.config import load_settings
from xiaozhi_bridge.mcp_catalog import suggestion_by_id

logger = logging.getLogger(__name__)

VALID_TRANSPORTS = frozenset({"http", "sse", "stdio"})


def normalize_server(raw: dict[str, Any]) -> dict[str, Any]:
    """Validate and fill defaults for one MCP server entry."""
    if not isinstance(raw, dict):
        raise ValueError("server entry must be an object")
    name = str(raw.get("name") or "").strip()
    if not name:
        raise ValueError("name required")
    transport = str(raw.get("transport") or "http").strip().lower()
    if transport not in VALID_TRANSPORTS:
        raise ValueError(f"transport must be one of {sorted(VALID_TRANSPORTS)}")
    enabled = bool(raw.get("enabled", True))
    sid = str(raw.get("id") or "").strip() or str(uuid.uuid4())
    catalog_id = str(raw.get("catalog_id") or "").strip() or None
    headers = raw.get("headers") if isinstance(raw.get("headers"), dict) else {}
    clean_headers = {str(k): str(v) for k, v in headers.items() if str(k).strip()}

    entry: dict[str, Any] = {
        "id": sid,
        "name": name,
        "enabled": enabled,
        "transport": transport,
        "catalog_id": catalog_id,
        "headers": clean_headers,
    }

    if transport == "stdio":
        command = str(raw.get("command") or "").strip()
        if not command:
            raise ValueError("command required for stdio transport")
        args = raw.get("args")
        if args is None:
            args_list: list[str] = []
        elif isinstance(args, list):
            args_list = [str(a) for a in args]
        else:
            raise ValueError("args must be a list for stdio transport")
        entry["command"] = command
        entry["args"] = args_list
        entry["url"] = ""
    else:
        url = str(raw.get("url") or "").strip()
        if not url:
            raise ValueError("url required for http/sse transport")
        entry["url"] = url
        entry["command"] = ""
        entry["args"] = []

    return entry


def normalize_servers(servers: list[Any]) -> list[dict[str, Any]]:
    if not isinstance(servers, list):
        raise ValueError("mcp_servers must be a list")
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in servers:
        norm = normalize_server(item)
        if norm["id"] in seen:
            raise ValueError(f"duplicate server id {norm['id']}")
        seen.add(norm["id"])
        out.append(norm)
    return out


def server_from_suggestion(
    catalog_id: str,
    *,
    shared_base_url: str = "",
    name_override: str = "",
) -> dict[str, Any]:
    """Build a server draft from catalog + user shared entrance base URL."""
    sug = suggestion_by_id(catalog_id)
    if not sug:
        raise ValueError(f"unknown catalog id {catalog_id}")
    if sug.get("kind") == "browser":
        raise ValueError(f"{catalog_id} is a browser endpoint, not MCP")

    base = shared_base_url.strip().rstrip("/")
    if not base:
        base = str(sug.get("in_cluster_url") or "").rstrip("/")
        path = ""
        if sug.get("path") and base and not base.endswith(str(sug["path"])):
            path = str(sug["path"])
        url = f"{base}{path}" if base else ""
    else:
        path = str(sug.get("path") or "")
        url = f"{base}{path}" if path else base

    return normalize_server(
        {
            "id": str(uuid.uuid4()),
            "name": name_override.strip() or str(sug["name"]),
            "enabled": True,
            "transport": sug.get("transport") or "http",
            "url": url,
            "catalog_id": catalog_id,
            "headers": {},
        }
    )


async def _probe_mcp_tools(server: dict[str, Any]) -> dict[str, Any]:
    started = time.perf_counter()
    transport = server["transport"]
    headers = dict(server.get("headers") or {})

    try:
        from contextlib import AsyncExitStack

        from mcp import ClientSession, StdioServerParameters
        from mcp.client.sse import sse_client
        from mcp.client.stdio import stdio_client
        from mcp.client.streamable_http import streamablehttp_client
    except ImportError as e:
        return {
            "ok": False,
            "error": "mcp package not installed on bridge (pip install 'mcp>=1.11')",
            "latency_ms": int((time.perf_counter() - started) * 1000),
        }

    try:
        async with AsyncExitStack() as stack:
            if transport == "stdio":
                params = StdioServerParameters(
                    command=server["command"],
                    args=list(server.get("args") or []),
                )
                streams = await stack.enter_async_context(stdio_client(params))
                read_stream, write_stream = streams[0], streams[1]
            elif transport == "sse":
                read_stream, write_stream = await stack.enter_async_context(
                    sse_client(url=server["url"], headers=headers or None)
                )
            else:
                read_stream, write_stream, _ = await stack.enter_async_context(
                    streamablehttp_client(url=server["url"], headers=headers or None)
                )

            session = await stack.enter_async_context(ClientSession(read_stream, write_stream))
            await session.initialize()
            listed = await session.list_tools()
            tools = [
                {
                    "name": t.name,
                    "description": (t.description or "")[:240],
                }
                for t in listed.tools
            ]
            latency_ms = int((time.perf_counter() - started) * 1000)
            return {
                "ok": True,
                "transport": transport,
                "url": server.get("url") or server.get("command"),
                "tool_count": len(tools),
                "tools": tools,
                "latency_ms": latency_ms,
            }
    except Exception as e:
        logger.exception("mcp probe failed name=%s", server.get("name"))
        return {
            "ok": False,
            "transport": transport,
            "url": server.get("url") or server.get("command"),
            "error": str(e),
            "latency_ms": int((time.perf_counter() - started) * 1000),
        }


async def _probe_http_health(url: str, *, headers: dict[str, str] | None = None) -> dict[str, Any]:
    started = time.perf_counter()
    cfg = load_settings()
    timeout = min(float(cfg.http_timeout or 30.0), 30.0)
    try:
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
            resp = await client.get(url, headers=headers or {})
        latency_ms = int((time.perf_counter() - started) * 1000)
        ok = resp.status_code < 500
        return {
            "ok": ok,
            "kind": "browser",
            "url": url,
            "status": resp.status_code,
            "detail": f"HTTP {resp.status_code}",
            "latency_ms": latency_ms,
        }
    except Exception as e:
        return {
            "ok": False,
            "kind": "browser",
            "url": url,
            "error": str(e),
            "latency_ms": int((time.perf_counter() - started) * 1000),
        }


async def test_server(server: dict[str, Any]) -> dict[str, Any]:
    """Test MCP list_tools or HTTP health for browser endpoints."""
    norm = normalize_server(server)
    catalog_id = norm.get("catalog_id")
    if catalog_id:
        sug = suggestion_by_id(str(catalog_id))
        if sug and sug.get("kind") == "browser":
            test_url = norm["url"] or str(sug.get("in_cluster_url", ""))
            path = str(sug.get("path") or "/docs")
            if test_url and not test_url.endswith(path):
                test_url = f"{test_url.rstrip('/')}{path}"
            result = await _probe_http_health(test_url, headers=norm.get("headers"))
            result["name"] = norm["name"]
            result["catalog_id"] = catalog_id
            return result

    result = await _probe_mcp_tools(norm)
    result["name"] = norm["name"]
    if catalog_id:
        result["catalog_id"] = catalog_id
    return result


async def test_server_by_id(server_id: str) -> dict[str, Any]:
    cfg = load_settings()
    for item in cfg.mcp_servers:
        if str(item.get("id")) == server_id:
            return await test_server(item)
    return {"ok": False, "error": f"server id not found: {server_id}"}


async def test_all_enabled() -> dict[str, Any]:
    cfg = load_settings()
    results: list[dict[str, Any]] = []
    for item in cfg.mcp_servers:
        if not item.get("enabled", True):
            continue
        row = await test_server(item)
        results.append(row)
    ok = all(r.get("ok") for r in results) if results else False
    return {"ok": ok, "results": results, "count": len(results)}
