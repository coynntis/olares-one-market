"""Shared MCP client session helpers (stdio / SSE / HTTP)."""

from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

from xiaozhi_bridge.mcp_catalog import suggestion_by_id
from xiaozhi_bridge.mcp_probe import normalize_server

logger = logging.getLogger(__name__)


def _is_mcp_server(server: dict[str, Any]) -> bool:
    catalog_id = server.get("catalog_id")
    if catalog_id:
        sug = suggestion_by_id(str(catalog_id))
        if sug and sug.get("kind") == "browser":
            return False
    return bool(server.get("enabled", True))


@asynccontextmanager
async def mcp_session(server: dict[str, Any]) -> AsyncIterator[Any]:
    """Yield initialized mcp.ClientSession for one server config."""
    norm = normalize_server(server)
    transport = norm["transport"]
    headers = dict(norm.get("headers") or {})

    from contextlib import AsyncExitStack

    from mcp import ClientSession, StdioServerParameters
    from mcp.client.sse import sse_client
    from mcp.client.stdio import stdio_client
    from mcp.client.streamable_http import streamablehttp_client

    async with AsyncExitStack() as stack:
        if transport == "stdio":
            params = StdioServerParameters(
                command=norm["command"],
                args=list(norm.get("args") or []),
            )
            streams = await stack.enter_async_context(stdio_client(params))
            read_stream, write_stream = streams[0], streams[1]
        elif transport == "sse":
            read_stream, write_stream = await stack.enter_async_context(
                sse_client(url=norm["url"], headers=headers or None)
            )
        else:
            read_stream, write_stream, _ = await stack.enter_async_context(
                streamablehttp_client(url=norm["url"], headers=headers or None)
            )

        session = await stack.enter_async_context(ClientSession(read_stream, write_stream))
        await session.initialize()
        yield session


async def list_server_tools(server: dict[str, Any]) -> list[dict[str, Any]]:
    """Return raw MCP tools for one server."""
    if not _is_mcp_server(server):
        return []
    try:
        async with mcp_session(server) as session:
            listed = await session.list_tools()
            out: list[dict[str, Any]] = []
            for tool in listed.tools:
                schema = getattr(tool, "inputSchema", None) or {}
                if hasattr(schema, "model_dump"):
                    schema = schema.model_dump()
                elif not isinstance(schema, dict):
                    schema = {}
                out.append(
                    {
                        "name": tool.name,
                        "description": (tool.description or "")[:500],
                        "inputSchema": schema,
                    }
                )
            return out
    except ImportError:
        raise
    except Exception as e:
        logger.warning("list_tools failed server=%s: %s", server.get("name"), e)
        return []


def _tool_result_text(result: Any) -> str:
    if result is None:
        return ""
    content = getattr(result, "content", None)
    if content is None and isinstance(result, dict):
        content = result.get("content")
    if not content:
        return json.dumps({"ok": True}, ensure_ascii=False)
    parts: list[str] = []
    for block in content:
        text = getattr(block, "text", None)
        if text is None and isinstance(block, dict):
            text = block.get("text")
        if text:
            parts.append(str(text))
        else:
            data = getattr(block, "data", None)
            if data is not None:
                parts.append(str(data))
    if parts:
        return "\n".join(parts)
    return json.dumps({"ok": True}, ensure_ascii=False)


async def call_server_tool(server: dict[str, Any], tool_name: str, arguments: dict[str, Any]) -> str:
    """Invoke one MCP tool and return text for the LLM."""
    async with mcp_session(server) as session:
        result = await session.call_tool(tool_name, arguments)
        return _tool_result_text(result)
