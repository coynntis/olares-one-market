"""Aggregate MCP tools into OpenAI function schema + dispatch calls."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any

from xiaozhi_bridge.browser_tool_bridge import invoke_browser_tool
from xiaozhi_bridge.builtin_tools import (
    builtin_tool_suffix,
    is_builtin_openai_name,
    normalize_builtin_tools,
    openai_builtin_tools,
)
from xiaozhi_bridge.config import load_settings
from xiaozhi_bridge.mcp_client import call_server_tool, list_server_tools
from xiaozhi_bridge.mcp_probe import normalize_server

logger = logging.getLogger(__name__)

SEP = "__"
_tools_cache: tuple[str, list[dict[str, Any]], "McpToolRouter"] | None = None


def _slug(name: str, server_id: str) -> str:
    base = re.sub(r"[^a-zA-Z0-9]+", "_", name.strip().lower()).strip("_") or "mcp"
    return f"{base[:24]}_{server_id[:8]}"


def namespaced_tool_name(server: dict[str, Any], tool_name: str) -> str:
    slug = _slug(str(server.get("name") or "mcp"), str(server.get("id") or ""))
    return f"{slug}{SEP}{tool_name}"


def split_namespaced_tool(name: str) -> tuple[str, str] | None:
    if SEP not in name:
        return None
    slug, tool = name.split(SEP, 1)
    if not slug or not tool:
        return None
    return slug, tool


@dataclass
class McpToolRouter:
    """Maps OpenAI tool names back to MCP server + original tool, or browser builtins."""

    by_openai_name: dict[str, tuple[dict[str, Any], str]] = field(default_factory=dict)
    device_id: str | None = None

    async def call(self, openai_name: str, arguments: dict[str, Any]) -> str:
        if is_builtin_openai_name(openai_name):
            if not self.device_id:
                return json.dumps(
                    {"error": "Built-in browser tools need an active browser session"},
                    ensure_ascii=False,
                )
            tool = builtin_tool_suffix(openai_name)
            return await invoke_browser_tool(self.device_id, tool, arguments)
        entry = self.by_openai_name.get(openai_name)
        if not entry:
            return json.dumps({"error": f"unknown tool {openai_name}"}, ensure_ascii=False)
        server, tool_name = entry
        try:
            return await call_server_tool(server, tool_name, arguments)
        except Exception as e:
            logger.exception("tool call failed tool=%s server=%s", openai_name, server.get("name"))
            return json.dumps({"error": str(e)}, ensure_ascii=False)


def invalidate_mcp_tools_cache() -> None:
    global _tools_cache
    _tools_cache = None


def _config_fingerprint(cfg: Any) -> str:
    enabled = [s for s in cfg.mcp_servers if s.get("enabled", True)]
    builtin = normalize_builtin_tools(getattr(cfg, "builtin_tools", None))
    return json.dumps({"mcp": enabled, "builtin": builtin}, sort_keys=True, default=str)


async def build_openai_tools(
    servers: list[dict[str, Any]],
    *,
    builtin: dict[str, bool],
    device_id: str | None = None,
) -> tuple[list[dict[str, Any]], McpToolRouter]:
    """List tools from enabled MCP servers + browser builtins → OpenAI tools[] + router."""
    router = McpToolRouter(device_id=device_id)
    openai_tools: list[dict[str, Any]] = []
    seen: set[str] = set()

    for tool_def in openai_builtin_tools(builtin):
        fn = tool_def.get("function") or {}
        oai_name = str(fn.get("name") or "")
        if oai_name:
            seen.add(oai_name)
        openai_tools.append(tool_def)

    for raw in servers:
        if not raw.get("enabled", True):
            continue
        try:
            server = normalize_server(raw)
        except ValueError as e:
            logger.warning("skip invalid mcp server: %s", e)
            continue
        tools = await list_server_tools(server)
        for tool in tools:
            oai_name = namespaced_tool_name(server, str(tool["name"]))
            if oai_name in seen:
                oai_name = f"{oai_name}_{server['id'][:4]}"
            seen.add(oai_name)
            router.by_openai_name[oai_name] = (server, str(tool["name"]))
            openai_tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": oai_name,
                        "description": tool.get("description") or f"MCP tool {tool['name']} on {server['name']}",
                        "parameters": tool.get("inputSchema") or {"type": "object", "properties": {}},
                    },
                }
            )
    if openai_tools and any(is_builtin_openai_name(str(t.get("function", {}).get("name", ""))) for t in openai_tools):
        logger.info("browser builtin tools enabled device=%s flags=%s", device_id, builtin)
    return openai_tools, router


async def get_mcp_openai_tools(*, device_id: str | None = None) -> tuple[list[dict[str, Any]], McpToolRouter | None]:
    """Cached OpenAI tools from configured MCP servers + browser builtins."""
    global _tools_cache
    cfg = load_settings()
    builtin = normalize_builtin_tools(getattr(cfg, "builtin_tools", None))
    enabled = [s for s in cfg.mcp_servers if s.get("enabled", True)]
    has_builtin = any(builtin.values())
    if not enabled and not has_builtin:
        return [], None
    fp = _config_fingerprint(cfg)
    if _tools_cache and _tools_cache[0] == fp:
        tools, router = _tools_cache[1], _tools_cache[2]
        router.device_id = device_id
        return tools, router
    try:
        tools, router = await build_openai_tools(enabled, builtin=builtin, device_id=device_id)
    except ImportError:
        logger.warning("mcp package missing — tool calling disabled")
        if has_builtin:
            tools, router = openai_builtin_tools(builtin), McpToolRouter(device_id=device_id)
        else:
            return [], None
    _tools_cache = (fp, tools, router)
    logger.info(
        "tools loaded count=%d mcp_servers=%d builtin=%s",
        len(tools),
        len(enabled),
        builtin,
    )
    return tools, router


def tools_configured() -> bool:
    cfg = load_settings()
    builtin = normalize_builtin_tools(getattr(cfg, "builtin_tools", None))
    if any(builtin.values()):
        return True
    return any(s.get("enabled", True) for s in cfg.mcp_servers)
