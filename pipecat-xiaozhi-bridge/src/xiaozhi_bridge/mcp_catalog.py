"""Olares One Market MCP / agent endpoint suggestions (from sibling app charts)."""

from __future__ import annotations

from typing import Any

# Shared entrance URL pattern (user pastes from Olares Settings after installing app):
#   http://<8-char-route>.shared.olares.com<path>
# In-cluster URLs work from other pods in the same Olares space.

OLARES_MCP_SUGGESTIONS: list[dict[str, Any]] = [
    {
        "id": "mageflowone",
        "name": "Mage Flow MCP",
        "kind": "mcp",
        "transport": "http",
        "path": "/mcp/mcp",
        "shared_entrance_name": "mageflowone",
        "in_cluster_url": "http://mageflowone:7860/mcp/mcp",
        "market_app": "mageflowone",
        "description": "Microsoft Mage-Flow Turbo T2I + Edit-Turbo. Tools: generate_image, edit_image, list_models, health_check, unload_model.",
        "install_hint": "Install mageflowone; shared entrance mageflowone + /mcp/mcp",
    },
    {
        "id": "krea2turboone",
        "name": "Krea 2 Turbo MCP",
        "kind": "mcp",
        "transport": "http",
        "path": "/mcp/mcp",
        "shared_entrance_name": "krea2turboone",
        "in_cluster_url": "http://krea2turboone:7860/mcp/mcp",
        "market_app": "krea2turboone",
        "description": "Krea-2-Turbo text-to-image on Olares One GPU. Tools: generate_image, list_loras, health_check, clear_vram, unload_model.",
        "install_hint": "Install krea2turboone from market; shared entrance krea2turboone + /mcp/mcp",
    },
    {
        "id": "openwebsearchone",
        "name": "Open WebSearch MCP",
        "kind": "mcp",
        "transport": "http",
        "path": "/mcp",
        "shared_entrance_name": "openwebsearchmcp",
        "in_cluster_url": "http://openwebsearchone:3000/mcp",
        "market_app": "openwebsearchone",
        "description": "Multi-engine web search (no API keys). Tools: search, fetchWebContent, fetchGithubReadme, and more.",
        "install_hint": "Install openwebsearchone from this market, then copy its shared entrance URL and append /mcp",
    },
    {
        "id": "dockerbuilderone",
        "name": "Docker Builder MCP",
        "kind": "mcp",
        "transport": "http",
        "path": "/mcp/mcp",
        "shared_entrance_name": "dockerbuildermcp",
        "in_cluster_url": "http://dockerbuilderone:8080/mcp/mcp",
        "market_app": "dockerbuilderone",
        "description": "Build and push images to ghcr.io. Tools: list_projects, start_build, get_build_logs, and more.",
        "install_hint": "Install dockerbuilderone, set GitHub PAT in Olares user env, use shared entrance + /mcp/mcp",
    },
    {
        "id": "splatlabone",
        "name": "SplatLab MCP",
        "kind": "mcp",
        "transport": "http",
        "path": "/mcp/mcp",
        "shared_entrance_name": "splatlabmcp",
        "in_cluster_url": "http://splatlabone:8080/mcp/mcp",
        "market_app": "splatlabone",
        "description": "3D Gaussian splatting on Olares One GPU. MCP for training jobs and artifacts.",
        "install_hint": "Install splatlabone from market; shared entrance + /mcp/mcp",
    },
    {
        "id": "browserlessone",
        "name": "Browserless Chromium",
        "kind": "browser",
        "transport": "http",
        "path": "/docs",
        "shared_entrance_name": "browserlessapi",
        "in_cluster_url": "http://browserlessone:3000",
        "ws_url": "ws://browserlessone:3000",
        "market_app": "browserlessone",
        "description": "Headless Chrome (Puppeteer/Playwright). Not an MCP server. Use with Open WebSearch PLAYWRIGHT_WS_ENDPOINT or custom agents.",
        "install_hint": "Install browserlessone; Playwright connects to ws://host:3000. Health/docs at /docs on HTTP entrance.",
    },
]


def list_suggestions() -> list[dict[str, Any]]:
    return [dict(item) for item in OLARES_MCP_SUGGESTIONS]


def suggestion_by_id(catalog_id: str) -> dict[str, Any] | None:
    for item in OLARES_MCP_SUGGESTIONS:
        if item["id"] == catalog_id:
            return dict(item)
    return None
