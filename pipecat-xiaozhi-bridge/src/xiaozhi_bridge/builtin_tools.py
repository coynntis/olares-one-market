"""Built-in browser tools (camera, Bluetooth, geolocation) — run in user's tab."""

from __future__ import annotations

from typing import Any

BUILTIN_SERVER_NAME = "Browser (this tab)"
BUILTIN_SLUG = "browser"

DEFAULT_BUILTIN_TOOLS: dict[str, bool] = {
    "camera": False,
    "bluetooth": False,
    "geolocation": False,
}

_BUILTIN_SCHEMAS: dict[str, dict[str, Any]] = {
    "take_picture": {
        "description": (
            "Capture a photo from the user's browser camera. "
            "Returns JPEG base64 (resized) for vision analysis."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "facing": {
                    "type": "string",
                    "enum": ["environment", "user"],
                    "description": "Rear (environment) or front (user) camera",
                },
            },
        },
    },
    "list_bluetooth_devices": {
        "description": (
            "List Bluetooth devices known to the browser (previously permitted). "
            "Web Bluetooth cannot scan silently; returns paired/remembered devices."
        ),
        "parameters": {"type": "object", "properties": {}},
    },
    "get_geolocation": {
        "description": "Read the browser's current GPS coordinates (requires user permission).",
        "parameters": {
            "type": "object",
            "properties": {
                "high_accuracy": {"type": "boolean", "description": "Use high accuracy GPS"},
            },
        },
    },
}

_TOOL_KEYS: dict[str, str] = {
    "take_picture": "camera",
    "list_bluetooth_devices": "bluetooth",
    "get_geolocation": "geolocation",
}


def normalize_builtin_tools(raw: dict[str, Any] | None) -> dict[str, bool]:
    out = dict(DEFAULT_BUILTIN_TOOLS)
    if not isinstance(raw, dict):
        return out
    for key in out:
        if key in raw:
            out[key] = bool(raw[key])
    return out


def enabled_builtin_tool_names(settings: dict[str, bool]) -> list[str]:
    names: list[str] = []
    for tool, cfg_key in _TOOL_KEYS.items():
        if settings.get(cfg_key):
            names.append(tool)
    return names


def openai_builtin_tools(enabled: dict[str, bool]) -> list[dict[str, Any]]:
    tools: list[dict[str, Any]] = []
    for tool_name in enabled_builtin_tool_names(enabled):
        schema = _BUILTIN_SCHEMAS.get(tool_name)
        if not schema:
            continue
        tools.append(
            {
                "type": "function",
                "function": {
                    "name": f"{BUILTIN_SLUG}{'__'}{tool_name}",
                    "description": schema["description"],
                    "parameters": schema.get("parameters") or {"type": "object", "properties": {}},
                },
            }
        )
    return tools


def is_builtin_openai_name(name: str) -> bool:
    return name.startswith(f"{BUILTIN_SLUG}__")


def builtin_tool_suffix(name: str) -> str:
    if not is_builtin_openai_name(name):
        return name
    return name.split("__", 1)[1]
