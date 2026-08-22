"""Lightweight agent tool helpers (no heavy deps)."""

from __future__ import annotations

import json
import logging
import os
import uuid
from typing import Any

logger = logging.getLogger(__name__)

MAX_TOOL_RESULT_CHARS = int(os.environ.get("MAX_TOOL_RESULT_CHARS", "12000"))


def clip_tool_content(text: str) -> str:
    """Keep MCP tool payloads from blowing LLM context after web search."""
    t = (text or "").strip()
    if len(t) <= MAX_TOOL_RESULT_CHARS:
        return t
    logger.info("truncating tool result chars=%d -> %d", len(t), MAX_TOOL_RESULT_CHARS)
    return t[:MAX_TOOL_RESULT_CHARS] + "\n…[truncated for LLM context]"


def process_mcp_tool_result(
    openai_name: str,
    raw: str,
) -> tuple[str, dict[str, Any] | None]:
    """Strip image_b64 from MCP JSON for LLM; return data URL meta for chat UI."""
    stripped = (raw or "").strip()
    if not stripped:
        return "", None

    data: dict[str, Any] | None = None
    if stripped.startswith("{"):
        try:
            parsed = json.loads(stripped)
            if isinstance(parsed, dict):
                data = parsed
        except json.JSONDecodeError:
            data = None

    b64 = ""
    mime = "image/png"
    if data:
        raw_b64 = data.get("image_b64")
        if isinstance(raw_b64, str) and raw_b64.strip():
            b64 = raw_b64.strip()
        mime = str(data.get("mime_type") or "image/png").strip() or "image/png"
    if not b64:
        # Truncated / wrapped payloads — still recover image for UI.
        import re

        m = re.search(r'"image_b64"\s*:\s*"([A-Za-z0-9+/=\s]+)"', stripped)
        if m:
            b64 = re.sub(r"\s+", "", m.group(1))
            logger.info(
                "recovered image_b64 via regex tool=%s chars=%d",
                openai_name,
                len(b64),
            )

    if not b64:
        return clip_tool_content(stripped), None

    if data:
        summary = {k: v for k, v in data.items() if k != "image_b64"}
    else:
        summary = {}
    summary["image"] = "[generated image shown in chat]"
    llm_text = clip_tool_content(json.dumps(summary, ensure_ascii=False, indent=2))
    gen_meta: dict[str, Any] = {
        "data_url": f"data:{mime};base64,{b64}",
        "mime_type": mime,
        "tool_name": openai_name,
        "seed": (data or {}).get("seed"),
        "width": (data or {}).get("width"),
        "height": (data or {}).get("height"),
        "prompt": (data or {}).get("prompt"),
        "timing": (data or {}).get("timing"),
    }
    logger.info(
        "mcp image ready tool=%s bytes≈%d %sx%s",
        openai_name,
        len(b64) * 3 // 4,
        gen_meta.get("width"),
        gen_meta.get("height"),
    )
    return llm_text, gen_meta


def generated_image_caption(gen: dict[str, Any]) -> str:
    prompt = str(gen.get("prompt") or "").strip()
    if prompt:
        return f"Generated: {prompt[:240]}"
    w, h = gen.get("width"), gen.get("height")
    if w and h:
        return f"Generated image ({w}×{h})"
    return "Generated image"


def light_generated_image_meta(gen: dict[str, Any]) -> dict[str, Any]:
    """Trace/SQLite meta without multi-MB data_url."""
    return {
        "has_image": True,
        "tool_name": gen.get("tool_name"),
        "seed": gen.get("seed"),
        "width": gen.get("width"),
        "height": gen.get("height"),
        "prompt": (str(gen.get("prompt") or ""))[:240],
        "mime_type": gen.get("mime_type") or "image/png",
    }


# Short UI/TTS labels — never include URLs or search queries.
_TOOL_KIND: dict[str, str] = {
    "search": "web",
    "fetchWebContent": "web",
    "fetchGithubReadme": "GitHub",
    "take_picture": "camera",
    "list_bluetooth_devices": "Bluetooth",
    "get_geolocation": "location",
    "generate_image": "image",
}


def _tool_suffix(openai_name: str) -> str:
    if "__" in openai_name:
        return openai_name.split("__", 1)[1]
    return openai_name


def human_tool_label(openai_name: str) -> str:
    """Short label for agent step rail (no URLs)."""
    suffix = _tool_suffix(openai_name)
    kind = _TOOL_KIND.get(suffix, suffix.replace("_", " "))
    if kind == "web":
        return "web search" if suffix == "search" else "web fetch"
    if kind == "camera":
        return "camera"
    if kind == "Bluetooth":
        return "Bluetooth"
    if kind == "location":
        return "location"
    if kind == "image":
        return "image gen"
    return kind


def tool_announce_message(openai_name: str) -> str:
    suffix = _tool_suffix(openai_name)
    if suffix == "search":
        return "Searching the web…"
    if suffix == "fetchWebContent":
        return "Fetching web content…"
    if suffix == "fetchGithubReadme":
        return "Fetching GitHub readme…"
    if suffix == "generate_image":
        return "Generating image…"
    if suffix == "take_picture":
        return "Opening camera…"
    if suffix == "list_bluetooth_devices":
        return "Reading Bluetooth devices…"
    if suffix == "get_geolocation":
        return "Reading location…"
    return "Running tool…"


def tool_running_message(openai_name: str) -> str:
    suffix = _tool_suffix(openai_name)
    if suffix in ("search",):
        return "Searching the web…"
    if suffix in ("fetchWebContent",):
        return "Fetching web content…"
    if suffix == "generate_image":
        return "Rendering image on GPU…"
    if suffix == "take_picture":
        return "Capturing photo…"
    if suffix == "list_bluetooth_devices":
        return "Scanning Bluetooth…"
    if suffix == "get_geolocation":
        return "Getting location…"
    return f"{human_tool_label(openai_name)}…"


def tool_done_message(openai_name: str) -> str:
    suffix = _tool_suffix(openai_name)
    if suffix == "search":
        return "Web search done"
    if suffix == "fetchWebContent":
        return "Web fetch done"
    if suffix == "generate_image":
        return "Image generated"
    if suffix == "take_picture":
        return "Photo captured"
    if suffix == "list_bluetooth_devices":
        return "Bluetooth scan done"
    if suffix == "get_geolocation":
        return "Location read"
    return f"{human_tool_label(openai_name)} done"


def ensure_tool_call_ids(tool_calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for tc in tool_calls:
        row = dict(tc)
        if not str(row.get("id") or "").strip():
            row["id"] = f"call_{uuid.uuid4().hex[:12]}"
        out.append(row)
    return out
