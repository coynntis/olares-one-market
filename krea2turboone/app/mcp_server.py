"""MCP (streamable HTTP) tools for Krea 2 Turbo One."""

from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import os
import sys
from typing import Any

from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings

LOG = logging.getLogger("krea2turboone.mcp")


def _mcp_transport_security() -> TransportSecuritySettings:
    enable = os.environ.get("MCP_DNS_REBINDING_PROTECTION", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )
    hosts = [h.strip() for h in os.environ.get("MCP_ALLOWED_HOSTS", "").split(",") if h.strip()]
    origins = [o.strip() for o in os.environ.get("MCP_ALLOWED_ORIGINS", "").split(",") if o.strip()]
    if enable and not hosts:
        hosts = [
            "127.0.0.1:*",
            "localhost:*",
            "[::1]:*",
            "krea2turboone:*",
        ]
    return TransportSecuritySettings(
        enable_dns_rebinding_protection=enable,
        allowed_hosts=hosts,
        allowed_origins=origins,
    )


mcp = FastMCP(
    "Krea 2 Turbo One",
    host="0.0.0.0",
    stateless_http=True,
    json_response=True,
    transport_security=_mcp_transport_security(),
)


def _json(data: object) -> str:
    return json.dumps(data, indent=2, ensure_ascii=False)


def _krea() -> Any:
    """App runs as `python app.py` — module is __main__, not app."""
    mod = sys.modules.get("__main__")
    if mod is None:
        raise RuntimeError("Krea app module not loaded")
    return mod


@mcp.tool()
async def health_check() -> str:
    """GPU, pipeline load state, quant/attention mode, available LoRAs."""
    k = _krea()
    ready = k._PIPE is not None
    return _json(
        {
            "status": "ok",
            "model": k.MODEL_ID,
            "quant": k.QUANT,
            "attention": k.ATTENTION,
            "sage3_status": k._SAGE3_STATUS,
            "pipeline_loaded": ready,
            "loaded_via": k._LOADED_VIA if ready else None,
            "cuda": k.torch.cuda.is_available(),
            "vram_gb": round(k._vram_gb(), 2) if k.torch.cuda.is_available() else 0,
            "vram": k._vram_report(),
            "loras": [name for name in k.LORA_CATALOG.keys() if name != "None"],
            "mcp": {"transport": "streamable-http", "url": "/mcp/mcp"},
        }
    )


@mcp.tool()
async def list_loras() -> str:
    """Official + extra style LoRAs (use name in generate_image lora param)."""
    k = _krea()
    out: dict[str, dict[str, str]] = {}
    for name, meta in k.LORA_CATALOG.items():
        if name == "None":
            continue
        out[name] = {key: str(val) for key, val in meta.items()}
    return _json({"loras": out})


@mcp.tool()
async def generate_image(
    prompt: str,
    negative_prompt: str = "",
    width: int = 768,
    height: int = 768,
    steps: int = 8,
    guidance: float = 0.0,
    seed: int = -1,
    lora: str = "None",
    lora_strength: float = 1.0,
    park_vram: bool = True,
    unload_after: bool = False,
    response_format: str = "b64_json",
) -> str:
    """Text-to-image with Krea-2-Turbo. Prefer ≤768 on 24GB; 1024 can OOM under HAMI.

    Returns PNG base64 (default) or timing-only if response_format=timing_only.
    """
    k = _krea()
    if not prompt or not str(prompt).strip():
        return _json({"error": "prompt is required"})
    LOG.info(
        "mcp generate_image prompt=%r size=%sx%s steps=%s lora=%s",
        str(prompt)[:120],
        width,
        height,
        steps,
        lora,
    )
    try:
        # generate_image() acquires _PIPE_LOCK internally — do NOT wrap again (deadlocks).
        image, seed_out, final_prompt, timing = await asyncio.to_thread(
            k.generate_image,
            prompt=str(prompt),
            negative_prompt=negative_prompt or "",
            width=int(width),
            height=int(height),
            steps=int(steps),
            guidance=float(guidance),
            seed=int(seed) if int(seed) >= 0 else 0,
            randomize_seed=int(seed) < 0,
            lora_key=lora or "None",
            lora_strength=float(lora_strength),
            park_vram=bool(park_vram),
            unload_after=bool(unload_after),
        )
    except ValueError as exc:
        LOG.warning("mcp generate_image validation error: %s", exc)
        return _json({"error": str(exc)})
    except Exception as exc:
        LOG.exception("mcp generate_image failed")
        return _json({"error": f"generate failed: {exc}"})

    LOG.info("mcp generate_image ok seed=%s %s", seed_out, timing)

    w = (int(width) // 64) * 64 or 64
    h = (int(height) // 64) * 64 or 64
    payload: dict[str, Any] = {
        "seed": seed_out,
        "width": w,
        "height": h,
        "prompt": final_prompt,
        "lora": lora or "None",
        "timing": timing,
        "mime_type": "image/png",
    }
    fmt = (response_format or "b64_json").strip().lower()
    if fmt != "timing_only":
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        payload["image_b64"] = base64.b64encode(buf.getvalue()).decode("ascii")
    return _json(payload)


@mcp.tool()
async def clear_vram() -> str:
    """Park pipeline on CPU and clear CUDA cache (frees GPU for other apps)."""
    k = _krea()

    def _run() -> str:
        with k._PIPE_LOCK:
            return k._force_clear_vram()

    detail = await asyncio.to_thread(_run)
    return _json({"ok": True, "detail": detail})


@mcp.tool()
async def unload_model() -> str:
    """Unload Krea pipeline from memory entirely."""
    k = _krea()

    def _run() -> str:
        with k._PIPE_LOCK:
            return k._unload_all_models()

    detail = await asyncio.to_thread(_run)
    return _json({"ok": True, "detail": detail})


class _McpClientCompatMiddleware:
    """Some MCP clients send Accept: */* — FastMCP needs json + sse."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope.get("type") == "http":
            headers = list(scope.get("headers") or [])
            accept_raw = next((v.decode("latin-1") for k, v in headers if k.lower() == b"accept"), "")
            needs_json = "application/json" not in accept_raw
            needs_sse = "text/event-stream" not in accept_raw
            accept = accept_raw.strip().lower()
            if accept in ("*/*", "*") or (needs_json and needs_sse):
                headers = [(k, v) for k, v in headers if k.lower() != b"accept"]
                headers.append((b"accept", b"application/json, text/event-stream"))
                scope = {**scope, "headers": headers}
        await self.app(scope, receive, send)


def mcp_http_app():
    return _McpClientCompatMiddleware(mcp.streamable_http_app())
