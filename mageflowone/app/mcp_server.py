"""MCP (streamable HTTP) tools for Mage Flow One."""

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
from PIL import Image

LOG = logging.getLogger("mageflowone.mcp")


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
        hosts = ["127.0.0.1:*", "localhost:*", "[::1]:*", "mageflowone:*"]
    return TransportSecuritySettings(
        enable_dns_rebinding_protection=enable,
        allowed_hosts=hosts,
        allowed_origins=origins,
    )


mcp = FastMCP(
    "Mage Flow One",
    host="0.0.0.0",
    stateless_http=True,
    json_response=True,
    transport_security=_mcp_transport_security(),
)


def _json(data: object) -> str:
    return json.dumps(data, indent=2, ensure_ascii=False)


def _mage() -> Any:
    mod = sys.modules.get("__main__")
    if mod is None:
        raise RuntimeError("Mage app module not loaded")
    return mod


def _b64_png(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _decode_image(image_b64: str) -> Image.Image:
    s = (image_b64 or "").strip()
    if s.startswith("data:"):
        s = s.split(",", 1)[-1]
    return Image.open(io.BytesIO(base64.b64decode(s))).convert("RGB")


@mcp.tool()
async def health_check() -> str:
    """VRAM, loaded model, available checkpoints."""
    k = _mage()
    return _json(
        {
            "status": "ok",
            "loaded": k._LOADED_KEY,
            "default_model": k.DEFAULT_MODEL,
            "models": k.list_models(),
            "cuda": k.torch.cuda.is_available(),
            "vram": k._vram_report(),
            "mcp": {"transport": "streamable-http", "url": "/mcp/mcp"},
        }
    )


@mcp.tool()
async def list_models() -> str:
    """List T2I and edit checkpoints (key, repo_id, default steps/cfg)."""
    return _json({"models": _mage().list_models()})


@mcp.tool()
async def generate_image(
    prompt: str,
    model: str = "Mage-Flow-Turbo",
    width: int = 1024,
    height: int = 1024,
    steps: int = 0,
    cfg: float = -1.0,
    seed: int = -1,
    negative_prompt: str = " ",
) -> str:
    """Text-to-image. Default model Mage-Flow-Turbo (4 steps, cfg=1). steps=0/cfg<0 → model defaults."""
    k = _mage()
    LOG.info("mcp generate_image model=%s prompt=%r", model, str(prompt)[:100])
    try:
        image, seed_out, final_prompt, timing = await asyncio.to_thread(
            k.generate_image,
            str(prompt),
            model_key=model,
            width=int(width),
            height=int(height),
            steps=None if int(steps) <= 0 else int(steps),
            cfg=None if float(cfg) < 0 else float(cfg),
            seed=int(seed),
            negative_prompt=negative_prompt or " ",
        )
    except ValueError as exc:
        return _json({"error": str(exc)})
    except Exception as exc:
        LOG.exception("mcp generate_image failed")
        return _json({"error": f"generate failed: {exc}"})
    return _json(
        {
            "seed": seed_out,
            "width": (int(width) // 16) * 16,
            "height": (int(height) // 16) * 16,
            "prompt": final_prompt,
            "model": model,
            "timing": timing,
            "mime_type": "image/png",
            "image_b64": _b64_png(image),
        }
    )


@mcp.tool()
async def edit_image(
    prompt: str,
    image_b64: str,
    model: str = "Mage-Flow-Edit-Turbo",
    max_size: int = 1024,
    steps: int = 0,
    cfg: float = -1.0,
    seed: int = -1,
) -> str:
    """Instruction image edit. image_b64 = PNG/JPEG base64. Default Mage-Flow-Edit-Turbo."""
    k = _mage()
    LOG.info("mcp edit_image model=%s prompt=%r", model, str(prompt)[:100])
    try:
        ref = _decode_image(image_b64)
        image, seed_out, final_prompt, timing = await asyncio.to_thread(
            k.edit_image,
            str(prompt),
            ref,
            model_key=model,
            max_size=int(max_size),
            steps=None if int(steps) <= 0 else int(steps),
            cfg=None if float(cfg) < 0 else float(cfg),
            seed=int(seed),
        )
    except ValueError as exc:
        return _json({"error": str(exc)})
    except Exception as exc:
        LOG.exception("mcp edit_image failed")
        return _json({"error": f"edit failed: {exc}"})
    w, h = image.size
    return _json(
        {
            "seed": seed_out,
            "width": w,
            "height": h,
            "prompt": final_prompt,
            "model": model,
            "timing": timing,
            "mime_type": "image/png",
            "image_b64": _b64_png(image),
        }
    )


@mcp.tool()
async def unload_model() -> str:
    """Unload loaded Mage-Flow pipeline from GPU."""
    k = _mage()

    def _run() -> str:
        with k._PIPE_LOCK:
            return k._unload_pipeline()

    detail = await asyncio.to_thread(_run)
    return _json({"ok": True, "detail": detail})


class _McpClientCompatMiddleware:
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
