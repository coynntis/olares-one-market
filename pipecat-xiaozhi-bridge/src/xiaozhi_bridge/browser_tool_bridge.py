"""Dispatch built-in browser tools to the connected web client over WebSocket."""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from typing import Any

from xiaozhi_bridge.ws_port import XiaozhiWsPort

logger = logging.getLogger(__name__)

BUILTIN_PREFIX = "browser__"
DEFAULT_TIMEOUT_S = 45.0

_device_ports: dict[str, XiaozhiWsPort] = {}
_pending: dict[str, asyncio.Future[str]] = {}


def register_device_port(device_id: str, port: XiaozhiWsPort) -> None:
    did = device_id.strip()
    if did:
        _device_ports[did] = port
        logger.debug("browser tools port registered device=%s", did)


def unregister_device_port(device_id: str) -> None:
    did = device_id.strip()
    _device_ports.pop(did, None)


def is_device_connected(device_id: str) -> bool:
    return device_id.strip() in _device_ports


async def invoke_browser_tool(
    device_id: str,
    tool: str,
    arguments: dict[str, Any],
    *,
    timeout_s: float = DEFAULT_TIMEOUT_S,
) -> str:
    """Ask the browser tab to run a built-in tool; returns JSON string."""
    did = device_id.strip()
    port = _device_ports.get(did)
    if not port:
        return json.dumps(
            {"error": "Browser tab not connected — keep Agent R open in this device"},
            ensure_ascii=False,
        )
    request_id = f"bt_{uuid.uuid4().hex[:16]}"
    loop = asyncio.get_running_loop()
    fut: asyncio.Future[str] = loop.create_future()
    _pending[request_id] = fut
    payload = json.dumps(
        {
            "type": "builtin_tool",
            "request_id": request_id,
            "tool": tool,
            "arguments": arguments,
        },
        ensure_ascii=False,
    )
    try:
        await port.send_text(payload)
        return await asyncio.wait_for(fut, timeout=timeout_s)
    except asyncio.TimeoutError:
        return json.dumps({"error": f"Browser tool timed out after {timeout_s:.0f}s"}, ensure_ascii=False)
    finally:
        _pending.pop(request_id, None)


def complete_browser_tool(request_id: str, *, result: str | None = None, error: str | None = None) -> bool:
    fut = _pending.get(request_id)
    if not fut or fut.done():
        return False
    if error:
        fut.set_result(json.dumps({"error": error}, ensure_ascii=False))
    else:
        fut.set_result(result or "{}")
    return True
