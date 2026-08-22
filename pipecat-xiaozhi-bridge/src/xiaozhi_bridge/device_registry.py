"""One live WebSocket connection per device-id."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from typing import Any

logger = logging.getLogger(__name__)

CloseFn = Callable[[], Awaitable[None]]

_lock = asyncio.Lock()
_slots: dict[str, tuple[Any, CloseFn]] = {}


async def claim(device_id: str, close_fn: CloseFn) -> Any:
    """Register device. Closes any previous connection for the same device-id."""
    device_id = device_id.strip()
    token = object()
    async with _lock:
        prev = _slots.get(device_id)
        if prev and prev[0] is not token:
            logger.info("device %s: closing previous connection", device_id)
            try:
                await prev[1]()
            except Exception:
                logger.exception("failed closing previous connection for %s", device_id)
        _slots[device_id] = (token, close_fn)
    return token


async def release(device_id: str, token: Any) -> None:
    device_id = device_id.strip()
    async with _lock:
        slot = _slots.get(device_id)
        if slot and slot[0] is token:
            del _slots[device_id]


def is_connected(device_id: str) -> bool:
    return device_id.strip() in _slots
