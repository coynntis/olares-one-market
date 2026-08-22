"""Map session_id → live xiaozhi voice session for UDP audio ingress."""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from xiaozhi_bridge.server import Session

logger = logging.getLogger(__name__)

OnAudio = Callable[[bytes], Awaitable[None]]

_entries: dict[str, tuple[Session, OnAudio]] = {}


def register(session_id: str, sess: Session, on_audio: OnAudio) -> None:
    _entries[session_id] = (sess, on_audio)
    logger.debug("session registry add %s (total=%d)", session_id, len(_entries))


def unregister(session_id: str) -> None:
    if _entries.pop(session_id, None):
        logger.debug("session registry remove %s (total=%d)", session_id, len(_entries))


async def dispatch_udp_audio(session_id: str, payload: bytes) -> None:
    entry = _entries.get(session_id)
    if not entry:
        return
    _, on_audio = entry
    await on_audio(payload)
