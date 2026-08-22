"""Optional UDP audio ingress (low-latency uplink alongside WebSocket control)."""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Awaitable, Callable

logger = logging.getLogger(__name__)

_MAGIC = b"XZ01"
# XZ01 | 36-byte session_id (utf-8, zero padded) | raw audio payload
_HEADER = 4 + 36

AudioHandler = Callable[[str, bytes], Awaitable[None]]

_hub: "UdpAudioHub | None" = None


class UdpAudioHub:
    def __init__(self, port: int) -> None:
        self.port = port
        self._transport: asyncio.DatagramTransport | None = None
        self._handler: AudioHandler | None = None

    def set_handler(self, handler: AudioHandler | None) -> None:
        self._handler = handler

    async def start(self) -> None:
        loop = asyncio.get_running_loop()
        self._transport, _ = await loop.create_datagram_endpoint(
            lambda: _UdpProtocol(self),
            local_addr=("0.0.0.0", self.port),
        )
        logger.info("UDP audio listening on :%d (magic XZ01 + session_id + payload)", self.port)

    def close(self) -> None:
        if self._transport:
            self._transport.close()
            self._transport = None

    async def handle_datagram(self, data: bytes, addr: tuple[str, int]) -> None:
        if len(data) < _HEADER or data[:4] != _MAGIC:
            return
        session_id = data[4:_HEADER].split(b"\x00", 1)[0].decode("utf-8", errors="ignore").strip()
        payload = data[_HEADER:]
        if not session_id or not payload or not self._handler:
            return
        try:
            await self._handler(session_id, payload)
        except Exception:
            logger.exception("udp audio handler failed from %s session=%s", addr, session_id)


class _UdpProtocol(asyncio.DatagramProtocol):
    def __init__(self, hub: UdpAudioHub) -> None:
        self._hub = hub

    def datagram_received(self, data: bytes, addr: tuple[str, int]) -> None:
        asyncio.create_task(self._hub.handle_datagram(data, addr))


def udp_port() -> int | None:
    raw = os.environ.get("UDP_AUDIO_PORT", "").strip()
    if not raw:
        return None
    try:
        port = int(raw)
        return port if port > 0 else None
    except ValueError:
        return None


def get_udp_hub() -> UdpAudioHub | None:
    return _hub


async def start_udp_hub() -> UdpAudioHub | None:
    global _hub
    port = udp_port()
    if not port:
        return None
    if _hub is None:
        _hub = UdpAudioHub(port)
        await _hub.start()
    return _hub


async def stop_udp_hub() -> None:
    global _hub
    if _hub:
        _hub.close()
        _hub = None
