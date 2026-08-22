"""Abstract send/close surface for xiaozhi session (websockets vs Starlette)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class XiaozhiWsPort(Protocol):
    async def send_text(self, data: str) -> None: ...
    async def send_bytes(self, data: bytes) -> None: ...
    async def close(self, code: int = 1000, reason: str = "") -> None: ...
