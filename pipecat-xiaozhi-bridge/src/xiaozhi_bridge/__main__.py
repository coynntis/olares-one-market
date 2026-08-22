"""Run ASGI app (static client + xiaozhi WebSocket)."""

from __future__ import annotations

import os
from pathlib import Path

import uvicorn


def main() -> None:
    here = Path(__file__).resolve().parent
    root = here.parent.parent
    dist = root / "client" / "dist"
    if dist.is_dir():
        os.environ.setdefault("STATIC_DIR", str(dist))
    uvicorn.run(
        "xiaozhi_bridge.asgi:app",
        host=os.environ.get("HOST", "0.0.0.0"),
        port=int(os.environ.get("PORT", "8000")),
        factory=False,
        log_level=os.environ.get("UVICORN_LOG_LEVEL", "info"),
    )


if __name__ == "__main__":
    main()
