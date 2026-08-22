"""Shared helpers for LingBot Gradio + FastAPI servers."""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any

LOG = logging.getLogger("lingbot")


def setup_logging(name: str) -> logging.Logger:
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO"),
        format=f"%(asctime)s [{name}] %(levelname)s %(message)s",
    )
    return logging.getLogger(name)


def env_path(key: str, default: str) -> Path:
    return Path(os.getenv(key, default).strip() or default)


def ensure_gradio_temp() -> str:
    d = (os.getenv("GRADIO_TEMP_DIR") or "/output/gradio").strip()
    os.makedirs(d, mode=0o755, exist_ok=True)
    os.environ["GRADIO_TEMP_DIR"] = d
    return os.path.realpath(d)


def append_bootstrap(msg: str) -> None:
    app = os.getenv("LINGBOT_APP", "lingbot")
    path = Path(os.getenv("LINGBOT_BOOT_ATTEMPTS_FILE", f"/workspace/{app}/bootstrap.log"))
    path.parent.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    with path.open("a", encoding="utf-8") as fh:
        fh.write(f"{ts} {msg}\n")
    print(f"[{app}] {msg}", flush=True)


def set_ready_phase(detail: str = "ready") -> None:
    app = os.getenv("LINGBOT_APP", "lingbot")
    phase = Path(os.getenv("LINGBOT_BOOT_PHASE_FILE", f"/workspace/{app}/.boot-phase"))
    phase.write_text(
        json.dumps({"status": "ready", "ready": True, "phase": "ready", "detail": detail}),
        encoding="utf-8",
    )


def vram_info() -> dict[str, Any]:
    try:
        import torch

        if not torch.cuda.is_available():
            return {"cuda": False}
        free, total = torch.cuda.mem_get_info()
        return {
            "cuda": True,
            "device": torch.cuda.get_device_name(0),
            "free_gb": round(free / 1e9, 2),
            "total_gb": round(total / 1e9, 2),
            "allocated_gb": round(torch.cuda.memory_allocated() / 1e9, 2),
        }
    except Exception as exc:
        return {"cuda": False, "error": str(exc)}


def find_weight_file(model_dir: Path, names: list[str]) -> Path | None:
    for n in names:
        p = model_dir / n
        if p.is_file():
            return p
    # search one level
    for n in names:
        matches = list(model_dir.rglob(n))
        if matches:
            return matches[0]
    return None


def mount_gradio(app, demo, path: str = "/ui"):
    """Mount Gradio under ``path`` with root_path so file URLs include the prefix.

    Olares ingress hits the app at the entrance root. Without root_path, Gradio
    emits ``/gradio_api/file=...`` which 404s; correct is ``/ui/gradio_api/...``.
    Also register redirects from bare ``/gradio_api/*`` → ``/ui/gradio_api/*``.
    """
    import gradio as gr
    from fastapi import Request
    from fastapi.responses import RedirectResponse
    from urllib.parse import quote

    mount = (path or "/ui").rstrip("/") or "/ui"
    root = (os.getenv("GRADIO_ROOT_PATH") or "").strip() or mount

    @app.api_route("/gradio_api/{rest:path}", methods=["GET", "HEAD"])
    def _gradio_api_redir(rest: str, request: Request):
        q = f"?{request.url.query}" if request.url.query else ""
        return RedirectResponse(
            url=f"{mount}/gradio_api/{quote(rest, safe='/=')}{q}",
            status_code=307,
        )

    return gr.mount_gradio_app(app, demo, path=mount, root_path=root)
