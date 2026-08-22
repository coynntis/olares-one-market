"""SplatLab One — FastAPI + static UI + MCP."""

from __future__ import annotations

import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

# App root on PYTHONPATH
APP_ROOT = Path(__file__).resolve().parent
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from api import ingest, jobs, live, meta, scenes
from deps import STATIC_DIR, ensure_dirs, manager
from mcp_server import mcp, mcp_http_app


@asynccontextmanager
async def lifespan(app: FastAPI):
    ensure_dirs()
    manager.ensure_dirs()
    manager._start_queue_worker()
    async with mcp.session_manager.run():
        yield


app = FastAPI(title="SplatLab One", version="1.0.0", lifespan=lifespan)

_cors_origin = os.environ.get("MCP_CORS_ORIGIN", "*").strip() or "*"
_cors_origins = [o.strip() for o in _cors_origin.split(",") if o.strip()]
_cors_allow_credentials = "*" not in _cors_origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=_cors_allow_credentials,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(ingest.router)
app.include_router(jobs.router)
app.include_router(scenes.router)
app.include_router(live.router)
app.include_router(meta.router)


@app.get("/healthz")
async def healthz() -> dict:
    return {
        "status": "ok",
        "gpu": manager.gpu_info(),
        "queue": manager.queue_status(),
        "datasets": len(manager.list_datasets()),
        "jobs": len(manager.list_jobs()),
        "mcp": {"transport": "streamable-http", "url": "/mcp/mcp"},
    }


@app.get("/api/mcp")
async def mcp_info() -> dict:
    return {
        "transport": "streamable-http",
        "endpoints": ["/mcp/mcp"],
        "tools": [
            "health_check",
            "list_presets",
            "get_preset",
            "get_guide_section",
            "list_datasets",
            "ingest_images_zip",
            "ingest_video_base64",
            "ingest_colmap_zip",
            "create_job",
            "get_job",
            "get_job_logs",
            "cancel_job",
            "list_scenes",
            "get_scene_urls",
            "start_live_session",
        ],
        "note": "Image builds: use dockerbuilderone MCP, not SplatLab MCP.",
    }


app.mount("/mcp", mcp_http_app())

if STATIC_DIR.is_dir():
    app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")
else:

    @app.get("/")
    async def root() -> FileResponse:
        raise HTTPException(500, "Static UI missing")
