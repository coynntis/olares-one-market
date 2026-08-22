"""Docker Builder One — upload Dockerfile folders, build, push to ghcr.io."""

from __future__ import annotations

import json
import os
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from builder import (
    DEFAULT_LOG_PAGE,
    build_backend,
    dockerfile_relpath,
    kaniko_executor_path,
)
from deps import STATIC_DIR, manager
from mcp_server import mcp, mcp_http_app


@asynccontextmanager
async def lifespan(app: FastAPI):
    manager.ensure_dirs()
    manager._start_queue_worker()
    async with mcp.session_manager.run():
        yield


app = FastAPI(title="Docker Builder One", version="1.1.0", lifespan=lifespan)

_cors_origin = os.environ.get("MCP_CORS_ORIGIN", "*").strip() or "*"
_cors_origins = [o.strip() for o in _cors_origin.split(",") if o.strip()]
# credentials + wildcard origin is invalid CORS; breaks some browser-based MCP clients.
_cors_allow_credentials = "*" not in _cors_origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=_cors_allow_credentials,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/livez")
async def livez() -> dict:
    """Liveness only — no disk / manager work. Heavy builds must not block this."""
    return {"status": "ok"}


@app.get("/health")
async def health() -> dict:
    return {
        "status": "ok",
        "build_backend": build_backend(),
        "kaniko_executor": str(kaniko_executor_path() or ""),
        "github_token_configured": bool(os.environ.get("GITHUB_TOKEN", "").strip()),
        "projects": len(manager.list_projects()),
        "queue": manager.queue_status(),
        "settings": manager.get_settings(),
        "mcp": {
            "transport": "streamable-http",
            "url": "/mcp/mcp",
            "note": "Mount path /mcp + SDK path /mcp. Use in-cluster http://dockerbuilderone:8080/mcp/mcp",
        },
    }


@app.get("/api/mcp")
async def mcp_info() -> dict:
    return {
        "transport": "streamable-http",
        "endpoints": ["/mcp/mcp"],
        "tools": [
            "health_check",
            "list_projects",
            "upload_project",
            "list_builds",
            "get_build_queue",
            "get_build",
            "get_build_logs",
            "get_settings",
            "set_kaniko_verbose",
            "start_build",
            "cancel_build",
        ],
    }


@app.get("/api/projects")
async def list_projects() -> dict:
    return {"projects": manager.list_projects()}


@app.post("/api/projects/upload")
async def upload_project(
    name: str = Form(...),
    archive: UploadFile = File(...),
) -> dict:
    if not archive.filename:
        raise HTTPException(400, "Missing file")
    if not archive.filename.lower().endswith(".zip"):
        raise HTTPException(400, "Upload a .zip of your Dockerfile folder")

    try:
        safe_name = manager.sanitize_name(name)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc

    with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
        tmp_path = Path(tmp.name)
        content = await archive.read()
        tmp.write(content)

    try:
        dest = await manager.create_project_from_zip(safe_name, tmp_path)
    except Exception as exc:
        raise HTTPException(400, str(exc)) from exc
    finally:
        tmp_path.unlink(missing_ok=True)

    rel = dockerfile_relpath(dest)
    return {
        "name": safe_name,
        "path": str(dest),
        "has_dockerfile": rel is not None,
        "dockerfile": rel,
        "files": sorted(p.name for p in dest.iterdir())[:40] if dest.is_dir() else [],
    }


@app.get("/api/builds")
async def list_builds() -> dict:
    return {"builds": manager.list_jobs(), "queue": manager.queue_status()}


@app.get("/api/builds/queue")
async def build_queue() -> dict:
    return manager.queue_status()


@app.get("/api/builds/current")
async def current_build() -> dict:
    job = manager.current_job()
    payload = None
    if job:
        payload = job.snapshot(queue_position=manager.queue_position(job.id))
    return {
        "build": payload,
        "queue": manager.queue_status(),
    }


@app.get("/api/settings")
async def get_settings() -> dict:
    return manager.get_settings()


@app.put("/api/settings")
async def update_settings(payload: dict) -> dict:
    if "kaniko_verbose" in payload:
        return manager.set_kaniko_verbose(bool(payload["kaniko_verbose"]))
    return manager.get_settings()


@app.post("/api/builds")
async def start_build(
    project: str = Form(...),
    image: str = Form(...),
    dockerfile: str = Form("Dockerfile"),
) -> dict:
    if not image.startswith("ghcr.io/"):
        raise HTTPException(400, "Image must start with ghcr.io/")
    try:
        job = await manager.enqueue_build(project, image, dockerfile)
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc
    snap = job.snapshot(queue_position=manager.queue_position(job.id))
    snap["queue"] = manager.queue_status()
    return snap


@app.post("/api/builds/{job_id}/cancel")
async def cancel_build(job_id: str) -> dict:
    try:
        job = await manager.cancel_build(job_id)
    except KeyError:
        raise HTTPException(404, "Build not found") from None
    except RuntimeError as exc:
        raise HTTPException(409, str(exc)) from exc
    snap = job.snapshot()
    snap["queue"] = manager.queue_status()
    return snap


@app.get("/api/builds/{job_id}")
async def get_build(job_id: str) -> dict:
    job = manager.get_job(job_id)
    if not job:
        raise HTTPException(404, "Build not found")
    return job.snapshot(queue_position=manager.queue_position(job_id))


@app.get("/api/builds/{job_id}/logs")
async def get_build_logs(
    job_id: str,
    limit: int = DEFAULT_LOG_PAGE,
    before: int | None = None,
) -> dict:
    if not manager.get_job(job_id):
        raise HTTPException(404, "Build not found")
    try:
        return manager.read_logs(job_id, before=before, limit=limit)
    except KeyError:
        raise HTTPException(404, "Build not found") from None


@app.get("/api/builds/{job_id}/stream")
async def stream_build(job_id: str, from_line: int = 0) -> StreamingResponse:
    job = manager.get_job(job_id)
    if not job:
        raise HTTPException(404, "Build not found")

    async def event_stream():
        async for chunk in manager.stream_logs(job_id, from_line):
            line = chunk.removesuffix("\n")
            yield f"data: {line}\n\n"
        job = manager.get_job(job_id)
        if job:
            payload = json.dumps(
                {"state": job.state.value, "error": job.error or ""},
                ensure_ascii=False,
            )
            yield f"event: done\ndata: {payload}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


# MCP streamable HTTP (Cursor, llama.cpp, agents). Client URL: <base>/mcp/mcp
app.mount("/mcp", mcp_http_app())


if STATIC_DIR.is_dir():
    app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")
else:

    @app.get("/")
    async def root() -> FileResponse:
        raise HTTPException(500, "Static UI missing")
