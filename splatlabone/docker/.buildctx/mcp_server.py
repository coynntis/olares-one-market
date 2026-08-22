"""MCP (streamable HTTP) tools for SplatLab One."""

from __future__ import annotations

import base64
import json
import os
import tempfile
from pathlib import Path

from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings

from deps import PRESETS_DIR, manager, new_id
from jobs.models import JobConfig, RealtimeMode, StageOverrides
from pipeline.ingest import ingest_colmap_zip
from pipeline.ingest import ingest_images_zip as do_ingest_images_zip
from pipeline.ingest import ingest_video
from pipeline.registry import get_guide_section, list_presets, load_preset

MCP_MAX_ZIP_MB = max(1, int(os.environ.get("MCP_MAX_ZIP_MB", "200")))


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
        hosts = ["127.0.0.1:*", "localhost:*", "[::1]:*", "splatlabone:*"]
    return TransportSecuritySettings(
        enable_dns_rebinding_protection=enable,
        allowed_hosts=hosts,
        allowed_origins=origins,
    )


mcp = FastMCP(
    "SplatLab One",
    host="0.0.0.0",
    stateless_http=True,
    json_response=True,
    transport_security=_mcp_transport_security(),
)


def _json(data: object) -> str:
    return json.dumps(data, indent=2, ensure_ascii=False)


@mcp.tool()
async def health_check() -> str:
    """GPU, queue, disk paths, dataset/job counts."""
    return _json(
        {
            "status": "ok",
            "gpu": manager.gpu_info(),
            "queue": manager.queue_status(),
            "paths": {
                "uploads": str(manager.uploads_dir),
                "workspaces": str(manager.workspaces_dir),
                "outputs": str(manager.outputs_dir),
            },
            "datasets": len(manager.list_datasets()),
            "jobs": len(manager.list_jobs()),
        }
    )


@mcp.tool()
async def list_presets() -> str:
    """Named presets and stage matrix summary."""
    return _json({"presets": list_presets(PRESETS_DIR)})


@mcp.tool()
async def get_preset(name: str) -> str:
    """Full preset config YAML-equivalent JSON."""
    try:
        return _json(load_preset(name, PRESETS_DIR))
    except KeyError:
        return _json({"error": f"Unknown preset: {name}"})


@mcp.tool()
async def get_guide_section(anchor: str) -> str:
    """SOTA guide snippet by anchor (overview, glomap, 2dgs, mcp, docker_build, ...)."""
    return _json({"anchor": anchor, "content": get_guide_section(anchor)})


@mcp.tool()
async def list_datasets() -> str:
    """Uploaded datasets under uploads/."""
    return _json({"datasets": manager.list_datasets()})


def _decode_zip(zip_base64: str) -> bytes:
    raw = base64.b64decode(zip_base64, validate=True)
    max_bytes = MCP_MAX_ZIP_MB * 1024 * 1024
    if len(raw) > max_bytes:
        raise ValueError(f"Zip too large ({len(raw)} bytes). Max {MCP_MAX_ZIP_MB} MiB.")
    if len(raw) < 4 or raw[:2] != b"PK":
        raise ValueError("Data does not look like a zip file")
    return raw


@mcp.tool()
async def ingest_images_zip(name: str, zip_base64: str) -> str:
    """Base64 zip of images → dataset_id."""
    try:
        raw = _decode_zip(zip_base64)
    except Exception as exc:
        return _json({"error": str(exc)})
    dataset_id = new_id("ds_")
    dataset_dir = manager.uploads_dir / dataset_id
    dataset_dir.mkdir(parents=True)
    with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
        tmp_path = Path(tmp.name)
        tmp.write(raw)
    try:
        meta = {"name": name or dataset_id}
        do_ingest_images_zip(tmp_path, dataset_dir, meta)
    except Exception as exc:
        return _json({"error": str(exc)})
    finally:
        tmp_path.unlink(missing_ok=True)
    return _json({"dataset_id": dataset_id, "meta": meta})


@mcp.tool()
async def ingest_video_base64(
    name: str,
    video_base64: str,
    fps: float = 2.0,
    max_frames: int = 300,
) -> str:
    """Base64 mp4/webm → extracted frames dataset."""
    try:
        raw = base64.b64decode(video_base64, validate=True)
    except Exception as exc:
        return _json({"error": str(exc)})
    dataset_id = new_id("ds_")
    dataset_dir = manager.uploads_dir / dataset_id
    dataset_dir.mkdir(parents=True)
    vpath = dataset_dir / "source.mp4"
    vpath.write_bytes(raw)
    meta = {"name": name or dataset_id}
    try:
        ingest_video(vpath, dataset_dir, meta, fps=fps, max_frames=max_frames)
    except Exception as exc:
        return _json({"error": str(exc)})
    return _json({"dataset_id": dataset_id, "meta": meta})


@mcp.tool()
async def ingest_colmap_zip(name: str, zip_base64: str) -> str:
    """Base64 COLMAP workspace (images/ + sparse/)."""
    try:
        raw = _decode_zip(zip_base64)
    except Exception as exc:
        return _json({"error": str(exc)})
    dataset_id = new_id("ds_")
    dataset_dir = manager.uploads_dir / dataset_id
    dataset_dir.mkdir(parents=True)
    with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
        tmp_path = Path(tmp.name)
        tmp.write(raw)
    try:
        meta = {"name": name or dataset_id}
        ingest_colmap_zip(tmp_path, dataset_dir, meta)
    except Exception as exc:
        return _json({"error": str(exc)})
    finally:
        tmp_path.unlink(missing_ok=True)
    return _json({"dataset_id": dataset_id, "meta": meta})


@mcp.tool()
async def create_job(
    dataset_id: str,
    preset: str = "quality",
    realtime_mode: str = "none",
    overrides_json: str = "{}",
) -> str:
    """Start pipeline job. realtime_mode: none | geometry_preview | progressive_splat."""
    try:
        overrides = StageOverrides.model_validate_json(overrides_json or "{}")
        mode = RealtimeMode(realtime_mode)
    except Exception as exc:
        return _json({"error": str(exc)})
    try:
        manager.dataset_path(dataset_id)
    except FileNotFoundError:
        return _json({"error": f"Dataset not found: {dataset_id}"})
    cfg = JobConfig(dataset_id=dataset_id, preset=preset, overrides=overrides, realtime_mode=mode)
    job = manager.create_job(cfg)
    return _json(job.snapshot().model_dump())


@mcp.tool()
async def get_job(job_id: str) -> str:
    """Job status, stage, metrics, artifact paths."""
    job = manager.get_job(job_id)
    if not job:
        return _json({"error": "Job not found"})
    return _json(job.snapshot().model_dump())


@mcp.tool()
async def get_job_logs(job_id: str, offset: int = 0, limit: int = 200) -> str:
    """Paginated pipeline log."""
    try:
        return _json(manager.read_logs(job_id, offset=offset, limit=limit))
    except KeyError:
        return _json({"error": "Job not found"})


@mcp.tool()
async def subscribe_job_events(job_id: str) -> str:
    """Document SSE polling: GET /api/v1/jobs/{id}/events or poll get_job."""
    return _json(
        {
            "job_id": job_id,
            "sse_url": f"/api/v1/jobs/{job_id}/events",
            "poll_tool": "get_job",
            "events": ["stage_start", "log", "checkpoint_ready", "geometry_preview", "job_complete", "error"],
        }
    )


@mcp.tool()
async def cancel_job(job_id: str) -> str:
    """Cancel queued or running job."""
    try:
        job = manager.cancel_job(job_id)
    except KeyError:
        return _json({"error": "Job not found"})
    except RuntimeError as exc:
        return _json({"error": str(exc)})
    return _json(job.snapshot().model_dump())


@mcp.tool()
async def list_scenes() -> str:
    """Completed jobs with splat artifacts."""
    return _json({"scenes": manager.list_scenes()})


@mcp.tool()
async def get_scene_urls(job_id: str, base_url: str = "") -> str:
    """Viewer, splat download, shared-entrance style URLs."""
    job = manager.get_job(job_id)
    if not job:
        return _json({"error": "Job not found"})
    base = base_url.rstrip("/")
    return _json(
        {
            "job_id": job_id,
            "viewer": f"{base}/viewer.html?job={job_id}" if base else f"/viewer.html?job={job_id}",
            "splat": f"{base}/api/v1/scenes/{job_id}/splat" if base else f"/api/v1/scenes/{job_id}/splat",
            "ply": f"{base}/api/v1/scenes/{job_id}/ply" if base else f"/api/v1/scenes/{job_id}/ply",
            "artifacts": job.artifacts,
            "shared_note": "External: http://{route-id}.shared.olares.com/...",
        }
    )


@mcp.tool()
async def start_live_session(mode: str = "geometry_preview") -> str:
    """Create geometry_preview WS session; connect to ws_url."""
    sess = manager.create_live_session(mode)
    return _json(
        {
            **sess,
            "ws_url": f"/api/v1/live/{sess['id']}",
            "protocol": "Send JSON {type:frame_base64,data:...} or binary JPEG chunks.",
        }
    )


def _normalize_accept_header(scope: dict) -> dict:
    if scope.get("type") != "http":
        return scope
    headers = list(scope.get("headers") or [])
    accept_raw = next((v.decode("latin-1") for k, v in headers if k.lower() == b"accept"), "")
    needs_json = "application/json" not in accept_raw
    needs_sse = "text/event-stream" not in accept_raw
    accept = accept_raw.strip().lower()
    if accept in ("*/*", "*") or (needs_json and needs_sse):
        headers = [(k, v) for k, v in headers if k.lower() != b"accept"]
        headers.append((b"accept", b"application/json, text/event-stream"))
        return {**scope, "headers": headers}
    return scope


class _McpClientCompatMiddleware:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        scope = _normalize_accept_header(scope)
        await self.app(scope, receive, send)


def mcp_http_app():
    return _McpClientCompatMiddleware(mcp.streamable_http_app())
