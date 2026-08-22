"""MCP (streamable HTTP) tools for Docker Builder One."""

from __future__ import annotations

import base64
import json
import os
import tempfile
from pathlib import Path

from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings

from builder import DEFAULT_LOG_PAGE, build_backend, dockerfile_relpath, kaniko_executor_path
from deps import manager

MCP_MAX_ZIP_MB = max(1, int(os.environ.get("MCP_MAX_ZIP_MB", "150")))


def _mcp_transport_security() -> TransportSecuritySettings:
    """FastMCP defaults host=127.0.0.1 and auto-enables localhost-only Host checks.

    Cursor/Olares reach /mcp/mcp with the entrance Host header (not 127.0.0.1) → 421 Invalid Host.
    """
    enable = os.environ.get("MCP_DNS_REBINDING_PROTECTION", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )
    hosts = [h.strip() for h in os.environ.get("MCP_ALLOWED_HOSTS", "").split(",") if h.strip()]
    origins = [o.strip() for o in os.environ.get("MCP_ALLOWED_ORIGINS", "").split(",") if o.strip()]
    if enable and not hosts:
        hosts = [
            "127.0.0.1:*",
            "localhost:*",
            "[::1]:*",
            "dockerbuilderone:*",
        ]
    return TransportSecuritySettings(
        enable_dns_rebinding_protection=enable,
        allowed_hosts=hosts,
        allowed_origins=origins,
    )


mcp = FastMCP(
    "Docker Builder One",
    host="0.0.0.0",
    stateless_http=True,
    json_response=True,
    transport_security=_mcp_transport_security(),
)


def _json(data: object) -> str:
    return json.dumps(data, indent=2, ensure_ascii=False)


@mcp.tool()
async def health_check() -> str:
    """Builder status: backend, Kaniko path, GitHub token, project count, queue."""
    return _json(
        {
            "status": "ok",
            "build_backend": build_backend(),
            "kaniko_executor": str(kaniko_executor_path() or ""),
            "github_token_configured": bool(os.environ.get("GITHUB_TOKEN", "").strip()),
            "projects": len(manager.list_projects()),
            "queue": manager.queue_status(),
            "settings": manager.get_settings(),
        }
    )


@mcp.tool()
async def list_projects() -> str:
    """List uploaded Dockerfile projects (name, dockerfile path, file sample)."""
    return _json({"projects": manager.list_projects()})


@mcp.tool()
async def upload_project(name: str, zip_base64: str) -> str:
    """Upload a project zip (base64). Name: alphanumeric/dash. Max size MCP_MAX_ZIP_MB (default 150)."""
    try:
        safe_name = manager.sanitize_name(name)
    except ValueError as exc:
        return _json({"error": str(exc)})

    try:
        raw = base64.b64decode(zip_base64, validate=True)
    except Exception as exc:
        return _json({"error": f"Invalid base64: {exc}"})

    max_bytes = MCP_MAX_ZIP_MB * 1024 * 1024
    if len(raw) > max_bytes:
        return _json(
            {
                "error": f"Zip too large ({len(raw)} bytes). Max {MCP_MAX_ZIP_MB} MiB.",
            }
        )
    if len(raw) < 4 or raw[:2] != b"PK":
        return _json({"error": "Data does not look like a zip file (expected PK header)."})

    with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
        tmp_path = Path(tmp.name)
        tmp.write(raw)

    try:
        dest = await manager.create_project_from_zip(safe_name, tmp_path)
    except Exception as exc:
        return _json({"error": str(exc)})
    finally:
        tmp_path.unlink(missing_ok=True)

    rel = dockerfile_relpath(dest)
    return _json(
        {
            "name": safe_name,
            "path": str(dest),
            "has_dockerfile": rel is not None,
            "dockerfile": rel,
            "files": sorted(p.name for p in dest.iterdir())[:40] if dest.is_dir() else [],
        }
    )


@mcp.tool()
async def list_builds() -> str:
    """List all builds (newest first) and current queue status."""
    return _json(
        {
            "builds": manager.list_jobs(),
            "queue": manager.queue_status(),
        }
    )


@mcp.tool()
async def get_build_queue() -> str:
    """Current running build and queued jobs."""
    job = manager.current_job()
    payload = None
    if job:
        payload = job.snapshot(queue_position=manager.queue_position(job.id))
    return _json({"build": payload, "queue": manager.queue_status()})


@mcp.tool()
async def get_build(job_id: str) -> str:
    """Get one build by id (state, image, error, log_lines, timestamps)."""
    job = manager.get_job(job_id)
    if not job:
        return _json({"error": "Build not found", "job_id": job_id})
    return _json(job.snapshot(queue_position=manager.queue_position(job_id)))


@mcp.tool()
async def get_build_logs(
    job_id: str,
    limit: int = DEFAULT_LOG_PAGE,
    before_line: int | None = None,
) -> str:
    """Tail build logs. before_line = 0-based line index to read older chunk (pagination)."""
    if not manager.get_job(job_id):
        return _json({"error": "Build not found", "job_id": job_id})
    try:
        data = manager.read_logs(job_id, before=before_line, limit=limit)
    except KeyError:
        return _json({"error": "Build not found", "job_id": job_id})
    return _json(data)


@mcp.tool()
async def get_settings() -> str:
    """Kaniko log verbosity and other builder UI settings (not env vars)."""
    return _json(manager.get_settings())


@mcp.tool()
async def set_kaniko_verbose(enabled: bool) -> str:
    """Toggle Kaniko debug logs. False=info (default, smaller logs). True=debug (verbose)."""
    return _json(manager.set_kaniko_verbose(enabled))


@mcp.tool()
async def start_build(
    project: str,
    image: str,
    dockerfile: str = "Dockerfile",
) -> str:
    """Queue a Kaniko build+push. image must start with ghcr.io/ (e.g. ghcr.io/user/app:1.0.0)."""
    if not image.startswith("ghcr.io/"):
        return _json({"error": "Image must start with ghcr.io/"})
    try:
        job = await manager.enqueue_build(project, image, dockerfile)
    except FileNotFoundError as exc:
        return _json({"error": str(exc)})
    except Exception as exc:
        return _json({"error": str(exc)})
    snap = job.snapshot(queue_position=manager.queue_position(job.id))
    snap["queue"] = manager.queue_status()
    return _json(snap)


@mcp.tool()
async def cancel_build(job_id: str) -> str:
    """Cancel a queued or running build."""
    try:
        job = await manager.cancel_build(job_id)
    except KeyError:
        return _json({"error": "Build not found", "job_id": job_id})
    except RuntimeError as exc:
        return _json({"error": str(exc)})
    snap = job.snapshot()
    snap["queue"] = manager.queue_status()
    return _json(snap)


def _normalize_accept_header(scope: dict) -> dict:
    """MCP SDK rejects Accept: */* with 406; Cursor often sends that on CallTool POST."""
    if scope.get("type") != "http":
        return scope
    headers = list(scope.get("headers") or [])
    accept_raw = next((v.decode("latin-1") for k, v in headers if k.lower() == b"accept"), "")
    accept = accept_raw.strip().lower()
    needs_json = "application/json" not in accept_raw
    needs_sse = "text/event-stream" not in accept_raw
    if accept in ("*/*", "*") or (needs_json and needs_sse):
        headers = [(k, v) for k, v in headers if k.lower() != b"accept"]
        headers.append((b"accept", b"application/json, text/event-stream"))
        return {**scope, "headers": headers}
    return scope


class _McpClientCompatMiddleware:
    """Work around strict MCP transport Accept checks for Cursor and other HTTP clients."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        scope = _normalize_accept_header(scope)
        await self.app(scope, receive, send)


def mcp_http_app():
    """ASGI app for streamable HTTP (mount on FastAPI)."""
    return _McpClientCompatMiddleware(mcp.streamable_http_app())
