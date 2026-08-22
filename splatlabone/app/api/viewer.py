"""REST + WebSocket proxy for gsplat Viser viewer sessions."""

from __future__ import annotations

import asyncio

import httpx
import websockets
from fastapi import APIRouter, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import Response
from starlette.websockets import WebSocketState

from deps import manager
from viewer.session import viewer_manager

router = APIRouter(prefix="/api/v1/scenes", tags=["viewer"])

_HOP_BY_HOP = frozenset(
    {
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailers",
        "transfer-encoding",
        "upgrade",
        "host",
        "content-length",
    }
)


def _get_session(job_id: str):
    sess = viewer_manager.get(job_id)
    if not sess or sess.proc.poll() is not None:
        raise HTTPException(404, "Viewer not running — POST /viewer/start first")
    return sess


@router.post("/{job_id}/viewer/start")
async def start_viewer(job_id: str) -> dict:
    job = manager.get_job(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    result_dir = manager.output_path(job_id) / "splat"
    try:
        sess = viewer_manager.start(job_id, result_dir)
    except Exception as exc:
        raise HTTPException(400, str(exc)) from exc
    return {
        "job_id": job_id,
        "port": sess.port,
        "ckpt": str(sess.ckpt_path),
        "embed_url": f"/viewer.html?job={job_id}",
        "proxy_prefix": sess.url_path,
        "ws_hint": f"/api/v1/scenes/{job_id}/viewer/proxy/",
    }


@router.post("/{job_id}/viewer/stop")
async def stop_viewer(job_id: str) -> dict:
    viewer_manager.stop(job_id)
    return {"job_id": job_id, "stopped": True}


@router.get("/{job_id}/viewer/status")
async def viewer_status(job_id: str) -> dict:
    sess = viewer_manager.get(job_id)
    if not sess:
        return {"active": False, "job_id": job_id}
    alive = sess.proc.poll() is None
    return {
        "active": alive,
        "job_id": job_id,
        "port": sess.port,
        "ckpt": str(sess.ckpt_path),
        "proxy_prefix": sess.url_path,
    }


async def _relay_ws(client: WebSocket, upstream_url: str) -> None:
    async with websockets.connect(
        upstream_url,
        max_size=None,
        ping_interval=None,
        close_timeout=5,
    ) as upstream:

        async def client_to_upstream() -> None:
            try:
                while True:
                    msg = await client.receive()
                    if msg.get("type") == "websocket.disconnect":
                        break
                    if "text" in msg:
                        await upstream.send(msg["text"])
                    elif "bytes" in msg:
                        await upstream.send(msg["bytes"])
            except WebSocketDisconnect:
                pass

        async def upstream_to_client() -> None:
            async for message in upstream:
                if client.client_state != WebSocketState.CONNECTED:
                    break
                if isinstance(message, str):
                    await client.send_text(message)
                else:
                    await client.send_bytes(message)

        tasks = [
            asyncio.create_task(client_to_upstream()),
            asyncio.create_task(upstream_to_client()),
        ]
        done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        for t in pending:
            t.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
        for t in done:
            if exc := t.exception():
                if not isinstance(exc, (WebSocketDisconnect, asyncio.CancelledError)):
                    raise exc


@router.websocket("/{job_id}/viewer/proxy")
@router.websocket("/{job_id}/viewer/proxy/{path:path}")
async def proxy_viewer_ws(websocket: WebSocket, job_id: str, path: str = "") -> None:
    sess = viewer_manager.get(job_id)
    if not sess or sess.proc.poll() is not None:
        await websocket.close(code=4404, reason="Viewer not running")
        return
    await websocket.accept()
    qs = websocket.scope.get("query_string", b"").decode()
    upstream_path = path or ""
    upstream_url = f"ws://127.0.0.1:{sess.port}/{upstream_path}"
    if qs:
        upstream_url += f"?{qs}"
    try:
        await _relay_ws(websocket, upstream_url)
    except Exception:
        if websocket.client_state == WebSocketState.CONNECTED:
            await websocket.close(code=1011)
    finally:
        viewer_manager.touch(job_id)


@router.api_route(
    "/{job_id}/viewer/proxy",
    methods=["GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"],
)
@router.api_route(
    "/{job_id}/viewer/proxy/{path:path}",
    methods=["GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"],
)
async def proxy_viewer_http(job_id: str, request: Request, path: str = "") -> Response:
    if request.headers.get("upgrade", "").lower() == "websocket":
        raise HTTPException(426, "Use WebSocket upgrade on this path")
    sess = _get_session(job_id)
    target = f"http://127.0.0.1:{sess.port}/{path}"
    if request.url.query:
        target += f"?{request.url.query}"
    headers = {k: v for k, v in request.headers.items() if k.lower() not in _HOP_BY_HOP}
    body = await request.body()
    async with httpx.AsyncClient(timeout=120.0, follow_redirects=True) as client:
        upstream = await client.request(request.method, target, headers=headers, content=body)
    resp_headers = {k: v for k, v in upstream.headers.items() if k.lower() not in _HOP_BY_HOP}
    viewer_manager.touch(job_id)
    return Response(content=upstream.content, status_code=upstream.status_code, headers=resp_headers)
