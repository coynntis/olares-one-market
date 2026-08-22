#!/usr/bin/env python3
"""LingBot-Map Gradio + REST — reconstruct + open viser at /viser/.

Inputs: image uploads, video upload, or in-pod folder path.
Viser serves on VISER_PORT; FastAPI proxies /viser/* (HTTP + WebSocket).
Gradio shows a link to open /viser/ in a new tab (no iframe).
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import os
import shutil
import sys
import threading
import time
import uuid
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
sys.path.insert(0, os.path.dirname(__file__))

from _common import (  # noqa: E402
    append_bootstrap,
    ensure_gradio_temp,
    find_weight_file,
    mount_gradio,
    set_ready_phase,
    setup_logging,
    vram_info,
)

ensure_gradio_temp()
LOG = setup_logging("lingbotmapone")


def _patch_matplotlib_get_cmap() -> None:
    """Upstream lingbot_map uses ``matplotlib.cm.get_cmap`` (removed in mpl 3.9)."""
    try:
        import matplotlib
        import matplotlib.cm as cm

        if hasattr(cm, "get_cmap"):
            return

        def _get_cmap(name, lut=None):  # noqa: ANN001
            cmap = matplotlib.colormaps[name]
            if lut is not None and hasattr(cmap, "resampled"):
                return cmap.resampled(lut)
            return cmap

        cm.get_cmap = _get_cmap  # type: ignore[attr-defined]
        append_bootstrap("patched matplotlib.cm.get_cmap for mpl>=3.9")
    except Exception as exc:
        append_bootstrap(f"matplotlib get_cmap patch soft-fail: {exc}")


_patch_matplotlib_get_cmap()

import gradio as gr  # noqa: E402
import httpx  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import uvicorn  # noqa: E402
import websockets  # noqa: E402
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect  # noqa: E402
from fastapi.responses import HTMLResponse, RedirectResponse, Response  # noqa: E402
from starlette.websockets import WebSocketState  # noqa: E402

SERVER_PORT = int(os.getenv("SERVER_PORT", "7860"))
VISER_PORT = int(os.getenv("VISER_PORT", "8090"))
MODEL_DIR = Path(os.getenv("LINGBOT_MODEL_DIR", "/models/lingbotmapone/weights"))
SRC = os.getenv("LINGBOT_SRC", "")
CKPT_NAME = os.getenv("LINGBOT_MAP_CKPT", "lingbot-map-long.pt")
HF_ID = os.getenv("HF_REPO_ID", "robbyant/lingbot-map")
RUNS_DIR = Path(os.getenv("LINGBOT_MAP_RUNS", "/output/lingbotmapone/runs"))
DEFAULT_FOLDER = os.getenv(
    "LINGBOT_MAP_DEFAULT_FOLDER",
    "/workspace/lingbotmapone/src/lingbot-map/example/courthouse",
)

_lock = threading.Lock()  # model load
_job_lock = threading.Lock()  # only one reconstruct at a time
_cancel = threading.Event()
_ready = False
_load_error: str | None = None
_ckpt: Path | None = None
_model = None
_demo = None  # imported demo.py module
_viewer = None
_viewer_thread: threading.Thread | None = None
_runs: list[dict] = []  # recent job history (metadata only)
_loaded_run_id: str | None = None
_job = {
    "id": None,
    "phase": "idle",
    "detail": "",
    "error": None,
    "frames": 0,
    "elapsed_s": 0.0,
    "viser_url": "/viser/",
    "loaded_run_id": None,
    "queue_note": "one reconstruct at a time — Cancel / Stop viser if stuck",
}

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
        "content-encoding",  # httpx decompresses body — never forward gzip/br
        "x-frame-options",
        "content-security-policy",
    }
)


def _ensure_src() -> None:
    if SRC and SRC not in sys.path:
        sys.path.insert(0, SRC)


def _load_demo_module():
    global _demo
    if _demo is not None:
        return _demo
    _ensure_src()
    demo_path = Path(SRC) / "demo.py" if SRC else Path()
    if not demo_path.is_file():
        raise FileNotFoundError(f"demo.py not found under LINGBOT_SRC={SRC!r}")
    spec = importlib.util.spec_from_file_location("lingbot_map_demo", demo_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {demo_path}")
    mod = importlib.util.module_from_spec(spec)
    # Avoid demo.py's argparse side effects; only load helpers.
    sys.modules["lingbot_map_demo"] = mod
    spec.loader.exec_module(mod)
    _demo = mod
    return mod


def _resolve_ckpt() -> Path:
    global _ckpt
    if _ckpt and _ckpt.is_file():
        return _ckpt
    found = find_weight_file(MODEL_DIR, [CKPT_NAME, "lingbot-map.pt", "lingbot-map-long.pt"])
    if not found:
        raise FileNotFoundError(f"No map ckpt under {MODEL_DIR}")
    _ckpt = found
    return found


def _smoke_or_load_model():
    """Load GCTStream once (weights stay on GPU/CPU)."""
    global _model, _ready, _load_error
    with _lock:
        if _model is not None:
            return _model
        t0 = time.time()
        try:
            demo = _load_demo_module()
            ckpt = _resolve_ckpt()
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            # Prefer SDPA on Olares One — FlashInfer may be missing / ABI-fragile.
            args = SimpleNamespace(
                model_path=str(ckpt),
                image_size=518,
                patch_size=14,
                mode="streaming",
                enable_3d_rope=True,
                max_frame_num=1024,
                num_scale_frames=8,
                kv_cache_sliding_window=64,
                camera_num_iterations=4,
                use_sdpa=True,
            )
            append_bootstrap(f"loading GCTStream from {ckpt}")
            _model = demo.load_model(args, device)
            _ready = True
            append_bootstrap(
                f"map model ready {time.time() - t0:.1f}s device={device} vram={vram_info()}"
            )
            set_ready_phase("map_model_ready")
            return _model
        except Exception as exc:
            _load_error = str(exc)
            append_bootstrap(f"map load FAIL: {exc}")
            raise


def _stop_viewer() -> None:
    global _viewer, _viewer_thread, _loaded_run_id
    if _viewer is None:
        _loaded_run_id = None
        return
    try:
        stop = getattr(_viewer.server, "stop", None)
        if callable(stop):
            stop()
    except Exception as exc:
        append_bootstrap(f"viser stop soft-fail: {exc}")
    _viewer = None
    _viewer_thread = None
    _loaded_run_id = None
    _job["loaded_run_id"] = None
    time.sleep(0.8)
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass


def _start_viewer(
    pred_vis: dict,
    image_folder: str,
    conf_threshold: float,
    downsample: int,
    run_id: str,
) -> None:
    """Start viser without blocking the Gradio worker.

    Upstream ``PointCloudViewer.run()`` calls ``animate()``, which ends in
    ``while True`` — that hung the first Gradio job forever and queued all later
    uploads. We only construct the viewer (binds :VISER_PORT) then run animate
    on a daemon thread.
    """
    global _viewer, _viewer_thread, _loaded_run_id
    _ensure_src()
    from lingbot_map.vis import PointCloudViewer  # type: ignore

    _stop_viewer()
    append_bootstrap(f"starting PointCloudViewer on 0.0.0.0:{VISER_PORT} run={run_id}")
    viewer = PointCloudViewer(
        pred_dict=pred_vis,
        port=VISER_PORT,
        vis_threshold=float(conf_threshold),
        downsample_factor=int(downsample),
        point_size=0.00001,
        mask_sky=False,
        image_folder=image_folder,
    )
    _viewer = viewer
    _loaded_run_id = run_id
    _job["loaded_run_id"] = run_id

    def _animate_loop() -> None:
        try:
            # Do NOT call viewer.run() — animate() never returns.
            viewer.animate()
        except Exception as exc:
            append_bootstrap(f"viser animate ended: {exc}")

    _viewer_thread = threading.Thread(
        target=_animate_loop, daemon=True, name=f"viser-animate-{run_id[:8]}"
    )
    _viewer_thread.start()
    # Brief wait so /viser is accepting before we return the link
    deadline = time.time() + 5.0
    while time.time() < deadline:
        try:
            import socket

            with socket.create_connection(("127.0.0.1", VISER_PORT), timeout=0.2):
                break
        except OSError:
            time.sleep(0.1)


def _check_cancel() -> None:
    if _cancel.is_set():
        raise RuntimeError("cancelled by user")


def _free_model_runtime() -> None:
    """Drop KV cache / CUDA freelist between jobs so next reconstruct can start."""
    global _model
    try:
        if _model is not None and hasattr(_model, "clean_kv_cache"):
            _model.clean_kv_cache()
    except Exception as exc:
        append_bootstrap(f"clean_kv_cache soft-fail: {exc}")
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass


def cancel_job() -> tuple[str, str]:
    """Signal running reconstruct to abort; Gradio cancel also uses this."""
    _cancel.set()
    _job["phase"] = "cancelling"
    _job["detail"] = "cancel requested"
    append_bootstrap("job cancel requested")
    return "Cancel requested — waiting for current step to stop…", json.dumps(_job)


def stop_viser_ui() -> tuple[str, str, str]:
    _stop_viewer()
    _free_model_runtime()
    append_bootstrap("viser stopped / unloaded")
    payload = json.dumps(
        {**_job, "phase": _job.get("phase") or "idle", "loaded_run_id": None}
    )
    return (
        "Viser stopped. GPU cache cleared. You can Reconstruct again.",
        _viser_link_html(False),
        payload,
    )


def refresh_status() -> tuple[str, str, str]:
    runs = json.dumps(_runs[-10:], indent=2)
    link = _viser_link_html(_viewer is not None)
    return (
        f"phase={_job.get('phase')} detail={_job.get('detail')} "
        f"loaded={_loaded_run_id} busy={_job_lock.locked()} vram={vram_info()}",
        link,
        json.dumps({**_job, "runs_tail": _runs[-5:]}, indent=2),
    )


def _viser_link_html(ready: bool = True) -> str:
    """Always show a real clickable /viser/ link (Gradio lives under /ui/)."""
    ts = int(time.time())
    href = f"/viser/?t={ts}"
    # Absolute path from entrance root — not relative to /ui/
    open_btn = (
        f'<p style="margin:0.75rem 0">'
        f'<a href="{href}" target="_blank" rel="noopener" '
        f'style="display:inline-block;padding:0.65rem 1.1rem;background:#1a7f37;'
        f'color:#fff;text-decoration:none;border-radius:6px;font-weight:600;'
        f'font-size:1.05rem">Open Viser ↗</a>'
        f' &nbsp; <code style="font-size:1rem"><a href="{href}" target="_blank" '
        f'rel="noopener" style="color:inherit">/viser/</a></code></p>'
    )
    if not ready:
        return (
            open_btn
            + "<p style='opacity:.8;margin:0'>Viewer empty until <b>Reconstruct</b> finishes "
            "(link still works — shows waiting page if not loaded).</p>"
            "<p style='opacity:.75;font-size:.9rem;margin:0.35rem 0 0'>Queued forever? "
            "<b>Cancel job</b> → <b>Stop viser</b> → Reconstruct.</p>"
        )
    rid = _loaded_run_id or "?"
    return (
        open_btn
        + f"<p style='margin:0'><strong>Scene loaded</strong> · run <code>{rid}</code></p>"
        + "<p style='opacity:.75;font-size:.9rem;margin:0.35rem 0 0'>"
        "Stop viser before a big new job if VRAM is tight.</p>"
    )


def _viser_header_html() -> str:
    return (
        '<div style="padding:0.75rem 1rem;border:1px solid #2a2a2a;border-radius:8px;'
        'background:#111;margin:0.5rem 0 1rem">'
        '<div style="font-size:0.85rem;opacity:0.75;margin-bottom:0.35rem">3D viewer</div>'
        '<a href="/viser/" target="_blank" rel="noopener" '
        'style="font-size:1.25rem;font-weight:700;color:#6bcb77;text-decoration:none">'
        "Open Viser → /viser/</a>"
        '<div style="font-size:0.8rem;opacity:0.65;margin-top:0.35rem">'
        "Same app host as this UI (not under /ui/). Reconstruct first, then open.</div>"
        "</div>"
    )


def _materialize_inputs(
    images,
    video,
    folder: str,
    max_frames: int,
    fps: int,
) -> tuple[str | None, str | None, Path]:
    """Return (image_folder, video_path, work_dir). Prefer video > images > folder."""
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    work = RUNS_DIR / time.strftime("%Y%m%d-%H%M%S") / uuid.uuid4().hex[:8]
    work.mkdir(parents=True, exist_ok=True)

    def _as_path(x) -> Path | None:
        if x is None:
            return None
        if isinstance(x, (list, tuple)):
            return None
        p = Path(str(x))
        return p if p.is_file() else None

    vid = _as_path(video)
    if vid is not None:
        dest = work / vid.name
        shutil.copy2(vid, dest)
        return None, str(dest), work

    # Gradio File(multiple) → list of paths; Gallery → list of (path, caption) or numpy
    paths: list[Path] = []
    if images:
        items = images if isinstance(images, (list, tuple)) else [images]
        for i, item in enumerate(items):
            if item is None:
                continue
            if isinstance(item, (list, tuple)) and item:
                item = item[0]
            if isinstance(item, (str, Path)) and Path(item).is_file():
                paths.append(Path(item))
            elif isinstance(item, np.ndarray):
                from PIL import Image

                out = work / "images" / f"{i:06d}.png"
                out.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(item.astype(np.uint8)).save(out)
                paths.append(out)

    if paths:
        img_dir = work / "images"
        img_dir.mkdir(parents=True, exist_ok=True)
        for i, src in enumerate(paths[: max(1, int(max_frames))]):
            ext = src.suffix.lower() or ".jpg"
            dest = img_dir / f"{i:06d}{ext}"
            if src.resolve() != dest.resolve():
                shutil.copy2(src, dest)
        return str(img_dir), None, work

    folder = (folder or "").strip()
    if folder and Path(folder).is_dir():
        return folder, None, work

    raise ValueError("Provide image uploads, a video, or a valid image folder path")


def reconstruct(
    images,
    video,
    folder: str,
    max_frames: int,
    fps: int,
    mode: str,
    conf_threshold: float,
    downsample: int,
    keyframe_interval: int,
):
    """Gradio generator: status updates + viser link. One job at a time."""
    global _job

    if not _job_lock.acquire(blocking=False):
        busy = {
            **_job,
            "phase": "busy",
            "detail": "another reconstruct is running — Cancel job or wait",
        }
        yield (
            "BUSY — another reconstruct holds the GPU. Click Cancel job, then retry.",
            _viser_link_html(_viewer is not None),
            json.dumps(busy),
        )
        return

    _cancel.clear()
    run_id = uuid.uuid4().hex[:12]
    t0 = time.time()
    _job = {
        "id": run_id,
        "phase": "running",
        "detail": "starting",
        "error": None,
        "frames": 0,
        "elapsed_s": 0.0,
        "viser_url": "/viser/",
        "loaded_run_id": _loaded_run_id,
    }
    images_t = None
    predictions = None
    yield f"[{run_id}] Preparing inputs…", _viser_link_html(False), json.dumps(_job)

    try:
        _check_cancel()
        image_folder, video_path, work = _materialize_inputs(
            images, video, folder, max_frames, fps
        )
        demo = _load_demo_module()
        model = _smoke_or_load_model()
        device = next(model.parameters()).device
        _check_cancel()

        _job["detail"] = "loading frames"
        yield (
            f"[{run_id}] Loading frames (max={max_frames}, fps={fps})…",
            _viser_link_html(False),
            json.dumps(_job),
        )

        images_t, paths, resolved_folder = demo.load_images(
            image_folder=image_folder,
            video_path=video_path,
            fps=int(fps),
            first_k=int(max_frames) if max_frames and max_frames > 0 else None,
            stride=1,
            image_size=518,
            patch_size=14,
        )
        n = int(images_t.shape[0])
        _job["frames"] = n
        if n < 2:
            raise RuntimeError(f"Need ≥2 frames, got {n}")
        _check_cancel()

        if torch.cuda.is_available():
            dtype = (
                torch.bfloat16
                if torch.cuda.get_device_capability()[0] >= 8
                else torch.float16
            )
        else:
            dtype = torch.float32

        if dtype != torch.float32 and getattr(model, "aggregator", None) is not None:
            model.aggregator = model.aggregator.to(dtype=dtype)

        images_t = images_t.to(device)
        kf = max(1, int(keyframe_interval))
        if mode == "streaming" and n > 320 and kf == 1:
            kf = (n + 319) // 320

        _job["detail"] = f"inference:{mode}:frames={n}"
        yield (
            f"[{run_id}] Running {mode} inference on {n} frames…",
            _viser_link_html(False),
            json.dumps(_job),
        )
        _check_cancel()

        with torch.no_grad(), torch.amp.autocast(
            "cuda", dtype=dtype, enabled=device.type == "cuda"
        ):
            if mode == "windowed":
                predictions = model.inference_windowed(
                    images_t,
                    window_size=64,
                    overlap_size=16,
                    overlap_keyframes=None,
                    num_scale_frames=8,
                    keyframe_interval=kf,
                    output_device=torch.device("cpu"),
                )
            else:
                predictions = model.inference_streaming(
                    images_t,
                    num_scale_frames=min(8, n),
                    keyframe_interval=kf,
                    output_device=torch.device("cpu"),
                )

        _check_cancel()
        _job["detail"] = "postprocess"
        yield (
            f"[{run_id}] Post-processing + launching viser…",
            _viser_link_html(False),
            json.dumps(_job),
        )

        predictions, images_cpu = demo.postprocess(predictions, images_t)
        # Free GPU activations before viser (animate thread is light)
        try:
            del images_t
            images_t = None
            _free_model_runtime()
        except Exception:
            pass

        pred_vis = demo.prepare_for_visualization(predictions, images_cpu)
        _start_viewer(pred_vis, resolved_folder, conf_threshold, downsample, run_id)

        elapsed = round(time.time() - t0, 1)
        _job.update(
            {
                "phase": "ready",
                "detail": f"ok frames={n} elapsed_s={elapsed}",
                "error": None,
                "elapsed_s": elapsed,
                "loaded_run_id": run_id,
            }
        )
        _runs.append(
            {
                "id": run_id,
                "frames": n,
                "elapsed_s": elapsed,
                "work": str(work),
                "folder": resolved_folder,
                "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
        )
        if len(_runs) > 30:
            del _runs[:-30]
        msg = (
            f"OK [{run_id}] — {n} frames in {elapsed}s · open /viser/ · "
            f"work={work} · vram={vram_info()}"
        )
        append_bootstrap(msg)
        yield msg, _viser_link_html(True), json.dumps(_job)
    except Exception as exc:
        _job.update({"phase": "error", "detail": str(exc), "error": str(exc)})
        append_bootstrap(f"reconstruct FAIL [{run_id}]: {exc}")
        yield f"FAIL [{run_id}]: {exc}", _viser_link_html(_viewer is not None), json.dumps(_job)
    finally:
        try:
            del images_t, predictions
        except Exception:
            pass
        _free_model_runtime()
        _cancel.clear()
        _job_lock.release()


app = FastAPI(title="LingBot Map One")


@app.on_event("startup")
def _startup():
    def run():
        try:
            _smoke_or_load_model()
        except Exception:
            # Still allow UI; reconstruct will retry
            try:
                _resolve_ckpt()
                global _ready
                _ready = True
                set_ready_phase("map_weights_ready")
                append_bootstrap("ckpt present — model load deferred to first reconstruct")
            except Exception as exc:
                append_bootstrap(f"startup soft-fail: {exc}")

    threading.Thread(target=run, daemon=True).start()


@app.get("/")
def root():
    return RedirectResponse("/ui/")


@app.get("/health")
def health():
    return {
        "status": "ready" if _ready else ("error" if _load_error else "loading"),
        "ready": _ready,
        "error": _load_error,
        "ckpt": str(_ckpt) if _ckpt else None,
        "model": HF_ID,
        "viser_port": VISER_PORT,
        "viser_url": "/viser/",
        "viser_loaded": _viewer is not None,
        "loaded_run_id": _loaded_run_id,
        "job": _job,
        "job_busy": _job_lock.locked(),
        "runs_tail": _runs[-5:],
        "vram": vram_info(),
        "app": "lingbotmapone",
    }


@app.get("/api/v1/info")
def info():
    return health()


@app.get("/api/v1/viser/status")
def viser_status():
    return {
        "active": _viewer is not None,
        "port": VISER_PORT,
        "proxy": "/viser/",
        "loaded_run_id": _loaded_run_id,
        "job": _job,
        "runs_tail": _runs[-10:],
    }


@app.post("/api/v1/job/cancel")
def api_cancel_job():
    msg, payload = cancel_job()
    return {"ok": True, "message": msg, "job": json.loads(payload)}


@app.post("/api/v1/viser/stop")
def api_stop_viser():
    msg, _link, payload = stop_viser_ui()
    return {"ok": True, "message": msg, "job": json.loads(payload)}


@app.get("/api/v1/runs")
def api_runs():
    return {"runs": _runs, "loaded_run_id": _loaded_run_id}


async def _relay_ws(
    client: WebSocket,
    upstream_url: str,
    *,
    subprotocols: list[str] | None = None,
) -> None:
    connect_kwargs: dict = {
        "max_size": None,
        "ping_interval": None,
        "close_timeout": 5,
    }
    # Forward Sec-WebSocket-Protocol — viser rejects Client:'unknown' without it.
    if subprotocols:
        connect_kwargs["subprotocols"] = subprotocols

    async with websockets.connect(upstream_url, **connect_kwargs) as upstream:

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


def _pick_viser_subprotocol(websocket: WebSocket) -> str | None:
    requested = list(websocket.scope.get("subprotocols") or [])
    for p in requested:
        if isinstance(p, str) and p.startswith("viser-"):
            return p
    return requested[0] if requested else None


@app.websocket("/viser")
@app.websocket("/viser/{path:path}")
async def proxy_viser_ws(websocket: WebSocket, path: str = "") -> None:
    if _viewer is None:
        await websocket.close(code=4404, reason="Viser not running — reconstruct first")
        return
    sub = _pick_viser_subprotocol(websocket)
    if sub:
        await websocket.accept(subprotocol=sub)
    else:
        await websocket.accept()
    qs = websocket.scope.get("query_string", b"").decode()
    # Viser listens at root; strip empty path
    upstream_path = path or ""
    upstream_url = f"ws://127.0.0.1:{VISER_PORT}/{upstream_path}"
    if qs:
        upstream_url += f"?{qs}"
    try:
        await _relay_ws(
            websocket,
            upstream_url,
            subprotocols=[sub] if sub else None,
        )
    except Exception as exc:
        append_bootstrap(f"viser ws proxy fail: {exc}")
        if websocket.client_state == WebSocketState.CONNECTED:
            await websocket.close(code=1011)


@app.api_route("/viser", methods=["GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"])
@app.api_route(
    "/viser/{path:path}",
    methods=["GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"],
)
async def proxy_viser_http(request: Request, path: str = "") -> Response:
    """Reverse-proxy to local viser.

    Critical: viser gzips HTML when Accept-Encoding includes gzip. httpx auto-decodes
    ``.content`` but if we forward Content-Encoding:gzip the browser double-decodes
    and shows a blank page. Strip encoding headers; prefer identity from upstream.
    """
    if request.headers.get("upgrade", "").lower() == "websocket":
        return HTMLResponse("Use WebSocket upgrade", status_code=426)
    # Canonical trailing slash so window.location.href WS path is stable
    if path == "" and not str(request.url.path).endswith("/"):
        q = f"?{request.url.query}" if request.url.query else ""
        return RedirectResponse(url=f"/viser/{q}", status_code=307)
    if _viewer is None:
        return HTMLResponse(
            "<h3>Viser not running</h3><p>Open Gradio /ui and click Reconstruct.</p>",
            status_code=503,
        )
    target = f"http://127.0.0.1:{VISER_PORT}/{path}"
    if request.url.query:
        target += f"?{request.url.query}"
    headers = {
        k: v
        for k, v in request.headers.items()
        if k.lower() not in _HOP_BY_HOP and k.lower() != "accept-encoding"
    }
    # Ask for uncompressed body so Content-Encoding stays identity end-to-end
    headers["accept-encoding"] = "identity"
    body = await request.body()
    async with httpx.AsyncClient(timeout=120.0, follow_redirects=True) as client:
        upstream = await client.request(request.method, target, headers=headers, content=body)
    resp_headers = {
        k: v for k, v in upstream.headers.items() if k.lower() not in _HOP_BY_HOP
    }
    media = upstream.headers.get("content-type")
    return Response(
        content=upstream.content,
        status_code=upstream.status_code,
        headers=resp_headers,
        media_type=media,
    )


def build_ui():
    with gr.Blocks(title="LingBot Map One") as demo:
        gr.Markdown("# LingBot Map One\nStreaming 3D reconstruction (LingBot-Map + viser).")
        gr.HTML(_viser_header_html())
        gr.Markdown(
            "**One job at a time.** Stuck queue? **Cancel job** → **Stop viser** → Reconstruct.\n\n"
            "**Inputs (pick one):** images, video, or in-pod folder."
        )
        with gr.Row():
            images = gr.File(
                label="Images (multi)",
                file_count="multiple",
                file_types=["image"],
                type="filepath",
            )
            video = gr.File(
                label="Video",
                file_count="single",
                file_types=["video"],
                type="filepath",
            )
        folder = gr.Textbox(label="Image folder (in pod)", value=DEFAULT_FOLDER)
        with gr.Row():
            max_frames = gr.Slider(4, 512, value=64, step=4, label="Max frames (first_k)")
            fps = gr.Slider(1, 30, value=10, step=1, label="Video sample FPS")
            mode = gr.Dropdown(["streaming", "windowed"], value="streaming", label="Mode")
        with gr.Row():
            conf = gr.Slider(0.5, 5.0, value=1.5, step=0.1, label="Vis conf threshold")
            down = gr.Slider(1, 40, value=10, step=1, label="Point downsample")
            kf = gr.Slider(1, 32, value=1, step=1, label="Keyframe interval")
        with gr.Row():
            btn = gr.Button("Reconstruct", variant="primary")
            btn_cancel = gr.Button("Cancel job", variant="stop")
            btn_stop = gr.Button("Stop viser / unload")
            btn_refresh = gr.Button("Refresh status")
        status = gr.Textbox(label="Status", lines=3)
        job_box = gr.Textbox(label="Job / runs JSON", lines=6)
        viser_link = gr.HTML(
            value=_viser_link_html(False),
            label="Viser link (always clickable)",
        )
        # Always-visible copyable path (Gradio markdown /ui breaks relative /viser)
        viser_url = gr.Textbox(
            label="Viser URL (copy)",
            value="/viser/",
            interactive=False,
        )

        run_evt = btn.click(
            reconstruct,
            [images, video, folder, max_frames, fps, mode, conf, down, kf],
            [status, viser_link, job_box],
            concurrency_limit=1,
        )
        btn_cancel.click(fn=cancel_job, outputs=[status, job_box], cancels=[run_evt])
        btn_stop.click(fn=stop_viser_ui, outputs=[status, viser_link, job_box])
        btn_refresh.click(fn=refresh_status, outputs=[status, viser_link, job_box])
    return demo


demo = build_ui()
# Small queue — reject pile-up; concurrency 1 matches GPU lock
try:
    demo.queue(default_concurrency_limit=1, max_size=2)
except TypeError:
    demo.queue()

app = mount_gradio(app, demo, path="/ui")

if __name__ == "__main__":
    append_bootstrap(f"uvicorn :{SERVER_PORT} viser_proxy=/{VISER_PORT}")
    uvicorn.run(app, host="0.0.0.0", port=SERVER_PORT, log_level="info")
