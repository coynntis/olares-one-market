"""MiniMax H3 Gradio app for Olares One."""

from __future__ import annotations

import asyncio
import logging
import os
import tempfile
import threading
from contextlib import asynccontextmanager
from io import BytesIO
from typing import Any

import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, RedirectResponse
from PIL import Image
from pydantic import BaseModel, Field

LOG = logging.getLogger("minimaxh3nvfp4one")

_LOCK = threading.Lock()
_PIPE = None
_PIPE_WORKFLOW = None
_GRADIO_MOUNT_PATH = "/ui"
_GRADIO_ROOT_PATH = (os.getenv("GRADIO_ROOT_PATH") or "").strip() or _GRADIO_MOUNT_PATH

MODEL_REPO = os.getenv("MINIMAX_MODEL_REPO", "MiniMaxAI/MiniMax-H3").strip() or "MiniMaxAI/MiniMax-H3"
DEFAULT_WORKFLOW = os.getenv("MINIMAX_WORKFLOW", "t2va").strip().lower() or "t2va"
OUTPUTS = ["videos", "audio", "sampling_rate"]


def _setup_logging() -> None:
    level = getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO)
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def _ensure_pipe(workflow: str):
    global _PIPE, _PIPE_WORKFLOW
    wf = workflow.strip().lower()
    if wf not in {"t2va", "fl2va"}:
        raise ValueError("workflow must be t2va or fl2va")
    if _PIPE is not None and _PIPE_WORKFLOW == wf:
        return _PIPE

    from diffusers import ComponentsManager, ModularPipeline

    manager = ComponentsManager()
    manager.enable_auto_cpu_offload(device="cuda")
    pipe = ModularPipeline.from_pretrained(MODEL_REPO, workflow=wf, components_manager=manager)
    pipe.load_components(dtype=torch.bfloat16)
    _PIPE = pipe
    _PIPE_WORKFLOW = wf
    return pipe


def _encode_result(results: dict[str, Any], output_name: str) -> str:
    from diffusers.utils.export_utils import encode_video

    out = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    out.close()
    encode_video(
        results["videos"][0],
        fps=24,
        output_path=out.name,
        audio=results["audio"][0],
        audio_sample_rate=results["sampling_rate"],
    )
    LOG.info("saved %s -> %s", output_name, out.name)
    return out.name


def _run_t2va(prompt: str, num_frames: int, height: int, width: int, seed: int) -> str:
    if not prompt.strip():
        raise ValueError("prompt is required")
    with _LOCK:
        pipe = _ensure_pipe("t2va")
        g = torch.Generator().manual_seed(int(seed))
        results = pipe(
            prompt=prompt.strip(),
            num_frames=int(num_frames),
            height=int(height),
            width=int(width),
            generator=g,
            output=OUTPUTS,
        )
        return _encode_result(results, "t2va")


def _run_fl2va(
    prompt: str,
    image: Image.Image,
    last_image: Image.Image | None,
    num_frames: int,
    seed: int,
) -> str:
    if not prompt.strip():
        raise ValueError("prompt is required")
    if image is None:
        raise ValueError("first frame image is required")

    with _LOCK:
        pipe = _ensure_pipe("fl2va")
        g = torch.Generator().manual_seed(int(seed))
        kwargs: dict[str, Any] = {
            "prompt": prompt.strip(),
            "image": image.convert("RGB"),
            "num_frames": int(num_frames),
            "generator": g,
            "output": OUTPUTS,
        }
        if last_image is not None:
            kwargs["last_image"] = last_image.convert("RGB")
        results = pipe(**kwargs)
        return _encode_result(results, "fl2va")


def build_gradio():
    import gradio as gr

    with gr.Blocks(title="MiniMax H3 NVFP4 One") as demo:
        gr.Markdown("# MiniMax H3 Video + Audio")
        gr.Markdown(
            "Gradio wrapper over Diffusers ModularPipeline MiniMax-H3. "
            "Note: coolthor NVFP4 repo is Comfy layout; this app uses diffusers-compatible MiniMax repo."
        )

        with gr.Tab("Text to Video + Audio"):
            p = gr.Textbox(label="Prompt", lines=4)
            nf = gr.Slider(124, 362, value=124, step=17, label="Frames")
            h = gr.Slider(512, 768, value=768, step=32, label="Height")
            w = gr.Slider(768, 1344, value=1344, step=32, label="Width")
            s = gr.Number(label="Seed", value=42, precision=0)
            btn = gr.Button("Generate T2VA", variant="primary")
            out = gr.Video(label="Output")
            btn.click(fn=_run_t2va, inputs=[p, nf, h, w, s], outputs=out)

        with gr.Tab("First/Last Frame to Video + Audio"):
            p2 = gr.Textbox(label="Prompt", lines=4)
            first = gr.Image(type="pil", label="First Frame", image_mode="RGB")
            last = gr.Image(type="pil", label="Last Frame (optional)", image_mode="RGB")
            nf2 = gr.Slider(124, 362, value=124, step=17, label="Frames")
            s2 = gr.Number(label="Seed", value=42, precision=0)
            btn2 = gr.Button("Generate FL2VA", variant="primary")
            out2 = gr.Video(label="Output")
            btn2.click(fn=_run_fl2va, inputs=[p2, first, last, nf2, s2], outputs=out2)

    return demo


class T2VARequest(BaseModel):
    prompt: str
    num_frames: int = 124
    height: int = 768
    width: int = 1344
    seed: int = Field(default=42)


@asynccontextmanager
async def lifespan(_: FastAPI):
    _setup_logging()
    LOG.info("starting minimax app repo=%s workflow=%s", MODEL_REPO, DEFAULT_WORKFLOW)
    yield


app = FastAPI(title="MiniMax H3 NVFP4 One", version="1.0.0", lifespan=lifespan)


@app.get("/")
def root():
    return RedirectResponse(url="/ui/", status_code=307)


@app.get("/api/v1/health")
def health():
    return {
        "status": "ok",
        "model_repo": MODEL_REPO,
        "pipe_workflow": _PIPE_WORKFLOW,
        "cuda": torch.cuda.is_available(),
    }


@app.post("/api/v1/t2va")
async def api_t2va(body: T2VARequest):
    loop = asyncio.get_running_loop()
    try:
        path = await loop.run_in_executor(None, lambda: _run_t2va(body.prompt, body.num_frames, body.height, body.width, body.seed))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return FileResponse(path, media_type="video/mp4", filename="minimax_t2va.mp4")


@app.post("/api/v1/fl2va")
async def api_fl2va(
    prompt: str = Form(...),
    image: UploadFile = File(...),
    last_image: UploadFile | None = File(default=None),
    num_frames: int = Form(124),
    seed: int = Form(42),
):
    raw = await image.read()
    try:
        first = Image.open(BytesIO(raw)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"invalid first image: {exc}") from exc

    last = None
    if last_image is not None:
        raw_last = await last_image.read()
        if raw_last:
            try:
                last = Image.open(BytesIO(raw_last)).convert("RGB")
            except Exception as exc:
                raise HTTPException(status_code=400, detail=f"invalid last image: {exc}") from exc

    loop = asyncio.get_running_loop()
    try:
        path = await loop.run_in_executor(None, lambda: _run_fl2va(prompt, first, last, num_frames, seed))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return FileResponse(path, media_type="video/mp4", filename="minimax_fl2va.mp4")


def _mount_gradio() -> None:
    import gradio as gr

    global app
    app = gr.mount_gradio_app(app, build_gradio(), path=_GRADIO_MOUNT_PATH, root_path=_GRADIO_ROOT_PATH)


_mount_gradio()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("SERVER_PORT", "7860")))
