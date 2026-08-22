#!/usr/bin/env python3
"""LingBot-Video Dense 1.3B Gradio + REST (single-GPU)."""

from __future__ import annotations

import os
import sys
import threading
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
sys.path.insert(0, os.path.dirname(__file__))

from _common import append_bootstrap, ensure_gradio_temp, mount_gradio, set_ready_phase, setup_logging, vram_info

ensure_gradio_temp()
LOG = setup_logging("lingbotvideoone")

import gradio as gr
import torch
import uvicorn
from fastapi import FastAPI
from fastapi.responses import RedirectResponse

SERVER_PORT = int(os.getenv("SERVER_PORT", "7860"))
MODEL_DIR = Path(os.getenv("LINGBOT_MODEL_DIR", "/models/lingbotvideoone/weights"))
SRC = os.getenv("LINGBOT_SRC", "")
HF_ID = os.getenv("HF_REPO_ID", "robbyant/lingbot-video-dense-1.3b")

_lock = threading.Lock()
_pipe = None
_ready = False
_load_error = None


def _load():
    global _pipe, _ready, _load_error
    with _lock:
        if _pipe is not None:
            return
        t0 = time.time()
        try:
            if SRC and SRC not in sys.path:
                sys.path.insert(0, SRC)
            # Prefer diffusers custom pipeline if registered; else load components lazily
            from diffusers import DiffusionPipeline

            local = str(MODEL_DIR) if (MODEL_DIR / "model_index.json").is_file() else HF_ID
            append_bootstrap(f"loading video dense from {local}")
            try:
                _pipe = DiffusionPipeline.from_pretrained(
                    local,
                    torch_dtype=torch.bfloat16,
                    trust_remote_code=True,
                )
            except Exception as e1:
                append_bootstrap(f"DiffusionPipeline fail ({e1}); trying AutoPipeline")
                from diffusers import AutoPipelineForText2Image

                _pipe = AutoPipelineForText2Image.from_pretrained(
                    local, torch_dtype=torch.bfloat16, trust_remote_code=True
                )
            if torch.cuda.is_available():
                # Keep text encoder on CPU if VRAM tight
                try:
                    _pipe.to("cuda")
                except torch.cuda.OutOfMemoryError:
                    append_bootstrap("OOM full GPU — enable_model_cpu_offload")
                    _pipe.enable_model_cpu_offload()
            _ready = True
            append_bootstrap(f"video ready {time.time()-t0:.1f}s vram={vram_info()}")
            set_ready_phase("video_ready")
        except Exception as exc:
            _load_error = str(exc)
            append_bootstrap(f"video load FAIL: {exc}")
            raise


def generate(prompt: str, steps: int = 30, height: int = 480, width: int = 832, seed: int = 0):
    _load()
    g = torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu")
    if seed >= 0:
        g = g.manual_seed(int(seed))
    with _lock:
        out = _pipe(
            prompt=prompt,
            num_inference_steps=int(steps),
            height=int(height),
            width=int(width),
            generator=g,
        )
    # T2I fallback returns images; T2V may return frames
    if hasattr(out, "images") and out.images:
        return out.images[0], None, str(vram_info())
    if hasattr(out, "frames") and out.frames:
        frames = out.frames[0]
        # save mp4
        import imageio
        import numpy as np

        path = f"/output/gradio/lingbot_video_{int(time.time())}.mp4"
        imageio.mimwrite(path, [np.asarray(f) for f in frames], fps=16)
        return None, path, str(vram_info())
    return None, None, str(vram_info())


app = FastAPI(title="LingBot Video Dense One")


@app.on_event("startup")
def _startup():
    threading.Thread(target=lambda: _load(), daemon=True).start()


@app.get("/")
def root():
    return RedirectResponse("/ui/")


@app.get("/health")
def health():
    return {
        "status": "ready" if _ready else ("error" if _load_error else "loading"),
        "ready": _ready,
        "error": _load_error,
        "model": HF_ID,
        "vram": vram_info(),
    }


@app.get("/api/v1/info")
def info():
    return health()


def build_ui():
    with gr.Blocks(title="LingBot Video Dense") as demo:
        gr.Markdown("# LingBot Video Dense 1.3B\nSingle-GPU T2I/T2V (diffusers path)")
        prompt = gr.Textbox(label="Prompt", lines=3)
        steps = gr.Slider(10, 50, value=30, step=1)
        h = gr.Slider(256, 720, value=480, step=16)
        w = gr.Slider(256, 1280, value=832, step=16)
        seed = gr.Number(value=0)
        btn = gr.Button("Generate", variant="primary")
        img = gr.Image(label="Image (T2I)")
        vid = gr.Video(label="Video (T2V)")
        vram = gr.Textbox(label="VRAM")
        btn.click(generate, [prompt, steps, h, w, seed], [img, vid, vram])
    return demo


demo = build_ui()
app = mount_gradio(app, demo, path="/ui")

if __name__ == "__main__":
    append_bootstrap(f"uvicorn :{SERVER_PORT}")
    uvicorn.run(app, host="0.0.0.0", port=SERVER_PORT, log_level="info")
