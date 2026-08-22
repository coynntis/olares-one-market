#!/usr/bin/env python3
"""LingBot-World NF4 Gradio + REST — T5 CPU + layer/sequential offload for 24GB.

Uses community prequant package `cahlen/lingbot-world-base-cam-nf4` (official World
v2 has no NF4 yet). Upstream generate_prequant already defaults t5_cpu=True.
Official card cites ~32GB; we force shorter frames + sequential DiT for 5090M.
"""

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
LOG = setup_logging("lingbotworldone")

import gradio as gr
import numpy as np
import torch
import uvicorn
from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from PIL import Image

SERVER_PORT = int(os.getenv("SERVER_PORT", "7860"))
MODEL_DIR = Path(os.getenv("LINGBOT_MODEL_DIR", "/models/lingbotworldone/weights"))
SRC = os.getenv("LINGBOT_SRC", "")
HF_ID = os.getenv("HF_REPO_ID", "cahlen/lingbot-world-base-cam-nf4")
T5_CPU = os.getenv("LINGBOT_WORLD_T5_CPU", "1").strip() not in ("0", "false", "no")
LAYER_OFFLOAD = os.getenv("LINGBOT_WORLD_LAYER_OFFLOAD", "1").strip() not in ("0", "false", "no")
MAX_FRAMES = int(os.getenv("LINGBOT_WORLD_MAX_FRAMES", "49"))

_lock = threading.Lock()
_ready = False
_load_error = None
_pipe = None


def _ensure_nf4_code() -> Path:
    """NF4 package ships generate_prequant.py + wan/ — prefer MODEL_DIR if present."""
    if (MODEL_DIR / "generate_prequant.py").is_file():
        return MODEL_DIR
    # Some downloads nest; search
    matches = list(MODEL_DIR.rglob("generate_prequant.py"))
    if matches:
        return matches[0].parent
    raise FileNotFoundError(
        f"NF4 package incomplete under {MODEL_DIR} — need generate_prequant.py + wan/"
    )


def _load():
    global _pipe, _ready, _load_error
    with _lock:
        if _ready:
            return
        t0 = time.time()
        try:
            root = _ensure_nf4_code()
            if str(root) not in sys.path:
                sys.path.insert(0, str(root))
            if SRC and SRC not in sys.path:
                sys.path.insert(0, SRC)
            append_bootstrap(
                f"World NF4 load root={root} t5_cpu={T5_CPU} layer_offload={LAYER_OFFLOAD} max_frames={MAX_FRAMES}"
            )
            from generate_prequant import WanI2V_PreQuant  # type: ignore

            _pipe = WanI2V_PreQuant(checkpoint_dir=str(root), device_id=0, t5_cpu=T5_CPU)
            # Aggressive: if both DiTs resident, move low-noise to CPU until needed
            if LAYER_OFFLOAD:
                for attr in ("low_noise_model", "high_noise_model", "model_low", "model_high"):
                    m = getattr(_pipe, attr, None)
                    if m is not None and hasattr(m, "to"):
                        try:
                            # Keep high on GPU, park low on CPU if both exist
                            if "low" in attr:
                                m.to("cpu")
                                append_bootstrap(f"parked {attr} on CPU")
                        except Exception as e:
                            append_bootstrap(f"park {attr} soft-fail: {e}")
                torch.cuda.empty_cache()
            _ready = True
            append_bootstrap(f"World NF4 ready {time.time()-t0:.1f}s vram={vram_info()}")
            set_ready_phase("world_nf4_ready")
        except Exception as exc:
            _load_error = str(exc)
            append_bootstrap(f"World NF4 load FAIL: {exc}")
            raise


def generate(image: np.ndarray, prompt: str, frames: int, steps: int, size: str):
    _load()
    frames = max(17, min(int(frames), MAX_FRAMES))
    # Prefer odd frame counts like upstream
    if frames % 2 == 0:
        frames += 1
    out_path = f"/output/gradio/lingbot_world_{int(time.time())}.mp4"
    tmp_img = f"/output/gradio/_in_{int(time.time())}.jpg"
    Image.fromarray(image.astype(np.uint8)).save(tmp_img)
    with _lock:
        # API differs slightly across package versions — try common call shapes
        gen = getattr(_pipe, "generate", None) or getattr(_pipe, "__call__", None)
        if gen is None:
            # Fallback: shell out to generate_prequant.py
            import subprocess

            root = _ensure_nf4_code()
            cmd = [
                sys.executable,
                str(root / "generate_prequant.py"),
                "--image",
                tmp_img,
                "--prompt",
                prompt,
                "--frame_num",
                str(frames),
                "--size",
                size,
                "--sampling_steps",
                str(int(steps)),
                "--output",
                out_path,
            ]
            append_bootstrap("exec " + " ".join(cmd))
            subprocess.check_call(cmd, cwd=str(root))
        else:
            try:
                gen(
                    image=tmp_img,
                    prompt=prompt,
                    frame_num=frames,
                    size=size,
                    sampling_steps=int(steps),
                    output=out_path,
                )
            except TypeError:
                # keyword mismatch — write via script
                import subprocess

                root = _ensure_nf4_code()
                subprocess.check_call(
                    [
                        sys.executable,
                        str(root / "generate_prequant.py"),
                        "--image",
                        tmp_img,
                        "--prompt",
                        prompt,
                        "--frame_num",
                        str(frames),
                        "--size",
                        size,
                        "--sampling_steps",
                        str(int(steps)),
                        "--output",
                        out_path,
                    ],
                    cwd=str(root),
                )
    return out_path if os.path.isfile(out_path) else None, str(vram_info())


app = FastAPI(title="LingBot World NF4 One")


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
        "t5_cpu": T5_CPU,
        "layer_offload": LAYER_OFFLOAD,
        "max_frames": MAX_FRAMES,
        "vram": vram_info(),
        "hint": "NF4 package targets ~32GB; 24GB uses T5 CPU + park low-noise DiT + short clips",
    }


@app.get("/api/v1/info")
def info():
    return health()


def build_ui():
    with gr.Blocks(title="LingBot World NF4") as demo:
        gr.Markdown(
            "# LingBot World NF4 One\n"
            "Community prequant `cahlen/lingbot-world-base-cam-nf4` (v2 has no official NF4 yet). "
            f"Defaults: T5 CPU, layer offload, max frames={MAX_FRAMES} for 5090M 24GB."
        )
        img = gr.Image(type="numpy", label="Start image")
        prompt = gr.Textbox(label="Prompt", lines=3, value="A cinematic camera move through the scene")
        frames = gr.Slider(17, MAX_FRAMES, value=min(33, MAX_FRAMES), step=4, label="Frames")
        steps = gr.Slider(10, 40, value=20, step=1, label="Steps")
        size = gr.Dropdown(choices=["480*832", "720*1280"], value="480*832")
        btn = gr.Button("Generate", variant="primary")
        vid = gr.Video(label="Output")
        vram = gr.Textbox(label="VRAM")
        btn.click(generate, [img, prompt, frames, steps, size], [vid, vram])
    return demo


demo = build_ui()
app = mount_gradio(app, demo, path="/ui")

if __name__ == "__main__":
    append_bootstrap(f"uvicorn :{SERVER_PORT}")
    uvicorn.run(app, host="0.0.0.0", port=SERVER_PORT, log_level="info")
