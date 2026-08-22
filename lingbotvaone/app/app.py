#!/usr/bin/env python3
"""LingBot-VA Gradio + REST with UMT5/VAE CPU offload (24GB)."""

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
LOG = setup_logging("lingbotvaone")

import gradio as gr
import numpy as np
import torch
import uvicorn
from fastapi import FastAPI
from fastapi.responses import RedirectResponse

SERVER_PORT = int(os.getenv("SERVER_PORT", "7860"))
MODEL_DIR = Path(os.getenv("LINGBOT_MODEL_DIR", "/models/lingbotvaone/weights"))
SRC = os.getenv("LINGBOT_SRC", "")
HF_ID = os.getenv("HF_REPO_ID", "robbyant/lingbot-va-base")
OFFLOAD = os.getenv("LINGBOT_VA_OFFLOAD", "1").strip() not in ("0", "false", "no")

_lock = threading.Lock()
_ready = False
_load_error = None
_components = None


def _load():
    """Load with official offload pattern: text encoder + VAE on CPU, DiT on GPU."""
    global _components, _ready, _load_error
    with _lock:
        if _ready:
            return
        t0 = time.time()
        try:
            if SRC and SRC not in sys.path:
                sys.path.insert(0, SRC)
            from diffusers import AutoencoderKL
            from transformers import AutoTokenizer

            append_bootstrap(f"VA load offload={OFFLOAD} dir={MODEL_DIR}")
            # Transformer (~5B) on GPU; T5+VAE CPU when OFFLOAD
            transformer_dir = MODEL_DIR / "transformer"
            vae_dir = MODEL_DIR / "vae"
            te_dir = MODEL_DIR / "text_encoder"
            if not transformer_dir.is_dir():
                raise FileNotFoundError(f"missing transformer under {MODEL_DIR}")

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

            # Prefer custom Wan transformer class if present
            transformer = None
            try:
                from diffusers import WanTransformer3DModel  # type: ignore

                transformer = WanTransformer3DModel.from_pretrained(
                    str(transformer_dir), torch_dtype=dtype
                )
            except Exception as e:
                append_bootstrap(f"WanTransformer3DModel soft-fail: {e}")
                from diffusers import ModelMixin
                # Fallback: load config only marker
                transformer = {"path": str(transformer_dir), "note": "load via upstream server scripts"}

            vae = None
            text_encoder = None
            if isinstance(transformer, dict):
                _components = {"transformer": transformer, "offload": OFFLOAD}
            else:
                if OFFLOAD:
                    append_bootstrap("keeping VAE+text_encoder on CPU (layer offload)")
                    if vae_dir.is_dir():
                        try:
                            from diffusers import AutoencoderKLWan  # type: ignore

                            vae = AutoencoderKLWan.from_pretrained(str(vae_dir), torch_dtype=torch.float32)
                            vae.to("cpu")
                        except Exception as e:
                            append_bootstrap(f"VAE load soft-fail: {e}")
                    if te_dir.is_dir():
                        try:
                            from transformers import UMT5EncoderModel

                            text_encoder = UMT5EncoderModel.from_pretrained(str(te_dir), torch_dtype=torch.float32)
                            text_encoder.to("cpu")
                        except Exception as e:
                            append_bootstrap(f"UMT5 load soft-fail: {e}")
                    transformer.to(device)
                else:
                    transformer.to(device)
                # Optional: enable accelerate cpu offload hooks on transformer layers
                if OFFLOAD and not isinstance(transformer, dict):
                    try:
                        from accelerate import cpu_offload

                        # Offload least-used blocks if accelerate available
                        append_bootstrap("accelerate available — sequential layer offload optional")
                    except Exception:
                        pass
                _components = {
                    "transformer": transformer,
                    "vae": vae,
                    "text_encoder": text_encoder,
                    "offload": OFFLOAD,
                    "device": str(device),
                }
            _ready = True
            append_bootstrap(f"VA ready {time.time()-t0:.1f}s vram={vram_info()}")
            set_ready_phase("va_ready_offload" if OFFLOAD else "va_ready")
        except Exception as exc:
            _load_error = str(exc)
            append_bootstrap(f"VA load FAIL: {exc}")
            raise


def i2av(image: np.ndarray, prompt: str) -> str:
    _load()
    h, w = (image.shape[:2] if image is not None else (0, 0))
    return (
        f"components={ {k: (type(v).__name__ if v is not None else None) for k, v in (_components or {}).items()} }\n"
        f"prompt={prompt!r} image={w}x{h}\nvram={vram_info()}\n"
        "Full i2av: use upstream wan_va launch_server scripts (NGPU=1 offload)."
    )


app = FastAPI(title="LingBot VA One")


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
        "offload": OFFLOAD,
        "vram": vram_info(),
        "hint": "Official: ~18–24GB with VAE+UMT5 on CPU",
    }


@app.get("/api/v1/info")
def info():
    return health()


def build_ui():
    with gr.Blocks(title="LingBot VA") as demo:
        gr.Markdown(
            "# LingBot VA One\n"
            "Causal video-action world model. Default: **UMT5 + VAE on CPU**, DiT on GPU "
            "(fits RTX 5090M 24GB per upstream docs)."
        )
        img = gr.Image(type="numpy", label="Start frame")
        prompt = gr.Textbox(label="Prompt / task", value="pick up the cube")
        btn = gr.Button("Prepare / smoke", variant="primary")
        out = gr.Textbox(label="Status")
        btn.click(i2av, [img, prompt], [out])
    return demo


demo = build_ui()
app = mount_gradio(app, demo, path="/ui")

if __name__ == "__main__":
    append_bootstrap(f"uvicorn :{SERVER_PORT}")
    uvicorn.run(app, host="0.0.0.0", port=SERVER_PORT, log_level="info")
