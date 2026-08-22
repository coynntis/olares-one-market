#!/usr/bin/env python3
"""LingBot-Vision PCA Gradio + REST."""

from __future__ import annotations

import os
import sys
import threading
import time

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
sys.path.insert(0, os.path.dirname(__file__))

from _common import append_bootstrap, ensure_gradio_temp, mount_gradio, set_ready_phase, setup_logging, vram_info

ensure_gradio_temp()
LOG = setup_logging("lingbotvisionone")

import gradio as gr
import numpy as np
import torch
import uvicorn
from fastapi import FastAPI
from fastapi.responses import RedirectResponse

SERVER_PORT = int(os.getenv("SERVER_PORT", "7860"))
MODEL_DIR = os.getenv("LINGBOT_MODEL_DIR", "/models/lingbotvisionone/weights")
SRC = os.getenv("LINGBOT_SRC", "")
VARIANT = os.getenv("LINGBOT_VISION_VARIANT", "large")
HF_ID = os.getenv("HF_REPO_ID", "robbyant/lingbot-vision-vit-large")

_lock = threading.Lock()
_backbone = None
_ready = False
_load_error = None


def _load():
    global _backbone, _ready, _load_error
    with _lock:
        if _backbone is not None:
            return
        t0 = time.time()
        try:
            if SRC and SRC not in sys.path:
                sys.path.insert(0, SRC)
            from lingbot_vision import load_pretrained_backbone

            # Prefer CUDA; RTX 5090M needs torch with sm_120 (CUDA 13 image).
            want_cuda = torch.cuda.is_available()
            if want_cuda:
                try:
                    major, minor = torch.cuda.get_device_capability(0)
                    append_bootstrap(
                        f"cuda device={torch.cuda.get_device_name(0)} "
                        f"capability=sm_{major}{minor} torch={torch.__version__} "
                        f"cuda_build={torch.version.cuda}"
                    )
                except Exception as exc:
                    append_bootstrap(f"cuda capability probe soft-fail: {exc}")

            local_pt = os.path.join(MODEL_DIR, "model.pt")
            if os.path.isfile(local_pt):
                repo = MODEL_DIR
                append_bootstrap(f"vision load local dir={repo}")
            else:
                repo = HF_ID
                append_bootstrap(f"vision load HF repo={repo}")

            def _try_load(device: str):
                return load_pretrained_backbone(
                    repo,
                    variant=VARIANT,
                    device=device,
                    dtype="auto",
                    local_files_only=os.path.isfile(local_pt),
                )

            device = "cuda" if want_cuda else "cpu"
            try:
                _backbone, _ = _try_load(device)
            except RuntimeError as exc:
                msg = str(exc)
                if want_cuda and (
                    "no kernel image" in msg
                    or "sm_120" in msg
                    or "CUDA error" in msg
                ):
                    append_bootstrap(
                        f"GPU load failed ({msg[:160]}) — falling back to CPU. "
                        "Fix: use pytorch/pytorch:2.12.0-cuda13.0-cudnn9-devel (sm_120)."
                    )
                    device = "cpu"
                    _backbone, _ = _try_load("cpu")
                else:
                    raise
            _backbone.eval()
            _ready = True
            append_bootstrap(
                f"vision {VARIANT} ready device={device} {time.time()-t0:.1f}s vram={vram_info()}"
            )
            set_ready_phase(f"vision_ready_{device}")
        except Exception as exc:
            _load_error = str(exc)
            append_bootstrap(f"vision load FAIL: {exc}")
            raise


def pca_map(image: np.ndarray, size: int = 512) -> np.ndarray:
    _load()
    from lingbot_vision import extract_patch_tokens, load_image
    from sklearn.decomposition import PCA

    # Upstream extract_patch_tokens expects device as str (uses .startswith("cuda")).
    param = next(_backbone.parameters())
    device = str(param.device)  # e.g. "cuda:0" or "cpu"
    dtype = param.dtype
    # save temp
    import tempfile
    from PIL import Image

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as fh:
        Image.fromarray(image.astype(np.uint8)).save(fh.name)
        path = fh.name
    img_norm, _, _ = load_image(path, size=size, patch_size=_backbone.patch_size, mode="square")
    tokens, grid = extract_patch_tokens(_backbone, img_norm, device, dtype)
    x = tokens[0].float().cpu().numpy()
    pca = PCA(n_components=3)
    rgb = pca.fit_transform(x)
    rgb = (rgb - rgb.min(0)) / (rgb.ptp(0) + 1e-8)
    h, w = grid
    vis = (rgb.reshape(h, w, 3) * 255).astype(np.uint8)
    # upsample to size
    from PIL import Image as PImage

    return np.array(PImage.fromarray(vis).resize((size, size), PImage.NEAREST))


app = FastAPI(title="LingBot Vision One")


@app.on_event("startup")
def _startup():
    threading.Thread(target=lambda: (_load() if True else None), daemon=True).start()


@app.get("/")
def root():
    return RedirectResponse("/ui/")


@app.get("/health")
def health():
    return {
        "status": "ready" if _ready else ("error" if _load_error else "loading"),
        "ready": _ready,
        "error": _load_error,
        "variant": VARIANT,
        "model": HF_ID,
        "vram": vram_info(),
    }


@app.get("/api/v1/info")
def info():
    return health()


def build_ui():
    with gr.Blocks(title="LingBot Vision One") as demo:
        gr.Markdown("# LingBot Vision One\nFrozen backbone PCA of patch tokens")
        inp = gr.Image(type="numpy", label="Image")
        size = gr.Slider(256, 1024, value=512, step=16, label="Size")
        btn = gr.Button("PCA", variant="primary")
        out = gr.Image(label="PCA map")
        btn.click(lambda im, s: pca_map(im, int(s)), [inp, size], [out])
    return demo


demo = build_ui()
app = mount_gradio(app, demo, path="/ui")

if __name__ == "__main__":
    append_bootstrap(f"uvicorn :{SERVER_PORT}")
    uvicorn.run(app, host="0.0.0.0", port=SERVER_PORT, log_level="info")
