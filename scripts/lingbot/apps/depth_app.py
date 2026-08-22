#!/usr/bin/env python3
"""LingBot-Depth Gradio + REST for Olares One."""

from __future__ import annotations

import os
import sys
import threading
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# Ensure /app-src helpers + workspace are importable when copied flat
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
LOG = setup_logging("lingbotdepthone")

import gradio as gr  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import uvicorn  # noqa: E402
from fastapi import FastAPI, HTTPException  # noqa: E402
from fastapi.responses import RedirectResponse  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

SERVER_PORT = int(os.getenv("SERVER_PORT", "7860"))
MODEL_DIR = os.getenv("LINGBOT_MODEL_DIR", "/models/lingbotdepthone/weights")
SRC = os.getenv("LINGBOT_SRC", "")
HF_ID = os.getenv("HF_REPO_ID", "robbyant/lingbot-depth-pretrain-vitl-14-v0.5")

_lock = threading.Lock()
_model = None
_load_error: str | None = None
_ready = False


def _resolve_pretrained() -> str:
    """MDMModel.from_pretrained expects a *file* (model.pt) or HF repo id — never a directory.

    Upstream: if Path(x).exists() it torch.loads(x); dirs exist → Errno 21 Is a directory.
    """
    found = find_weight_file(Path(MODEL_DIR), ["model.pt"])
    if found is not None:
        return str(found)
    return HF_ID


def _load() -> None:
    global _model, _load_error, _ready
    with _lock:
        if _model is not None:
            return
        t0 = time.time()
        pretrained = _resolve_pretrained()
        append_bootstrap(f"loading depth model pretrained={pretrained}")
        try:
            if SRC and SRC not in sys.path:
                sys.path.insert(0, SRC)
            from mdm.model.v2 import MDMModel  # type: ignore

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            _model = MDMModel.from_pretrained(pretrained).to(device)
            _model.eval()
            _ready = True
            append_bootstrap(f"depth model ready in {time.time()-t0:.1f}s device={device} vram={vram_info()}")
            set_ready_phase("depth_ready")
        except Exception as exc:
            _load_error = str(exc)
            append_bootstrap(f"depth load FAIL: {exc}")
            raise


def refine(image_rgb: np.ndarray, depth_m: np.ndarray, fx: float, fy: float, cx: float, cy: float) -> dict[str, Any]:
    _load()
    assert _model is not None
    device = next(_model.parameters()).device
    h, w = image_rgb.shape[:2]
    img = torch.tensor(image_rgb / 255.0, dtype=torch.float32, device=device).permute(2, 0, 1)[None]
    depth = torch.tensor(depth_m, dtype=torch.float32, device=device)[None]
    K = torch.tensor([[fx / w, 0, cx / w], [0, fy / h, cy / h], [0, 0, 1]], dtype=torch.float32, device=device)[None]
    # MDM expects intrinsics as 3x3 normalized — match upstream example layout when possible
    try:
        intr = torch.tensor([fx / w, fy / h, cx / w, cy / h], dtype=torch.float32, device=device)
        # Prefer full matrix if model wants it
        out = _model.infer(img, depth_in=depth, intrinsics=K)
    except Exception:
        out = _model.infer(img, depth_in=depth, intrinsics=K)
    depth_pred = out["depth"]
    if torch.is_tensor(depth_pred):
        depth_pred = depth_pred.detach().float().cpu().numpy()
    depth_pred = np.squeeze(depth_pred)
    return {"depth": depth_pred, "vram": vram_info()}


def _gradio_run(rgb, depth_img, fx, fy, cx, cy):
    if rgb is None or depth_img is None:
        raise gr.Error("Need RGB + depth images")
    import cv2

    if depth_img.ndim == 3:
        depth_m = depth_img[:, :, 0].astype(np.float32)
    else:
        depth_m = depth_img.astype(np.float32)
    # Heuristic: 16-bit mm vs meters
    if depth_m.max() > 100:
        depth_m = depth_m / 1000.0
    h, w = rgb.shape[:2]
    fx = float(fx) if fx else float(w)
    fy = float(fy) if fy else float(w)
    cx = float(cx) if cx else w / 2
    cy = float(cy) if cy else h / 2
    with _lock:
        res = refine(rgb, depth_m, fx, fy, cx, cy)
    d = res["depth"]
    vis = (np.clip(d / (np.percentile(d, 99) + 1e-6), 0, 1) * 255).astype(np.uint8)
    vis = cv2.applyColorMap(vis, cv2.COLORMAP_INFERNO)[:, :, ::-1]
    return vis, d, str(res["vram"])


class InferBody(BaseModel):
    # base64 paths omitted for simplicity — Gradio primary
    note: str = Field(default="Use Gradio /ui for image upload; POST multipart coming later")


app = FastAPI(title="LingBot Depth One")
_ui_built = False


@app.on_event("startup")
def _startup() -> None:
    def run():
        try:
            _load()
        except Exception:
            pass

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
        "model": HF_ID,
        "vram": vram_info(),
        "app": "lingbotdepthone",
    }


@app.get("/api/v1/info")
def info():
    return {"model": HF_ID, "model_dir": MODEL_DIR, "ready": _ready, "vram": vram_info()}


def build_ui():
    with gr.Blocks(title="LingBot Depth One") as demo:
        gr.Markdown("# LingBot Depth One\nMetric depth refinement / completion (ViT-L v0.5)")
        with gr.Row():
            rgb = gr.Image(label="RGB", type="numpy")
            depth = gr.Image(label="Raw depth", type="numpy")
        with gr.Row():
            fx = gr.Number(label="fx", value=0)
            fy = gr.Number(label="fy", value=0)
            cx = gr.Number(label="cx", value=0)
            cy = gr.Number(label="cy", value=0)
        btn = gr.Button("Refine", variant="primary")
        out_vis = gr.Image(label="Refined depth viz")
        out_raw = gr.Numpy(label="Depth meters")
        out_vram = gr.Textbox(label="VRAM")
        btn.click(_gradio_run, [rgb, depth, fx, fy, cx, cy], [out_vis, out_raw, out_vram])
    return demo


demo = build_ui()
app = mount_gradio(app, demo, path="/ui")

if __name__ == "__main__":
    append_bootstrap(f"uvicorn :{SERVER_PORT}")
    uvicorn.run(app, host="0.0.0.0", port=SERVER_PORT, log_level="info")
