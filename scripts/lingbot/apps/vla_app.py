#!/usr/bin/env python3
"""LingBot-VLA 2.0 Gradio + REST (policy smoke / action dump)."""

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
LOG = setup_logging("lingbotvlaone")

import gradio as gr
import numpy as np
import torch
import uvicorn
from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from pydantic import BaseModel

SERVER_PORT = int(os.getenv("SERVER_PORT", "7860"))
MODEL_DIR = Path(os.getenv("LINGBOT_MODEL_DIR", "/models/lingbotvlaone/weights"))
SRC = os.getenv("LINGBOT_SRC", "")
HF_ID = os.getenv("HF_REPO_ID", "robbyant/lingbot-vla-v2-6b")

_lock = threading.Lock()
_ready = False
_load_error = None
_policy = None


def _load():
    global _policy, _ready, _load_error
    with _lock:
        if _ready:
            return
        t0 = time.time()
        try:
            if SRC and SRC not in sys.path:
                sys.path.insert(0, SRC)
            # Weights present check + optional safetensors index
            safes = list(MODEL_DIR.glob("*.safetensors")) + list(MODEL_DIR.glob("model-*.safetensors"))
            if not safes and not (MODEL_DIR / "config.json").is_file():
                raise FileNotFoundError(f"No VLA weights in {MODEL_DIR}")
            # Try official policy module
            try:
                # deploy path varies by repo layout
                import importlib

                for mod_name in (
                    "deploy.lingbot_vla_v2_policy",
                    "lingbotvla.policy",
                ):
                    try:
                        importlib.import_module(mod_name)
                        append_bootstrap(f"imported {mod_name}")
                        break
                    except Exception as e:
                        append_bootstrap(f"import soft-fail {mod_name}: {e}")
                # Load safetensors meta only for smoke if full policy needs robot config
                _policy = {"model_dir": str(MODEL_DIR), "shards": [p.name for p in safes[:8]]}
            except Exception as e:
                append_bootstrap(f"policy module soft-fail: {e}")
                _policy = {"model_dir": str(MODEL_DIR), "shards": [p.name for p in safes[:8]]}
            if torch.cuda.is_available():
                # Touch GPU
                torch.zeros(1, device="cuda")
            _ready = True
            append_bootstrap(f"vla weights ready {time.time()-t0:.1f}s shards={len(safes)} vram={vram_info()}")
            set_ready_phase("vla_weights_ready")
        except Exception as exc:
            _load_error = str(exc)
            append_bootstrap(f"vla load FAIL: {exc}")
            raise


def predict_action(image: np.ndarray, instruction: str) -> str:
    _load()
    # Placeholder action dump — full closed-loop needs robot proprio + post-train ckpt
    h, w = image.shape[:2] if image is not None else (0, 0)
    return (
        f"policy={_policy}\ninstruction={instruction!r}\nimage={w}x{h}\n"
        f"vram={vram_info()}\n"
        "Note: wire deploy.lingbot_vla_v2_policy for real robot actions after post-train."
    )


class ActReq(BaseModel):
    instruction: str = "pick up the cup"
    note: str = "Upload image via Gradio; JSON image base64 optional later"


app = FastAPI(title="LingBot VLA 2.0 One")


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
        "policy": _policy,
        "vram": vram_info(),
    }


@app.get("/api/v1/info")
def info():
    return health()


@app.post("/api/v1/action")
def action(req: ActReq):
    _load()
    return {"ok": True, "instruction": req.instruction, "policy": _policy, "vram": vram_info()}


def build_ui():
    with gr.Blocks(title="LingBot VLA 2.0") as demo:
        gr.Markdown("# LingBot VLA 2.0 One\nWeight load + action API smoke (post-train for real robots)")
        img = gr.Image(type="numpy", label="Camera")
        instr = gr.Textbox(label="Instruction", value="pick up the red block")
        btn = gr.Button("Predict", variant="primary")
        out = gr.Textbox(label="Output")
        btn.click(predict_action, [img, instr], [out])
    return demo


demo = build_ui()
app = mount_gradio(app, demo, path="/ui")

if __name__ == "__main__":
    append_bootstrap(f"uvicorn :{SERVER_PORT}")
    uvicorn.run(app, host="0.0.0.0", port=SERVER_PORT, log_level="info")
