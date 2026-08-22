"""LocateAnything-3B: FastAPI + Gradio on Olares One."""

from __future__ import annotations

import asyncio
import base64
import io
import logging
import os
import threading
from contextlib import asynccontextmanager
from typing import Literal

import gradio as gr
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ImageDraw
from pydantic import BaseModel, Field
from starlette.responses import RedirectResponse

# Before model/magi_attention import: Blackwell needs FA4, not Hopper flex_flash_attn (sm90).
# v1.1.1+ defaults to kernel_backend=ffa; force fa4 on RTX 5090M (sm_120).
os.environ.setdefault("MAGI_ATTENTION_PREBUILD_FFA", "0")
if os.getenv("MAGI_ATTENTION_KERNEL_BACKEND"):
    pass
elif os.getenv("MAGI_ATTENTION_FA4_BACKEND"):
    pass
else:
    os.environ.setdefault("MAGI_ATTENTION_KERNEL_BACKEND", "fa4")

from worker import LocateAnythingWorker

LOG = logging.getLogger("locateanything3bone")

_GENERATION_MODES = ("fast", "slow", "hybrid")


def _fa4_backend_ready() -> bool:
    """FA4 on Blackwell needs flash_attn_cute from MagiAttention v1.1+ build."""
    try:
        import importlib.util

        return importlib.util.find_spec("flash_attn_cute") is not None
    except Exception:
        return False


def _resolved_generation_mode(requested: str | None = None) -> str:
    """hybrid/MTP needs FA4 on sm100; fall back to slow if image lacks flash_attn_cute."""
    mode = (requested or os.getenv("GENERATION_MODE", "") or "hybrid").strip().lower()
    if mode != "hybrid":
        return mode
    kb = os.getenv("MAGI_ATTENTION_KERNEL_BACKEND", "").strip().lower()
    fa4_legacy = os.getenv("MAGI_ATTENTION_FA4_BACKEND", "").strip().lower()
    fa4_off = kb not in ("", "fa4") and fa4_legacy in ("0", "false", "no")
    if kb in ("ffa", "sdpa", "sdpa_ol") or fa4_off:
        return mode
    if _fa4_backend_ready():
        return mode
    LOG.warning(
        "hybrid needs MagiAttention FA4 (flash_attn_cute) on Blackwell; "
        "using slow mode. Rebuild with MagiAttention v1.1+ or set GENERATION_MODE=slow."
    )
    return "slow"
_TASKS = (
    "freeform",
    "detect",
    "ground_single",
    "ground_multi",
    "ground_text",
    "detect_text",
    "ground_gui_box",
    "ground_gui_point",
    "point",
)

_inference_lock = threading.Lock()
_worker: LocateAnythingWorker | None = None


def _model_path() -> str:
    p = (os.getenv("MODEL_PATH") or os.getenv("MODEL_DIR") or "").strip()
    if not p:
        raise RuntimeError("MODEL_PATH or MODEL_DIR required")
    return p


def _dtype_from_env() -> torch.dtype:
    name = os.getenv("DTYPE", "bfloat16").lower()
    mapping = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    if name not in mapping:
        raise ValueError(f"Unsupported DTYPE={name}")
    return mapping[name]


def _load_worker() -> None:
    global _worker
    device = os.getenv("DEVICE", "cuda")
    dtype = _dtype_from_env()
    path = _model_path()
    LOG.info("loading model from %s device=%s dtype=%s", path, device, dtype)
    _worker = LocateAnythingWorker(path, device=device, dtype=dtype)
    try:
        import magi_attention  # noqa: F401

        LOG.info("MagiAttention available (MTP fast path)")
    except ImportError:
        LOG.info("MagiAttention not installed; using PyTorch SDPA fallback")
    LOG.info("model ready")


def _require_worker() -> LocateAnythingWorker:
    if _worker is None:
        raise HTTPException(status_code=503, detail="model not loaded")
    return _worker


def _locked(fn):
    with _inference_lock:
        return fn()


def _pil_from_upload(img) -> Image.Image:
    if img is None:
        raise HTTPException(status_code=400, detail="image required")
    if isinstance(img, Image.Image):
        return img.convert("RGB")
    return Image.fromarray(img).convert("RGB")


def _draw_overlay(image: Image.Image, answer: str) -> Image.Image:
    w, h = image.size
    out = image.copy()
    draw = ImageDraw.Draw(out)
    boxes = LocateAnythingWorker.parse_boxes(answer, w, h)
    for i, b in enumerate(boxes):
        draw.rectangle([b["x1"], b["y1"], b["x2"], b["y2"]], outline=(255, 64, 64), width=3)
        draw.text((b["x1"] + 2, max(0, b["y1"] - 14)), f"box{i + 1}", fill=(255, 220, 64))
    for i, p in enumerate(LocateAnythingWorker.parse_points(answer, w, h)):
        r = 6
        draw.ellipse([p["x"] - r, p["y"] - r, p["x"] + r, p["y"] + r], fill=(64, 200, 255))
        draw.text((p["x"] + 8, p["y"] - 8), f"p{i + 1}", fill=(200, 240, 255))
    return out


def _run_task(
    worker: LocateAnythingWorker,
    task: str,
    image: Image.Image,
    phrase: str,
    categories_csv: str,
    generation_mode: str,
    max_new_tokens: int,
    temperature: float,
) -> dict:
    kwargs = {
        "generation_mode": _resolved_generation_mode(generation_mode),
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "verbose": os.getenv("VERBOSE_GENERATION", "0").strip().lower() in ("1", "true", "yes"),
    }
    if task == "freeform":
        if not phrase.strip():
            raise HTTPException(status_code=400, detail="prompt required for freeform")
        return worker.predict(image, phrase.strip(), **kwargs)
    if task == "detect":
        cats = [c.strip() for c in categories_csv.split(",") if c.strip()]
        if not cats:
            raise HTTPException(status_code=400, detail="categories required (comma-separated)")
        return worker.detect(image, cats, **kwargs)
    if task == "ground_single":
        return worker.ground_single(image, phrase, **kwargs)
    if task == "ground_multi":
        return worker.ground_multi(image, phrase, **kwargs)
    if task == "ground_text":
        return worker.ground_text(image, phrase, **kwargs)
    if task == "detect_text":
        return worker.detect_text(image, **kwargs)
    if task == "ground_gui_box":
        return worker.ground_gui(image, phrase, output_type="box", **kwargs)
    if task == "ground_gui_point":
        return worker.ground_gui(image, phrase, output_type="point", **kwargs)
    if task == "point":
        return worker.point(image, phrase, **kwargs)
    raise HTTPException(status_code=400, detail=f"unknown task {task}")


class PredictRequest(BaseModel):
    task: Literal[
        "freeform",
        "detect",
        "ground_single",
        "ground_multi",
        "ground_text",
        "detect_text",
        "ground_gui_box",
        "ground_gui_point",
        "point",
    ] = "ground_multi"
    image_base64: str
    phrase: str = ""
    categories: list[str] = Field(default_factory=list)
    generation_mode: Literal["fast", "slow", "hybrid"] = "hybrid"
    max_new_tokens: int = 2048
    temperature: float = 0.7


@asynccontextmanager
async def lifespan(_app: FastAPI):
    logging.basicConfig(
        level=getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, _load_worker)
    yield


app = FastAPI(title="LocateAnything-3B", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return RedirectResponse(url="/ui/", status_code=307)


@app.get("/healthz")
def healthz():
    if _worker is None:
        raise HTTPException(status_code=503, detail="loading")
    return {"status": "ok"}


@app.get("/api/v1/capabilities")
def capabilities():
    magi = False
    try:
        import magi_attention  # noqa: F401

        magi = True
    except ImportError:
        pass
    fa4 = _fa4_backend_ready()
    effective = _resolved_generation_mode()
    return {
        "model": os.getenv("MODEL_REPO", "nvidia/LocateAnything-3B"),
        "tasks": list(_TASKS),
        "generation_modes": list(_GENERATION_MODES),
        "magi_attention": magi,
        "magi_fa4_backend": fa4,
        "effective_generation_mode": effective,
        "magi_env": {
            "MAGI_ATTENTION_PREBUILD_FFA": os.getenv("MAGI_ATTENTION_PREBUILD_FFA", "0"),
            "MAGI_ATTENTION_KERNEL_BACKEND": os.getenv(
                "MAGI_ATTENTION_KERNEL_BACKEND", "fa4"
            ),
            "MAGI_ATTENTION_FA4_BACKEND": os.getenv("MAGI_ATTENTION_FA4_BACKEND", ""),
        },
        "endpoints": {"predict": "POST /api/v1/predict"},
        "notes": [
            "v1.1.1 defaults kernel_backend=ffa (sm90); set MAGI_ATTENTION_KERNEL_BACKEND=fa4 on 5090M",
            "hybrid needs flash_attn_cute from install_flash_attn_cute.sh sm100",
            "max_new_tokens 8192 for long dense scenes if VRAM allows",
        ],
    }


def _decode_b64_image(data: str) -> Image.Image:
    s = data.strip()
    if "," in s and s.lower().startswith("data:"):
        s = s.split(",", 1)[1]
    raw = base64.standard_b64decode(s)
    return Image.open(io.BytesIO(raw)).convert("RGB")


@app.post("/api/v1/predict")
def api_predict(body: PredictRequest):
    worker = _require_worker()
    image = _decode_b64_image(body.image_base64)
    cats_csv = ",".join(body.categories)

    def go():
        return _run_task(
            worker,
            body.task,
            image,
            body.phrase,
            cats_csv,
            body.generation_mode,
            body.max_new_tokens,
            body.temperature,
        )

    result = _locked(go)
    w, h = image.size
    answer = result.get("answer", "")
    return {
        **result,
        "boxes": LocateAnythingWorker.parse_boxes(answer, w, h),
        "points": LocateAnythingWorker.parse_points(answer, w, h),
    }


def _ui_run(task, img, phrase, categories, mode, max_tokens, temperature):
    try:
        if img is None:
            raise gr.Error("upload an image")
        pil = _pil_from_upload(img)
        buf = io.BytesIO()
        pil.save(buf, format="PNG")
        b64 = base64.standard_b64encode(buf.getvalue()).decode("ascii")
        req = PredictRequest(
            task=task,
            image_base64=b64,
            phrase=phrase or "",
            categories=[c.strip() for c in (categories or "").split(",") if c.strip()],
            generation_mode=mode,
            max_new_tokens=int(max_tokens),
            temperature=float(temperature),
        )
        out = api_predict(req)
        answer = out.get("answer", "")
        overlay = _draw_overlay(pil, answer)
        stats = out.get("stats")
        stats_txt = "" if stats is None else str(stats)
        return answer, overlay, stats_txt
    except HTTPException as he:
        raise gr.Error(str(he.detail)) from he
    except Exception as exc:
        LOG.exception("gradio predict")
        raise gr.Error(str(exc)) from exc


def build_gradio() -> gr.Blocks:
    default_mode = _resolved_generation_mode()
    with gr.Blocks(title="LocateAnything-3B") as demo:
        gr.Markdown(
            "# LocateAnything-3B\n"
            "Visual grounding (boxes / points) on Olares One. "
            "**API:** `POST /api/v1/predict` · **Docs:** `/docs`"
        )
        with gr.Row():
            task = gr.Dropdown(list(_TASKS), value="ground_multi", label="task")
            mode = gr.Dropdown(list(_GENERATION_MODES), value=default_mode, label="generation_mode")
        img = gr.Image(label="image", type="numpy")
        phrase = gr.Textbox(label="phrase / freeform prompt", lines=2)
        categories = gr.Textbox(
            label="categories (detect only, comma-separated)",
            placeholder="person, car, bicycle",
        )
        with gr.Row():
            max_tok = gr.Slider(256, 8192, value=2048, step=256, label="max_new_tokens")
            temp = gr.Slider(0.0, 1.5, value=0.7, label="temperature")
        go = gr.Button("Run", variant="primary")
        answer = gr.Textbox(label="raw answer", lines=8)
        vis = gr.Image(label="overlay")
        stats = gr.Textbox(label="stats (if any)", lines=3)
        go.click(_ui_run, [task, img, phrase, categories, mode, max_tok, temp], [answer, vis, stats])
    return demo


_GRADIO_MOUNT = "/ui"
_GRADIO_ROOT = (os.getenv("GRADIO_ROOT_PATH") or "").strip() or _GRADIO_MOUNT

app = gr.mount_gradio_app(
    app,
    build_gradio(),
    path=_GRADIO_MOUNT,
    root_path=_GRADIO_ROOT,
)


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("SERVER_PORT", "7860"))
    uvicorn.run(app, host="0.0.0.0", port=port)
