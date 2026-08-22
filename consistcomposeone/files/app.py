"""ConsistCompose-BAGEL-7B-MoT layout compose — FastAPI + Gradio (NF4 for 24GB)."""

from __future__ import annotations

import asyncio
import base64
import io
import logging
import os
import random
import sys
import tempfile
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel, Field
from starlette.responses import RedirectResponse

LOG = logging.getLogger("consistcomposeone")

_GRADIO_MOUNT_PATH = "/ui"
_GRADIO_ROOT_PATH = (os.getenv("GRADIO_ROOT_PATH") or "").strip() or _GRADIO_MOUNT_PATH

_inference_lock = threading.Lock()
_runtime: Any = None
_load_error: str | None = None
_attn_note: str = "unknown"
_precision: str = "nf4"


def _env_truthy(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).strip().lower() in ("1", "true", "yes", "on")


def _ensure_gradio_temp_dir() -> str:
    d = (os.getenv("GRADIO_TEMP_DIR") or "").strip() or "/output/gradio"
    os.environ["GRADIO_TEMP_DIR"] = d
    os.makedirs(d, mode=0o755, exist_ok=True)
    return os.path.realpath(d)


_GRADIO_TEMP_DIR = _ensure_gradio_temp_dir()

import gradio as gr  # noqa: E402

ComposeMode = Literal[
    "layout_t2i",
    "layout_subject_driven",
    "generate",
    "think_generate",
    "edit",
    "think_edit",
    "understanding",
]


def _ensure_pythonpath() -> Path:
    """ConsistCompose clone root (PYTHONPATH / CONSIST_COMPOSE_ROOT)."""
    root = (os.getenv("CONSIST_COMPOSE_ROOT") or "/workspace/ConsistCompose").strip()
    p = Path(root)
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
    return p


def ensure_model_dir() -> str:
    """snapshot_download ConsistCompose weights into MODEL_DIR."""
    model_dir = (os.getenv("MODEL_DIR") or "").strip()
    model_repo = (
        os.getenv("MODEL_REPO") or "sensenova/ConsistCompose-BAGEL-7B-MoT"
    ).strip()
    if not model_dir:
        model_dir = f"/models/{model_repo.split('/')[-1]}"
    marker = os.path.join(model_dir, "model.safetensors")
    if os.path.isfile(marker):
        LOG.info("model already present at %s", model_dir)
        return model_dir

    os.makedirs(model_dir, exist_ok=True)
    token = (os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN") or "").strip() or None
    endpoint = (os.getenv("HF_ENDPOINT") or "").strip()
    if endpoint:
        os.environ["HF_ENDPOINT"] = endpoint

    from huggingface_hub import snapshot_download

    kwargs: dict[str, Any] = {"repo_id": model_repo, "local_dir": model_dir}
    if token:
        kwargs["token"] = token
    LOG.info("snapshot_download start repo=%s dest=%s", model_repo, model_dir)
    t0 = time.monotonic()
    snapshot_download(**kwargs)
    LOG.info("snapshot_download done elapsed_s=%.1f", time.monotonic() - t0)
    return model_dir


def resolve_attn_note() -> str:
    """Detect flash_attn availability (inference uses patched SDPA fallback if missing)."""
    global _attn_note
    backend = (os.getenv("ATTENTION_BACKEND") or "sdpa").strip().lower()
    if backend in ("sdpa", "eager"):
        _attn_note = "sdpa"
        return _attn_note
    try:
        import flash_attn  # noqa: F401

        _attn_note = "flash_attn"
        LOG.info("flash_attn import ok")
    except Exception as exc:  # noqa: BLE001
        LOG.warning("flash_attn unavailable (%s); ConsistCompose navit patched → SDPA", exc)
        _attn_note = "sdpa_fallback"
    return _attn_note


def set_seed(seed: int | None) -> None:
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def decode_b64_image(data: str) -> Image.Image:
    s = data.strip()
    if "," in s and s.lower().startswith("data:"):
        s = s.split(",", 1)[1]
    return Image.open(io.BytesIO(base64.standard_b64decode(s))).convert("RGB")


def image_to_b64_png(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="PNG")
    return base64.standard_b64encode(buf.getvalue()).decode("ascii")


def _save_temp_images(images: list[Image.Image], tmpdir: str) -> list[str]:
    paths: list[str] = []
    for i, im in enumerate(images):
        p = os.path.join(tmpdir, f"ref_{i:02d}.png")
        im.convert("RGB").save(p, format="PNG")
        paths.append(p)
    return paths


class ConsistComposeRuntime:
    """Wraps ConsistComposeBagelModel with NF4 (BAGEL Gradio mode 2 equivalent)."""

    def __init__(self, model_path: str) -> None:
        global _precision, _attn_note
        _ensure_pythonpath()
        resolve_attn_note()
        precision = (os.getenv("MODEL_PRECISION") or "nf4").strip().lower()
        if precision not in ("nf4", "int8", "bf16"):
            LOG.warning("unknown MODEL_PRECISION=%r; forcing nf4", precision)
            precision = "nf4"
        _precision = precision
        if precision == "bf16":
            LOG.warning(
                "bf16 weights ~29GB — will NOT fit RTX 5090M 24GB; prefer nf4"
            )

        out_dir = (os.getenv("OUT_IMG_DIR") or "/output/consistcompose").strip()
        os.makedirs(out_dir, exist_ok=True)
        # accelerate offload_folder defaults to cwd-relative "offload"
        offload = (os.getenv("OFFLOAD_DIR") or "/workspace/offload").strip()
        os.makedirs(offload, exist_ok=True)
        try:
            os.chdir("/workspace")
        except OSError:
            pass

        from consist_compose import ConsistComposeBagelModel

        LOG.info(
            "loading ConsistCompose model_path=%s precision=%s attn=%s",
            model_path,
            precision,
            _attn_note,
        )
        t0 = time.monotonic()
        self.model = ConsistComposeBagelModel(
            model_path=model_path,
            out_img_dir=out_dir,
            dtype=precision,
        )
        LOG.info(
            "model ready elapsed_s=%.1f precision=%s attn=%s",
            time.monotonic() - t0,
            precision,
            _attn_note,
        )

    def compose(
        self,
        *,
        prompt: str,
        mode: str,
        images: list[Image.Image] | None,
        vis_bbox: bool,
        resize_short_edge: int,
        seed: int | None,
    ) -> Image.Image:
        set_seed(seed)
        images = images or []
        # layout_t2i: no <image> tokens. subject_driven: prompt must contain matching <image>.
        with tempfile.TemporaryDirectory(prefix="cc_refs_", dir=_GRADIO_TEMP_DIR) as td:
            img_paths = _save_temp_images(images, td) if images else None
            path_or_text = self.model.generate(
                question=prompt,
                images=img_paths,
                mode=mode,
                vis_bbox=vis_bbox,
                resize_short_edge=resize_short_edge,
            )
        if path_or_text is None:
            raise RuntimeError("generate() returned None")
        if isinstance(path_or_text, str) and os.path.isfile(path_or_text):
            return Image.open(path_or_text).convert("RGB")
        if isinstance(path_or_text, Image.Image):
            return path_or_text.convert("RGB")
        # understanding / text modes
        raise RuntimeError(f"expected image path, got {type(path_or_text)!r}: {path_or_text!r}")


def _load_runtime() -> None:
    global _runtime, _load_error
    try:
        path = ensure_model_dir()
        _runtime = ConsistComposeRuntime(path)
        _load_error = None
    except Exception as exc:  # noqa: BLE001
        LOG.exception("model load failed")
        _load_error = str(exc)
        _runtime = None
        raise


def _ensure_runtime() -> ConsistComposeRuntime:
    global _runtime
    if _runtime is not None:
        return _runtime
    with _inference_lock:
        if _runtime is None:
            _load_runtime()
        if _runtime is None:
            raise HTTPException(status_code=503, detail=_load_error or "model not loaded")
        return _runtime


def _require_runtime() -> ConsistComposeRuntime:
    if _runtime is None:
        if _env_truthy("LOAD_LAZY", "0"):
            return _ensure_runtime()
        raise HTTPException(status_code=503, detail=_load_error or "model not loaded")
    return _runtime


class ComposeRequest(BaseModel):
    prompt: str = Field(
        ...,
        description=(
            "Layout prompt with linguistic bbox tokens, e.g. "
            "'a dragon <bbox>[0.38, 0.08, 0.76, 0.67]</bbox> ...' "
            "For layout_subject_driven insert <image> markers matching images_base64."
        ),
    )
    mode: ComposeMode = "layout_t2i"
    image_base64: str | None = None
    images_base64: list[str] = Field(default_factory=list)
    vis_bbox: bool = False
    resize_short_edge: int = 768
    seed: int | None = 42


@asynccontextmanager
async def lifespan(_app: FastAPI):
    logging.basicConfig(
        level=getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    if not _env_truthy("LOAD_LAZY", "0"):
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, _load_runtime)
    else:
        LOG.info("LOAD_LAZY=1 — model downloads/loads on first /api/v1/compose")
    yield


app = FastAPI(
    title="ConsistCompose BAGEL-7B-MoT",
    description=(
        "Layout-controlled multi-instance image composition (ConsistCompose). "
        "NF4 / bitsandbytes for 24GB VRAM. For SenseNova-U1 generation use sensenovau1serveone. "
        "SenseNova-Vision skipped (CC-BY-NC)."
    ),
    lifespan=lifespan,
)
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


@app.get("/health")
@app.get("/healthz")
def health():
    if _runtime is None and not _env_truthy("LOAD_LAZY", "0"):
        raise HTTPException(status_code=503, detail=_load_error or "loading")
    return {
        "status": "ok" if _runtime is not None else "loading",
        "model_ready": _runtime is not None,
        "precision": _precision,
        "attention": _attn_note,
        "load_error": _load_error,
        "modality": "layout_compose",
    }


@app.get("/api/v1/capabilities")
def capabilities():
    return {
        "modalities": ["layout_t2i", "layout_subject_driven", "generate"],
        "model_repo": os.getenv("MODEL_REPO", "sensenova/ConsistCompose-BAGEL-7B-MoT"),
        "precision": _precision,
        "attention": _attn_note,
        "model_ready": _runtime is not None,
        "endpoints": {"compose": "POST /api/v1/compose"},
        "notes": [
            "NF4 (BAGEL Gradio mode 2) required for RTX 5090M 24GB — BF16 weights ~29GB.",
            "Embed layout as <bbox>[x0, y0, x1, y1]</bbox> (normalized) in the prompt.",
            "flash_attn soft-fails to SDPA (no flash_attn==2.5.8 pin).",
            "For SenseNova-U1 T2I/editing use sensenovau1serveone. Vision-7B skipped (CC-BY-NC).",
        ],
    }


@app.post("/api/v1/compose")
def api_compose(body: ComposeRequest):
    images_b64 = list(body.images_base64)
    if body.image_base64:
        images_b64 = [body.image_base64] + images_b64
    images = [decode_b64_image(x) for x in images_b64]
    rt = _require_runtime()

    def go():
        with _inference_lock:
            return rt.compose(
                prompt=body.prompt,
                mode=body.mode,
                images=images,
                vis_bbox=body.vis_bbox,
                resize_short_edge=body.resize_short_edge,
                seed=body.seed,
            )

    try:
        img = go()
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        LOG.exception("compose failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return {
        "image_base64": image_to_b64_png(img),
        "mode": body.mode,
        "precision": _precision,
        "attention": _attn_note,
    }


_DEFAULT_LAYOUT_PROMPT = (
    "In a dimly lit cavern, a powerful dragon "
    "<bbox>[0.380, 0.086, 0.768, 0.673]</bbox> stands majestically. "
    "Beside it, a brave man <bbox>[0.155, 0.231, 0.439, 0.717]</bbox> "
    "clad in armor stands with a gleaming sword "
    "<bbox>[0.166, 0.401, 0.577, 0.663]</bbox>."
)


def _load_ref_images(ref_files) -> list[Image.Image]:
    images: list[Image.Image] = []
    if not ref_files:
        return images
    # Gradio File(file_count=multiple) → list of paths or single path
    paths = ref_files if isinstance(ref_files, list) else [ref_files]
    for p in paths:
        if not p:
            continue
        path = p if isinstance(p, str) else getattr(p, "name", None) or str(p)
        images.append(Image.open(path).convert("RGB"))
    return images


def _ui_compose(prompt, mode, ref_files, vis_bbox, resize_short_edge, seed):
    try:
        images = _load_ref_images(ref_files)
        out = api_compose(
            ComposeRequest(
                prompt=prompt or "",
                mode=mode or "layout_t2i",
                images_base64=[image_to_b64_png(im) for im in images],
                vis_bbox=bool(vis_bbox),
                resize_short_edge=int(resize_short_edge),
                seed=int(seed) if seed is not None else None,
            )
        )
        raw = base64.standard_b64decode(out["image_base64"])
        return Image.open(io.BytesIO(raw)).convert("RGB")
    except HTTPException as he:
        raise gr.Error(str(he.detail)) from he
    except Exception as exc:  # noqa: BLE001
        LOG.exception("gradio compose")
        raise gr.Error(str(exc)) from exc


def build_gradio() -> gr.Blocks:
    with gr.Blocks(title="ConsistCompose One") as demo:
        gr.Markdown(
            "# ConsistCompose (BAGEL-7B-MoT)\n"
            "Layout-controlled multi-instance image composition — **NF4** for 24GB.\n"
            "Embed boxes as `<bbox>[x0, y0, x1, y1]</bbox>` (normalized 0–1). "
            "Subject-driven: put `<image>` in prompt and upload refs.\n"
            "API: `POST /api/v1/compose` · docs: `/docs` · "
            "For U1 gen use **sensenovau1serveone**."
        )
        prompt = gr.Textbox(
            label="layout prompt",
            lines=8,
            value=_DEFAULT_LAYOUT_PROMPT,
        )
        with gr.Row():
            mode = gr.Dropdown(
                choices=[
                    "layout_t2i",
                    "layout_subject_driven",
                    "generate",
                    "think_generate",
                ],
                value="layout_t2i",
                label="mode",
            )
            vis_bbox = gr.Checkbox(value=False, label="vis_bbox overlay")
            resize = gr.Slider(256, 1024, value=768, step=32, label="resize_short_edge")
            seed = gr.Number(value=42, precision=0, label="seed")
        refs = gr.File(
            label="reference images (subject-driven; match <image> tokens)",
            file_count="multiple",
            type="filepath",
            file_types=["image"],
        )
        go = gr.Button("Compose", variant="primary")
        out = gr.Image(label="output", type="pil")
        go.click(_ui_compose, [prompt, mode, refs, vis_bbox, resize, seed], [out])
    return demo


app = gr.mount_gradio_app(
    app,
    build_gradio(),
    path=_GRADIO_MOUNT_PATH,
    root_path=_GRADIO_ROOT_PATH,
    allowed_paths=[_GRADIO_TEMP_DIR, tempfile.gettempdir(), "/output"],
)


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("SERVER_PORT", "7860"))
    uvicorn.run(app, host="0.0.0.0", port=port)
