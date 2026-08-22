"""Mage-Flow Gradio + REST + MCP for Olares One (RTX 5090M 24GB)."""

from __future__ import annotations

import base64
import gc
import io
import logging
import os
import random
import threading
import time
from contextlib import asynccontextmanager
from typing import Any

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")


def _ensure_gradio_temp_dir() -> str:
    d = (os.getenv("GRADIO_TEMP_DIR") or "").strip()
    if not d:
        d = "/output/gradio"
        os.environ["GRADIO_TEMP_DIR"] = d
    os.makedirs(d, mode=0o755, exist_ok=True)
    return os.path.realpath(d)


_GRADIO_TEMP_DIR = _ensure_gradio_temp_dir()
_GRADIO_MOUNT_PATH = "/ui"
_GRADIO_ROOT_PATH = (os.getenv("GRADIO_ROOT_PATH") or "").strip() or _GRADIO_MOUNT_PATH

import gradio as gr  # noqa: E402
import torch  # noqa: E402
import uvicorn  # noqa: E402
from fastapi import FastAPI, File, Form, HTTPException, UploadFile  # noqa: E402
from fastapi.responses import RedirectResponse  # noqa: E402
from PIL import Image  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from mcp_server import mcp, mcp_http_app  # noqa: E402

LOG = logging.getLogger("mageflowone")

SERVER_PORT = int(os.getenv("SERVER_PORT", "7860"))
DEFAULT_MODEL = os.getenv("MAGE_DEFAULT_MODEL", "Mage-Flow-Turbo").strip() or "Mage-Flow-Turbo"
PRELOAD = os.getenv("MAGE_PRELOAD", "0").strip().lower() in ("1", "true", "yes", "on")
MAX_SEED = 2**31 - 1

# One chart — pick checkpoint. Peak ~18–20GB; only one loaded at a time.
# microsoft/* repos are gated (401 without HF access) → hub reports 404.
# Public mirrors: mage-flow-community/* (same weights / MageFlowPipeline layout).
MODEL_CATALOG: dict[str, dict[str, Any]] = {
    "Mage-Flow-Turbo": {
        "repo_id": "mage-flow-community/Mage-Flow-Turbo",
        "task": "t2i",
        "steps": 4,
        "cfg": 1.0,
        "label": "T2I Turbo (4-step)",
    },
    "Mage-Flow-Edit-Turbo": {
        "repo_id": "mage-flow-community/Mage-Flow-Edit-Turbo",
        "task": "edit",
        "steps": 4,
        "cfg": 1.0,
        "label": "Edit Turbo (4-step)",
    },
    "Mage-Flow": {
        "repo_id": "mage-flow-community/Mage-Flow",
        "task": "t2i",
        "steps": 20,
        "cfg": 5.0,
        "label": "T2I RL-aligned",
    },
    "Mage-Flow-Edit": {
        "repo_id": "mage-flow-community/Mage-Flow-Edit",
        "task": "edit",
        "steps": 30,
        "cfg": 5.0,
        "label": "Edit RL-aligned",
    },
    "Mage-Flow-Base": {
        "repo_id": "mage-flow-community/Mage-Flow-Base",
        "task": "t2i",
        "steps": 30,
        "cfg": 5.0,
        "label": "T2I Base",
    },
    "Mage-Flow-Edit-Base": {
        "repo_id": "mage-flow-community/Mage-Flow-Edit-Base",
        "task": "edit",
        "steps": 30,
        "cfg": 5.0,
        "label": "Edit Base",
    },
}

_PIPE_LOCK = threading.Lock()
_PIPE: Any = None
_LOADED_KEY: str | None = None


def _setup_logging() -> None:
    level = getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO)
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def _vram_report() -> str:
    if not torch.cuda.is_available():
        return "CUDA unavailable"
    free, total = torch.cuda.mem_get_info()
    alloc = torch.cuda.memory_allocated()
    reserved = torch.cuda.memory_reserved()
    return (
        f"VRAM free {free / 1024**3:.2f} / {total / 1024**3:.2f} GB  "
        f"({100 * free / total:.0f}% free)\n"
        f"PyTorch alloc {alloc / 1024**3:.2f} GB  reserved {reserved / 1024**3:.2f} GB"
    )


def _free_cuda(label: str = "", *, hard: bool = False) -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        if hard:
            try:
                torch.cuda.ipc_collect()
            except Exception:
                pass
    if label:
        LOG.info("%s cuda cache cleared%s", label, " (hard)" if hard else "")


def _unload_pipeline() -> str:
    global _PIPE, _LOADED_KEY
    if _PIPE is not None:
        try:
            del _PIPE
        except Exception:
            pass
        _PIPE = None
        _LOADED_KEY = None
    _free_cuda("after unload", hard=True)
    return f"Unloaded.\n\n{_vram_report()}"


def _resolve_model_key(key: str) -> str:
    k = (key or "").strip()
    if k in MODEL_CATALOG:
        return k
    # Allow full HF id
    for name, meta in MODEL_CATALOG.items():
        if meta["repo_id"] == k:
            return name
    raise ValueError(f"Unknown model {key!r}. Choose: {', '.join(MODEL_CATALOG)}")


def _flash_attn_available() -> bool:
    try:
        import flash_attn  # noqa: F401

        return True
    except Exception:
        return False


def _configure_attn_backend() -> None:
    """Mage defaults attn_type=flash2; without flash-attn, Qwen3-VL TextEncoder crashes.

    Mage supports SDPA via ModelConfig.attn_type=sdpa and VF_HF_ATTN_IMPL for HF.
    """
    want = (os.getenv("MAGE_ATTN_TYPE") or "").strip().lower()
    if not want:
        want = "flash2" if _flash_attn_available() else "sdpa"
    if want in ("sdpa", "torch_sdpa", "scaled_dot_product_attention"):
        os.environ.setdefault("VF_HF_ATTN_IMPL", "sdpa")
    LOG.info("mage attn backend prefer=%s VF_HF_ATTN_IMPL=%s", want, os.environ.get("VF_HF_ATTN_IMPL"))

    from mage_flow.models.mage_flow import MageFlowModel

    if getattr(MageFlowModel.__init__, "_mage_attn_patched", False):
        return

    _orig = MageFlowModel.__init__

    def _init(self, config, *args, **kwargs):  # type: ignore[no-untyped-def]
        cfg = config
        try:
            cur = str(getattr(cfg, "attn_type", "flash2") or "flash2").lower()
            if want == "sdpa" and cur not in ("sdpa", "torch_sdpa", "scaled_dot_product_attention"):
                cfg = cfg.model_copy(update={"attn_type": "sdpa"})
            elif want in ("flash2", "flash4") and cur != want:
                cfg = cfg.model_copy(update={"attn_type": want})
        except Exception as exc:
            LOG.warning("attn_type patch skipped: %s", exc)
        return _orig(self, cfg, *args, **kwargs)

    _init._mage_attn_patched = True  # type: ignore[attr-defined]
    MageFlowModel.__init__ = _init  # type: ignore[method-assign]


def _load_pipeline(model_key: str):
    global _PIPE, _LOADED_KEY
    key = _resolve_model_key(model_key)
    if _PIPE is not None and _LOADED_KEY == key:
        return _PIPE, MODEL_CATALOG[key]
    if _PIPE is not None:
        LOG.info("switching model %s → %s", _LOADED_KEY, key)
        _unload_pipeline()

    from mage_flow import MageFlowPipeline

    _configure_attn_backend()

    meta = MODEL_CATALOG[key]
    repo = meta["repo_id"]
    LOG.info("loading %s (%s)…", key, repo)
    t0 = time.perf_counter()
    pipe = MageFlowPipeline.from_pretrained(repo, device="cuda")
    _PIPE = pipe
    _LOADED_KEY = key
    LOG.info("loaded %s in %.1fs\n%s", key, time.perf_counter() - t0, _vram_report())
    return pipe, meta


def generate_image(
    prompt: str,
    *,
    model_key: str = DEFAULT_MODEL,
    width: int = 1024,
    height: int = 1024,
    steps: int | None = None,
    cfg: float | None = None,
    seed: int = -1,
    negative_prompt: str = " ",
    park_vram: bool = False,
):
    if not prompt or not str(prompt).strip():
        raise ValueError("prompt is required")
    key = _resolve_model_key(model_key)
    meta = MODEL_CATALOG[key]
    if meta["task"] != "t2i":
        raise ValueError(f"{key} is an edit model — use edit_image /api/v1/edit")

    width = max(512, min(2048, int(width)))
    height = max(512, min(2048, int(height)))
    width = (width // 16) * 16 or 512
    height = (height // 16) * 16 or 512
    use_steps = int(steps) if steps is not None else int(meta["steps"])
    use_cfg = float(cfg) if cfg is not None else float(meta["cfg"])
    if seed is None or int(seed) < 0:
        seed = random.randint(0, MAX_SEED)
    else:
        seed = int(seed)

    t0 = time.perf_counter()
    with _PIPE_LOCK:
        pipe, meta = _load_pipeline(key)
        t_load = time.perf_counter()
        imgs = pipe.generate(
            [str(prompt).strip()],
            steps=use_steps,
            cfg=use_cfg,
            heights=[height],
            widths=[width],
            seeds=[seed],
            neg_prompts=[negative_prompt or " "],
        )
        image = imgs[0]
        denoise_s = time.perf_counter() - t_load
        if park_vram:
            _free_cuda("after generate")
    total_s = time.perf_counter() - t0
    timing = (
        f"model={key}  {width}×{height}  steps={use_steps}  cfg={use_cfg}\n"
        f"denoise   {denoise_s:7.2f}s\n"
        f"total     {total_s:7.2f}s\n\n{_vram_report()}"
    )
    LOG.info("[t2i] model=%s seed=%s %.2fs %sx%s", key, seed, total_s, width, height)
    return image, seed, str(prompt).strip(), timing


def edit_image(
    prompt: str,
    ref_image: Image.Image,
    *,
    model_key: str = "Mage-Flow-Edit-Turbo",
    width: int | None = None,
    height: int | None = None,
    max_size: int = 1024,
    steps: int | None = None,
    cfg: float | None = None,
    seed: int = -1,
    park_vram: bool = False,
):
    if not prompt or not str(prompt).strip():
        raise ValueError("prompt is required")
    if ref_image is None:
        raise ValueError("reference image is required")
    key = _resolve_model_key(model_key)
    meta = MODEL_CATALOG[key]
    if meta["task"] != "edit":
        raise ValueError(f"{key} is a T2I model — use generate_image /api/v1/generate")

    use_steps = int(steps) if steps is not None else int(meta["steps"])
    use_cfg = float(cfg) if cfg is not None else float(meta["cfg"])
    if seed is None or int(seed) < 0:
        seed = random.randint(0, MAX_SEED)
    else:
        seed = int(seed)

    kw: dict[str, Any] = {
        "steps": use_steps,
        "cfg": use_cfg,
        "seeds": [seed],
    }
    if width and height:
        w = max(512, min(2048, (int(width) // 16) * 16 or 512))
        h = max(512, min(2048, (int(height) // 16) * 16 or 512))
        kw["widths"] = [w]
        kw["heights"] = [h]
    else:
        kw["max_size"] = int(max_size) if max_size else 1024

    t0 = time.perf_counter()
    with _PIPE_LOCK:
        pipe, meta = _load_pipeline(key)
        t_load = time.perf_counter()
        imgs = pipe.edit([str(prompt).strip()], [ref_image.convert("RGB")], **kw)
        image = imgs[0]
        denoise_s = time.perf_counter() - t_load
        if park_vram:
            _free_cuda("after edit")
    total_s = time.perf_counter() - t0
    ow, oh = image.size
    timing = (
        f"model={key}  out={ow}×{oh}  steps={use_steps}  cfg={use_cfg}\n"
        f"denoise   {denoise_s:7.2f}s\n"
        f"total     {total_s:7.2f}s\n\n{_vram_report()}"
    )
    LOG.info("[edit] model=%s seed=%s %.2fs", key, seed, total_s)
    return image, seed, str(prompt).strip(), timing


def list_models() -> list[dict[str, Any]]:
    return [
        {"key": k, "repo_id": v["repo_id"], "task": v["task"], "steps": v["steps"], "cfg": v["cfg"], "label": v["label"]}
        for k, v in MODEL_CATALOG.items()
    ]


def build_gradio() -> gr.Blocks:
    t2i_keys = [k for k, v in MODEL_CATALOG.items() if v["task"] == "t2i"]
    edit_keys = [k for k, v in MODEL_CATALOG.items() if v["task"] == "edit"]
    default_t2i = DEFAULT_MODEL if DEFAULT_MODEL in t2i_keys else "Mage-Flow-Turbo"
    default_edit = "Mage-Flow-Edit-Turbo"

    with gr.Blocks(title="Mage Flow One") as demo:
        gr.Markdown(
            "# Mage Flow One\n"
            "Microsoft **Mage-Flow** (4B) on Olares One — pick Turbo for speed, "
            "Edit-Turbo for instruction edits. One model loaded at a time (~18–20 GB)."
        )
        with gr.Tabs():
            with gr.Tab("Text → Image"):
                with gr.Row():
                    with gr.Column():
                        t2i_model = gr.Dropdown(t2i_keys, value=default_t2i, label="Model")
                        t2i_prompt = gr.Textbox(label="Prompt", lines=3, placeholder="A close-up portrait…")
                        t2i_neg = gr.Textbox(label="Negative prompt", value=" ", lines=1)
                        with gr.Row():
                            t2i_w = gr.Slider(512, 2048, value=1024, step=16, label="Width")
                            t2i_h = gr.Slider(512, 2048, value=1024, step=16, label="Height")
                        with gr.Row():
                            t2i_steps = gr.Slider(1, 50, value=4, step=1, label="Steps (Turbo=4)")
                            t2i_cfg = gr.Slider(0.0, 10.0, value=1.0, step=0.1, label="CFG (Turbo=1)")
                        t2i_seed = gr.Number(label="Seed (−1 = random)", value=-1, precision=0)
                        t2i_run = gr.Button("Generate", variant="primary")
                    with gr.Column():
                        t2i_out = gr.Image(label="Output", type="pil")
                        t2i_timing = gr.Textbox(label="Timing", lines=6, interactive=False)
                        t2i_seed_out = gr.Number(label="Seed used", precision=0)

                def _on_t2i_model(m: str):
                    meta = MODEL_CATALOG.get(m) or MODEL_CATALOG["Mage-Flow-Turbo"]
                    return int(meta["steps"]), float(meta["cfg"])

                t2i_model.change(_on_t2i_model, inputs=[t2i_model], outputs=[t2i_steps, t2i_cfg])

                def _t2i(prompt, model, w, h, steps, cfg, seed, neg):
                    img, s, _p, timing = generate_image(
                        prompt,
                        model_key=model,
                        width=int(w),
                        height=int(h),
                        steps=int(steps),
                        cfg=float(cfg),
                        seed=int(seed),
                        negative_prompt=neg or " ",
                    )
                    return img, s, timing

                t2i_run.click(
                    _t2i,
                    inputs=[t2i_prompt, t2i_model, t2i_w, t2i_h, t2i_steps, t2i_cfg, t2i_seed, t2i_neg],
                    outputs=[t2i_out, t2i_seed_out, t2i_timing],
                )

            with gr.Tab("Image Edit"):
                with gr.Row():
                    with gr.Column():
                        edit_model = gr.Dropdown(edit_keys, value=default_edit, label="Model")
                        edit_prompt = gr.Textbox(
                            label="Edit instruction",
                            lines=2,
                            placeholder="Replace the background with a field of sunflowers",
                        )
                        edit_ref = gr.Image(label="Reference", type="pil")
                        edit_max = gr.Slider(512, 2048, value=1024, step=16, label="Max size (long edge)")
                        with gr.Row():
                            edit_steps = gr.Slider(1, 50, value=4, step=1, label="Steps")
                            edit_cfg = gr.Slider(0.0, 10.0, value=1.0, step=0.1, label="CFG")
                        edit_seed = gr.Number(label="Seed (−1 = random)", value=-1, precision=0)
                        edit_run = gr.Button("Edit", variant="primary")
                    with gr.Column():
                        edit_out = gr.Image(label="Output", type="pil")
                        edit_timing = gr.Textbox(label="Timing", lines=6, interactive=False)
                        edit_seed_out = gr.Number(label="Seed used", precision=0)

                def _on_edit_model(m: str):
                    meta = MODEL_CATALOG.get(m) or MODEL_CATALOG["Mage-Flow-Edit-Turbo"]
                    return int(meta["steps"]), float(meta["cfg"])

                edit_model.change(_on_edit_model, inputs=[edit_model], outputs=[edit_steps, edit_cfg])

                def _edit(prompt, model, ref, max_size, steps, cfg, seed):
                    img, s, _p, timing = edit_image(
                        prompt,
                        ref,
                        model_key=model,
                        max_size=int(max_size),
                        steps=int(steps),
                        cfg=float(cfg),
                        seed=int(seed),
                    )
                    return img, s, timing

                edit_run.click(
                    _edit,
                    inputs=[edit_prompt, edit_model, edit_ref, edit_max, edit_steps, edit_cfg, edit_seed],
                    outputs=[edit_out, edit_seed_out, edit_timing],
                )

            with gr.Tab("Memory"):
                mem = gr.Textbox(label="VRAM", lines=4, value=_vram_report(), interactive=False)
                with gr.Row():
                    btn_ref = gr.Button("Refresh")
                    btn_unload = gr.Button("Unload model", variant="stop")
                btn_ref.click(lambda: _vram_report(), outputs=[mem])
                btn_unload.click(lambda: _unload_pipeline(), outputs=[mem])

    return demo


class GenerateRequest(BaseModel):
    prompt: str
    model: str = DEFAULT_MODEL
    width: int = Field(default=1024, ge=512, le=2048)
    height: int = Field(default=1024, ge=512, le=2048)
    steps: int | None = Field(default=None, ge=1, le=50)
    cfg: float | None = Field(default=None, ge=0.0, le=20.0)
    seed: int = -1
    negative_prompt: str = " "


class EditRequest(BaseModel):
    prompt: str
    model: str = "Mage-Flow-Edit-Turbo"
    max_size: int = Field(default=1024, ge=512, le=2048)
    width: int | None = Field(default=None, ge=512, le=2048)
    height: int | None = Field(default=None, ge=512, le=2048)
    steps: int | None = Field(default=None, ge=1, le=50)
    cfg: float | None = Field(default=None, ge=0.0, le=20.0)
    seed: int = -1
    image_b64: str = Field(..., description="PNG/JPEG base64 (raw or data URL)")


def _pil_to_b64(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _b64_to_pil(raw: str) -> Image.Image:
    s = (raw or "").strip()
    if s.startswith("data:"):
        s = s.split(",", 1)[-1]
    data = base64.b64decode(s)
    return Image.open(io.BytesIO(data)).convert("RGB")


def build_api(demo: gr.Blocks) -> FastAPI:
    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        if PRELOAD:
            try:
                with _PIPE_LOCK:
                    _load_pipeline(DEFAULT_MODEL)
            except Exception:
                LOG.exception("preload failed")
        async with mcp.session_manager.run():
            yield

    api = FastAPI(title="Mage Flow One", version="1.0.0", docs_url="/docs", lifespan=lifespan)

    @api.get("/")
    def root():
        return RedirectResponse(url="/ui/", status_code=307)

    @api.get("/gradio_api/{path:path}")
    def _gradio_api_prefix_fix(path: str):
        from urllib.parse import quote

        return RedirectResponse(url=f"/ui/gradio_api/{quote(path, safe='/')}", status_code=307)

    @api.get("/health")
    def health():
        return {
            "status": "ok",
            "loaded": _LOADED_KEY,
            "default_model": DEFAULT_MODEL,
            "models": list_models(),
            "cuda": torch.cuda.is_available(),
            "vram": _vram_report(),
            "mcp": {"transport": "streamable-http", "url": "/mcp/mcp"},
        }

    @api.get("/api/mcp")
    def mcp_info():
        return {
            "transport": "streamable-http",
            "endpoints": ["/mcp/mcp"],
            "tools": [
                "health_check",
                "list_models",
                "generate_image",
                "edit_image",
                "unload_model",
            ],
            "in_cluster_url": "http://mageflowone:7860/mcp/mcp",
        }

    @api.get("/api/v1/models")
    def api_models():
        return {"models": list_models(), "loaded": _LOADED_KEY}

    @api.post("/api/v1/unload")
    def api_unload():
        with _PIPE_LOCK:
            return {"ok": True, "detail": _unload_pipeline()}

    @api.post("/api/v1/generate")
    def api_generate(body: GenerateRequest):
        try:
            image, seed_out, final_prompt, timing = generate_image(
                body.prompt,
                model_key=body.model,
                width=body.width,
                height=body.height,
                steps=body.steps,
                cfg=body.cfg,
                seed=body.seed,
                negative_prompt=body.negative_prompt,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            LOG.exception("api generate failed")
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        return {
            "seed": seed_out,
            "width": (body.width // 16) * 16,
            "height": (body.height // 16) * 16,
            "prompt": final_prompt,
            "model": body.model,
            "image_b64": _pil_to_b64(image),
            "mime_type": "image/png",
            "timing": timing,
        }

    @api.post("/api/v1/edit")
    def api_edit(body: EditRequest):
        try:
            ref = _b64_to_pil(body.image_b64)
            image, seed_out, final_prompt, timing = edit_image(
                body.prompt,
                ref,
                model_key=body.model,
                max_size=body.max_size,
                width=body.width,
                height=body.height,
                steps=body.steps,
                cfg=body.cfg,
                seed=body.seed,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            LOG.exception("api edit failed")
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        w, h = image.size
        return {
            "seed": seed_out,
            "width": w,
            "height": h,
            "prompt": final_prompt,
            "model": body.model,
            "image_b64": _pil_to_b64(image),
            "mime_type": "image/png",
            "timing": timing,
        }

    @api.post("/api/v1/edit/upload")
    async def api_edit_upload(
        prompt: str = Form(...),
        model: str = Form("Mage-Flow-Edit-Turbo"),
        max_size: int = Form(1024),
        steps: int | None = Form(None),
        cfg: float | None = Form(None),
        seed: int = Form(-1),
        file: UploadFile = File(...),
    ):
        raw = await file.read()
        ref = Image.open(io.BytesIO(raw)).convert("RGB")
        try:
            image, seed_out, final_prompt, timing = edit_image(
                prompt,
                ref,
                model_key=model,
                max_size=max_size,
                steps=steps,
                cfg=cfg,
                seed=seed,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            LOG.exception("api edit upload failed")
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        w, h = image.size
        return {
            "seed": seed_out,
            "width": w,
            "height": h,
            "prompt": final_prompt,
            "model": model,
            "image_b64": _pil_to_b64(image),
            "mime_type": "image/png",
            "timing": timing,
        }

    def _gradio_allowed_paths() -> list[str]:
        order = (_GRADIO_TEMP_DIR, "/tmp/gradio", "/output")
        return list(dict.fromkeys(os.path.realpath(p) for p in order))

    api.mount("/mcp", mcp_http_app())
    return gr.mount_gradio_app(
        api,
        demo.queue(default_concurrency_limit=1),
        path=_GRADIO_MOUNT_PATH,
        root_path=_GRADIO_ROOT_PATH,
        allowed_paths=_gradio_allowed_paths(),
    )


def main() -> None:
    _setup_logging()
    LOG.info("starting Mage Flow One on 0.0.0.0:%s default=%s", SERVER_PORT, DEFAULT_MODEL)
    demo = build_gradio()
    app = build_api(demo)
    uvicorn.run(app, host="0.0.0.0", port=SERVER_PORT, log_level=os.getenv("LOG_LEVEL", "info").lower())


if __name__ == "__main__":
    main()
