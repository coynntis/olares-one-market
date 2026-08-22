#!/usr/bin/env python3
"""SenseNova-Vision Gradio + REST with timing (Olares One soft-boot)."""

from __future__ import annotations

import base64
import io
import os
import sys
import tempfile
import threading
import time
from typing import Any, Optional

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
sys.path.insert(0, os.path.dirname(__file__))

from _common import (  # noqa: E402
    append_bootstrap,
    ensure_gradio_temp,
    mount_gradio,
    set_ready_phase,
    setup_logging,
    vram_info,
)

ensure_gradio_temp()
LOG = setup_logging("sensenovavisionone")

import gradio as gr  # noqa: E402
import uvicorn  # noqa: E402
from fastapi import FastAPI, File, Form, UploadFile  # noqa: E402
from fastapi.responses import JSONResponse, RedirectResponse  # noqa: E402
from PIL import Image  # noqa: E402

SERVER_PORT = int(os.getenv("SERVER_PORT", "7860"))
MODEL_DIR = os.getenv("SENSENOVA_MODEL_DIR", "/models/sensenovavisionone/weights")
SRC = os.getenv("SENSENOVA_SRC", "")
HF_ID = os.getenv("HF_REPO_ID", "sensenova/SenseNova-Vision-7B-MoT")
DTYPE = os.getenv("SENSENOVA_DTYPE", "nf4").strip() or "nf4"
DEVICE = os.getenv("SENSENOVA_DEVICE", "cuda").strip() or "cuda"
MAX_MEM = os.getenv("SENSENOVA_MAX_MEM_PER_GPU", "20GiB").strip() or "20GiB"
OFFLOAD = os.getenv(
    "SENSENOVA_OFFLOAD_FOLDER",
    f"/workspace/{os.getenv('APP_NAME', 'sensenovavisionone')}/offload",
)
OUTPUT_DIR = os.getenv("SENSENOVA_OUTPUT_DIR", "/output/sensenova")
LOAD_LAZY = os.getenv("LOAD_LAZY", "0").strip() in {"1", "true", "yes", "on"}

TASK_ORDER = [
    "raw_query",
    "depth",
    "normal",
    "binary_seg",
    "pan_seg",
    "gcg_seg",
    "bbox_detection",
    "point_detection",
    "keypoint",
    "ocr",
    "recon3d",
    "camera_pose",
]

_lock = threading.Lock()
_model = None
_ready = False
_load_error: Optional[str] = None
_load_elapsed_s: Optional[float] = None
_timings: list[dict[str, Any]] = []
_TIMINGS_MAX = 50


def _ensure_src() -> None:
    if SRC and SRC not in sys.path:
        sys.path.insert(0, SRC)


def _load() -> None:
    global _model, _ready, _load_error, _load_elapsed_s
    with _lock:
        if _model is not None:
            return
        t0 = time.time()
        try:
            _ensure_src()
            os.makedirs(OFFLOAD, exist_ok=True)
            os.makedirs(OUTPUT_DIR, exist_ok=True)

            import torch
            from inference.sensenova_vision import SenseNovaVisionModel

            want_cuda = torch.cuda.is_available() and DEVICE != "cpu"
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

            ema = os.path.join(MODEL_DIR, "ema.safetensors")
            if not os.path.isfile(ema):
                raise FileNotFoundError(
                    f"Missing {ema}. Wait for HF download or set HF_REPO_ID / model path."
                )

            device = "cuda" if want_cuda else "cpu"
            append_bootstrap(
                f"loading SenseNovaVisionModel path={MODEL_DIR} dtype={DTYPE} "
                f"device={device} max_mem={MAX_MEM} offload={OFFLOAD}"
            )
            _model = SenseNovaVisionModel(
                model_path=MODEL_DIR,
                checkpoint_name="ema.safetensors",
                dtype=DTYPE,
                device=device,
                max_mem_per_gpu=MAX_MEM,
                offload_folder=OFFLOAD,
                download_local_files_only=True,
            )
            _ready = True
            _load_elapsed_s = round(time.time() - t0, 1)
            append_bootstrap(
                f"model ready {_load_elapsed_s}s vram={vram_info()} dtype={DTYPE}"
            )
            set_ready_phase(f"model_ready_{device}_{DTYPE}")
        except Exception as exc:
            _load_error = str(exc)
            append_bootstrap(f"model load FAIL: {exc}")
            raise


def _record_timing(task: str, elapsed_s: float, ok: bool, detail: str = "") -> None:
    entry = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "task": task,
        "elapsed_s": round(elapsed_s, 3),
        "ok": ok,
        "detail": detail[:200],
        "vram": vram_info(),
    }
    _timings.append(entry)
    if len(_timings) > _TIMINGS_MAX:
        del _timings[: len(_timings) - _TIMINGS_MAX]
    append_bootstrap(
        f"infer task={task} ok={ok} elapsed_s={entry['elapsed_s']} {detail[:80]}"
    )


def _save_pil(image: Image.Image, suffix: str = ".png") -> str:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    fd, path = tempfile.mkstemp(suffix=suffix, prefix="snv_", dir=OUTPUT_DIR)
    os.close(fd)
    image.save(path)
    return path


def _pil_to_b64(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def run_task(
    image: Image.Image,
    task: str,
    query: str = "",
    mode: str = "",
    seed: int = 42,
    extra_images: Optional[list[Image.Image]] = None,
) -> dict[str, Any]:
    """Run one SenseNova-Vision task; returns dict with timing + outputs."""
    _load()
    assert _model is not None
    _ensure_src()

    from inference.inference_demo import (  # noqa: WPS433
        build_prompt,
        model_question,
        resolve_request_mode,
    )

    task = (task or "depth").strip()
    if task not in TASK_ORDER:
        raise ValueError(f"Unsupported task={task}; choose from {TASK_ORDER}")

    t0 = time.time()
    try:
        prompt = build_prompt(task, query or "")
        resolved_mode = resolve_request_mode(task, mode or "")
        paths: list[str] = []
        paths.append(_save_pil(image.convert("RGB")))
        for extra in extra_images or []:
            if extra is not None:
                paths.append(_save_pil(extra.convert("RGB")))

        if task == "recon3d":
            result = _model.reconstruct_3d(
                images=paths,
                prompt=prompt,
                noise_seed=int(seed),
                glb_output=os.path.join(
                    OUTPUT_DIR, f"recon3d_{int(time.time())}.glb"
                ),
            )
            out_image = None
            out_text = str(result.get("text") or "")
            glb = None
            scene = result.get("scene")
            if scene is not None:
                glb_path = os.path.join(OUTPUT_DIR, f"recon3d_{int(time.time())}.glb")
                try:
                    scene.export(file_obj=glb_path)
                    glb = glb_path
                except Exception as exc:
                    append_bootstrap(f"glb export soft-fail: {exc}")
            elapsed = time.time() - t0
            _record_timing(task, elapsed, True, f"mode={resolved_mode}")
            return {
                "ok": True,
                "task": task,
                "mode": resolved_mode,
                "prompt": prompt,
                "elapsed_s": round(elapsed, 3),
                "text": out_text,
                "image": out_image,
                "glb_path": glb,
                "vram": vram_info(),
            }

        result = _model.generate(
            question=model_question(prompt, len(paths)),
            images=paths,
            mode=resolved_mode,
            noise_seed=int(seed),
            return_intermediate_outputs=True,
        )
        out_image = None
        out_text = None
        if isinstance(result, dict):
            out_image = result.get("image")
            out_text = result.get("text")
        elif isinstance(result, Image.Image):
            out_image = result
        else:
            out_text = str(result) if result is not None else None

        elapsed = time.time() - t0
        _record_timing(task, elapsed, True, f"mode={resolved_mode}")
        return {
            "ok": True,
            "task": task,
            "mode": resolved_mode,
            "prompt": prompt,
            "elapsed_s": round(elapsed, 3),
            "text": out_text,
            "image": out_image,
            "vram": vram_info(),
        }
    except Exception as exc:
        elapsed = time.time() - t0
        _record_timing(task, elapsed, False, str(exc))
        raise


def _gradio_run(image, task, query, seed):
    if image is None:
        raise gr.Error("Upload an image first")
    if isinstance(image, Image.Image):
        pil = image
    else:
        pil = Image.fromarray(image.astype("uint8")).convert("RGB")
    out = run_task(pil, task=task, query=query or "", seed=int(seed or 42))
    img = out.get("image")
    text = out.get("text") or ""
    meta = (
        f"task={out['task']} mode={out['mode']} elapsed_s={out['elapsed_s']}\n"
        f"vram={out.get('vram')}\n"
        f"prompt={out.get('prompt')}"
    )
    return img, text, meta


app = FastAPI(title="SenseNova Vision One")


@app.on_event("startup")
def _startup():
    if not LOAD_LAZY:
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
        "model_dir": MODEL_DIR,
        "dtype": DTYPE,
        "device": DEVICE,
        "max_mem_per_gpu": MAX_MEM,
        "load_elapsed_s": _load_elapsed_s,
        "vram": vram_info(),
        "license": "CC-BY-NC-4.0",
        "note": (
            "Official demo recommends ~80GB GPU; this chart uses accelerate "
            "offload (SENSENOVA_MAX_MEM_PER_GPU) for RTX 5090M 24GB + 96GB RAM."
        ),
    }


@app.get("/api/v1/info")
def info():
    return {
        **health(),
        "tasks": TASK_ORDER,
        "timings_tail": _timings[-10:],
    }


@app.get("/api/v1/timings")
def timings(limit: int = 20):
    n = max(1, min(int(limit or 20), _TIMINGS_MAX))
    return {"timings": _timings[-n:], "count": len(_timings)}


@app.post("/api/v1/predict")
async def predict(
    file: UploadFile = File(...),
    task: str = Form("depth"),
    query: str = Form(""),
    mode: str = Form(""),
    seed: int = Form(42),
):
    raw = await file.read()
    pil = Image.open(io.BytesIO(raw)).convert("RGB")
    try:
        out = run_task(pil, task=task, query=query, mode=mode, seed=int(seed))
    except Exception as exc:
        return JSONResponse(
            status_code=500,
            content={"ok": False, "error": str(exc), "task": task},
        )
    payload: dict[str, Any] = {
        "ok": True,
        "task": out["task"],
        "mode": out["mode"],
        "prompt": out["prompt"],
        "elapsed_s": out["elapsed_s"],
        "text": out.get("text"),
        "vram": out.get("vram"),
        "glb_path": out.get("glb_path"),
    }
    if isinstance(out.get("image"), Image.Image):
        payload["image_png_base64"] = _pil_to_b64(out["image"])
    return payload


def build_ui():
    with gr.Blocks(title="SenseNova Vision One") as demo:
        gr.Markdown(
            "# SenseNova Vision One\n"
            "Unified multimodal CV (depth, normals, seg, detect, OCR, …) "
            "via SenseNova-Vision-7B-MoT.\n\n"
            f"**dtype** `{DTYPE}` · **max GPU mem** `{MAX_MEM}` · "
            "CC BY-NC 4.0 · REST: `POST /api/v1/predict` · timings: `GET /api/v1/timings`"
        )
        with gr.Row():
            with gr.Column():
                inp = gr.Image(type="pil", label="Image")
                task = gr.Dropdown(TASK_ORDER, value="depth", label="Task")
                query = gr.Textbox(
                    label="Query (needed for raw_query / seg / detect)",
                    placeholder="person, car  OR  free-form question",
                )
                seed = gr.Number(value=42, precision=0, label="Seed")
                btn = gr.Button("Run", variant="primary")
            with gr.Column():
                out_img = gr.Image(type="pil", label="Output image")
                out_txt = gr.Textbox(label="Output text", lines=8)
                out_meta = gr.Textbox(label="Timing / metadata", lines=4)
        btn.click(_gradio_run, [inp, task, query, seed], [out_img, out_txt, out_meta])
    return demo


demo = build_ui()
app = mount_gradio(app, demo, path="/ui")

if __name__ == "__main__":
    append_bootstrap(f"uvicorn :{SERVER_PORT}")
    uvicorn.run(app, host="0.0.0.0", port=SERVER_PORT, log_level="info")
