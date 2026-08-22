"""SenseNova-SI-1.5 InternVL3-8B spatial VQA — FastAPI + Gradio (understand only, no image gen)."""

from __future__ import annotations

import asyncio
import base64
import io
import logging
import math
import os
import tempfile
import threading
import time
from contextlib import asynccontextmanager
from typing import Any

import torch
import torchvision.transforms as T
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel, Field
from starlette.responses import RedirectResponse
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoConfig, AutoModel, AutoTokenizer

LOG = logging.getLogger("sensenovasi15")

_GRADIO_MOUNT_PATH = "/ui"
_GRADIO_ROOT_PATH = (os.getenv("GRADIO_ROOT_PATH") or "").strip() or _GRADIO_MOUNT_PATH

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

_inference_lock = threading.Lock()
_runtime: "InternVLVQARuntime | None" = None
_load_error: str | None = None
_attn_used: str = "unknown"


def _env_truthy(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).strip().lower() in ("1", "true", "yes", "on")


def _ensure_gradio_temp_dir() -> str:
    d = (os.getenv("GRADIO_TEMP_DIR") or "").strip() or "/output/gradio"
    os.environ["GRADIO_TEMP_DIR"] = d
    os.makedirs(d, mode=0o755, exist_ok=True)
    return os.path.realpath(d)


_GRADIO_TEMP_DIR = _ensure_gradio_temp_dir()

import gradio as gr  # noqa: E402


# --- InternVL image utils (mirrors OpenSenseNova/SenseNova-SI sensenova_si/utils.py) ---


def build_transform(input_size: int):
    return T.Compose(
        [
            T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image, min_num=1, max_num=6, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height
    target_ratios = set(
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        processed_images.append(resized_img.crop(box))
    if use_thumbnail and len(processed_images) != 1:
        processed_images.append(image.resize((image_size, image_size)))
    return processed_images


def load_image_pil(image: Image.Image, input_size=448, max_num=6) -> torch.Tensor:
    image = image.convert("RGB")
    transform = build_transform(input_size=input_size)
    tiles = dynamic_preprocess(
        image, image_size=input_size, use_thumbnail=True, max_num=max_num
    )
    return torch.stack([transform(t) for t in tiles])


def reorganize_prompt(message: list[dict[str, str]], image_num: int) -> str:
    if image_num == 1:
        return "<image>\n" + "\n".join(x["value"] for x in message if x["type"] == "text")
    prompt = "".join(x["value"] for x in message if x["type"] == "text")
    return "".join(f"Image-{i + 1}: <image>\n" for i in range(image_num)) + prompt


def split_model_device_map(model_path: str) -> dict[str, int]:
    """Single-GPU-friendly map from SenseNova-SI split_model (all on cuda:0 when 1 GPU)."""
    device_map: dict[str, int] = {}
    world_size = max(1, torch.cuda.device_count())
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    num_layers = config.llm_config.num_hidden_layers
    num_layers_per_gpu = math.ceil(num_layers / (world_size - 0.5))
    num_layers_per_gpu_list = [num_layers_per_gpu] * world_size
    num_layers_per_gpu_list[0] = math.ceil(num_layers_per_gpu_list[0] * 0.5)
    layer_cnt = 0
    for i, num_layer in enumerate(num_layers_per_gpu_list):
        for _ in range(num_layer):
            if layer_cnt >= num_layers:
                break
            device_map[f"language_model.model.layers.{layer_cnt}"] = i
            layer_cnt += 1
    device_map["vision_model"] = 0
    device_map["mlp1"] = 0
    device_map["language_model.model.tok_embeddings"] = 0
    device_map["language_model.model.embed_tokens"] = 0
    device_map["language_model.output"] = 0
    device_map["language_model.model.norm"] = 0
    device_map["language_model.model.rotary_emb"] = 0
    device_map["language_model.lm_head"] = 0
    device_map[f"language_model.model.layers.{num_layers - 1}"] = 0
    return device_map


def resolve_attn_implementation() -> str:
    """Prefer SDPA on sm_120; soft-fail flash_attn → sdpa. Never pin ancient flash_attn."""
    global _attn_used
    backend = (os.getenv("ATTENTION_BACKEND") or "sdpa").strip().lower()
    if backend in ("sdpa", "eager"):
        _attn_used = backend
        return backend
    if backend in ("flash", "flash_attention_2", "auto"):
        try:
            import flash_attn  # noqa: F401

            _attn_used = "flash_attention_2"
            LOG.info("flash_attn import ok → attn_implementation=flash_attention_2")
            return "flash_attention_2"
        except Exception as exc:
            LOG.warning("flash_attn unavailable (%s); falling back to sdpa", exc)
            _attn_used = "sdpa"
            return "sdpa"
    LOG.warning("unknown ATTENTION_BACKEND=%r; using sdpa", backend)
    _attn_used = "sdpa"
    return "sdpa"


def decode_b64_image(data: str) -> Image.Image:
    s = data.strip()
    if "," in s and s.lower().startswith("data:"):
        s = s.split(",", 1)[1]
    return Image.open(io.BytesIO(base64.standard_b64decode(s))).convert("RGB")


def ensure_model_dir() -> str:
    """snapshot_download into MODEL_DIR (or resolve MODEL_REPO under HF_HOME)."""
    model_dir = (os.getenv("MODEL_DIR") or "").strip()
    model_repo = (os.getenv("MODEL_REPO") or "sensenova/SenseNova-SI-1.5-InternVL3-8B").strip()
    if not model_dir:
        model_dir = f"/models/{model_repo.split('/')[-1]}"
    marker = os.path.join(model_dir, "config.json")
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


class InternVLVQARuntime:
    """InternVL chat path matching SenseNova-SI example.py / SenseNovaSIInternVLModel."""

    def __init__(self, model_path: str) -> None:
        attn = resolve_attn_implementation()
        dtype = torch.bfloat16
        device_map = split_model_device_map(model_path)
        LOG.info("loading InternVL model_path=%s attn=%s dtype=bf16", model_path, attn)
        t0 = time.monotonic()
        try:
            self.model = AutoModel.from_pretrained(
                model_path,
                dtype=dtype,
                attn_implementation=attn,
                load_in_8bit=False,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
                device_map=device_map,
            ).eval()
        except Exception as exc:
            if attn != "sdpa":
                LOG.warning("load with attn=%s failed (%s); retrying sdpa", attn, exc)
                global _attn_used
                _attn_used = "sdpa"
                self.model = AutoModel.from_pretrained(
                    model_path,
                    dtype=dtype,
                    attn_implementation="sdpa",
                    load_in_8bit=False,
                    low_cpu_mem_usage=True,
                    trust_remote_code=True,
                    device_map=device_map,
                ).eval()
            else:
                raise
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True, use_fast=False
        )
        self.max_num_per_image = int(os.getenv("MAX_NUM_PER_IMAGE", "6"))
        self.total_max_num = int(os.getenv("TOTAL_MAX_NUM", "64"))
        LOG.info("model ready elapsed_s=%.1f attn=%s", time.monotonic() - t0, _attn_used)

    def _pixel_values(self, images: list[Image.Image]) -> tuple[torch.Tensor, list[int]]:
        if len(images) > 1:
            max_num = max(1, min(self.max_num_per_image, self.total_max_num // len(images)))
        else:
            max_num = self.max_num_per_image
        tiles: list[torch.Tensor] = []
        num_patches: list[int] = []
        for im in images:
            pv = load_image_pil(im, max_num=max_num).to(torch.bfloat16).cuda()
            num_patches.append(pv.size(0))
            tiles.append(pv)
        if len(tiles) == 1:
            return tiles[0], num_patches
        return torch.cat(tiles, dim=0), num_patches

    @torch.inference_mode()
    def run_vqa(
        self,
        *,
        images: list[Image.Image],
        question: str,
        max_new_tokens: int,
        do_sample: bool,
        temperature: float,
        top_p: float,
        top_k: int | None,
        repetition_penalty: float | None,
        num_beams: int,
    ) -> str:
        if not images:
            raise ValueError("at least one image required")
        message: list[dict[str, str]] = [{"type": "image", "value": ""} for _ in images]
        message.append({"type": "text", "value": question})
        prompt = reorganize_prompt(message, len(images))
        pixel_values, num_patches_list = self._pixel_values(images)
        generation_config: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "num_beams": num_beams,
        }
        if do_sample:
            generation_config["temperature"] = temperature
            generation_config["top_p"] = top_p
            if top_k is not None:
                generation_config["top_k"] = top_k
        if repetition_penalty is not None:
            generation_config["repetition_penalty"] = repetition_penalty
        return self.model.chat(
            self.tokenizer,
            pixel_values=pixel_values,
            num_patches_list=num_patches_list,
            question=prompt,
            generation_config=generation_config,
            history=None,
        )


def _load_runtime() -> None:
    global _runtime, _load_error
    try:
        path = ensure_model_dir()
        _runtime = InternVLVQARuntime(path)
        _load_error = None
    except Exception as exc:
        LOG.exception("model load failed")
        _load_error = str(exc)
        _runtime = None
        raise


def _ensure_runtime() -> InternVLVQARuntime:
    global _runtime
    if _runtime is not None:
        return _runtime
    with _inference_lock:
        if _runtime is None:
            _load_runtime()
        if _runtime is None:
            raise HTTPException(status_code=503, detail=_load_error or "model not loaded")
        return _runtime


def _require_runtime() -> InternVLVQARuntime:
    if _runtime is None:
        if _env_truthy("LOAD_LAZY", "0"):
            return _ensure_runtime()
        raise HTTPException(status_code=503, detail=_load_error or "model not loaded")
    return _runtime


class VQARequest(BaseModel):
    question: str
    image_base64: str | None = None
    images_base64: list[str] = Field(default_factory=list)
    max_new_tokens: int = 1024
    do_sample: bool = False
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int | None = None
    repetition_penalty: float | None = 1.0
    num_beams: int = 1


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
        LOG.info("LOAD_LAZY=1 — model downloads/loads on first /api/v1/vqa request")
    yield


app = FastAPI(
    title="SenseNova-SI-1.5 VQA",
    description=(
        "Spatial VQA / vision understanding only (not image generation). "
        "For U1 T2I/editing use sensenovau1serveone. SenseNova-Vision skipped (CC-BY-NC)."
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
        "attention": _attn_used,
        "load_error": _load_error,
        "modality": "vqa_understand_only",
    }


@app.get("/api/v1/capabilities")
def capabilities():
    return {
        "modalities": ["vqa"],
        "model_repo": os.getenv("MODEL_REPO", "sensenova/SenseNova-SI-1.5-InternVL3-8B"),
        "attention": _attn_used,
        "model_ready": _runtime is not None,
        "endpoints": {"vqa": "POST /api/v1/vqa"},
        "notes": [
            "Spatial VQA / vision understanding only — no image generation.",
            "For SenseNova-U1 T2I / editing / interleave use sensenovau1serveone.",
            "SenseNova-Vision is skipped (CC-BY-NC).",
            "ATTENTION_BACKEND=sdpa|auto|flash — flash soft-fails to sdpa (no ancient flash_attn pin).",
        ],
    }


@app.post("/api/v1/vqa")
def api_vqa(body: VQARequest):
    images_b64 = list(body.images_base64)
    if body.image_base64:
        images_b64 = [body.image_base64] + images_b64
    if not images_b64:
        raise HTTPException(status_code=400, detail="image_base64 or images_base64 required")
    images = [decode_b64_image(x) for x in images_b64]
    rt = _require_runtime()

    def go():
        with _inference_lock:
            return rt.run_vqa(
                images=images,
                question=body.question,
                max_new_tokens=body.max_new_tokens,
                do_sample=body.do_sample,
                temperature=body.temperature,
                top_p=body.top_p,
                top_k=body.top_k,
                repetition_penalty=body.repetition_penalty,
                num_beams=body.num_beams,
            )

    try:
        answer = go()
    except HTTPException:
        raise
    except Exception as exc:
        LOG.exception("vqa failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return {"answer": answer, "attention": _attn_used}


def _ui_vqa(img, question, max_new_tokens, do_sample, temperature, top_p):
    try:
        if img is None:
            raise gr.Error("image required")
        if isinstance(img, Image.Image):
            pil = img.convert("RGB")
        else:
            pil = Image.fromarray(img).convert("RGB")
        buf = io.BytesIO()
        pil.save(buf, format="PNG")
        b64 = base64.standard_b64encode(buf.getvalue()).decode("ascii")
        out = api_vqa(
            VQARequest(
                question=question or "",
                image_base64=b64,
                max_new_tokens=int(max_new_tokens),
                do_sample=bool(do_sample),
                temperature=float(temperature),
                top_p=float(top_p),
            )
        )
        return out["answer"]
    except HTTPException as he:
        raise gr.Error(str(he.detail)) from he
    except Exception as exc:
        LOG.exception("gradio vqa")
        raise gr.Error(str(exc)) from exc


def build_gradio() -> gr.Blocks:
    with gr.Blocks(title="SenseNova-SI-1.5 VQA") as demo:
        gr.Markdown(
            "# SenseNova-SI-1.5 (InternVL3-8B)\n"
            "Spatial **VQA / vision understanding only** — not image generation.\n"
            "API: `POST /api/v1/vqa` · docs: `/docs` · "
            "For U1 generation use **sensenovau1serveone**."
        )
        im = gr.Image(label="image", type="pil")
        q = gr.Textbox(label="question", lines=3, value="Please describe the image in detail.")
        with gr.Row():
            mnt = gr.Slider(64, 4096, value=1024, step=64, label="max_new_tokens")
            ds = gr.Checkbox(value=False, label="do_sample")
            temp = gr.Slider(0.0, 1.5, value=0.0, label="temperature")
            tp = gr.Slider(0.0, 1.0, value=1.0, label="top_p")
        go = gr.Button("Ask", variant="primary")
        ans = gr.Textbox(label="answer", lines=14)
        go.click(_ui_vqa, [im, q, mnt, ds, temp, tp], [ans])
    return demo


app = gr.mount_gradio_app(
    app,
    build_gradio(),
    path=_GRADIO_MOUNT_PATH,
    root_path=_GRADIO_ROOT_PATH,
    allowed_paths=[_GRADIO_TEMP_DIR, tempfile.gettempdir()],
)


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("SERVER_PORT", "7860"))
    uvicorn.run(app, host="0.0.0.0", port=port)
