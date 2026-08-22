"""SenseNova-U1 Infographic-V2 FastAPI + Gradio: t2i, vqa, editing, interleave."""

from __future__ import annotations

import asyncio
import base64
import builtins
import gc
import io
import logging
import os
import re
import threading
import time
from contextlib import asynccontextmanager, contextmanager
from typing import Any, Literal

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import RedirectResponse
from PIL import Image
from pydantic import BaseModel, Field


def _ensure_gradio_temp_dir() -> str:
    """Where Gradio stores generated assets (GRADIO_TEMP_DIR). Default /output/gradio = host volume."""
    d = (os.getenv("GRADIO_TEMP_DIR") or "").strip()
    if not d:
        d = "/output/gradio"
        os.environ["GRADIO_TEMP_DIR"] = d
    os.makedirs(d, mode=0o755, exist_ok=True)
    return os.path.realpath(d)


_GRADIO_TEMP_DIR = _ensure_gradio_temp_dir()

import gradio as gr  # noqa: E402
import sensenova_u1
from sensenova_u1.models.neo_unify.utils import load_image_native, smart_resize
from sensenova_u1.utils import (
    DEFAULT_IMAGE_PATCH_SIZE,
    DEFAULT_VRAM_MODE,
    VRAM_MODE_OPTIONS,
    infer_input_device,
    load_model_and_tokenizer,
    make_offload_ctx,
    vram_mode_to_prefetch_count,
)

LOG = logging.getLogger("sensenovau1serve")


def _env_truthy(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).strip().lower() in ("1", "true", "yes", "on")


@contextmanager
def _inference_timer(label: str, *, sync_cuda: bool) -> Any:
    """Wall time with optional CUDA sync. Always records; logs when INFERENCE_TIMING=1."""
    do_log = _env_truthy("INFERENCE_TIMING", "1")
    if sync_cuda and torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    try:
        yield
    finally:
        if sync_cuda and torch.cuda.is_available():
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        _LAST_TIMING[label] = dt
        if do_log:
            LOG.info("timing %s %.3fs (cuda_sync=%s)", label, dt, sync_cuda)


_LAST_TIMING: dict[str, float] = {}
PARK_AFTER_GENERATE = _env_truthy("SENSENOVA_PARK_AFTER_GENERATE", "1")
UNLOAD_AFTER_GENERATE = _env_truthy("SENSENOVA_UNLOAD_AFTER_GENERATE", "0")


def _vram_report() -> str:
    lines: list[str] = []
    if _runtime is None:
        lines.append("model: —")
    else:
        gguf = _loaded_gguf_abs or "(safetensors)"
        lines.append(f"model: loaded  gguf={os.path.basename(gguf) if gguf.startswith('/') else gguf}")
        lines.append(f"device={_runtime.device}  prefetch={_runtime.prefetch_count}")
    if not torch.cuda.is_available():
        lines.append("CUDA: unavailable")
        return "\n".join(lines)
    free_b, total_b = torch.cuda.mem_get_info()
    alloc = torch.cuda.memory_allocated()
    reserved = torch.cuda.memory_reserved()
    peak = torch.cuda.max_memory_allocated() if hasattr(torch.cuda, "max_memory_allocated") else 0
    lines.append(
        f"VRAM free {free_b / 1024**3:.2f} / {total_b / 1024**3:.2f} GB  "
        f"({100.0 * free_b / total_b:.0f}% free)"
    )
    lines.append(
        f"PyTorch alloc {alloc / 1024**3:.2f} GB  reserved {reserved / 1024**3:.2f} GB  "
        f"peak {peak / 1024**3:.2f} GB"
    )
    if _LAST_TIMING:
        parts = "  ".join(f"{k}={v:.2f}s" for k, v in _LAST_TIMING.items())
        lines.append(f"last timing: {parts}")
    return "\n".join(lines)


def _park_runtime_cpu() -> str:
    """Park loaded model on CPU between runs (VRAM free). Cold GPU reload next request."""
    global _runtime
    if _runtime is None:
        return f"Nothing loaded.\n\n{_vram_report()}"
    try:
        _runtime.model.to("cpu")
    except Exception as exc:
        LOG.warning("park model.to(cpu) failed: %s", exc)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    LOG.info("parked SenseNova model on CPU")
    return f"Parked model on CPU.\n\n{_vram_report()}"


def _unload_runtime() -> str:
    """Drop runtime entirely (full free). Next request reloads from disk."""
    global _runtime, _loaded_gguf_abs
    if _runtime is None:
        return f"Nothing loaded.\n\n{_vram_report()}"
    try:
        _runtime.model.to("cpu")
    except Exception:
        pass
    _runtime = None
    _loaded_gguf_abs = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass
    LOG.info("unloaded SenseNova runtime")
    return f"Unloaded model.\n\n{_vram_report()}"


def _ensure_runtime_on_device() -> SenseNovaRuntime:
    """Reload if unloaded; move back to GPU if parked on CPU (VRAM_MODE=full)."""
    rt = _require_runtime()
    if rt.prefetch_count > 0:
        # Layer-offload modes already park via make_offload_ctx; leave as-is.
        return rt
    try:
        # If weights were parked on CPU, bring back for full-GPU mode.
        p = next(rt.model.parameters(), None)
        if p is not None and p.device.type == "cpu":
            LOG.info("restoring parked model to %s", rt.device)
            rt.model.to(rt.device)
    except Exception:
        LOG.exception("ensure on device failed")
    return rt


@contextmanager
def _tee_interleave_prints() -> Any:
    """Mirror upstream interleave_gen print() to LOG (needs INTERLEAVE_VERBOSE + INTERLEAVE_LOG_PRINTS)."""
    if not (_env_truthy("INTERLEAVE_VERBOSE") and _env_truthy("INTERLEAVE_LOG_PRINTS")):
        yield
        return
    orig_print = builtins.print

    def _dup_print(*args: Any, **kwargs: Any) -> None:
        end = kwargs.get("end", "\n")
        sep = kwargs.get("sep", " ")
        msg = sep.join(str(a) for a in args)
        if end not in (None, "\n"):
            msg = msg + str(end)
        if isinstance(msg, str) and msg.strip():
            LOG.info("interleave_stream %s", msg.rstrip("\n")[:8000])
        return orig_print(*args, **kwargs)

    builtins.print = _dup_print
    try:
        yield
    finally:
        builtins.print = orig_print


# Gradio mounted at /ui; client must request /ui/gradio_api/..., not /gradio_api/... (else 404).
_GRADIO_MOUNT_PATH = "/ui"
# If Olares ingress uses another public prefix, set GRADIO_ROOT_PATH to that full path (e.g. /apps/foo/ui).
_GRADIO_ROOT_PATH = (os.getenv("GRADIO_ROOT_PATH") or "").strip() or _GRADIO_MOUNT_PATH

# Buckets from examples/t2i/inference.py
T2I_RESOLUTIONS: dict[str, tuple[int, int]] = {
    "1:1": (2048, 2048),
    "16:9": (2720, 1536),
    "9:16": (1536, 2720),
    "3:2": (2496, 1664),
    "2:3": (1664, 2496),
    "4:3": (2368, 1760),
    "3:4": (1760, 2368),
    "1:2": (1440, 2880),
    "2:1": (2880, 1440),
    "1:3": (1152, 3456),
    "3:1": (3456, 1152),
}

# examples/interleave/inference.py (different from t2i)
INTERLEAVE_RESOLUTIONS: dict[str, tuple[int, int]] = {
    "1:1": (1536, 1536),
    "16:9": (2048, 1152),
    "9:16": (1152, 2048),
    "3:2": (1888, 1248),
    "2:3": (1248, 1888),
    "4:3": (1760, 1312),
    "3:4": (1312, 1760),
    "1:2": (1088, 2144),
    "2:1": (2144, 1088),
    "1:3": (864, 2592),
    "3:1": (2592, 864),
}

DEFAULT_SYSTEM_MESSAGE = """You are a multimodal assistant capable of reasoning with both text and images. You support two modes:\n\nThink Mode: When reasoning is needed, you MUST start with a <think></think> block and place all reasoning inside it. You MUST interleave text with generated images using tags like <image1>, <image2>. Images can ONLY be generated between <think> and </think>, and may be referenced in the final answer.\n\nNon-Think Mode: When no reasoning is needed, directly provide the answer without reasoning. Do not use tags like <image1>, <image2>; present any images naturally alongside the text.\n\nAfter the think block, always provide a concise, user-facing final answer. The answer may include text, images, or both. Match the user's language in both reasoning and the final answer."""

NORM_MEAN = (0.5, 0.5, 0.5)
NORM_STD = (0.5, 0.5, 0.5)

_IMAGE_GRID_FACTOR = DEFAULT_IMAGE_PATCH_SIZE
DEFAULT_TARGET_PIXELS = 2048 * 2048
DEFAULT_INPUT_MAX_PIXELS = 2048 * 2048
MIN_INPUT_MAX_PIXELS = 512 * 512

# Hugging Face file sizes (approx) + usage hints for smthem Infographic GGUFs.
GGUF_VARIANT_GUIDE: dict[str, dict[str, Any]] = {
    "SenseNova-U1-8B-MoT-Infographic-Q4_K_S.gguf": {
        "approx_gb": 13.9,
        "role": "Infographic specialist Q4 — lowest VRAM; use if Q6 OOMs.",
    },
    "SenseNova-U1-8B-MoT-Infographic-Q6_K.gguf": {
        "approx_gb": 16.1,
        "role": "Infographic specialist Q6 — default on Olares One 24GB (VRAM_MODE=full).",
    },
    "SenseNova-U1-8B-MoT-Infographic-Q8_0.gguf": {
        "approx_gb": 20.0,
        "role": "Infographic specialist Q8 — highest quality; tight on 24GB (may need balanced).",
    },
}

_GGUF_BASENAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*\.gguf$")

_inference_lock = threading.Lock()
_runtime: SenseNovaRuntime | None = None
_loaded_gguf_abs: str | None = None


def _denorm(x: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(NORM_MEAN, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    std = torch.tensor(NORM_STD, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    return (x * std + mean).clamp(0, 1)


def _to_pil_list(batch: torch.Tensor) -> list[Image.Image]:
    arr = _denorm(batch.float()).permute(0, 2, 3, 1).cpu().numpy()
    arr = (arr * 255.0).round().astype(np.uint8)
    return [Image.fromarray(a) for a in arr]


def _pil_to_b64_png(im: Image.Image) -> str:
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return base64.standard_b64encode(buf.getvalue()).decode("ascii")


def _decode_b64_image(data: str) -> Image.Image:
    s = data.strip()
    if "," in s and s.lower().startswith("data:"):
        s = s.split(",", 1)[1]
    raw = base64.standard_b64decode(s)
    return Image.open(io.BytesIO(raw)).convert("RGB")


def _gguf_weights_enabled() -> bool:
    return os.getenv("USE_GGUF_WEIGHTS", "0").strip().lower() in ("1", "true", "yes", "on")


def _gguf_dir_path() -> str:
    return os.getenv("GGUF_DIR", "/models/gguf")


def _norm_ckpt(p: str | None) -> str | None:
    if not p:
        return None
    return os.path.realpath(p)


def _safe_gguf_basename_path(basename: str) -> str:
    base = basename.strip()
    if not _GGUF_BASENAME_RE.fullmatch(base):
        raise HTTPException(status_code=400, detail=f"invalid gguf_filename: {basename!r}")
    full = os.path.realpath(os.path.join(_gguf_dir_path(), base))
    root = os.path.realpath(_gguf_dir_path())
    if not full.startswith(root + os.sep) and full != root:
        raise HTTPException(status_code=400, detail="gguf path escapes GGUF_DIR")
    if not os.path.isfile(full):
        raise HTTPException(status_code=400, detail=f"GGUF not on disk: {base} (under {_gguf_dir_path()})")
    return full


def scan_gguf_files() -> list[dict[str, Any]]:
    d = _gguf_dir_path()
    out: list[dict[str, Any]] = []
    if not os.path.isdir(d):
        return out
    for name in sorted(os.listdir(d)):
        if not name.endswith(".gguf"):
            continue
        fp = os.path.join(d, name)
        if not os.path.isfile(fp):
            continue
        try:
            sz = os.path.getsize(fp)
        except OSError:
            continue
        if sz < 1_000_000_000:
            continue
        row: dict[str, Any] = {"name": name, "path": fp, "bytes": sz}
        if name in GGUF_VARIANT_GUIDE:
            row["guide"] = GGUF_VARIANT_GUIDE[name]
        out.append(row)
    return out


def _default_env_gguf_abs() -> str | None:
    ck = (os.getenv("GGUF_CHECKPOINT") or "").strip()
    if ck:
        return os.path.realpath(ck)
    gf = (os.getenv("GGUF_FILE") or "").strip()
    if gf and _gguf_weights_enabled():
        return _safe_gguf_basename_path(gf)
    return None


def _desired_gguf_abs(requested_basename: str | None) -> str | None:
    if not _gguf_weights_enabled():
        return None
    if requested_basename and requested_basename.strip():
        return _safe_gguf_basename_path(requested_basename.strip())
    return _default_env_gguf_abs()


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


def _auto_input_max_pixels(num_images: int) -> int:
    full_res_image_budget = 2
    if num_images <= full_res_image_budget:
        return DEFAULT_INPUT_MAX_PIXELS
    total_budget = full_res_image_budget * DEFAULT_INPUT_MAX_PIXELS
    return max(MIN_INPUT_MAX_PIXELS, total_budget // max(1, num_images))


def _resize_to_max_budget(img: Image.Image, input_max_pixels: int) -> Image.Image:
    resized_h, resized_w = smart_resize(
        height=img.height,
        width=img.width,
        factor=_IMAGE_GRID_FACTOR,
        min_pixels=input_max_pixels,
        max_pixels=input_max_pixels,
    )
    if (resized_w, resized_h) == img.size:
        return img
    return img.resize((resized_w, resized_h), Image.LANCZOS)


def _resolve_edit_output_size(
    input_images: list[Image.Image],
    explicit: tuple[int, int] | None,
    target_pixels: int,
) -> tuple[int, int]:
    if explicit is not None:
        width, height = explicit
        if width % _IMAGE_GRID_FACTOR or height % _IMAGE_GRID_FACTOR:
            raise HTTPException(
                status_code=400,
                detail=f"width/height must be multiples of {_IMAGE_GRID_FACTOR}",
            )
        return width, height
    w, h = input_images[0].size
    resized_h, resized_w = smart_resize(
        height=h,
        width=w,
        factor=_IMAGE_GRID_FACTOR,
        min_pixels=target_pixels,
        max_pixels=target_pixels,
    )
    return resized_w, resized_h


class SenseNovaRuntime:
    """One load_model_and_tokenizer; all modalities share weights."""

    def __init__(
        self,
        *,
        model_path: str,
        device: str,
        dtype: torch.dtype,
        gguf_checkpoint: str | None,
        vram_mode: str,
        device_map: str | None,
        max_memory: str | None,
    ) -> None:
        self.device = device
        self.prefetch_count = vram_mode_to_prefetch_count(vram_mode)
        self.model, self.tokenizer = load_model_and_tokenizer(
            model_path,
            dtype=dtype,
            device=device,
            gguf_checkpoint=gguf_checkpoint,
            for_offload=self.prefetch_count > 0,
            device_map=device_map,
            max_memory=max_memory,
        )
        self.vqa_device = str(infer_input_device(self.model, fallback=device)) if device_map else device
        self._last_think_text: str = ""

    @property
    def last_think_text(self) -> str:
        return self._last_think_text

    def _offload(self):
        return make_offload_ctx(self.model, self.prefetch_count, self.device)

    @torch.inference_mode()
    def run_t2i(
        self,
        *,
        prompt: str,
        width: int,
        height: int,
        cfg_scale: float,
        cfg_norm: str,
        timestep_shift: float,
        num_steps: int,
        batch_size: int,
        seed: int,
        think_mode: bool,
    ) -> tuple[list[Image.Image], str]:
        with _inference_timer("t2i", sync_cuda=True):
            with self._offload() as offloaded:
                out = offloaded.t2i_generate(
                    self.tokenizer,
                    prompt,
                    image_size=(width, height),
                    cfg_scale=cfg_scale,
                    cfg_norm=cfg_norm,
                    timestep_shift=timestep_shift,
                    cfg_interval=(0.0, 1.0),
                    num_steps=num_steps,
                    batch_size=batch_size,
                    seed=seed,
                    think_mode=think_mode,
                )
        if think_mode:
            tensor, think_text = out
            self._last_think_text = think_text
        else:
            tensor = out
            self._last_think_text = ""
        return _to_pil_list(tensor), self._last_think_text

    @torch.inference_mode()
    def run_vqa(
        self,
        *,
        image: Image.Image,
        question: str,
        max_new_tokens: int,
        do_sample: bool,
        temperature: float,
        top_p: float,
        top_k: int | None,
        repetition_penalty: float | None,
    ) -> str:
        pixel_values, grid_hw = load_image_native(image)
        pixel_values = pixel_values.to(self.vqa_device, dtype=self.model.dtype)
        grid_hw = grid_hw.to(self.vqa_device)
        generation_config: dict = dict(max_new_tokens=max_new_tokens, do_sample=do_sample)
        if do_sample:
            generation_config["temperature"] = temperature
            generation_config["top_p"] = top_p
            if top_k is not None:
                generation_config["top_k"] = top_k
        if repetition_penalty is not None:
            generation_config["repetition_penalty"] = repetition_penalty
        with _inference_timer("vqa", sync_cuda=True):
            with self._offload() as offloaded:
                response, _hist = offloaded.chat(
                    self.tokenizer,
                    pixel_values,
                    question,
                    generation_config,
                    history=None,
                    return_history=True,
                    grid_hw=grid_hw,
                )
        return response

    @torch.inference_mode()
    def run_editing(
        self,
        *,
        prompt: str,
        images: list[Image.Image],
        image_size: tuple[int, int],
        cfg_scale: float,
        img_cfg_scale: float,
        cfg_norm: str,
        timestep_shift: float,
        num_steps: int,
        batch_size: int,
        think_mode: bool,
        seed: int,
    ) -> tuple[list[Image.Image], str]:
        with _inference_timer("editing", sync_cuda=True):
            with self._offload() as offloaded:
                output = offloaded.it2i_generate(
                    self.tokenizer,
                    prompt,
                    list(images),
                    image_size=image_size,
                    cfg_scale=cfg_scale,
                    img_cfg_scale=img_cfg_scale,
                    cfg_norm=cfg_norm,
                    timestep_shift=timestep_shift,
                    cfg_interval=(0.0, 1.0),
                    num_steps=num_steps,
                    batch_size=batch_size,
                    think_mode=think_mode,
                    seed=seed,
                )
        if think_mode:
            return _to_pil_list(output[0]), output[1]
        return _to_pil_list(output), ""

    @torch.inference_mode()
    def run_interleave(
        self,
        *,
        prompt: str,
        input_images: list[Image.Image],
        width: int,
        height: int,
        cfg_scale: float,
        img_cfg_scale: float,
        timestep_shift: float,
        num_steps: int,
        think_mode: bool,
        seed: int,
        system_message: str,
    ) -> tuple[str, list[Image.Image]]:
        verbose = _env_truthy("INTERLEAVE_VERBOSE")
        with _inference_timer("interleave", sync_cuda=True):
            with self._offload() as offloaded:
                with _tee_interleave_prints():
                    text, image_tensors = offloaded.interleave_gen(
                        self.tokenizer,
                        prompt,
                        images=list(input_images),
                        image_size=(width, height),
                        cfg_scale=cfg_scale,
                        img_cfg_scale=img_cfg_scale,
                        timestep_shift=timestep_shift,
                        cfg_interval=(0.0, 1.0),
                        num_steps=num_steps,
                        system_message=system_message,
                        think_mode=think_mode,
                        seed=seed,
                        verbose=verbose,
                    )
        return text, [_to_pil_list(t)[0] for t in image_tensors]


def _load_runtime_inner(gguf_checkpoint: str | None) -> None:
    global _runtime, _loaded_gguf_abs
    model_path = os.environ["MODEL_PATH"]
    device = os.getenv("DEVICE", "cuda")
    dtype = _dtype_from_env()
    vram_mode = os.getenv("VRAM_MODE", DEFAULT_VRAM_MODE)
    if vram_mode not in VRAM_MODE_OPTIONS:
        raise ValueError(f"VRAM_MODE must be one of {VRAM_MODE_OPTIONS}")
    device_map = (os.getenv("DEVICE_MAP") or "").strip() or None
    max_memory = (os.getenv("MAX_MEMORY") or "").strip() or None
    sensenova_u1.set_attn_backend(os.getenv("ATTENTION_BACKEND", "auto"))
    LOG.info(
        "loading model_path=%s gguf=%s vram_mode=%s attn=%s",
        model_path,
        gguf_checkpoint or "(safetensors)",
        vram_mode,
        sensenova_u1.effective_attn_backend(),
    )
    _runtime = SenseNovaRuntime(
        model_path=model_path,
        device=device,
        dtype=dtype,
        gguf_checkpoint=gguf_checkpoint,
        vram_mode=vram_mode,
        device_map=device_map,
        max_memory=max_memory,
    )
    _loaded_gguf_abs = _norm_ckpt(gguf_checkpoint)
    LOG.info("model ready loaded_gguf=%s", _loaded_gguf_abs or "(safetensors)")


def _load_runtime() -> None:
    """Initial load from env (startup)."""
    gguf = _desired_gguf_abs(None)
    _load_runtime_inner(gguf)


def _ensure_gguf_checkpoint(requested_basename: str | None) -> None:
    """Reload weights if request asks for a different GGUF than currently loaded."""
    global _runtime, _loaded_gguf_abs
    if not _gguf_weights_enabled():
        if requested_basename and requested_basename.strip():
            raise HTTPException(status_code=400, detail="USE_GGUF_WEIGHTS is off; omit gguf_filename")
        return
    target = _desired_gguf_abs(requested_basename)
    if target is None:
        raise HTTPException(status_code=400, detail="no GGUF checkpoint resolved (set GGUF_CHECKPOINT / GGUF_FILE)")
    cur = _norm_ckpt(_loaded_gguf_abs)
    tgt = _norm_ckpt(target)
    if _runtime is not None and cur == tgt:
        return
    with _inference_lock:
        cur2 = _norm_ckpt(_loaded_gguf_abs)
        tgt2 = _norm_ckpt(target)
        if _runtime is not None and cur2 == tgt2:
            return
        LOG.warning("reloading model for GGUF switch -> %s", tgt2)
        _runtime = None
        _loaded_gguf_abs = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        _load_runtime_inner(target)


def _require_runtime() -> SenseNovaRuntime:
    if _runtime is None:
        LOG.info("model not loaded — loading now")
        _load_runtime()
    if _runtime is None:
        raise HTTPException(status_code=503, detail="model not loaded")
    return _runtime


def _locked(callable_fn):
    with _inference_lock:
        return callable_fn()


# --- Pydantic ---


class T2IRequest(BaseModel):
    prompt: str
    aspect_ratio: str = "1:1"
    cfg_scale: float = 4.0
    cfg_norm: Literal["none", "global", "channel"] = "none"
    timestep_shift: float = 3.0
    num_steps: int = 50
    batch_size: int = 1
    seed: int = 0
    think_mode: bool = False
    gguf_filename: str | None = Field(default=None, description="Basename under GGUF_DIR; reloads weights if different")


class VQARequest(BaseModel):
    question: str
    image_base64: str
    max_new_tokens: int = 1024
    do_sample: bool = False
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int | None = None
    repetition_penalty: float | None = None
    gguf_filename: str | None = Field(default=None, description="Basename under GGUF_DIR; reloads weights if different")


class EditingRequest(BaseModel):
    prompt: str
    images_base64: list[str] = Field(min_length=1)
    width: int | None = None
    height: int | None = None
    target_pixels: int = DEFAULT_TARGET_PIXELS
    do_resize: bool = True
    input_max_pixels: int | Literal["auto"] | None = "auto"
    cfg_scale: float = 4.0
    img_cfg_scale: float = 1.0
    cfg_norm: Literal["none", "global", "channel"] = "none"
    timestep_shift: float = 3.0
    num_steps: int = 50
    batch_size: int = 1
    think_mode: bool = False
    seed: int = 0
    gguf_filename: str | None = Field(default=None, description="Basename under GGUF_DIR; reloads weights if different")


class InterleaveRequest(BaseModel):
    prompt: str
    images_base64: list[str] = Field(default_factory=list)
    aspect_ratio: str = "16:9"
    cfg_scale: float = 4.0
    img_cfg_scale: float = 1.0
    timestep_shift: float = 3.0
    num_steps: int = 50
    think_mode: bool = True
    seed: int = 0
    system_message: str | None = None
    gguf_filename: str | None = Field(default=None, description="Basename under GGUF_DIR; reloads weights if different")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.basicConfig(
        level=getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, _load_runtime)
    yield


app = FastAPI(title="SenseNova-U1 Infographic-V2", lifespan=lifespan)
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
    # Stay healthy even when user unloaded weights (avoid CrashLoop).
    return {"status": "ok", "model_loaded": _runtime is not None}


@app.get("/api/v1/capabilities")
def capabilities():
    use_g = _gguf_weights_enabled()
    files = scan_gguf_files()
    loaded = None
    if _loaded_gguf_abs:
        loaded = os.path.basename(_loaded_gguf_abs)
    return {
        "modalities": ["t2i", "vqa", "editing", "interleave"],
        "weight_loading": "gguf" if use_g else "safetensors",
        "gguf_enabled": use_g,
        "gguf_dir": _gguf_dir_path(),
        "gguf_files_on_disk": files,
        "loaded_gguf_basename": loaded,
        "gguf_variant_guide": GGUF_VARIANT_GUIDE,
        "endpoints": {
            "t2i": "POST /api/v1/t2i",
            "vqa": "POST /api/v1/vqa",
            "editing": "POST /api/v1/editing",
            "interleave": "POST /api/v1/interleave",
            "vram": "GET /api/v1/vram",
            "clear_vram": "POST /api/v1/clear_vram",
            "unload": "POST /api/v1/unload",
        },
        "vram_modes": list(VRAM_MODE_OPTIONS),
        "vram_report": _vram_report(),
        "park_after_generate_default": PARK_AFTER_GENERATE,
        "unload_after_generate_default": UNLOAD_AFTER_GENERATE,
        "t2i_aspect_ratios": list(T2I_RESOLUTIONS.keys()),
        "interleave_aspect_ratios": list(INTERLEAVE_RESOLUTIONS.keys()),
        "attention_backend_options": ["auto", "flash", "sdpa"],
        "attention_backend_notes": (
            "NeoChat MoT only supports auto|flash|sdpa. SageAttention3 is DiT-oriented and lacks NEO-Unify hybrid "
            "multimodal masks — do not set ATTENTION_BACKEND to sage. "
            "OSS path uses flash_attn.flash_attn_func when ATTENTION_BACKEND=flash or auto+importable flash_attn "
            "(FlashAttention-2 style package; upstream pyproject flash extra pins flash-attn<3). "
            "There is no separate FA3 env: FA3 in SenseNova docs refers to their LightLLM / forked stack, not this single-process server. "
            "Without flash_attn installed, auto falls back to torch.nn.functional.scaled_dot_product_attention. "
            "Helm: set INSTALL_FLASH_ATTN=1 and preferably FLASH_ATTN_PIP_SPEC to a matching prebuilt .whl (see https://flashattn.dev)."
        ),
        "diagnostics_env": {
            "INTERLEAVE_VERBOSE": "1 → upstream streams decoded text every ~16 tokens to stdout; wraps each image diffusion loop in tqdm (stderr). Good qualitative signal: long gaps between chunks = autoregressive decode slow; tqdm crawling = diffusion slow.",
            "INTERLEAVE_LOG_PRINTS": "1 → with INTERLEAVE_VERBOSE, also mirror those print() lines to app logger (pod logs via uvicorn). tqdm may still only appear on stderr.",
            "INFERENCE_TIMING": "1 → INFO log one line per request with wall seconds and CUDA sync for t2i, vqa, editing, interleave (total only, not text vs image split).",
            "LOG_LEVEL": "DEBUG may add noise from dependencies; prefer targeted flags above.",
            "INSTALL_FLASH_ATTN": "1 → container startup pip-installs flash_attn before uvicorn. Prefer FLASH_ATTN_PIP_SPEC (prebuilt .whl): source build compiles dozens of CUDA translation units; parallel nvcc/link jobs each use multi‑GB RAM so high MAX_JOBS can OOM the pod or get OOMKilled.",
            "FLASH_ATTN_PIP_SPEC": "Optional single pip argument, e.g. https://.../flash_attn-2.8.3+cu128torch2xx...whl",
            "FLASH_ATTN_MAX_JOBS": "Parallel compile cap (default 2 in chart); raise only if node has plenty of free RAM during build.",
            "ATTENTION_BACKEND": "auto|flash|sdpa — use flash only after flash_attn installs successfully.",
        },
        "notes": [
            "Optional body field gguf_filename (basename) switches Infographic GGUF under GGUF_DIR; triggers full reload when different.",
            "t2i uses T2I resolution buckets; interleave uses interleave buckets when no input image.",
            "With input images, interleave output size follows first image (smart_resize), matching upstream examples.",
            "Default Infographic-Q6_K; switch to Infographic-Q4_K_S if VRAM tight. Q8 is tight on 24GB.",
        ],
    }


def _maybe_park_or_unload(*, park: bool | None = None, unload: bool | None = None) -> str:
    """Post-inference VRAM policy. Defaults from env; Gradio can override."""
    do_unload = UNLOAD_AFTER_GENERATE if unload is None else unload
    do_park = PARK_AFTER_GENERATE if park is None else park
    if do_unload:
        return _unload_runtime()
    if do_park:
        return _park_runtime_cpu()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return _vram_report()


@app.get("/api/v1/vram")
def api_vram():
    return {"report": _vram_report(), "model_loaded": _runtime is not None}


@app.post("/api/v1/clear_vram")
def api_clear_vram():
    with _inference_lock:
        return {"ok": True, "detail": _park_runtime_cpu()}


@app.post("/api/v1/unload")
def api_unload():
    with _inference_lock:
        return {"ok": True, "detail": _unload_runtime()}


@app.post("/api/v1/t2i")
def api_t2i(body: T2IRequest):
    _ensure_gguf_checkpoint(body.gguf_filename)
    if body.aspect_ratio not in T2I_RESOLUTIONS:
        raise HTTPException(status_code=400, detail=f"aspect_ratio must be one of {list(T2I_RESOLUTIONS)}")
    w, h = T2I_RESOLUTIONS[body.aspect_ratio]
    rt = _ensure_runtime_on_device()
    t0 = time.perf_counter()

    def go():
        return rt.run_t2i(
            prompt=body.prompt,
            width=w,
            height=h,
            cfg_scale=body.cfg_scale,
            cfg_norm=body.cfg_norm,
            timestep_shift=body.timestep_shift,
            num_steps=body.num_steps,
            batch_size=body.batch_size,
            seed=body.seed,
            think_mode=body.think_mode,
        )

    images, think_text = _locked(go)
    wall = time.perf_counter() - t0
    infer = _LAST_TIMING.get("t2i", wall)
    mem = _maybe_park_or_unload()
    return {
        "images_png_base64": [_pil_to_b64_png(im) for im in images],
        "think_text": think_text,
        "width": w,
        "height": h,
        "timing": {
            "infer_s": round(infer, 3),
            "wall_s": round(wall, 3),
            "vram": mem,
        },
    }


@app.post("/api/v1/vqa")
def api_vqa(body: VQARequest):
    _ensure_gguf_checkpoint(body.gguf_filename)
    rt = _ensure_runtime_on_device()
    image = _decode_b64_image(body.image_base64)

    def go():
        return rt.run_vqa(
            image=image,
            question=body.question,
            max_new_tokens=body.max_new_tokens,
            do_sample=body.do_sample,
            temperature=body.temperature,
            top_p=body.top_p,
            top_k=body.top_k,
            repetition_penalty=body.repetition_penalty,
        )

    answer = _locked(go)
    mem = _maybe_park_or_unload()
    return {
        "answer": answer,
        "timing": {"infer_s": round(_LAST_TIMING.get("vqa", 0.0), 3), "vram": mem},
    }


@app.post("/api/v1/editing")
def api_editing(body: EditingRequest):
    _ensure_gguf_checkpoint(body.gguf_filename)
    rt = _ensure_runtime_on_device()
    pil_images = [_decode_b64_image(x) for x in body.images_base64]
    n = len(pil_images)
    imp_raw = body.input_max_pixels
    if imp_raw == "auto" or imp_raw is None:
        input_max = _auto_input_max_pixels(n)
    else:
        input_max = int(imp_raw)
    processed: list[Image.Image] = []
    for im in pil_images:
        im = im.convert("RGB")
        if body.do_resize:
            im = _resize_to_max_budget(im, input_max)
        processed.append(im)
    explicit = None
    if body.width is not None and body.height is not None:
        explicit = (body.width, body.height)
    ow, oh = _resolve_edit_output_size(processed, explicit, body.target_pixels)

    def go():
        return rt.run_editing(
            prompt=body.prompt,
            images=processed,
            image_size=(ow, oh),
            cfg_scale=body.cfg_scale,
            img_cfg_scale=body.img_cfg_scale,
            cfg_norm=body.cfg_norm,
            timestep_shift=body.timestep_shift,
            num_steps=body.num_steps,
            batch_size=body.batch_size,
            think_mode=body.think_mode,
            seed=body.seed,
        )

    images, think_side = _locked(go)
    mem = _maybe_park_or_unload()
    return {
        "images_png_base64": [_pil_to_b64_png(im) for im in images],
        "think_text": think_side,
        "width": ow,
        "height": oh,
        "timing": {"infer_s": round(_LAST_TIMING.get("editing", 0.0), 3), "vram": mem},
    }


@app.post("/api/v1/interleave")
def api_interleave(body: InterleaveRequest):
    _ensure_gguf_checkpoint(body.gguf_filename)
    rt = _ensure_runtime_on_device()
    input_images = [_decode_b64_image(x) for x in body.images_base64]
    if body.aspect_ratio not in INTERLEAVE_RESOLUTIONS:
        raise HTTPException(
            status_code=400,
            detail=f"aspect_ratio must be one of {list(INTERLEAVE_RESOLUTIONS)}",
        )
    fb_w, fb_h = INTERLEAVE_RESOLUTIONS[body.aspect_ratio]
    if input_images:
        w0, h0 = input_images[0].size
        rh, rw = smart_resize(h0, w0)
        w, h = rw, rh
    else:
        w, h = fb_w, fb_h
    sys_msg = body.system_message or DEFAULT_SYSTEM_MESSAGE

    def go():
        return rt.run_interleave(
            prompt=body.prompt,
            input_images=input_images,
            width=w,
            height=h,
            cfg_scale=body.cfg_scale,
            img_cfg_scale=body.img_cfg_scale,
            timestep_shift=body.timestep_shift,
            num_steps=body.num_steps,
            think_mode=body.think_mode,
            seed=body.seed,
            system_message=sys_msg,
        )

    text, images = _locked(go)
    mem = _maybe_park_or_unload()
    return {
        "text": text,
        "images_png_base64": [_pil_to_b64_png(im) for im in images],
        "width": w,
        "height": h,
        "timing": {"infer_s": round(_LAST_TIMING.get("interleave", 0.0), 3), "vram": mem},
    }


# --- Gradio (calls same runtime under lock) ---


def _format_timing_box(out: dict, modality: str) -> str:
    t = out.get("timing") or {}
    infer = t.get("infer_s")
    wall = t.get("wall_s")
    parts = [f"modality={modality}"]
    if infer is not None:
        parts.append(f"infer={infer:.3f}s")
    if wall is not None:
        parts.append(f"wall={wall:.3f}s")
    if modality in _LAST_TIMING:
        parts.append(f"cuda_sync={_LAST_TIMING[modality]:.3f}s")
    vram = t.get("vram") or _vram_report()
    return "  ".join(parts) + f"\n\n{vram}"


def _ui_t2i(gguf_sel, prompt, aspect_ratio, cfg_scale, cfg_norm, timestep_shift, num_steps, seed, think_mode):
    try:
        req = T2IRequest(
            prompt=prompt or "",
            aspect_ratio=aspect_ratio or "1:1",
            cfg_scale=float(cfg_scale),
            cfg_norm=cfg_norm,
            timestep_shift=float(timestep_shift),
            num_steps=int(num_steps),
            seed=int(seed),
            think_mode=bool(think_mode),
            gguf_filename=gguf_sel if gguf_sel else None,
        )
        out = api_t2i(req)
        imgs = [Image.open(io.BytesIO(base64.standard_b64decode(x))) for x in out["images_png_base64"]]
        return imgs, out.get("think_text") or "", _format_timing_box(out, "t2i")
    except HTTPException as he:
        raise gr.Error(str(he.detail)) from he
    except Exception as exc:
        LOG.exception("gradio t2i")
        raise gr.Error(str(exc)) from exc


def _ui_vqa(gguf_sel, img, question, do_sample, temperature, top_p, top_k, max_new_tokens, repetition_penalty):
    try:
        if img is None:
            raise gr.Error("image required")
        buf = io.BytesIO()
        Image.fromarray(img).save(buf, format="PNG")
        b64 = base64.standard_b64encode(buf.getvalue()).decode("ascii")
        req = VQARequest(
            question=question or "",
            image_base64=b64,
            max_new_tokens=int(max_new_tokens),
            do_sample=bool(do_sample),
            temperature=float(temperature),
            top_p=float(top_p),
            top_k=int(top_k) if top_k and int(top_k) > 0 else None,
            repetition_penalty=float(repetition_penalty) if repetition_penalty and float(repetition_penalty) > 0 else None,
            gguf_filename=gguf_sel if gguf_sel else None,
        )
        out = api_vqa(req)
        timing = f"modality=vqa  infer={_LAST_TIMING.get('vqa', 0):.3f}s\n\n{_vram_report()}"
        return out["answer"], timing
    except HTTPException as he:
        raise gr.Error(str(he.detail)) from he
    except Exception as exc:
        LOG.exception("gradio vqa")
        raise gr.Error(str(exc)) from exc


def _ui_edit(gguf_sel, prompt, *imgs):
    try:
        pil_list = []
        for im in imgs:
            if im is not None:
                pil_list.append(Image.fromarray(im))
        if not pil_list:
            raise gr.Error("at least one image required")
        b64s = []
        for p in pil_list:
            b64s.append(_pil_to_b64_png(p))
        req = EditingRequest(prompt=prompt or "", images_base64=b64s, gguf_filename=gguf_sel if gguf_sel else None)
        out = api_editing(req)
        first = out["images_png_base64"][0]
        return (
            Image.open(io.BytesIO(base64.standard_b64decode(first))),
            out.get("think_text") or "",
            _format_timing_box(out, "editing"),
        )
    except HTTPException as he:
        raise gr.Error(str(he.detail)) from he
    except Exception as exc:
        LOG.exception("gradio edit")
        raise gr.Error(str(exc)) from exc


def _ui_interleave(gguf_sel, prompt, img, aspect_ratio, think_mode, seed):
    try:
        imgs_b64: list[str] = []
        if img is not None:
            imgs_b64.append(_pil_to_b64_png(Image.fromarray(img)))
        req = InterleaveRequest(
            prompt=prompt or "",
            images_base64=imgs_b64,
            aspect_ratio=aspect_ratio or "16:9",
            think_mode=bool(think_mode),
            seed=int(seed),
            gguf_filename=gguf_sel if gguf_sel else None,
        )
        out = api_interleave(req)
        gal = [Image.open(io.BytesIO(base64.standard_b64decode(x))) for x in out["images_png_base64"]]
        return out["text"], gal, _format_timing_box(out, "interleave")
    except HTTPException as he:
        raise gr.Error(str(he.detail)) from he
    except Exception as exc:
        LOG.exception("gradio interleave")
        raise gr.Error(str(exc)) from exc


def build_gradio() -> gr.Blocks:
    ar_t2i = list(T2I_RESOLUTIONS.keys())
    ar_il = list(INTERLEAVE_RESOLUTIONS.keys())
    cfg_norm_choices: list = ["none", "global", "channel"]
    use_g = _gguf_weights_enabled()
    disk_names = [x["name"] for x in scan_gguf_files()]
    gf_default_file = (os.getenv("GGUF_FILE") or "").strip()

    with gr.Blocks(title="SenseNova Infographic V2") as demo:
        gr.Markdown(
            "# SenseNova-U1 Infographic V2\n"
            "Specialist weights for charts / posters / structured layouts. "
            "Use **API** under `/api/v1/*` for agents; this UI shares the same loaded model.\n\n"
            "Park-on-CPU after generate frees VRAM (default ON). Unload drops weights entirely."
        )
        if use_g and disk_names:
            gf_init = gf_default_file if gf_default_file in disk_names else disk_names[0]
            gf_comp = gr.Dropdown(
                disk_names,
                value=gf_init,
                label="gguf_filename",
                info="Switching file reloads weights (slow). See GET /api/v1/capabilities for hints.",
            )
        elif use_g:
            gf_comp = gr.State(None)
            gr.Markdown("*GGUF mode on but no large `.gguf` files found under GGUF_DIR — check init download.*")
        else:
            gf_comp = gr.State(None)
        with gr.Tab("T2I"):
            with gr.Row():
                p = gr.Textbox(label="prompt", lines=4)
                aspect = gr.Dropdown(ar_t2i, value="1:1", label="aspect_ratio")
            cfg = gr.Slider(0.1, 10.0, value=4.0, label="cfg_scale")
            cn = gr.Dropdown(cfg_norm_choices, value="none", label="cfg_norm")
            ts = gr.Slider(-1.0, 10.0, value=3.0, label="timestep_shift")
            steps = gr.Slider(1, 100, value=50, step=1, label="num_steps")
            sd = gr.Number(value=0, label="seed")
            th = gr.Checkbox(value=False, label="think_mode")
            go = gr.Button("Generate")
            gal = gr.Gallery(label="images", columns=2)
            tt = gr.Textbox(label="think_text", lines=6)
            timing_t2i = gr.Textbox(label="timing / VRAM", lines=6, interactive=False)
            go.click(_ui_t2i, [gf_comp, p, aspect, cfg, cn, ts, steps, sd, th], [gal, tt, timing_t2i])
        with gr.Tab("VQA"):
            im = gr.Image(label="image")
            q = gr.Textbox(label="question", lines=3)
            ds = gr.Checkbox(value=False, label="do_sample")
            temp = gr.Slider(0.0, 1.0, value=0.7, label="temperature")
            tp = gr.Slider(0.0, 1.0, value=0.9, label="top_p")
            tk = gr.Number(value=0, label="top_k (0=off)")
            mnt = gr.Slider(256, 8192, value=1024, step=256, label="max_new_tokens")
            rp = gr.Number(value=0.0, label="repetition_penalty (0=off)")
            go2 = gr.Button("Ask")
            ans = gr.Textbox(label="answer", lines=12)
            timing_vqa = gr.Textbox(label="timing / VRAM", lines=6, interactive=False)
            go2.click(_ui_vqa, [gf_comp, im, q, ds, temp, tp, tk, mnt, rp], [ans, timing_vqa])
        with gr.Tab("Editing"):
            pe = gr.Textbox(label="prompt", lines=3)
            i1 = gr.Image(label="image 1")
            i2 = gr.Image(label="image 2 (optional)")
            i3 = gr.Image(label="image 3 (optional)")
            go3 = gr.Button("Edit")
            out_im = gr.Image(label="output")
            te = gr.Textbox(label="think_text", lines=4)
            timing_edit = gr.Textbox(label="timing / VRAM", lines=6, interactive=False)
            go3.click(_ui_edit, [gf_comp, pe, i1, i2, i3], [out_im, te, timing_edit])
        with gr.Tab("Interleave"):
            pi = gr.Textbox(label="prompt", lines=5)
            ii = gr.Image(label="optional input image")
            ai = gr.Dropdown(ar_il, value="16:9", label="aspect_ratio (text-only fallback)")
            tm = gr.Checkbox(value=True, label="think_mode")
            si = gr.Number(value=0, label="seed")
            go4 = gr.Button("Run")
            tx = gr.Textbox(label="text", lines=10)
            g2 = gr.Gallery(label="images", columns=2)
            timing_il = gr.Textbox(label="timing / VRAM", lines=6, interactive=False)
            go4.click(_ui_interleave, [gf_comp, pi, ii, ai, tm, si], [tx, g2, timing_il])
        with gr.Tab("Memory / VRAM"):
            gr.Markdown(
                "Park = weights on CPU (fast reload next run). Unload = drop entirely (slow disk reload).\n\n"
                f"Env defaults: `SENSENOVA_PARK_AFTER_GENERATE`={'ON' if PARK_AFTER_GENERATE else 'OFF'}, "
                f"`SENSENOVA_UNLOAD_AFTER_GENERATE`={'ON' if UNLOAD_AFTER_GENERATE else 'OFF'}."
            )
            mem_status = gr.Textbox(label="status", lines=10, value=_vram_report(), interactive=False)
            with gr.Row():
                btn_refresh = gr.Button("Refresh")
                btn_park = gr.Button("Force clear VRAM (park CPU)")
                btn_unload = gr.Button("Unload model", variant="stop")

            def _ui_refresh():
                return _vram_report()

            def _ui_park():
                with _inference_lock:
                    return _park_runtime_cpu()

            def _ui_unload():
                with _inference_lock:
                    return _unload_runtime()

            btn_refresh.click(_ui_refresh, outputs=[mem_status])
            btn_park.click(_ui_park, outputs=[mem_status])
            btn_unload.click(_ui_unload, outputs=[mem_status])
    return demo


def _gradio_allowed_paths() -> list[str]:
    """Serve generated files from Gradio temp dir and legacy /tmp/gradio."""
    order = (_GRADIO_TEMP_DIR, "/tmp/gradio")
    return list(dict.fromkeys(os.path.realpath(p) for p in order))


app = gr.mount_gradio_app(
    app,
    build_gradio(),
    path=_GRADIO_MOUNT_PATH,
    root_path=_GRADIO_ROOT_PATH,
    allowed_paths=_gradio_allowed_paths(),
)


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("SERVER_PORT", "7860"))
    uvicorn.run(app, host="0.0.0.0", port=port)
