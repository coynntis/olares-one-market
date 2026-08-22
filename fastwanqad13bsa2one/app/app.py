import contextlib
import gc
import logging
import os
import tempfile
import threading
import time

# FastVideo reads attention backend at import time.
os.environ.setdefault("FASTVIDEO_ATTENTION_BACKEND", "SAGE_ATTN")
os.environ.setdefault("FASTVIDEO_DISABLE_ATTENTION_COMPILE", "1")
os.environ.setdefault("FASTWAN_INPROCESS_EXECUTOR", "1")
os.environ.setdefault(
    "FASTVIDEO_WORKER_MULTIPROC_METHOD",
    os.getenv("FASTVIDEO_WORKER_MULTIPROC_METHOD", "spawn"),
)

try:
    import fastwan_worker_patch

    if not fastwan_worker_patch.apply(source="app.py import"):
        logging.getLogger("fastwanqad13bsa2one").error(
            "FASTWAN worker patch apply() failed at import — see fastwan.worker_patch logs"
        )
    else:
        fastwan_worker_patch.log_status("app.py import")
except ImportError:
    logging.getLogger("fastwanqad13bsa2one").error(
        "FASTWAN worker patch module missing at import — upgrade chart ConfigMap"
    )

import gradio as gr
import imageio
import torch
from fastvideo import VideoGenerator
from fastvideo.configs.pipelines.base import PipelineConfig
from fastvideo.layers.quantization.nvfp4_qat_config import NVFP4QATConfig

MODEL_ID = os.getenv("FASTWAN_MODEL", "FastVideo/FastWan-QAD-1.3B-SA2")
TAEHV_CHECKPOINT = os.getenv("TAEHV_CHECKPOINT", "/models/taehv/taew2_1.pth")
TAEHV_DEVICE = os.getenv("FASTWAN_TAEHV_DEVICE", "cpu")
INFER_STEPS = int(os.getenv("FASTWAN_INFER_STEPS", "3"))
GUIDANCE = float(os.getenv("FASTWAN_GUIDANCE", "1.0"))
ENABLE_COMPILE = os.getenv("FASTWAN_ENABLE_COMPILE", "0").lower() in ("1", "true", "yes")
WARMUPS = int(os.getenv("FASTWAN_WARMUPS", "5" if ENABLE_COMPILE else "0"))
PRELOAD = os.getenv("FASTWAN_PRELOAD", "0").lower() in ("1", "true", "yes")
SKIP_BACKEND_VERIFY = os.getenv("FASTWAN_SKIP_BACKEND_VERIFY", "1").lower() in (
    "1",
    "true",
    "yes",
)
TEXT_ENCODER_CPU_OFFLOAD = os.getenv("FASTWAN_TEXT_ENCODER_CPU_OFFLOAD", "1").lower() in (
    "1",
    "true",
    "yes",
)

_GENERATOR = None
_TAEHV = None
_LOCK = threading.Lock()
_LOG = logging.getLogger("fastwanqad13bsa2one")


def _verify_sa2_attention_backend() -> None:
    """SA2 model needs SageAttention2++ (SAGE_ATTN), not SageAttention3."""
    try:
        from sageattention import sageattn  # noqa: F401
    except ImportError as exc:
        raise gr.Error(
            "sageattention (SageAttention2++ / SAGE_ATTN) is not installed. "
            "FastWan-QAD-1.3B-SA2 requires it — check /workspace/fastwan-sageattention-build.log"
        ) from exc

    if SKIP_BACKEND_VERIFY:
        _LOG.info("attention backend: SAGE_ATTN (install-time verify skipped)")
        return

    from fastvideo.platforms import AttentionBackendEnum, current_platform

    backend_cls = current_platform.get_attn_backend_cls(
        AttentionBackendEnum.SAGE_ATTN,
        128,
        torch.bfloat16,
    )
    if "sage_attn.SageAttentionBackend" not in backend_cls:
        raise gr.Error(
            f"SAGE_ATTN backend unavailable (got {backend_cls}). "
            "Do not switch to SAGE_ATTN_THREE — that is for non-SA2 FastWan-QAD-1.3B only."
        )
    _LOG.info("attention backend OK: SAGE_ATTN (SageAttention2++) -> %s", backend_cls)


class TaehvDecoder:
    def __init__(self, checkpoint_path: str, device: str | None = None):
        from taehv import TAEHV

        self.device = device or TAEHV_DEVICE
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.model = TAEHV(checkpoint_path=checkpoint_path).to(self.device, self.dtype).eval()

    @torch.no_grad()
    def decode(self, latents: torch.Tensor):
        latents = latents.permute(0, 2, 1, 3, 4).to(self.device, self.dtype)
        decoded = self.model.decode_video(latents, parallel=False, show_progress_bar=False)
        frames = (decoded[0].clamp(0, 1) * 255).to(torch.uint8)
        return frames.permute(0, 2, 3, 1).cpu().numpy()


def _log_gpu_memory(tag: str) -> None:
    if not torch.cuda.is_available():
        return
    free, total = torch.cuda.mem_get_info()
    _LOG.info(
        "%s VRAM: %.2f GiB free / %.2f GiB total",
        tag,
        free / (1024**3),
        total / (1024**3),
    )


def _release_parent_cuda_cache() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    _log_gpu_memory("parent cuda release")


def _ensure_worker_patch() -> None:
    try:
        import fastwan_worker_patch
    except ImportError:
        _LOG.error("FASTWAN worker patch MISSING — ConfigMap stale; upgrade chart")
        return
    if not fastwan_worker_patch.apply(source="app._ensure_worker_patch"):
        _LOG.error("FASTWAN worker patch apply() FAILED before VideoGenerator load")
        return
    fastwan_worker_patch.log_status("app._ensure_worker_patch")


def _build_generator() -> VideoGenerator:
    _ensure_worker_patch()
    pipeline_config = PipelineConfig.from_pretrained(MODEL_ID)
    pipeline_config.dit_precision = "bf16"
    pipeline_config.vae_precision = "bf16"
    pipeline_config.text_encoder_precisions = ("bf16",)
    pipeline_config.dit_config.quant_config = NVFP4QATConfig()

    _release_parent_cuda_cache()
    return VideoGenerator.from_pretrained(
        MODEL_ID,
        pipeline_config=pipeline_config,
        num_gpus=1,
        use_fsdp_inference=False,
        dit_cpu_offload=False,
        dit_layerwise_offload=False,
        vae_cpu_offload=True,
        text_encoder_cpu_offload=TEXT_ENCODER_CPU_OFFLOAD,
        pin_cpu_memory=False,
        enable_torch_compile=ENABLE_COMPILE,
        enable_torch_compile_text_encoder=ENABLE_COMPILE,
        enable_torch_compile_vae=False,
        output_type="latent",
    )


def _ensure_taehv() -> TaehvDecoder:
    global _TAEHV
    if _TAEHV is None:
        _LOG.info("loading TAEHV on %s (after denoise)", TAEHV_DEVICE)
        _TAEHV = TaehvDecoder(TAEHV_CHECKPOINT)
    return _TAEHV


def _unload_taehv() -> None:
    global _TAEHV
    if _TAEHV is None:
        return
    del _TAEHV.model
    _TAEHV = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def _reset_generator() -> None:
    global _GENERATOR
    if _GENERATOR is not None:
        try:
            _GENERATOR.shutdown()
        except Exception:
            _LOG.exception("generator shutdown failed")
    _GENERATOR = None
    _unload_taehv()


@contextlib.contextmanager
def _silence_request_log():
    vg_logger = logging.getLogger("fastvideo.entrypoints.video_generator")
    prev_level = vg_logger.level
    vg_logger.setLevel(logging.WARNING)
    try:
        yield
    finally:
        vg_logger.setLevel(prev_level)


def _generation_request(prompt: str, *, steps: int) -> dict:
    return {
        "prompt": prompt,
        "sampling": {"num_inference_steps": steps, "guidance_scale": GUIDANCE},
        # FastVideo gates result.samples on return_frames even when output_type=latent.
        "output": {"save_video": False, "return_frames": True},
    }


def _run_warmups(gen: VideoGenerator) -> None:
    import fastwan_worker_patch

    with _silence_request_log():
        for i in range(WARMUPS):
            _LOG.info("warmup %s/%s", i + 1, WARMUPS)
            _release_parent_cuda_cache()
            try:
                warm = gen.generate(request=_generation_request("warmup", steps=2))
                _ensure_taehv().decode(fastwan_worker_patch.extract_latents_cpu(warm))
                fastwan_worker_patch.shutdown_worker_after_generate(gen)
            except Exception:
                _LOG.exception("warmup %s/%s failed", i + 1, WARMUPS)
                raise
            finally:
                _unload_taehv()


def _ensure_ready() -> VideoGenerator:
    global _GENERATOR
    if _GENERATOR is not None:
        return _GENERATOR

    if not torch.cuda.is_available():
        raise gr.Error("CUDA GPU required.")

    cap = torch.cuda.get_device_capability()
    if cap[0] < 10:
        raise gr.Error(
            f"FastWan NVFP4 needs Blackwell (sm100+, capability 10.0+). Got {cap[0]}.{cap[1]}."
        )

    if not os.path.isfile(TAEHV_CHECKPOINT):
        raise gr.Error(f"Missing TAEHV weights at {TAEHV_CHECKPOINT}")

    _verify_sa2_attention_backend()
    _LOG.info(
        "loading FastWan SA2 model=%s compile=%s text_encoder_cpu_offload=%s taehv_device=%s",
        MODEL_ID,
        ENABLE_COMPILE,
        TEXT_ENCODER_CPU_OFFLOAD,
        TAEHV_DEVICE,
    )
    _GENERATOR = _build_generator()
    if WARMUPS > 0:
        _run_warmups(_GENERATOR)
        _LOG.info("warmup complete (%s runs)", WARMUPS)
    return _GENERATOR


def _generate_once(gen: VideoGenerator, prompt: str):
    import fastwan_worker_patch

    _unload_taehv()
    _release_parent_cuda_cache()
    _log_gpu_memory("parent before denoise")
    result = gen.generate(request=_generation_request(prompt.strip(), steps=INFER_STEPS))
    denoise = result.generation_time
    latents = fastwan_worker_patch.extract_latents_cpu(result)
    fastwan_worker_patch.shutdown_worker_after_generate(gen)
    _LOG.info("denoise %.2fs, TAEHV decode on %s shape=%s", denoise, TAEHV_DEVICE, tuple(latents.shape))
    frames = _ensure_taehv().decode(latents)
    return frames, denoise


def generate_video(prompt: str, progress=gr.Progress()):
    if not prompt.strip():
        raise gr.Error("Prompt required.")

    with _LOCK:
        progress(0.05, desc="Preparing model...")
        gen = _ensure_ready()

        progress(0.2, desc=f"Denoising ({INFER_STEPS} steps)...")
        t0 = time.perf_counter()
        try:
            frames, denoise = _generate_once(gen, prompt)
        except RuntimeError as exc:
            if "Forward execution thread failed" not in str(exc):
                raise
            _LOG.exception("generation failed, resetting generator and retrying once")
            _reset_generator()
            gen = _ensure_ready()
            frames, denoise = _generate_once(gen, prompt)
        elapsed = time.perf_counter() - t0

    out = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    imageio.mimsave(out.name, frames, fps=16, format="mp4")
    stats = f"total {elapsed:.2f}s | denoise {denoise:.2f}s | 81 frames @ 480p"
    return out.name, stats


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    if PRELOAD:
        _LOG.info("preloading FastWan on main thread before Gradio")
        try:
            _ensure_ready()
        except Exception:
            _LOG.exception("preload failed — see traceback above")
            raise
    else:
        _LOG.info("lazy load — model loads on first Generate")

    with gr.Blocks(title="FastWan QAD 1.3B SA2 One") as demo:
        gr.Markdown(
            "# FastWan QAD 1.3B SA2 One\n"
            "3-step QAD text-to-video — ~2s for 5s 480p on RTX 5090M (NVFP4 + SageAttention2++)."
        )
        prompt = gr.Textbox(
            label="Prompt",
            lines=4,
            value=(
                "A curious raccoon peers through a vibrant field of yellow sunflowers, "
                "its eyes wide with interest. Soft natural light, warm cheerful tones."
            ),
        )
        btn = gr.Button("Generate video", variant="primary")
        video = gr.Video(label="Output")
        stats = gr.Textbox(label="Timing", interactive=False)
        btn.click(generate_video, inputs=[prompt], outputs=[video, stats])

    demo.queue(default_concurrency_limit=1).launch(server_name="0.0.0.0", server_port=7860, show_error=True)


if __name__ == "__main__":
    main()
