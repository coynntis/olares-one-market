import logging
import os
import tempfile
import time
from typing import Any, Dict, Optional

import gradio as gr
import torch
from diffusers import AdaptiveProjectedGuidance, DiffusionPipeline
from diffusers.utils import export_to_video
from PIL import Image

MODEL_ID = "Motif-Technologies/Motif-Video-2B"

# Recommended by the model card for <=24GB GPUs.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

_PIPELINE: Optional[DiffusionPipeline] = None
_OFFLOAD_MODE = True

_LOG = logging.getLogger("motif_video_gradio")


def _env_truthy(name: str, default: str = "1") -> bool:
    val = os.getenv(name, default).strip().lower()
    return val not in {"", "0", "false", "no", "off"}


def _setup_logging() -> None:
    if getattr(_setup_logging, "_done", False):
        return

    level_name = os.getenv("MOTIF_LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)

    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.INFO)

    if _env_truthy("MOTIF_HF_DEBUG", "1"):
        os.environ.setdefault("HF_HUB_VERBOSITY", "debug")
        os.environ.setdefault("TRANSFORMERS_VERBOSITY", "debug")
        os.environ.setdefault("DIFFUSERS_VERBOSITY", "debug")
        os.environ.setdefault("TORCH_LOGS", os.getenv("TORCH_LOGS", "+dynamo"))

        try:
            import huggingface_hub

            huggingface_hub.utils.logging.set_verbosity_debug()
        except Exception:
            _LOG.exception("Failed to enable huggingface_hub debug verbosity")

        try:
            import transformers

            transformers.logging.set_verbosity_debug()
        except Exception:
            _LOG.exception("Failed to enable transformers debug verbosity")

        try:
            import diffusers

            diffusers.utils.logging.set_verbosity_debug()
        except Exception:
            _LOG.exception("Failed to enable diffusers debug verbosity")

    _setup_logging._done = True  # type: ignore[attr-defined]


def _log_cuda_mem(prefix: str) -> None:
    if not torch.cuda.is_available():
        return
    torch.cuda.synchronize()
    alloc = torch.cuda.memory_allocated() / (1024**3)
    reserved = torch.cuda.memory_reserved() / (1024**3)
    _LOG.info("%s cuda_mem allocated=%.3fGiB reserved=%.3fGiB", prefix, alloc, reserved)


def _make_step_logger(total_steps: int):
    last_t = time.perf_counter()

    def _cb(pipe, step: int, timestep, callback_kwargs: Dict[str, Any]):
        nonlocal last_t
        now = time.perf_counter()
        dt = now - last_t
        last_t = now
        _LOG.info(
            "diffusion step %s/%s timestep=%s dt=%.3fs",
            step + 1,
            total_steps,
            timestep,
            dt,
        )
        return callback_kwargs

    return _cb


_setup_logging()


def _gpu_total_memory_gb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.get_device_properties(0).total_memory / (1024**3)


def _build_pipeline(enable_cpu_offload: bool) -> DiffusionPipeline:
    t0 = time.perf_counter()
    _LOG.info(
        "loading pipeline repo=%s offload=%s hf_home=%s hf_endpoint=%s",
        MODEL_ID,
        enable_cpu_offload,
        os.getenv("HF_HOME", ""),
        os.getenv("HF_ENDPOINT", ""),
    )

    guider = AdaptiveProjectedGuidance(
        guidance_scale=8.0,
        adaptive_projected_guidance_rescale=12.0,
        adaptive_projected_guidance_momentum=0.1,
        use_original_formulation=True,
    )

    pipe = DiffusionPipeline.from_pretrained(
        MODEL_ID,
        custom_pipeline="pipeline_motif_video",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        guider=guider,
    )
    _LOG.info("from_pretrained done in %.1fs", time.perf_counter() - t0)
    _log_cuda_mem("after_from_pretrained")

    if not torch.cuda.is_available():
        raise gr.Error("CUDA GPU is required for this model.")

    t1 = time.perf_counter()
    if enable_cpu_offload:
        _LOG.info("enable_model_cpu_offload()")
        pipe.enable_model_cpu_offload()
    else:
        _LOG.info("pipe.to(cuda)")
        pipe = pipe.to("cuda")
    _LOG.info("device placement done in %.1fs", time.perf_counter() - t1)
    _log_cuda_mem("after_device_placement")

    return pipe


def get_pipeline(force_cpu_offload: bool) -> DiffusionPipeline:
    global _PIPELINE
    global _OFFLOAD_MODE
    if _PIPELINE is None or _OFFLOAD_MODE != force_cpu_offload:
        _PIPELINE = _build_pipeline(force_cpu_offload)
        _OFFLOAD_MODE = force_cpu_offload
    return _PIPELINE


def _run_generation(
    prompt: str,
    image: Optional[Image.Image],
    width: int,
    height: int,
    num_frames: int,
    num_inference_steps: int,
    guidance_scale: float,
    apg_rescale: float,
    apg_momentum: float,
    fps: int,
    seed: int,
    force_cpu_offload: bool,
) -> str:
    if not prompt.strip():
        raise gr.Error("Prompt is required.")

    t0 = time.perf_counter()
    pipe = get_pipeline(force_cpu_offload=force_cpu_offload)
    _LOG.info("pipeline ready in %.1fs", time.perf_counter() - t0)

    # Update guidance parameters per request without rebuilding full pipeline.
    if hasattr(pipe, "guider") and pipe.guider is not None:
        pipe.guider.guidance_scale = guidance_scale
        pipe.guider.adaptive_projected_guidance_rescale = apg_rescale
        pipe.guider.adaptive_projected_guidance_momentum = apg_momentum

    kwargs: Dict[str, Any] = dict(
        prompt=prompt,
        width=width,
        height=height,
        num_frames=num_frames,
        num_inference_steps=num_inference_steps,
        generator=torch.Generator(device="cuda").manual_seed(int(seed)),
        callback_on_step_end=_make_step_logger(num_inference_steps),
    )
    if image is not None:
        kwargs["image"] = image.convert("RGB")

    _LOG.info(
        "starting inference w=%s h=%s frames=%s steps=%s offload=%s seed=%s",
        width,
        height,
        num_frames,
        num_inference_steps,
        force_cpu_offload,
        int(seed),
    )
    _log_cuda_mem("before_inference")

    t1 = time.perf_counter()
    output = pipe(**kwargs)
    _LOG.info("inference done in %.1fs", time.perf_counter() - t1)
    _log_cuda_mem("after_inference")

    frames = output.frames[0]

    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp.close()
    export_to_video(frames, tmp.name, fps=fps)
    _LOG.info("wrote video %s", tmp.name)
    return tmp.name


def generate_text_to_video(
    prompt: str,
    width: int,
    height: int,
    num_frames: int,
    num_inference_steps: int,
    guidance_scale: float,
    apg_rescale: float,
    apg_momentum: float,
    fps: int,
    seed: int,
    force_cpu_offload: bool,
) -> str:
    return _run_generation(
        prompt=prompt,
        image=None,
        width=width,
        height=height,
        num_frames=num_frames,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        apg_rescale=apg_rescale,
        apg_momentum=apg_momentum,
        fps=fps,
        seed=seed,
        force_cpu_offload=force_cpu_offload,
    )


def generate_image_to_video(
    prompt: str,
    image: Image.Image,
    width: int,
    height: int,
    num_frames: int,
    num_inference_steps: int,
    guidance_scale: float,
    apg_rescale: float,
    apg_momentum: float,
    fps: int,
    seed: int,
    force_cpu_offload: bool,
) -> str:
    if image is None:
        raise gr.Error("Input image is required for image-to-video.")
    return _run_generation(
        prompt=prompt,
        image=image,
        width=width,
        height=height,
        num_frames=num_frames,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        apg_rescale=apg_rescale,
        apg_momentum=apg_momentum,
        fps=fps,
        seed=seed,
        force_cpu_offload=force_cpu_offload,
    )


def build_ui() -> gr.Blocks:
    gpu_mem = _gpu_total_memory_gb()
    offload_default = gpu_mem <= 24.5 if gpu_mem > 0 else True

    description = (
        "Motif-Video 2B with diffusers (text-to-video and image-to-video).\n"
        "For <=24GB GPUs (e.g. RTX 4090 / 3090), CPU offload is enabled by default.\n"
        f"Detected GPU memory: {gpu_mem:.1f} GB"
    )

    with gr.Blocks(title="Motif Video 2B Gradio") as demo:
        gr.Markdown("# Motif Video 2B Generator")
        gr.Markdown(description)

        with gr.Row():
            with gr.Column():
                force_cpu_offload = gr.Checkbox(
                    label="Use enable_model_cpu_offload()",
                    value=offload_default,
                    info="Recommended for GPUs with 24GB VRAM or less.",
                )
                prompt = gr.Textbox(
                    label="Prompt",
                    lines=4,
                    placeholder="Describe the video you want to generate...",
                )
                with gr.Row():
                    width = gr.Slider(512, 1280, value=1280, step=64, label="Width")
                    height = gr.Slider(512, 736, value=736, step=64, label="Height")
                with gr.Row():
                    frames = gr.Slider(16, 121, value=121, step=1, label="Frames")
                    steps = gr.Slider(
                        10, 80, value=50, step=1, label="Inference Steps"
                    )
                with gr.Row():
                    guidance = gr.Slider(
                        1.0, 12.0, value=8.0, step=0.1, label="Guidance Scale"
                    )
                    fps = gr.Slider(8, 30, value=24, step=1, label="FPS")
                with gr.Row():
                    apg_rescale = gr.Slider(
                        1.0, 20.0, value=12.0, step=0.1, label="APG Rescale"
                    )
                    apg_momentum = gr.Slider(
                        0.0, 1.0, value=0.1, step=0.01, label="APG Momentum"
                    )
                seed = gr.Number(label="Seed", value=42, precision=0)

            with gr.Column():
                output_video = gr.Video(label="Generated Video", format="mp4")

        with gr.Tabs():
            with gr.TabItem("Text to Video"):
                t2v_btn = gr.Button("Generate Text to Video", variant="primary")
                t2v_btn.click(
                    fn=generate_text_to_video,
                    inputs=[
                        prompt,
                        width,
                        height,
                        frames,
                        steps,
                        guidance,
                        apg_rescale,
                        apg_momentum,
                        fps,
                        seed,
                        force_cpu_offload,
                    ],
                    outputs=output_video,
                )

            with gr.TabItem("Image to Video"):
                i2v_image = gr.Image(
                    type="pil", label="Input Image", image_mode="RGB", height=320
                )
                i2v_btn = gr.Button("Generate Image to Video", variant="primary")
                i2v_btn.click(
                    fn=generate_image_to_video,
                    inputs=[
                        prompt,
                        i2v_image,
                        width,
                        height,
                        frames,
                        steps,
                        guidance,
                        apg_rescale,
                        apg_momentum,
                        fps,
                        seed,
                        force_cpu_offload,
                    ],
                    outputs=output_video,
                )

    return demo


if __name__ == "__main__":
    app = build_ui()
    app.queue(default_concurrency_limit=1).launch(server_name="0.0.0.0", server_port=7860)
