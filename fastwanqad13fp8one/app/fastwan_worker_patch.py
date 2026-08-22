"""Site-packages bootstrap: spawn workers + parent both import via .pth."""

from __future__ import annotations

import gc
import logging
import os
import sys
import types

_LOG = logging.getLogger("fastwan.worker_patch")
PATCH_REV = "8"
_APPLIED = False
_APPLY_PID: int | None = None
_APPLY_SOURCE = "unknown"
_ACTIVE_PATCHES: list[str] = []
_LAST_LATENTS = None


def _mark_patch(name: str) -> None:
    if name not in _ACTIVE_PATCHES:
        _ACTIVE_PATCHES.append(name)


def is_applied() -> bool:
    return _APPLIED


def active_patches() -> tuple[str, ...]:
    return tuple(_ACTIVE_PATCHES)


def log_status(caller: str = "") -> None:
    """Emit a grep-friendly status line (call from app.py after apply)."""
    if not _APPLIED:
        _LOG.error(
            "FASTWAN worker patch NOT applied caller=%s — stale ConfigMap or apply() failed",
            caller or "unknown",
        )
        return
    _LOG.info(
        "FASTWAN worker patch OK rev=%s caller=%s pid=%s module=%s patches=[%s]",
        PATCH_REV,
        caller or "unknown",
        _APPLY_PID,
        __file__,
        ", ".join(_ACTIVE_PATCHES) or "none",
    )


def _log_patch_banner(*, success: bool) -> None:
    status = "APPLIED" if success else "FAILED"
    lines = [
        "=" * 78,
        f"FASTWAN worker patch {status} rev={PATCH_REV} pid={os.getpid()} source={_APPLY_SOURCE}",
        f"  module: {__file__}",
        f"  patches: {', '.join(_ACTIVE_PATCHES) or 'none'}",
        f"  env: INPROCESS={os.getenv('FASTWAN_INPROCESS_EXECUTOR', '1')} "
        f"ATTN={os.getenv('FASTVIDEO_ATTENTION_BACKEND', '')} "
        f"ALLOW_FA4={os.getenv('FASTWAN_ALLOW_FA4', '0')} "
        f"ALLOW_CUDA_IPC={os.getenv('FASTWAN_ALLOW_CUDA_IPC', '0')}",
        "=" * 78,
    ]
    for line in lines:
        if success:
            _LOG.info(line)
        else:
            _LOG.error(line)


def _truthy(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).lower() in ("1", "true", "yes")


def _patch_inprocess_executor() -> None:
    """Run FastVideo in the main process — Olares HAMI breaks MultiprocExecutor CUDA IPC."""
    if not _truthy("FASTWAN_INPROCESS_EXECUTOR", "1"):
        _LOG.info("FASTWAN_INPROCESS_EXECUTOR=0 — using FastVideo MultiprocExecutor")
        _mark_patch("multiproc_executor (default)")
        return

    from fastvideo.fastvideo_args import FastVideoArgs
    from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
    from fastvideo.utils import get_distributed_init_method, get_loopback_ip, get_open_port
    from fastvideo.worker.executor import Executor
    from fastvideo.worker.gpu_worker import Worker
    import fastvideo.envs as envs
    import numpy as np
    import torch

    if getattr(Executor, "_fastwan_inprocess", False):
        _mark_patch("inprocess_executor")
        return
    Executor._fastwan_inprocess = True
    _orig_get_class = Executor.get_class

    class InProcessExecutor(Executor):
        def _init_executor(self) -> None:
            master_port = get_open_port(self.fastvideo_args.master_port)
            dist_init = get_distributed_init_method(get_loopback_ip(), master_port)
            os.environ["LOCAL_RANK"] = "0"
            os.environ["RANK"] = "0"
            os.environ["WORLD_SIZE"] = str(self.fastvideo_args.num_gpus)
            self._worker = Worker(
                fastvideo_args=self.fastvideo_args,
                local_rank=0,
                rank=0,
                distributed_init_method=dist_init,
            )
            self._worker.init_device()
            _LOG.info("InProcessExecutor: single-process GPU (no MultiprocExecutor worker)")

        @staticmethod
        def _ensure_cpu_tensor(value):
            if isinstance(value, torch.Tensor):
                return value.detach().cpu() if value.is_cuda else value.detach().clone()
            if isinstance(value, np.ndarray):
                arr = np.ascontiguousarray(value)
                tensor = torch.from_numpy(arr)
                return tensor.float() if tensor.dtype != torch.float32 else tensor
            if isinstance(value, (list, tuple)):
                return type(value)(InProcessExecutor._ensure_cpu_tensor(v) for v in value)
            return value

        def collective_rpc(self, method, timeout=None, args=(), kwargs=None):
            kwargs = kwargs or {}
            if method == "execute_forward":
                output_batch = self._worker.execute_forward(
                    kwargs["forward_batch"], kwargs["fastvideo_args"]
                )
                logging_info = (
                    output_batch.logging_info if envs.FASTVIDEO_STAGE_LOGGING else None
                )
                extra = dict(output_batch.extra or {})
                out_tensor = self._ensure_cpu_tensor(output_batch.output)
                _stash_latents(out_tensor)
                _LOG.info(
                    "InProcessExecutor output: type=%s shape=%s device=%s",
                    type(out_tensor).__name__,
                    getattr(out_tensor, "shape", None),
                    getattr(out_tensor, "device", None)
                    if isinstance(out_tensor, torch.Tensor)
                    else None,
                )
                return [
                    {
                        "output_batch": out_tensor,
                        "logging_info": logging_info,
                        "extra": extra,
                        "trajectory_latents": self._ensure_cpu_tensor(
                            output_batch.trajectory_latents
                        ),
                        "trajectory_timesteps": output_batch.trajectory_timesteps,
                    }
                ]
            if method == "shutdown":
                return [self._worker.shutdown()]
            if method in ("set_log_queue", "clear_log_queue"):
                return [{"status": "ok"}]
            if isinstance(method, str) and hasattr(self._worker, method):
                fn = getattr(self._worker, method)
                if method.startswith("execute_"):
                    return [fn(*args, **kwargs)]
                return [fn(*args, **kwargs)]
            raise NotImplementedError(f"InProcessExecutor unsupported RPC: {method!r}")

        def execute_forward(self, forward_batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> ForwardBatch:
            responses = self.collective_rpc(
                "execute_forward",
                kwargs={"forward_batch": forward_batch, "fastvideo_args": fastvideo_args},
            )
            output = responses[0]["output_batch"]
            logging_info = responses[0].get("logging_info")
            extra = responses[0].get("extra", {})
            return ForwardBatch(
                data_type=forward_batch.data_type,
                output=output,
                logging_info=logging_info,
                extra=extra,
                trajectory_latents=responses[0].get("trajectory_latents"),
                trajectory_timesteps=responses[0].get("trajectory_timesteps"),
            )

        def set_lora_adapter(
            self, lora_nickname: str, lora_path: str | None = None, strength: float = 1.0, accumulate: bool = False
        ) -> None:
            self.collective_rpc(
                "set_lora_adapter",
                kwargs={
                    "lora_nickname": lora_nickname,
                    "lora_path": lora_path,
                    "strength": strength,
                    "accumulate": accumulate,
                },
            )

        def unmerge_lora_weights(self) -> None:
            self.collective_rpc("unmerge_lora_weights")

        def merge_lora_weights(self) -> None:
            self.collective_rpc("merge_lora_weights")

        def set_log_queue(self, log_queue) -> None:
            self._log_queue = log_queue

        def clear_log_queue(self) -> None:
            self._log_queue = None

        def shutdown(self) -> None:
            try:
                self.collective_rpc("shutdown")
            except Exception:
                _LOG.exception("InProcessExecutor shutdown failed")

    @staticmethod
    def get_class_patched(fastvideo_args: FastVideoArgs) -> type[Executor]:
        if (
            fastvideo_args.distributed_executor_backend == "mp"
            and fastvideo_args.num_gpus == 1
        ):
            return InProcessExecutor
        return _orig_get_class(fastvideo_args)

    Executor.get_class = staticmethod(get_class_patched)
    _mark_patch("inprocess_executor")
    _LOG.info("patched Executor.get_class -> InProcessExecutor for num_gpus=1 (Olares HAMI)")


def _sanitize_worker_ipc_payload(payload):
    """Convert CUDA tensors to CPU torch before multiprocessing pipe (HAMI cannot do CUDA IPC)."""
    import numpy as np
    import torch

    if not isinstance(payload, dict) or "output_batch" not in payload:
        return payload

    def _to_cpu_tensor(value):
        if isinstance(value, torch.Tensor):
            return value.detach().cpu() if value.is_cuda else value.detach().clone()
        if isinstance(value, np.ndarray):
            arr = np.ascontiguousarray(value)
            tensor = torch.from_numpy(arr)
            return tensor.float() if tensor.dtype != torch.float32 else tensor
        if isinstance(value, (list, tuple)):
            return type(value)(_to_cpu_tensor(v) for v in value)
        return value

    out = dict(payload)
    out["output_batch"] = _to_cpu_tensor(out.get("output_batch"))
    _stash_latents(out["output_batch"])
    for key in ("trajectory_latents", "trajectory_timesteps"):
        if key in out and out[key] is not None:
            out[key] = _to_cpu_tensor(out[key])
    out["__fastwan_cpu_tensor__"] = True
    return out


def _patch_multiproc_connection_send() -> None:
    """FastVideo worker uses CUDA IPC for output_batch; Olares HAMI segfaults on parent recv."""
    if _truthy("FASTWAN_INPROCESS_EXECUTOR", "1"):
        return
    if _truthy("FASTWAN_ALLOW_CUDA_IPC", "0"):
        _LOG.info("FASTWAN_ALLOW_CUDA_IPC=1 — not patching multiprocessing pipe send")
        return

    from multiprocessing.connection import Connection

    if getattr(Connection, "_fastwan_ipc_patch", False):
        return
    Connection._fastwan_ipc_patch = True
    orig_send = Connection.send

    def send(self, obj):
        if isinstance(obj, dict) and "output_batch" in obj:
            obj = _sanitize_worker_ipc_payload(obj)
        return orig_send(self, obj)

    Connection.send = send
    _mark_patch("ipc_numpy_pipe")
    _LOG.info("patched Connection.send — worker results use CPU numpy not CUDA IPC (Olares HAMI)")


def _block_flash_attn_cute_fa4() -> None:
    """FA4 cute JIT crashes cross-attn on Olares HAMI (tma_partition _trait=None).

    Force fastvideo/attention/backends/flash_attn.py to use FA2/FA3 when layers
    fall back from SAGE_ATTN / ATTN_QAT_INFER.
    """
    if _truthy("FASTWAN_ALLOW_FA4", "0"):
        _LOG.info("FASTWAN_ALLOW_FA4=1 — not blocking flash_attn_cute")
        return

    stub = types.ModuleType("fastvideo.attention.utils.flash_attn_cute")
    sys.modules["fastvideo.attention.utils.flash_attn_cute"] = stub
    for mod in list(sys.modules):
        if mod == "fastvideo.attention.backends.flash_attn" or mod.startswith(
            "fastvideo.attention.backends.flash_attn."
        ):
            del sys.modules[mod]
    _mark_patch("block_fa4_cute")
    _LOG.info("blocked flash_attn_cute (FA4) — FLASH_ATTN fallback will use FA2/FA3")


def _stash_latents(value) -> None:
    """Keep last denoise output when FastVideo omits samples (return_frames=False)."""
    global _LAST_LATENTS
    import numpy as np
    import torch

    if value is None:
        return
    if isinstance(value, torch.Tensor):
        _LAST_LATENTS = value.detach().cpu() if value.is_cuda else value.detach().clone()
        return
    if isinstance(value, np.ndarray):
        arr = np.ascontiguousarray(value)
        tensor = torch.from_numpy(arr)
        _LAST_LATENTS = tensor.float() if tensor.dtype != torch.float32 else tensor


def _coerce_latent_tensor(raw, source: str):
    import numpy as np
    import torch

    if raw is None:
        return None
    if isinstance(raw, dict):
        raw = raw.get("samples") or raw.get("frames")
        if raw is None:
            return None
    if isinstance(raw, (list, tuple)):
        raw = raw[0]
    if isinstance(raw, np.ndarray):
        _LOG.info("latents from %s numpy shape=%s dtype=%s", source, raw.shape, raw.dtype)
        arr = np.ascontiguousarray(raw)
        tensor = torch.from_numpy(arr)
        return tensor.float() if tensor.dtype != torch.float32 else tensor
    if isinstance(raw, torch.Tensor):
        _LOG.info(
            "latents from %s tensor device=%s shape=%s",
            source,
            raw.device,
            tuple(raw.shape),
        )
        if raw.is_cuda:
            arr = raw.detach().cpu().numpy()
            return torch.from_numpy(np.ascontiguousarray(arr)).float()
        return raw.detach().clone().float()
    return None


def extract_latents_cpu(result) -> "torch.Tensor":
    """Copy worker latents to CPU without parent CUDA (avoids HAMI IPC segfault)."""
    from collections.abc import Mapping

    candidates: list[tuple[str, object]] = []
    if isinstance(result, Mapping):
        for key in ("samples", "frames", "output"):
            if key in result and result[key] is not None:
                candidates.append((f"result[{key}]", result[key]))
    else:
        for attr in ("samples", "frames", "output"):
            raw = getattr(result, attr, None)
            if raw is not None:
                candidates.append((f"result.{attr}", raw))
        extra = getattr(result, "extra", None)
        if isinstance(extra, Mapping):
            for key in ("samples", "frames", "output"):
                if key in extra and extra[key] is not None:
                    candidates.append((f"result.extra.{key}", extra[key]))

    for source, raw in candidates:
        tensor = _coerce_latent_tensor(raw, source)
        if tensor is not None:
            return tensor

    if _LAST_LATENTS is not None:
        _LOG.info("latents from executor stash shape=%s", tuple(_LAST_LATENTS.shape))
        return _LAST_LATENTS.clone().float()

    present = []
    if isinstance(result, Mapping):
        present = [f"{k}={type(result[k]).__name__}" for k in ("samples", "frames", "output") if k in result]
    else:
        for attr in ("samples", "frames", "output"):
            val = getattr(result, attr, None)
            if val is not None:
                present.append(f"{attr}={type(val).__name__}")
    hint = ", ".join(present) if present else f"type={type(result).__name__}"
    raise RuntimeError(
        "generate returned no latents (checked samples, frames, output; "
        f"return_frames must be True for output_type=latent; saw {hint})"
    )


def shutdown_worker_after_generate(gen) -> None:
    """Release GPU worker before parent TAEHV decode (MultiprocExecutor only)."""
    if _truthy("FASTWAN_INPROCESS_EXECUTOR", "1"):
        return
    if not _truthy("FASTWAN_SHUTDOWN_WORKER_AFTER_GEN", "1"):
        return
    try:
        gen.shutdown()
        _LOG.info("GPU worker shutdown after generate")
    except Exception:
        _LOG.exception("post-generate worker shutdown failed")


def _log_vram(tag: str) -> None:
    try:
        import torch

        if not torch.cuda.is_available():
            return
        free, total = torch.cuda.mem_get_info()
        alloc = torch.cuda.memory_allocated() / (1024**3)
        reserved = torch.cuda.memory_reserved() / (1024**3)
        _LOG.info(
            "%s VRAM: %.2f GiB free / %.2f GiB total | torch alloc=%.2f reserved=%.2f",
            tag,
            free / (1024**3),
            total / (1024**3),
            alloc,
            reserved,
        )
    except Exception:
        _LOG.exception("%s VRAM log failed", tag)


def _verify_worker_attn_backend() -> None:
    backend = os.getenv("FASTVIDEO_ATTENTION_BACKEND", "")
    if backend == "ATTN_QAT_INFER":
        try:
            from fastvideo.attention.backends.attn_qat_infer import is_attn_qat_infer_available
            from fastvideo.platforms import AttentionBackendEnum, current_platform

            avail = is_attn_qat_infer_available()
            _LOG.info("worker ATTN_QAT_INFER available=%s", avail)
            if avail:
                cls = current_platform.get_attn_backend_cls(
                    AttentionBackendEnum.ATTN_QAT_INFER, 128, __import__("torch").bfloat16
                )
                _LOG.info("worker ATTN_QAT_INFER backend -> %s", cls)
        except Exception:
            _LOG.exception("worker ATTN_QAT_INFER verification failed")
    elif backend == "SAGE_ATTN":
        try:
            from sageattention import sageattn  # noqa: F401
            from fastvideo.platforms import AttentionBackendEnum, current_platform

            cls = current_platform.get_attn_backend_cls(
                AttentionBackendEnum.SAGE_ATTN, 128, __import__("torch").bfloat16
            )
            _LOG.info("worker SAGE_ATTN backend -> %s", cls)
        except Exception:
            _LOG.exception("worker SAGE_ATTN verification failed")

    try:
        from fastvideo.attention.backends import flash_attn as fa_mod

        _LOG.info("worker FlashAttention fallback version=%s", getattr(fa_mod, "fa_version", "?"))
    except Exception:
        _LOG.warning("worker flash_attn backend not importable yet")


def _patch_te_gpu_encode() -> None:
    if not _truthy("FASTWAN_TE_GPU_ENCODE", "0"):
        return

    import torch
    from fastvideo.distributed import get_local_torch_device
    from fastvideo.pipelines.stages.text_encoding import TextEncodingStage

    orig_forward = TextEncodingStage.forward

    @torch.no_grad()
    def patched_forward(self, batch, fastvideo_args):
        skip_encode = batch.prompt_embeds is not None and len(batch.prompt_embeds) > 0
        use_hop = (
            fastvideo_args.text_encoder_cpu_offload
            and not skip_encode
            and torch.cuda.is_available()
        )
        if use_hop:
            device = get_local_torch_device()
            _LOG.info(
                "FASTWAN_TE_GPU_ENCODE: moving %d text encoder(s) to %s",
                len(self.text_encoders),
                device,
            )
            for te in self.text_encoders:
                te.to(device)

        try:
            return orig_forward(self, batch, fastvideo_args)
        finally:
            if use_hop:
                for te in self.text_encoders:
                    te.to("cpu")
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()

    TextEncodingStage.forward = patched_forward
    _mark_patch("te_gpu_encode")
    _LOG.info("FASTWAN_TE_GPU_ENCODE patch active")


def _patch_denoising_vram_log() -> None:
    from fastvideo.pipelines.stages.denoising import DenoisingStage

    orig_forward = DenoisingStage.forward
    first = {"done": False}

    def patched_forward(self, batch, fastvideo_args):
        if not first["done"]:
            first["done"] = True
            _log_vram("worker before first denoise step")
        try:
            return orig_forward(self, batch, fastvideo_args)
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                _log_vram("worker CUDA OOM during denoise")
            raise

    DenoisingStage.forward = patched_forward
    _mark_patch("denoise_vram_log")
    _LOG.info("DenoisingStage VRAM logging patch active")


def apply(source: str = "unknown") -> bool:
    global _APPLIED, _APPLY_PID, _APPLY_SOURCE
    _APPLY_SOURCE = source
    if _APPLIED:
        _LOG.info(
            "fastwan worker patch already active rev=%s pid=%s source=%s module=%s patches=[%s]",
            PATCH_REV,
            _APPLY_PID,
            _APPLY_SOURCE,
            __file__,
            ", ".join(_ACTIVE_PATCHES) or "none",
        )
        return True
    _ACTIVE_PATCHES.clear()
    try:
        if _truthy("FASTWAN_FORCE_TORCH_SDPA", "0"):
            os.environ["FASTVIDEO_ATTENTION_BACKEND"] = "TORCH_SDPA"
            _mark_patch("force_torch_sdpa")
            _LOG.info("FASTWAN_FORCE_TORCH_SDPA=1 — all attention via TORCH_SDPA")
        _block_flash_attn_cute_fa4()
        _patch_inprocess_executor()
        _patch_multiproc_connection_send()
        _verify_worker_attn_backend()
        _patch_te_gpu_encode()
        _patch_denoising_vram_log()
        _APPLIED = True
        _APPLY_PID = os.getpid()
        _log_patch_banner(success=True)
    except Exception:
        _APPLIED = False
        _APPLY_PID = None
        _LOG.exception(
            "fastwan worker patch apply() FAILED rev=%s source=%s module=%s",
            PATCH_REV,
            source,
            __file__,
        )
        _log_patch_banner(success=False)
        return False
    return True
