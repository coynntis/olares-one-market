"""FastWan: GPU text-encode hop when TE weights live on CPU (FastVideo v1-style)."""

from __future__ import annotations

import gc
import logging
import os

_LOG = logging.getLogger("fastwan.te_gpu_encode_patch")
_PATCHED = False


def _truthy(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).lower() in ("1", "true", "yes")


def _log_vram(tag: str) -> None:
    import torch

    if not torch.cuda.is_available():
        return
    free, total = torch.cuda.mem_get_info()
    _LOG.info("%s VRAM: %.2f GiB free / %.2f GiB total", tag, free / (1024**3), total / (1024**3))


def apply() -> bool:
    """Monkey-patch TextEncodingStage: cuda encode, then TE back to CPU."""
    global _PATCHED
    if _PATCHED:
        return True
    if not _truthy("FASTWAN_TE_GPU_ENCODE", "0"):
        return False

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
                "FASTWAN_TE_GPU_ENCODE: moving %d text encoder(s) to %s for encode",
                len(self.text_encoders),
                device,
            )
            for te in self.text_encoders:
                te.to(device)
            _log_vram("before TE GPU encode")

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
                _log_vram("after TE offloaded to CPU")

    TextEncodingStage.forward = patched_forward
    _PATCHED = True
    _LOG.info(
        "FASTWAN_TE_GPU_ENCODE patch active (text_encoder_cpu_offload=%s)",
        _truthy("FASTWAN_TEXT_ENCODER_CPU_OFFLOAD", "1"),
    )
    return True
