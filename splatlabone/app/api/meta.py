"""REST: presets + guide + capabilities."""

from __future__ import annotations

import shutil
from pathlib import Path

from fastapi import APIRouter, HTTPException

from deps import PRESETS_DIR, manager
from pipeline.models.cache import MODEL_REGISTRY, is_cached
from pipeline.registry import guide_section, list_presets, load_preset

router = APIRouter(prefix="/api/v1", tags=["meta"])


@router.get("/capabilities")
async def capabilities() -> dict:
    trainer = Path("/opt/gsplat/examples/simple_trainer.py")
    viewer = Path("/opt/gsplat/examples/simple_viewer.py")
    backends = {
        "colmap": shutil.which("colmap") is not None,
        "glomap": shutil.which("glomap") is not None,
        "ffmpeg": shutil.which("ffmpeg") is not None,
        "gsplat": trainer.is_file(),
        "viewer": "gsplat_viser" if viewer.is_file() else None,
        "vggt_omega": Path("/opt/vggt-omega").is_dir(),
        "da3": Path("/opt/da3").is_dir(),
        "da3_cpp": Path("/opt/depth-anything-cpp").is_dir()
        or shutil.which("da3-cli") is not None
        or shutil.which("da3") is not None,
        "lingbot_map": Path("/opt/lingbot-map").is_dir(),
        "instant_splat": Path("/opt/instantsplat").is_dir(),
        "gluemap": Path("/opt/gluemap").is_dir(),
        "fastmap": Path("/opt/fastmap").is_dir(),
        "hloc": Path("/opt/hloc").is_dir(),
        "dense_sfm": Path("/opt/dense-sfm/dense_sfm").is_dir()
        or Path("/opt/dense-sfm/run_matching.py").is_file(),
        "dense_sfm_refine": Path("/opt/dense-sfm-refine").is_dir(),
    }
    model_hubs = {k: v.get("hub", "huggingface") for k, v in MODEL_REGISTRY.items()}
    models_cached = {k: is_cached(k) for k in MODEL_REGISTRY}
    return {
        "backends": backends,
        "viewer": backends.get("viewer"),
        "model_hubs": model_hubs,
        "models_cached": models_cached,
        "gpu": manager.gpu_info(),
        "artifixer": {"available": False, "reason": "deferred — requires 48GB+ VRAM"},
    }


@router.get("/presets")
async def presets() -> dict:
    return {"presets": list_presets(PRESETS_DIR)}


@router.get("/presets/{name}")
async def preset_detail(name: str) -> dict:
    try:
        return load_preset(name, PRESETS_DIR)
    except KeyError:
        raise HTTPException(404, f"Unknown preset: {name}") from None


@router.get("/guide/{anchor}")
async def guide(anchor: str) -> dict:
    return {"anchor": anchor, "content": guide_section(anchor)}
