"""REST: presets + guide."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from deps import PRESETS_DIR, manager
from pipeline.registry import guide_section, list_presets, load_preset

router = APIRouter(prefix="/api/v1", tags=["meta"])


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
