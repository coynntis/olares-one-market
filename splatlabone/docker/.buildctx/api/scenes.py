"""REST: scene / splat artifacts."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from deps import manager

router = APIRouter(prefix="/api/v1/scenes", tags=["scenes"])


@router.get("")
async def list_scenes() -> dict:
    return {"scenes": manager.list_scenes()}


@router.get("/{job_id}")
async def scene_meta(job_id: str) -> dict:
    job = manager.get_job(job_id)
    if not job:
        raise HTTPException(404, "Scene not found")
    snap = job.snapshot()
    return snap.model_dump()


@router.get("/{job_id}/splat")
async def download_splat(job_id: str) -> FileResponse:
    job = manager.get_job(job_id)
    if not job:
        raise HTTPException(404, "Scene not found")
    splat = job.artifacts.get("splat")
    if not splat:
        out = manager.output_path(job_id) / "splat" / "scene.splat"
        if out.is_file():
            splat = str(out)
        else:
            raise HTTPException(404, "No splat artifact yet")
    path = Path(splat)
    if not path.is_file():
        raise HTTPException(404, "Splat file missing")
    return FileResponse(path, media_type="application/octet-stream", filename=path.name)


@router.get("/{job_id}/ply")
async def download_ply(job_id: str) -> FileResponse:
    job = manager.get_job(job_id)
    if not job:
        raise HTTPException(404, "Scene not found")
    ply = job.artifacts.get("ply")
    if not ply:
        out = manager.output_path(job_id) / "splat" / "point_cloud.ply"
        if out.is_file():
            ply = str(out)
        else:
            raise HTTPException(404, "No PLY artifact yet")
    path = Path(ply)
    if not path.is_file():
        raise HTTPException(404, "PLY file missing")
    return FileResponse(path, media_type="application/octet-stream", filename=path.name)
