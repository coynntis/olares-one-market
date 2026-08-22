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


@router.get("/{job_id}/ply_compressed")
async def download_ply_compressed(job_id: str) -> FileResponse:
    job = manager.get_job(job_id)
    if not job:
        raise HTTPException(404, "Scene not found")
    path_str = job.artifacts.get("ply_compressed")
    if not path_str:
        out = manager.output_path(job_id) / "splat" / "scene.ply_compressed"
        if out.is_file():
            path_str = str(out)
        else:
            raise HTTPException(404, "No ply_compressed artifact yet")
    path = Path(path_str)
    if not path.is_file():
        raise HTTPException(404, "File missing")
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


@router.get("/{job_id}/ckpt")
async def download_ckpt(job_id: str) -> FileResponse:
    job = manager.get_job(job_id)
    if not job:
        raise HTTPException(404, "Scene not found")
    ckpt = job.artifacts.get("ckpt") or job.artifacts.get("checkpoint")
    if not ckpt:
        ckpt_dir = manager.output_path(job_id) / "train" / "ckpts"
        if ckpt_dir.is_dir():
            pts = sorted(ckpt_dir.glob("*.pt"), key=lambda p: p.stat().st_mtime)
            if pts:
                ckpt = str(pts[-1])
    if not ckpt:
        raise HTTPException(404, "No checkpoint yet")
    path = Path(ckpt)
    if not path.is_file():
        raise HTTPException(404, "Checkpoint missing")
    return FileResponse(path, media_type="application/octet-stream", filename=path.name)


@router.get("/{job_id}/artifacts")
async def list_artifacts(job_id: str) -> dict:
    from pipeline.stages import list_stage_artifacts

    job = manager.get_job(job_id)
    if not job:
        raise HTTPException(404, "Scene not found")
    out = manager.output_path(job_id)
    return list_stage_artifacts(job, out)


@router.get("/{job_id}/artifact/{key}")
async def download_artifact(job_id: str, key: str) -> FileResponse:
    from pipeline.stages import resolve_artifact

    job = manager.get_job(job_id)
    if not job:
        raise HTTPException(404, "Scene not found")
    out = manager.output_path(job_id)
    path = resolve_artifact(job, out, key)
    if not path or not path.is_file():
        raise HTTPException(404, f"Artifact not found: {key}")
    media = "application/octet-stream"
    if path.suffix.lower() == ".json":
        media = "application/json"
    elif path.suffix.lower() == ".ply":
        media = "application/octet-stream"
    elif path.suffix.lower() == ".glb":
        media = "model/gltf-binary"
    return FileResponse(path, media_type=media, filename=path.name)


@router.get("/{job_id}/infer_gs")
async def download_infer_gs(job_id: str) -> FileResponse:
    """Shortcut for DA3 feed-forward Gaussian PLY."""
    return await download_artifact(job_id, "infer_gs_ply")
