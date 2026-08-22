"""REST: multipart ingest."""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from deps import manager, new_id
from pipeline.ingest import ingest_colmap_zip, ingest_images_zip, ingest_video

router = APIRouter(prefix="/api/v1/ingest", tags=["ingest"])


@router.get("/datasets")
async def list_datasets() -> dict:
    return {"datasets": manager.list_datasets()}


@router.post("/images")
async def ingest_images(
    files: list[UploadFile] = File(...),
    name: str = Form(""),
) -> dict:
    dataset_id = new_id("ds_")
    dataset_dir = manager.uploads_dir / dataset_id
    dataset_dir.mkdir(parents=True)
    images = dataset_dir / "images"
    images.mkdir()
    count = 0
    for uf in files:
        if not uf.filename:
            continue
        ext = Path(uf.filename).suffix.lower() or ".jpg"
        dest = images / f"{count:05d}{ext}"
        content = await uf.read()
        dest.write_bytes(content)
        count += 1
    meta = {"name": name or dataset_id, "type": "images", "frame_count": count}
    (dataset_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    return {"dataset_id": dataset_id, "frame_count": count, "path": str(dataset_dir)}


@router.post("/images-zip")
async def ingest_images_zip_upload(archive: UploadFile = File(...), name: str = Form("")) -> dict:
    dataset_id = new_id("ds_")
    dataset_dir = manager.uploads_dir / dataset_id
    dataset_dir.mkdir(parents=True)
    with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
        tmp.write(await archive.read())
        tmp_path = Path(tmp.name)
    try:
        meta = {"name": name or dataset_id}
        ingest_images_zip(tmp_path, dataset_dir, meta)
    finally:
        tmp_path.unlink(missing_ok=True)
    return {"dataset_id": dataset_id, "meta": meta}


@router.post("/video")
async def ingest_video_upload(
    video: UploadFile = File(...),
    name: str = Form(""),
    fps: float = Form(2.0),
    max_frames: int = Form(300),
) -> dict:
    dataset_id = new_id("ds_")
    dataset_dir = manager.uploads_dir / dataset_id
    dataset_dir.mkdir(parents=True)
    ext = Path(video.filename or "video.mp4").suffix
    vpath = dataset_dir / f"source{ext}"
    vpath.write_bytes(await video.read())
    meta = {"name": name or dataset_id}
    ingest_video(vpath, dataset_dir, meta, fps=fps, max_frames=max_frames)
    return {"dataset_id": dataset_id, "meta": meta}


@router.post("/colmap")
async def ingest_colmap(archive: UploadFile = File(...), name: str = Form("")) -> dict:
    dataset_id = new_id("ds_")
    dataset_dir = manager.uploads_dir / dataset_id
    dataset_dir.mkdir(parents=True)
    with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
        tmp.write(await archive.read())
        tmp_path = Path(tmp.name)
    try:
        meta = {"name": name or dataset_id}
        ingest_colmap_zip(tmp_path, dataset_dir, meta)
    finally:
        tmp_path.unlink(missing_ok=True)
    return {"dataset_id": dataset_id, "meta": meta}
