"""Ingest: images, video, COLMAP zip."""

from __future__ import annotations

import json
import shutil
import subprocess
import zipfile
from pathlib import Path
from typing import Any


def _safe_extract(zip_path: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.namelist():
            target = (dest / member).resolve()
            if not str(target).startswith(str(dest.resolve())):
                raise ValueError(f"Zip slip: {member}")
        zf.extractall(dest)


def ingest_images_dir(src: Path, dataset_dir: Path, meta: dict[str, Any]) -> Path:
    images = dataset_dir / "images"
    images.mkdir(parents=True, exist_ok=True)
    exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
    count = 0
    if src.is_dir():
        for f in sorted(src.rglob("*")):
            if f.suffix.lower() in exts and f.is_file():
                shutil.copy2(f, images / f"{count:05d}{f.suffix.lower()}")
                count += 1
    meta.update({"type": "images", "frame_count": count})
    (dataset_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    return dataset_dir


def ingest_images_zip(zip_path: Path, dataset_dir: Path, meta: dict[str, Any]) -> Path:
    tmp = dataset_dir / "_zip"
    if tmp.exists():
        shutil.rmtree(tmp)
    _safe_extract(zip_path, tmp)
    return ingest_images_dir(tmp, dataset_dir, meta)


def ingest_colmap_zip(zip_path: Path, dataset_dir: Path, meta: dict[str, Any]) -> Path:
    tmp = dataset_dir / "_zip"
    if tmp.exists():
        shutil.rmtree(tmp)
    _safe_extract(zip_path, tmp)
    images = dataset_dir / "images"
    sparse = dataset_dir / "sparse" / "0"
    images.mkdir(parents=True, exist_ok=True)
    sparse.mkdir(parents=True, exist_ok=True)
    for sub in ("images", "sparse/0", "sparse"):
        candidate = tmp / sub
        if candidate.is_dir():
            if "images" in sub:
                for f in candidate.iterdir():
                    if f.is_file():
                        shutil.copy2(f, images / f.name)
            elif sub.endswith("sparse/0") or sub == "sparse/0":
                for f in candidate.iterdir():
                    if f.is_file():
                        shutil.copy2(f, sparse / f.name)
            elif sub == "sparse":
                zero = candidate / "0"
                if zero.is_dir():
                    for f in zero.iterdir():
                        if f.is_file():
                            shutil.copy2(f, sparse / f.name)
    # flat layout at zip root
    for f in tmp.iterdir():
        if f.is_file() and f.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            shutil.copy2(f, images / f.name)
    meta.update({"type": "colmap", "frame_count": len(list(images.glob("*")))})
    (dataset_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    shutil.rmtree(tmp, ignore_errors=True)
    return dataset_dir


def ingest_video(
    video_path: Path,
    dataset_dir: Path,
    meta: dict[str, Any],
    fps: float = 2.0,
    max_frames: int = 300,
) -> Path:
    images = dataset_dir / "images"
    images.mkdir(parents=True, exist_ok=True)
    out_pattern = str(images / "frame_%05d.jpg")
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-vf",
        f"fps={fps}",
        "-frames:v",
        str(max_frames),
        out_pattern,
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except FileNotFoundError:
        # dev fallback: copy video as placeholder
        shutil.copy2(video_path, dataset_dir / video_path.name)
        meta["warning"] = "ffmpeg not found; video stored without frame extraction"
    except subprocess.CalledProcessError as exc:
        meta["warning"] = f"ffmpeg failed: {exc.stderr[:200]}"
    count = len(list(images.glob("*.jpg"))) + len(list(images.glob("*.png")))
    meta.update({"type": "video", "frame_count": count, "fps": fps, "max_frames": max_frames})
    (dataset_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    return dataset_dir
