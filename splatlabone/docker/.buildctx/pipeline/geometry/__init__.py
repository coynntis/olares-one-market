"""Geometry front-ends: VGGT-X, DA3, InstantSplat stubs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable


def run_vggt_x(workspace: Path, images: Path, log: Callable[[str], None]) -> Path:
    log("VGGT-X geometry: estimating poses + depth (chunked)")
    out = workspace / "geometry" / "vggt_x"
    out.mkdir(parents=True, exist_ok=True)
    meta = {"backend": "vggt_x", "images": len(list(images.glob("*"))), "status": "stub_or_model"}
    (out / "poses.json").write_text(json.dumps(meta, indent=2))
    sparse = workspace / "sparse" / "0"
    sparse.mkdir(parents=True, exist_ok=True)
    if not (sparse / "cameras.txt").exists():
        (sparse / "cameras.txt").write_text("# vggt_x init\n1 SIMPLE_PINHOLE 1920 1080 1000 960 540\n")
        (sparse / "images.txt").write_text("# vggt_x init\n")
        (sparse / "points3D.txt").write_text("# vggt_x init\n")
    return out


def run_da3(workspace: Path, images: Path, log: Callable[[str], None], chunk_size: int = 8) -> Path:
    log(f"DA3 infer_gs geometry preview (chunk_size={chunk_size})")
    out = workspace / "geometry" / "da3"
    out.mkdir(parents=True, exist_ok=True)
    frames = sorted(images.glob("*"))[:chunk_size * 4]
    (out / "preview.json").write_text(
        json.dumps({"backend": "da3", "frames": len(frames), "license": "check model"}, indent=2)
    )
    return out


def run_instant_splat(workspace: Path, images: Path, log: Callable[[str], None]) -> Path:
    log("InstantSplat / MASt3R sparse init")
    out = workspace / "geometry" / "instant_splat"
    out.mkdir(parents=True, exist_ok=True)
    n = len(list(images.glob("*")))
    (out / "init.json").write_text(json.dumps({"backend": "instant_splat", "images": n}, indent=2))
    return out
