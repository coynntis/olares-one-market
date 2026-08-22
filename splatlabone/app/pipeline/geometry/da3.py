"""Depth Anything 3 geometry backend."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Callable

from pipeline.colmap_io import copy_colmap_sparse, write_colmap_model, write_points_ply
from pipeline.models.cache import resolve
from pipeline.models.download import download_model


def _list_images(images: Path) -> list[Path]:
    exts = {".jpg", ".jpeg", ".png", ".webp"}
    return sorted(p for p in images.iterdir() if p.suffix.lower() in exts)


def _find_gs_ply(root: Path) -> Path | None:
    """Locate DA3 infer_gs PLY under export tree."""
    candidates: list[Path] = []
    for pat in ("**/gs_ply/**/*.ply", "**/*gs*.ply", "**/gaussians*.ply", "**/*.ply"):
        candidates.extend(root.glob(pat))
    if not candidates:
        return None
    # Prefer paths mentioning gs / gaussian; else largest file (usually the splat ply)
    scored = sorted(
        candidates,
        key=lambda p: (
            "gs" in p.as_posix().lower() or "gaussian" in p.name.lower(),
            p.stat().st_size,
        ),
        reverse=True,
    )
    return scored[0]


def run_da3(
    workspace: Path,
    images: Path,
    log: Callable[[str], None],
    *,
    chunk_size: int = 8,
) -> tuple[Path, dict[str, str]]:
    log(f"DA3 infer_gs + COLMAP export (chunk_size={chunk_size})")
    out = workspace / "geometry" / "da3"
    out.mkdir(parents=True, exist_ok=True)
    img_list = _list_images(images)
    if not img_list:
        raise RuntimeError("no images for DA3")

    model_id = os.environ.get("DA3_MODEL", "depth-anything/DA3NESTED-GIANT-LARGE")
    if resolve("da3") is None:
        log(f"downloading DA3 weights ({model_id})...")
        download_model("da3")

    names = [str(p) for p in img_list]
    export_dir = out / "export"
    export_dir.mkdir(parents=True, exist_ok=True)
    sparse = workspace / "sparse" / "0"
    artifacts: dict[str, str] = {}

    import numpy as np
    import torch
    from depth_anything_3.api import DepthAnything3

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DepthAnything3.from_pretrained(model_id).to(device)
    log(f"DA3 inference on {len(names)} frames (colmap + infer_gs export)")

    gs_ply_src: Path | None = None
    for start in range(0, len(names), chunk_size):
        chunk_names = names[start : start + chunk_size]
        log(f"DA3 chunk {start // chunk_size + 1}: {len(chunk_names)} frames")
        chunk_export = export_dir / f"chunk_{start:04d}"
        chunk_export.mkdir(parents=True, exist_ok=True)
        model.inference(
            chunk_names,
            export_dir=str(chunk_export),
            export_format="colmap-gs_ply-glb",
            infer_gs=True,
            align_to_input_ext_scale=True,
        )
        colmap_src = chunk_export / "sparse" / "0"
        if colmap_src.is_dir() and (colmap_src / "images.txt").is_file():
            copy_colmap_sparse(colmap_src, sparse)
            log(f"DA3 wrote COLMAP from export -> {sparse}")
        hit = _find_gs_ply(chunk_export)
        if hit:
            gs_ply_src = hit

    infer_gs_dest = out / "infer_gs.ply"
    if gs_ply_src and gs_ply_src.is_file():
        shutil.copy2(gs_ply_src, infer_gs_dest)
        artifacts["infer_gs_ply"] = str(infer_gs_dest)
        log(f"DA3 infer_gs PLY -> {infer_gs_dest.name} ({infer_gs_dest.stat().st_size // 1024} KB)")

    glb_hits = sorted(export_dir.glob("**/*.glb"), key=lambda p: p.stat().st_size, reverse=True)
    if glb_hits:
        glb_dest = out / "preview.glb"
        shutil.copy2(glb_hits[0], glb_dest)
        artifacts["geometry_glb"] = str(glb_dest)

    if not ((sparse / "images.txt").is_file() or (sparse / "images.bin").is_file()):
        log("DA3 COLMAP export empty — building from prediction tensors (first chunk)")
        chunk_names = names[:chunk_size]
        prediction = model.inference(
            chunk_names,
            export_dir=str(export_dir / "tensor_fallback"),
            export_format="mini_npz",
            infer_gs=True,
            align_to_input_ext_scale=True,
        )
        if not hasattr(prediction, "extrinsics") or prediction.extrinsics is None:
            raise RuntimeError("DA3 failed to produce COLMAP or pose tensors")
        ext = np.asarray(prediction.extrinsics)
        ixt = np.asarray(prediction.intrinsics)
        basenames = [Path(p).name for p in chunk_names]
        c2w_list = []
        for e in ext:
            e4 = np.eye(4)
            e4[:3, :4] = e
            c2w_list.append(np.linalg.inv(e4)[:3, :4])
        k_stack = ixt if ixt.ndim == 3 else np.repeat(ixt[None, ...], len(basenames), axis=0)
        write_colmap_model(
            sparse,
            image_names=basenames,
            extrinsics_c2w=np.stack(c2w_list, axis=0),
            intrinsics=k_stack,
            shared_camera=False,
        )

    meta = {
        "backend": "da3",
        "model": model_id,
        "frames": len(names),
        "license": "check model (Giant=NC)",
        "status": "ok",
        "colmap": str(sparse),
        "infer_gs_ply": artifacts.get("infer_gs_ply"),
        "geometry_glb": artifacts.get("geometry_glb"),
    }
    meta_path = out / "preview.json"
    meta_path.write_text(json.dumps(meta, indent=2))
    artifacts["geometry_meta"] = str(meta_path)

    log("DA3 done")
    return out, artifacts
