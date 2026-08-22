"""VGGT-Omega geometry backend."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import numpy as np

from pipeline.colmap_io import unproject_depth_points, write_colmap_model, write_points_ply
from pipeline.models.cache import resolve
from pipeline.models.download import download_model


def _list_images(images: Path) -> list[Path]:
    exts = {".jpg", ".jpeg", ".png", ".webp"}
    return sorted(p for p in images.iterdir() if p.suffix.lower() in exts)


def run_vggt_omega(
    workspace: Path,
    images: Path,
    log: Callable[[str], None],
    *,
    chunk_size: int = 8,
) -> tuple[Path, dict[str, str]]:
    log("VGGT-Omega: estimating poses + depth")
    out = workspace / "geometry" / "vggt_omega"
    out.mkdir(parents=True, exist_ok=True)
    img_list = _list_images(images)
    if not img_list:
        raise RuntimeError("no images for VGGT-Omega")

    ckpt = resolve("vggt_omega")
    if ckpt is None:
        log("downloading VGGT-Omega weights (ModelScope)...")
        ckpt = download_model("vggt_omega")

    ckpt_path = ckpt if ckpt.suffix == ".pt" else next(ckpt.rglob("vggt_omega_1b_512.pt"), ckpt)

    import torch
    from vggt_omega.models import VGGTOmega
    from vggt_omega.utils.load_fn import load_and_preprocess_images
    from vggt_omega.utils.pose_enc import encoding_to_camera

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VGGTOmega().to(device).eval()
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu", weights_only=True))

    sparse = workspace / "sparse" / "0"
    all_names: list[str] = []
    all_ext: list[np.ndarray] = []
    all_k: list[np.ndarray] = []
    all_pts: list[np.ndarray] = []
    all_rgb: list[np.ndarray] = []

    for start in range(0, len(img_list), chunk_size):
        chunk = img_list[start : start + chunk_size]
        log(f"VGGT-Omega chunk {start // chunk_size + 1}: {len(chunk)} images")
        names = [str(p) for p in chunk]
        imgs = load_and_preprocess_images(names, image_resolution=512).to(device)
        with torch.inference_mode():
            preds = model(imgs)

        pose_enc = preds["pose_enc"]
        hw = preds["images"].shape[-2:]
        extri, intri = encoding_to_camera(pose_enc, hw, build_intrinsics=True)
        extri_np = extri.detach().cpu().numpy()
        intri_np = intri.detach().cpu().numpy()

        # Drop batch dim if present: (B, S, 3, 4) -> (S, 3, 4)
        if extri_np.ndim == 4 and extri_np.shape[0] == 1:
            extri_np = extri_np[0]
            intri_np = intri_np[0]
        if extri_np.ndim == 2:
            extri_np = extri_np[None, ...]
            intri_np = intri_np[None, ...]

        n_frames = extri_np.shape[0]

        depth = preds.get("depth")
        depth_conf = preds.get("depth_conf")

        for j in range(n_frames):
            idx = len(all_names)
            if idx >= len(chunk):
                break
            all_names.append(chunk[j].name)
            all_ext.append(extri_np[j])
            all_k.append(intri_np[j])
            if depth is not None:
                d = depth[j].detach().cpu().numpy()
                conf = depth_conf[j].detach().cpu().numpy() if depth_conf is not None else None
                pts, rgb = unproject_depth_points(d, intri_np[j], extri_np[j], conf=conf)
                if pts.size:
                    all_pts.append(pts)
                    all_rgb.append(rgb)

    points = np.concatenate(all_pts, axis=0) if all_pts else None
    colors = np.concatenate(all_rgb, axis=0) if all_rgb else None
    if points is not None and points.shape[0] > 200_000:
        sel = np.linspace(0, points.shape[0] - 1, 200_000, dtype=int)
        points = points[sel]
        colors = colors[sel] if colors is not None else None

    write_colmap_model(
        sparse,
        image_names=all_names,
        extrinsics_c2w=np.stack(all_ext, axis=0),
        intrinsics=np.stack(all_k, axis=0),
        points3d=points,
        points_rgb=colors,
        shared_camera=False,
    )

    meta = {
        "backend": "vggt_omega",
        "images": len(all_names),
        "checkpoint": str(ckpt_path),
        "points3d": int(points.shape[0]) if points is not None else 0,
        "status": "ok",
    }
    (out / "poses.json").write_text(json.dumps(meta, indent=2))
    log(f"VGGT-Omega done: {len(all_names)} images -> COLMAP sparse/0")
    artifacts: dict[str, str] = {"geometry_meta": str(out / "poses.json")}
    if points is not None and points.size:
        preview_ply = out / "sparse_preview.ply"
        write_points_ply(preview_ply, points, colors)
        artifacts["geometry_sparse_ply"] = str(preview_ply)
    return out, artifacts
