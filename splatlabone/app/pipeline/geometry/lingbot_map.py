"""LingBot-Map streaming geometry backend."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Callable

import numpy as np

from pipeline.colmap_io import unproject_depth_points, write_colmap_model, write_points_ply
from pipeline.models.cache import model_dir, resolve
from pipeline.models.download import download_model


def _list_images(images: Path) -> list[Path]:
    exts = {".jpg", ".jpeg", ".png", ".webp"}
    return sorted(p for p in images.iterdir() if p.suffix.lower() in exts)


def _find_lingbot_ckpt() -> Path:
    variant = os.environ.get("LINGBOT_VARIANT", "lingbot-map-long")
    resolved = resolve("lingbot")
    if resolved is not None:
        if resolved.suffix == ".pt":
            return resolved
        for hit in sorted(resolved.rglob("*.pt")):
            if variant.replace("-", "_") in hit.name or variant in hit.name:
                return hit
        pts = sorted(resolved.rglob("*.pt"))
        if pts:
            return pts[0]
    base = model_dir("lingbot")
    for hit in sorted(base.rglob("*.pt")):
        return hit
    raise FileNotFoundError(f"LingBot checkpoint not found (variant={variant})")


def run_lingbot_map(
    workspace: Path,
    images: Path,
    log: Callable[[str], None],
    *,
    chunk_size: int = 16,
) -> tuple[Path, dict[str, str]]:
    log(f"LingBot-Map streaming geometry (chunk_size={chunk_size})")
    out = workspace / "geometry" / "lingbot_map"
    out.mkdir(parents=True, exist_ok=True)
    frames = _list_images(images)
    if not frames:
        raise RuntimeError("no frames for LingBot-Map")

    if resolve("lingbot") is None:
        log("downloading LingBot-Map weights (ModelScope)...")
        download_model("lingbot")

    ckpt_path = _find_lingbot_ckpt()
    paths = [str(p) for p in frames]
    image_size = int(os.environ.get("LINGBOT_IMAGE_SIZE", "518"))
    patch_size = int(os.environ.get("LINGBOT_PATCH_SIZE", "14"))
    keyframe_interval = int(os.environ.get("LINGBOT_KEYFRAME_INTERVAL", "6"))
    num_scale_frames = int(os.environ.get("LINGBOT_SCALE_FRAMES", "8"))

    import torch
    from lingbot_map.models.gct_stream import GCTStream
    from lingbot_map.utils.geometry import closed_form_inverse_se3_general
    from lingbot_map.utils.load_fn import load_and_preprocess_images
    from lingbot_map.utils.pose_enc import pose_encoding_to_extri_intri

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Loading LingBot-Map checkpoint: {ckpt_path.name}")
    model = GCTStream(
        img_size=image_size,
        patch_size=patch_size,
        enable_3d_rope=True,
        max_frame_num=max(1024, len(frames) + num_scale_frames),
        kv_cache_sliding_window=512,
        kv_cache_scale_frames=num_scale_frames,
        kv_cache_cross_frame_special=True,
        kv_cache_include_scale_frames=True,
        use_sdpa=True,
        camera_num_iterations=4,
    ).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = ckpt.get("model", ckpt)
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    log(f"Preprocessing {len(paths)} frames...")
    images_t = load_and_preprocess_images(
        paths, mode="crop", image_size=image_size, patch_size=patch_size
    ).to(device)

    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    log("LingBot-Map streaming inference...")
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=dtype, enabled=device.type == "cuda"):
        predictions = model.inference_streaming(
            images_t,
            num_scale_frames=min(num_scale_frames, max(1, len(frames) - 1)),
            keyframe_interval=keyframe_interval,
            output_device=torch.device("cpu"),
        )

    # postprocess: w2c -> c2w
    extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], images_t.shape[-2:])
    extrinsic_4x4 = torch.zeros((*extrinsic.shape[:-2], 4, 4), device=extrinsic.device, dtype=extrinsic.dtype)
    extrinsic_4x4[..., :3, :4] = extrinsic
    extrinsic_4x4[..., 3, 3] = 1.0
    extrinsic_4x4 = closed_form_inverse_se3_general(extrinsic_4x4)
    extrinsic_c2w = extrinsic_4x4[..., :3, :4].cpu().numpy()
    intrinsic_np = intrinsic.cpu().numpy()

    n = min(len(frames), extrinsic_c2w.shape[0])
    names = [frames[i].name for i in range(n)]
    ext = extrinsic_c2w[:n]
    k = intrinsic_np[:n] if intrinsic_np.ndim == 3 else np.repeat(intrinsic_np[None, ...], n, axis=0)

    all_pts: list[np.ndarray] = []
    all_rgb: list[np.ndarray] = []
    depth_key = "depth" if "depth" in predictions else "depth_map"
    if depth_key in predictions:
        depth = predictions[depth_key]
        if isinstance(depth, torch.Tensor):
            depth = depth.cpu().numpy()
        for i in range(n):
            if i >= depth.shape[0]:
                break
            conf = None
            if "depth_conf" in predictions:
                conf = predictions["depth_conf"][i].cpu().numpy()
            pts, rgb = unproject_depth_points(depth[i], k[i], ext[i], conf=conf)
            if pts.size:
                all_pts.append(pts)
                all_rgb.append(rgb)

    points = np.concatenate(all_pts, axis=0) if all_pts else None
    colors = np.concatenate(all_rgb, axis=0) if all_pts else None
    if points is not None and points.shape[0] > 200_000:
        sel = np.linspace(0, points.shape[0] - 1, 200_000, dtype=int)
        points = points[sel]
        colors = colors[sel] if colors is not None else None

    sparse = workspace / "sparse" / "0"
    write_colmap_model(
        sparse,
        image_names=names,
        extrinsics_c2w=ext,
        intrinsics=k,
        points3d=points,
        points_rgb=colors,
        shared_camera=False,
    )

    (out / "stream.json").write_text(
        json.dumps(
            {
                "backend": "lingbot_map",
                "checkpoint": str(ckpt_path),
                "frames": n,
                "points3d": int(points.shape[0]) if points is not None else 0,
                "status": "ok",
            },
            indent=2,
        )
    )
    log(f"LingBot-Map done: {n} frames -> COLMAP sparse/0")
    artifacts: dict[str, str] = {
        "geometry_meta": str(out / "stream.json"),
    }
    if points is not None and points.size:
        preview_ply = out / "sparse_preview.ply"
        write_points_ply(preview_ply, points, colors)
        artifacts["geometry_sparse_ply"] = str(preview_ply)
    return out, artifacts
