"""InstantSplat / MASt3R sparse init backend."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Callable

from pipeline.colmap_io import export_sparse_pointcloud_ply
from pipeline.models.cache import resolve
from pipeline.models.download import download_model


def _list_images(images: Path) -> list[Path]:
    exts = {".jpg", ".jpeg", ".png", ".webp"}
    return sorted(p for p in images.iterdir() if p.suffix.lower() in exts)


def run_instant_splat(
    workspace: Path,
    images: Path,
    log: Callable[[str], None],
) -> tuple[Path, dict[str, str]]:
    log("InstantSplat / MASt3R sparse init")
    out = workspace / "geometry" / "instant_splat"
    out.mkdir(parents=True, exist_ok=True)
    img_list = _list_images(images)
    n = len(img_list)
    if n < 3:
        raise RuntimeError("InstantSplat needs at least 3 images")

    ckpt = resolve("mast3r")
    if ckpt is None:
        log("downloading MASt3R checkpoint...")
        ckpt = download_model("mast3r")

    ckpt_path = ckpt if ckpt.suffix == ".pt" else ckpt / "MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth"
    if not ckpt_path.is_file():
        found = list(ckpt.rglob("MASt3R*.pth")) if ckpt.is_dir() else []
        if not found:
            raise FileNotFoundError(f"MASt3R checkpoint missing at {ckpt_path}")
        ckpt_path = found[0]

    init_geo = Path("/opt/instantsplat/init_geo.py")
    if not init_geo.is_file():
        raise RuntimeError("InstantSplat repo not present in image (/opt/instantsplat/init_geo.py)")

    n_views = min(n, int(os.environ.get("INSTANTSPLAT_N_VIEWS", str(n))))
    model_path = out / "mast3r_run"
    model_path.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["PYTHONPATH"] = ":".join(
        p
        for p in (
            "/opt/instantsplat",
            "/opt/instantsplat/mast3r",
            "/opt/instantsplat/dust3r",
            env.get("PYTHONPATH", ""),
        )
        if p
    )

    cmd = [
        sys.executable,
        str(init_geo),
        "-s",
        str(workspace),
        "-m",
        str(model_path),
        "--ckpt_path",
        str(ckpt_path),
        "--n_views",
        str(n_views),
        "--focal_avg",
        "--co_vis_dsp",
        "--conf_aware_ranking",
        "--infer_video",
        "--device",
        "cuda" if __import__("torch").cuda.is_available() else "cpu",
    ]
    log(f"Running InstantSplat init_geo: {' '.join(cmd)}")
    proc = subprocess.run(cmd, cwd=str(init_geo.parent), env=env, capture_output=True, text=True)
    if proc.stdout:
        for line in proc.stdout.strip().splitlines()[-30:]:
            log(line)
    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "")[-2000:]
        raise RuntimeError(f"init_geo failed ({proc.returncode}): {err}")

    sparse = workspace / "sparse" / "0"
    if not (sparse / "images.txt").is_file() and not (sparse / "images.bin").is_file():
        raise RuntimeError("init_geo completed but sparse/0 COLMAP model missing")

    (out / "init.json").write_text(
        json.dumps(
            {
                "backend": "instant_splat",
                "images": n_views,
                "mast3r_ckpt": str(ckpt_path),
                "sparse": str(sparse),
                "status": "ok",
            },
            indent=2,
        )
    )
    log("InstantSplat init done — COLMAP sparse/0 from MASt3R")
    sparse_ply = out / "sparse_preview.ply"
    artifacts: dict[str, str] = {"geometry_meta": str(out / "init.json")}
    if export_sparse_pointcloud_ply(sparse, sparse_ply):
        artifacts["geometry_sparse_ply"] = str(sparse_ply)
    return out, artifacts
