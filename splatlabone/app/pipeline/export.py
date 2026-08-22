"""Export gsplat checkpoints to PLY / SPLAT / SuperSplat formats."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Callable


def find_latest_checkpoint(result_dir: Path) -> Path | None:
    candidates: list[Path] = []
    for pattern in ("ckpts/ckpt_*_rank0.pt", "ckpts/ckpt_*.pt", "**/ckpt_*_rank0.pt", "**/ckpt_*.pt"):
        candidates.extend(result_dir.glob(pattern))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def find_latest_ply(result_dir: Path) -> Path | None:
    ply_dir = result_dir / "ply"
    if ply_dir.is_dir():
        plies = sorted(ply_dir.glob("point_cloud_*.ply"))
        if plies:
            return plies[-1]
    plies = sorted(result_dir.rglob("point_cloud_*.ply"))
    return plies[-1] if plies else None


def export_from_checkpoint(
    ckpt_path: Path,
    result_dir: Path,
    log: Callable[[str], None] | None = None,
) -> dict[str, str]:
    """Export trained checkpoint to ply, splat, ply_compressed."""
    result_dir.mkdir(parents=True, exist_ok=True)
    ply_final = result_dir / "point_cloud.ply"
    splat_final = result_dir / "scene.splat"
    compressed_final = result_dir / "scene.ply_compressed"
    artifacts: dict[str, str] = {"ckpt": str(ckpt_path), "dir": str(result_dir)}

    def _log(msg: str) -> None:
        if log:
            log(msg)

    try:
        import torch
        from gsplat import export_splats

        data = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        splats = data.get("splats") or data
        means = splats["means"]
        scales = splats["scales"]
        quats = splats["quats"]
        opacities = splats["opacities"]
        sh0 = splats.get("sh0")
        shN = splats.get("shN")

        for fmt, path in (
            ("ply", ply_final),
            ("splat", splat_final),
            ("ply_compressed", compressed_final),
        ):
            export_splats(
                means=means,
                scales=scales,
                quats=quats,
                opacities=opacities,
                sh0=sh0,
                shN=shN,
                format=fmt,
                save_to=str(path),
            )
            _log(f"exported {fmt} -> {path.name}")
            artifacts[fmt.replace("ply_compressed", "ply_compressed")] = str(path)

        artifacts["ply"] = str(ply_final)
        artifacts["splat"] = str(splat_final)
        artifacts["ply_compressed"] = str(compressed_final)
        return artifacts
    except Exception as exc:
        _log(f"gsplat export_splats failed ({exc}); falling back to PLY copy")

    src_ply = find_latest_ply(result_dir)
    if src_ply and src_ply.is_file():
        shutil.copy2(src_ply, ply_final)
        artifacts["ply"] = str(ply_final)
        _log(f"copied {src_ply.name} -> point_cloud.ply")
    elif not ply_final.is_file():
        ply_final.write_text("ply\ncomment splatlab export fallback\n")

    if not splat_final.is_file():
        splat_final.write_text("# splatlab fallback — re-export when gsplat available\n")
    artifacts["splat"] = str(splat_final)
    return artifacts
