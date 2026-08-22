"""Publish intermediate pipeline artifacts for inspection."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pipeline.colmap_io import export_sparse_pointcloud_ply

if TYPE_CHECKING:
    from jobs.worker import Job

STAGE_LABELS: dict[str, str] = {
    "sfm_sparse_ply": "SfM sparse point cloud",
    "geometry_sparse_ply": "Geometry sparse point cloud",
    "infer_gs_ply": "DA3 infer_gs Gaussians (feed-forward)",
    "geometry_glb": "Geometry GLB preview",
    "geometry_meta": "Geometry metadata JSON",
}


def publish_stage(
    job: "Job",
    stage: str,
    *,
    workspace: Path,
    output_dir: Path,
    extra: dict[str, str] | None = None,
) -> dict[str, str]:
    """Snapshot COLMAP sparse + extra files into output_dir/stages/ and register on job."""
    stages_dir = output_dir / "stages"
    stages_dir.mkdir(parents=True, exist_ok=True)
    published: dict[str, str] = dict(extra or {})

    sparse = workspace / "sparse" / "0"
    ply_key = f"{stage}_sparse_ply" if stage != "geometry" else "geometry_sparse_ply"
    if stage == "sfm":
        ply_key = "sfm_sparse_ply"
    ply_dest = stages_dir / f"{stage}_sparse.ply"
    if export_sparse_pointcloud_ply(sparse, ply_dest):
        published[ply_key] = str(ply_dest)

    for key, src in list(published.items()):
        p = Path(src)
        if not p.is_file():
            continue
        if str(p).startswith(str(stages_dir)):
            continue
        dest = stages_dir / f"{key.replace('_', '-')}{p.suffix or '.bin'}"
        shutil.copy2(p, dest)
        published[key] = str(dest)

    manifest = output_dir / "stages" / "manifest.json"
    existing: dict[str, Any] = {}
    if manifest.is_file():
        try:
            existing = json.loads(manifest.read_text())
        except Exception:
            existing = {}
    existing[stage] = published
    manifest.write_text(json.dumps(existing, indent=2))

    job.artifacts.update(published)
    job.emit(
        "stage_artifacts",
        stage=stage,
        artifacts=published,
        urls={k: f"/api/v1/scenes/{job.id}/artifact/{k}" for k in published},
    )
    return published


def list_stage_artifacts(job: "Job", output_dir: Path) -> dict[str, Any]:
    """Structured artifact list for API / UI."""
    manifest_path = output_dir / "stages" / "manifest.json"
    stages: dict[str, dict[str, str]] = {}
    if manifest_path.is_file():
        try:
            stages = json.loads(manifest_path.read_text())
        except Exception:
            stages = {}

    items: list[dict[str, Any]] = []
    for stage, arts in stages.items():
        for key, path in arts.items():
            items.append(
                {
                    "stage": stage,
                    "key": key,
                    "label": STAGE_LABELS.get(key, key),
                    "path": path,
                    "url": f"/api/v1/scenes/{job.id}/artifact/{key}",
                    "viewer": _viewer_hint(key),
                }
            )

    train_keys = {
        "ckpt": ("Trained gsplat checkpoint", "gsplat_viser"),
        "ply": ("Trained point cloud", "download"),
        "splat": ("Trained .splat export", "download"),
        "ply_compressed": ("SuperSplat edit export", "supersplat"),
    }
    for key, (label, viewer) in train_keys.items():
        p = job.artifacts.get(key)
        if p and Path(p).is_file():
            items.append(
                {
                    "stage": "train",
                    "key": key,
                    "label": label,
                    "path": p,
                    "url": f"/api/v1/scenes/{job.id}/{key}",
                    "viewer": viewer,
                }
            )

    # Keys registered on job but not yet in manifest (e.g. mid-run)
    seen = {(i["stage"], i["key"]) for i in items}
    for key, label in (
        ("infer_gs_ply", "DA3 infer_gs Gaussians (feed-forward)"),
        ("geometry_sparse_ply", "Geometry sparse point cloud"),
        ("sfm_sparse_ply", "SfM sparse point cloud"),
        ("geometry_glb", "Geometry GLB preview"),
        ("geometry_meta", "Geometry metadata JSON"),
    ):
        p = job.artifacts.get(key)
        if not p or not Path(p).is_file():
            continue
        stage = "geometry" if key.startswith(("infer_gs", "geometry_")) else "sfm" if key.startswith("sfm_") else "geometry"
        if (stage, key) in seen:
            continue
        items.append(
            {
                "stage": stage,
                "key": key,
                "label": label,
                "path": p,
                "url": f"/api/v1/scenes/{job.id}/artifact/{key}",
                "viewer": _viewer_hint(key),
            }
        )

    return {"job_id": job.id, "artifacts": items, "stages": stages}


def _viewer_hint(key: str) -> str:
    if key.endswith("_sparse_ply"):
        return "pointcloud"
    if key == "infer_gs_ply":
        return "supersplat"
    if key == "geometry_glb":
        return "download"
    return "download"


def resolve_artifact(job: "Job", output_dir: Path, key: str) -> Path | None:
    if key in job.artifacts and Path(job.artifacts[key]).is_file():
        return Path(job.artifacts[key])
    manifest_path = output_dir / "stages" / "manifest.json"
    if manifest_path.is_file():
        try:
            stages = json.loads(manifest_path.read_text())
            for arts in stages.values():
                if key in arts and Path(arts[key]).is_file():
                    return Path(arts[key])
        except Exception:
            pass
    # Disk fallbacks
    fallbacks = {
        "infer_gs_ply": output_dir / "stages" / "infer-gs-ply.ply",
        "geometry_sparse_ply": output_dir / "stages" / "geometry_sparse.ply",
        "sfm_sparse_ply": output_dir / "stages" / "sfm_sparse.ply",
    }
    p = fallbacks.get(key)
    return p if p and p.is_file() else None
