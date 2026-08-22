"""Pipeline stage orchestration."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any

from jobs.models import RealtimeMode
from pipeline.geometry import run_geometry
from pipeline.export import find_latest_checkpoint
from pipeline.matching import run_matching
from pipeline.registry import merge_config
from pipeline.sfm import run_sfm
from pipeline.stages import publish_stage
from pipeline.train import run_gsplat_train

if TYPE_CHECKING:
    from jobs.worker import Job, JobManager


class PipelineRunner:
    def __init__(self, manager: "JobManager", job: "Job") -> None:
        self.manager = manager
        self.job = job

    def log(self, msg: str) -> None:
        self.job.append_log(msg)

    def cancelled(self) -> bool:
        return self.job._cancel.is_set()

    def run(self) -> None:
        cfg = merge_config(
            self.job.config.preset,
            self.job.config.overrides.model_dump(exclude_none=True),
            self.manager.presets_dir,
        )
        ws = self.manager.workspace_path(self.job.id)
        out = self.manager.output_path(self.job.id)
        dataset = self.manager.dataset_path(self.job.config.dataset_id)

        images_src = dataset / "images"
        ws_images = ws / "images"
        if images_src.is_dir():
            if not ws_images.exists():
                ws_images.symlink_to(images_src, target_is_directory=True)
        else:
            ws_images.mkdir(parents=True, exist_ok=True)
            for f in dataset.iterdir():
                if f.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                    shutil.copy2(f, ws_images / f.name)

        sparse_ds = dataset / "sparse" / "0"
        if sparse_ds.is_dir():
            ws_sparse = ws / "sparse" / "0"
            ws_sparse.mkdir(parents=True, exist_ok=True)
            for f in sparse_ds.iterdir():
                shutil.copy2(f, ws_sparse / f.name)

        stages = cfg.get("stages", {})
        realtime = self.job.config.realtime_mode

        # --- Matching (optional; hloc / dense_sfm before SfM) ---
        matching = stages.get("matching", "sift")
        if matching not in ("none", "sift", "", None):
            self.job.set_stage("matching", 0.05)
            run_matching(matching, ws, ws_images, self.log)

        if self.cancelled():
            return

        # --- SfM ---
        sfm = stages.get("sfm", "glomap")
        if sfm != "skip":
            self.job.set_stage("sfm", 0.1)
            run_sfm(sfm, ws, ws_images, self.log)
            publish_stage(self.job, "sfm", workspace=ws, output_dir=out)
        else:
            self.log("SfM skipped (poses provided or geometry front-end)")

        if self.cancelled():
            return

        # --- Geometry ---
        geom = stages.get("geometry", "none")
        geom_artifacts: dict[str, str] = {}
        extras = stages.get("extras") or []
        if isinstance(extras, str):
            extras = [extras]
        if geom != "none":
            self.job.set_stage("geometry", 0.25)
            _, geom_artifacts = run_geometry(
                geom,
                ws,
                ws_images,
                self.log,
                chunk_size=int(cfg.get("chunk_size", 8)),
                extras=list(extras),
            )
            published = publish_stage(
                self.job,
                "geometry",
                workspace=ws,
                output_dir=out,
                extra=geom_artifacts,
            )

            if realtime == RealtimeMode.geometry_preview:
                preview_path = (
                    published.get("infer_gs_ply")
                    or published.get("geometry_sparse_ply")
                    or self.job.artifacts.get("infer_gs_ply")
                    or self.job.artifacts.get("geometry_sparse_ply")
                )
                self.job.emit(
                    "geometry_preview",
                    path=preview_path,
                    artifacts=published,
                    infer_gs=published.get("infer_gs_ply"),
                )
                manifest = {
                    "job_id": self.job.id,
                    "preset": self.job.config.preset,
                    "dataset_id": self.job.config.dataset_id,
                    "artifacts": dict(self.job.artifacts),
                    "config": cfg,
                    "preview_only": True,
                }
                (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
                self.log("Geometry preview complete — skipping gsplat train")
                self.job.set_stage("done", 1.0)
                return

        if self.cancelled():
            return

        # --- Train ---
        self.job.set_stage("train", 0.4)
        base_port = int(os.environ.get("SPLATLAB_VISER_BASE_PORT", "8780"))
        live_viewer = realtime == RealtimeMode.progressive_splat

        def on_checkpoint(it: int, ply: Path, splat: Path) -> None:
            self.job.artifacts["splat"] = str(splat)
            self.job.artifacts["ply"] = str(ply)
            ckpt = find_latest_checkpoint(out / "splat")
            if ckpt:
                self.job.artifacts["ckpt"] = str(ckpt)
            self.job.artifacts[f"checkpoint_{it}"] = str(splat)
            self.job.emit("checkpoint_ready", iteration=it, splat=str(splat), ply=str(ply), ckpt=str(ckpt) if ckpt else None)
            self.job.progress = 0.4 + 0.5 * min(1.0, it / int(cfg.get("iterations", 30000)))

        ckpt_cb = on_checkpoint if realtime == RealtimeMode.progressive_splat else None
        if live_viewer:
            self.job.emit(
                "viewer_ready",
                port=base_port,
                hint="Live Viser during training (progressive_splat)",
            )

        artifacts = run_gsplat_train(
            ws,
            out,
            cfg,
            self.log,
            on_checkpoint=ckpt_cb,
            cancel_check=self.cancelled,
            enable_live_viewer=live_viewer,
            viewer_port=base_port if live_viewer else None,
        )

        if self.cancelled():
            return

        # --- Export ---
        self.job.set_stage("export", 0.95)
        self.job.artifacts.update(artifacts)
        manifest = {
            "job_id": self.job.id,
            "preset": self.job.config.preset,
            "dataset_id": self.job.config.dataset_id,
            "artifacts": artifacts,
            "config": cfg,
        }
        (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
        self.log("Pipeline complete")
        self.job.set_stage("done", 1.0)
