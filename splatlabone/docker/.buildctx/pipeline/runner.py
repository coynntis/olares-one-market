"""Pipeline stage orchestration."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any

from jobs.models import RealtimeMode
from pipeline.geometry import run_da3, run_instant_splat, run_vggt_x
from pipeline.registry import merge_config
from pipeline.sfm import run_colmap, run_glomap
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

        # --- SfM ---
        sfm = stages.get("sfm", "glomap")
        if sfm != "skip":
            self.job.set_stage("sfm", 0.1)
            if sfm == "glomap":
                run_glomap(ws, ws_images, self.log)
            elif sfm == "colmap":
                run_colmap(ws, ws_images, self.log)
        else:
            self.log("SfM skipped (poses provided or geometry front-end)")

        if self.cancelled():
            return

        # --- Geometry ---
        geom = stages.get("geometry", "none")
        if geom != "none":
            self.job.set_stage("geometry", 0.25)
            if geom == "vggt_x":
                run_vggt_x(ws, ws_images, self.log)
            elif geom == "da3":
                run_da3(ws, ws_images, self.log, chunk_size=int(cfg.get("chunk_size", 8)))
            elif geom == "instant_splat":
                run_instant_splat(ws, ws_images, self.log)

        if self.cancelled():
            return

        # --- geometry_preview early export ---
        if realtime == RealtimeMode.geometry_preview:
            preview = ws / "geometry" / "preview.ply"
            preview.parent.mkdir(parents=True, exist_ok=True)
            preview.write_text("ply\ncomment geometry preview\n")
            self.job.artifacts["preview_ply"] = str(preview)
            self.job.emit("geometry_preview", path=str(preview))

        # --- Train ---
        self.job.set_stage("train", 0.4)

        def on_checkpoint(it: int, ply: Path, splat: Path) -> None:
            rel_splat = str(splat)
            self.job.artifacts["splat"] = rel_splat
            self.job.artifacts[f"checkpoint_{it}"] = rel_splat
            self.job.emit("checkpoint_ready", iteration=it, splat=rel_splat, ply=str(ply))
            self.job.progress = 0.4 + 0.5 * min(1.0, it / int(cfg.get("iterations", 30000)))

        ckpt_cb = on_checkpoint if realtime == RealtimeMode.progressive_splat else None
        artifacts = run_gsplat_train(
            ws,
            out,
            cfg,
            self.log,
            on_checkpoint=ckpt_cb,
            cancel_check=self.cancelled,
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
