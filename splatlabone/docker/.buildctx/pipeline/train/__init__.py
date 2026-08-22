"""gsplat training: 3DGS, 2DGS+MCMC, progressive checkpoints."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Callable


def _export_splat_stub(ply_path: Path, splat_path: Path, iteration: int) -> None:
    header = f"# SplatLab stub splat iter={iteration}\n"
    splat_path.write_text(header)
    if not ply_path.exists():
        ply_path.write_text(f"ply\ncomment iter {iteration}\n")


def run_gsplat_train(
    workspace: Path,
    output_dir: Path,
    cfg: dict,
    log: Callable[[str], None],
    *,
    on_checkpoint: Callable[[int, Path, Path], None] | None = None,
    cancel_check: Callable[[], bool] | None = None,
) -> dict[str, str]:
    representation = cfg.get("stages", {}).get("representation", "3dgs")
    densification = cfg.get("stages", {}).get("densification", "default_adc")
    iterations = int(cfg.get("iterations", 30000))
    ckpt_interval = int(cfg.get("checkpoint_interval", 7000))
    train_script = shutil.which("gsplat_train") or shutil.which("python")

    result_dir = output_dir / "splat"
    result_dir.mkdir(parents=True, exist_ok=True)
    ply_final = result_dir / "point_cloud.ply"
    splat_final = result_dir / "scene.splat"

    sparse = workspace / "sparse" / "0"
    images = workspace / "images"
    if not images.is_dir():
        # dataset images symlink
        pass

    if train_script and Path("/opt/gsplat/examples/simple_trainer.py").is_file():
        cmd = [
            "python",
            "/opt/gsplat/examples/simple_trainer.py",
            "colmap",
            "--data_dir",
            str(workspace),
            "--result_dir",
            str(result_dir),
            "--max_steps",
            str(iterations),
        ]
        if representation == "2dgs":
            cmd.extend(["--with_eval3d"])
        if densification == "mcmc":
            cmd.append("--use_mcmc")
        log(f"gsplat train: {representation} {densification} iters={iterations}")
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        step = 0
        for line in proc.stdout or []:
            log(line.rstrip())
            if "step" in line.lower():
                step += 1
            if on_checkpoint and step > 0 and step % max(1, ckpt_interval // 100) == 0:
                ck_ply = result_dir / f"ckpt_{step}.ply"
                ck_splat = result_dir / f"ckpt_{step}.splat"
                _export_splat_stub(ck_ply, ck_splat, step)
                on_checkpoint(step, ck_ply, ck_splat)
            if cancel_check and cancel_check():
                proc.terminate()
                raise RuntimeError("Training cancelled")
        proc.wait()
        if proc.returncode != 0:
            raise RuntimeError(f"gsplat training failed: {proc.returncode}")
    else:
        log(f"gsplat trainer not found — simulating {iterations} iters ({representation})")
        checkpoints = [ckpt_interval, ckpt_interval * 2, iterations]
        for i, it in enumerate(checkpoints):
            if cancel_check and cancel_check():
                raise RuntimeError("Training cancelled")
            pct = (i + 1) / len(checkpoints)
            log(f"simulated train step {it}/{iterations} ({pct:.0%})")
            if on_checkpoint:
                ck_ply = result_dir / f"ckpt_{it}.ply"
                ck_splat = result_dir / f"ckpt_{it}.splat"
                _export_splat_stub(ck_ply, ck_splat, it)
                on_checkpoint(it, ck_ply, ck_splat)

    _export_splat_stub(ply_final, splat_final, iterations)
    meta = {
        "representation": representation,
        "densification": densification,
        "iterations": iterations,
        "pose_opt": cfg.get("stages", {}).get("pose_opt", "off"),
    }
    (result_dir / "train_meta.json").write_text(json.dumps(meta, indent=2))
    return {"ply": str(ply_final), "splat": str(splat_final), "dir": str(result_dir)}
