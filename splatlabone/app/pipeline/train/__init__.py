"""gsplat training: 3DGS, 2DGS+MCMC, progressive checkpoints."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Callable

from pipeline.export import export_from_checkpoint, find_latest_checkpoint

TRAINER_PATH = Path("/opt/gsplat/examples/simple_trainer.py")


def _build_trainer_cmd(
    workspace: Path,
    result_dir: Path,
    cfg: dict,
    *,
    enable_live_viewer: bool = False,
    viewer_port: int | None = None,
) -> list[str]:
    representation = cfg.get("stages", {}).get("representation", "3dgs")
    densification = cfg.get("stages", {}).get("densification", "default_adc")
    pose_opt = cfg.get("stages", {}).get("pose_opt", "off")
    iterations = int(cfg.get("iterations", 30000))
    ckpt_interval = int(cfg.get("checkpoint_interval", 7000))

    # gsplat subcommands: default | mcmc (2DGS+MCMC requires mcmc)
    if densification == "mcmc" or representation == "2dgs":
        subcommand = "mcmc"
    else:
        subcommand = "default"

    save_steps = sorted({ckpt_interval, ckpt_interval * 2, iterations})
    ply_steps = save_steps[:]

    cmd = [
        "python",
        str(TRAINER_PATH),
        subcommand,
        "--data-type",
        "colmap",
        "--data_dir",
        str(workspace),
        "--result_dir",
        str(result_dir),
        "--max_steps",
        str(iterations),
        "--save_ply",
        "--disable_video",
    ]
    for step in save_steps:
        cmd.extend(["--save-steps", str(step)])
    for step in ply_steps:
        cmd.extend(["--ply-steps", str(step)])

    if representation == "2dgs":
        cmd.append("--with_eval3d")
    if pose_opt == "joint":
        cmd.append("--pose_opt")

    if enable_live_viewer and viewer_port:
        cmd.extend(["--port", str(viewer_port)])
    else:
        cmd.append("--disable_viewer")

    return cmd


def run_gsplat_train(
    workspace: Path,
    output_dir: Path,
    cfg: dict,
    log: Callable[[str], None],
    *,
    on_checkpoint: Callable[[int, Path, Path], None] | None = None,
    cancel_check: Callable[[], bool] | None = None,
    enable_live_viewer: bool = False,
    viewer_port: int | None = None,
) -> dict[str, str]:
    representation = cfg.get("stages", {}).get("representation", "3dgs")
    densification = cfg.get("stages", {}).get("densification", "default_adc")
    iterations = int(cfg.get("iterations", 30000))
    ckpt_interval = int(cfg.get("checkpoint_interval", 7000))

    result_dir = output_dir / "splat"
    result_dir.mkdir(parents=True, exist_ok=True)

    if not TRAINER_PATH.is_file():
        log(f"gsplat trainer not found at {TRAINER_PATH} — simulating {iterations} iters")
        ply_final = result_dir / "point_cloud.ply"
        splat_final = result_dir / "scene.splat"
        ply_final.write_text("ply\ncomment simulated\n")
        splat_final.write_text("# simulated\n")
        return {"ply": str(ply_final), "splat": str(splat_final), "dir": str(result_dir)}

    cmd = _build_trainer_cmd(
        workspace,
        result_dir,
        cfg,
        enable_live_viewer=enable_live_viewer,
        viewer_port=viewer_port,
    )
    log(f"gsplat train: {' '.join(cmd)}")

    env = os.environ.copy()
    # Keep site-packages ahead of /opt/gsplat so the CUDA wheel wins; examples cwd supplies scripts
    py = env.get("PYTHONPATH", "")
    parts = [p for p in py.split(":") if p and p != "/opt/gsplat"]
    env["PYTHONPATH"] = ":".join(parts)

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
        cwd=str(TRAINER_PATH.parent),
    )
    last_ckpt_step = 0
    for line in proc.stdout or []:
        log(line.rstrip())
        if "Step:" in line and on_checkpoint:
            last_ckpt_step += 1
            ckpt = find_latest_checkpoint(result_dir)
            if ckpt:
                try:
                    partial = export_from_checkpoint(ckpt, result_dir, log)
                    ply_p = Path(partial.get("ply", result_dir / "point_cloud.ply"))
                    splat_p = Path(partial.get("splat", result_dir / "scene.splat"))
                    on_checkpoint(last_ckpt_step * ckpt_interval, ply_p, splat_p)
                except Exception as exc:
                    log(f"checkpoint export warn: {exc}")
        if cancel_check and cancel_check():
            proc.terminate()
            raise RuntimeError("Training cancelled")

    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"gsplat training failed: {proc.returncode}")

    ckpt = find_latest_checkpoint(result_dir)
    if not ckpt:
        raise RuntimeError("training finished but no checkpoint found")

    artifacts = export_from_checkpoint(ckpt, result_dir, log)
    artifacts["ckpt"] = str(ckpt)

    meta = {
        "representation": representation,
        "densification": densification,
        "iterations": iterations,
        "pose_opt": cfg.get("stages", {}).get("pose_opt", "off"),
        "ckpt": str(ckpt),
    }
    (result_dir / "train_meta.json").write_text(json.dumps(meta, indent=2))
    return artifacts
