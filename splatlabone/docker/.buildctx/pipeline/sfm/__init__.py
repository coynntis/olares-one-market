"""SfM backends: GLOMAP, COLMAP."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Callable


def _run(cmd: list[str], log: Callable[[str], None], cwd: Path | None = None) -> None:
    log(f"$ {' '.join(cmd)}")
    proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if proc.stdout:
        for line in proc.stdout.strip().splitlines()[-20:]:
            log(line)
    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "")[-500:]
        raise RuntimeError(f"Command failed ({proc.returncode}): {err}")


def _tool(name: str) -> str | None:
    return shutil.which(name)


def run_glomap(workspace: Path, images: Path, log: Callable[[str], None]) -> Path:
    sparse = workspace / "sparse" / "0"
    sparse.mkdir(parents=True, exist_ok=True)
    db = workspace / "database.db"
    if _tool("glomap"):
        _run(
            [
                "glomap",
                "mapper",
                "--image_path",
                str(images),
                "--output_path",
                str(sparse.parent),
            ],
            log,
        )
        return sparse
    if _tool("colmap"):
        log("Using COLMAP global_mapper (GLOMAP successor in COLMAP 4.x)")
        _run(
            [
                "colmap",
                "feature_extractor",
                "--database_path",
                str(db),
                "--image_path",
                str(images),
            ],
            log,
            workspace,
        )
        _run(["colmap", "exhaustive_matcher", "--database_path", str(db)], log, workspace)
        sparse_out = workspace / "sparse"
        sparse_out.mkdir(parents=True, exist_ok=True)
        _run(
            [
                "colmap",
                "global_mapper",
                "--database_path",
                str(db),
                "--image_path",
                str(images),
                "--output_path",
                str(sparse_out),
            ],
            log,
            workspace,
        )
        src = sparse_out / "0"
        if src.is_dir():
            for f in src.iterdir():
                shutil.copy2(f, sparse / f.name)
        return sparse
    log("No glomap/colmap — writing stub sparse model for dev")
    (sparse / "cameras.txt").write_text(
        "# stub\n1 SIMPLE_PINHOLE 1920 1080 1000 960 540\n"
    )
    (sparse / "images.txt").write_text("# stub\n")
    (sparse / "points3D.txt").write_text("# stub\n")
    return sparse


def run_colmap(workspace: Path, images: Path, log: Callable[[str], None]) -> Path:
    sparse = workspace / "sparse" / "0"
    sparse.mkdir(parents=True, exist_ok=True)
    db = workspace / "database.db"
    if not _tool("colmap"):
        log("colmap not found — stub sparse")
        (sparse / "cameras.txt").write_text("# stub\n")
        return sparse
    _run(
        [
            "colmap",
            "feature_extractor",
            "--database_path",
            str(db),
            "--image_path",
            str(images),
        ],
        log,
        workspace,
    )
    _run(["colmap", "exhaustive_matcher", "--database_path", str(db)], log, workspace)
    sparse_tmp = workspace / "sparse_tmp"
    sparse_tmp.mkdir(exist_ok=True)
    _run(
        [
            "colmap",
            "mapper",
            "--database_path",
            str(db),
            "--image_path",
            str(images),
            "--output_path",
            str(sparse_tmp),
        ],
        log,
        workspace,
    )
    src = sparse_tmp / "0"
    if src.is_dir():
        for f in src.iterdir():
            shutil.copy2(f, sparse / f.name)
    return sparse
