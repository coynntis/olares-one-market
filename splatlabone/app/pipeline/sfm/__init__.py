"""SfM backends: GLOMAP, COLMAP, FastMap."""

from __future__ import annotations

import shutil
import sqlite3
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


def _database_has_features_and_matches(db: Path) -> bool:
    if not db.is_file() or db.stat().st_size == 0:
        return False
    try:
        conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        try:
            kp = conn.execute("SELECT COUNT(*) FROM keypoints").fetchone()[0]
            mt = conn.execute("SELECT COUNT(*) FROM matches").fetchone()[0]
        finally:
            conn.close()
        return kp > 0 and mt > 0
    except sqlite3.Error:
        return False


def database_ready_for_mapper(workspace: Path) -> bool:
    """True when hloc/dense matching finished or COLMAP DB has features+matches."""
    if (workspace / ".matching_hloc_done").is_file():
        return True
    if (workspace / ".matching_dense_done").is_file():
        return True
    return _database_has_features_and_matches(workspace / "database.db")


def _ensure_colmap_database(
    workspace: Path,
    images: Path,
    db: Path,
    log: Callable[[str], None],
) -> None:
    _extract_and_match(workspace, images, db, log)


def run_glomap(workspace: Path, images: Path, log: Callable[[str], None]) -> Path:
    sparse = workspace / "sparse" / "0"
    sparse.mkdir(parents=True, exist_ok=True)
    db = workspace / "database.db"
    if database_ready_for_mapper(workspace) and _tool("colmap"):
        log(
            "Pre-matched COLMAP database found — skipping feature_extractor / "
            "exhaustive_matcher, running global_mapper only"
        )
        return _global_mapper(workspace, images, db, sparse, log)
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
        _ensure_colmap_database(workspace, images, db, log)
        return _global_mapper(workspace, images, db, sparse, log)
    log("No glomap/colmap — writing stub sparse model for dev")
    (sparse / "cameras.txt").write_text(
        "# stub\n1 SIMPLE_PINHOLE 1920 1080 1000 960 540\n"
    )
    (sparse / "images.txt").write_text("# stub\n")
    (sparse / "points3D.txt").write_text("# stub\n")
    return sparse


def _extract_and_match(
    workspace: Path,
    images: Path,
    db: Path,
    log: Callable[[str], None],
) -> None:
    if database_ready_for_mapper(workspace):
        log("COLMAP database already has features/matches — skipping extract+match")
        return
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


def _global_mapper(
    workspace: Path,
    images: Path,
    db: Path,
    sparse: Path,
    log: Callable[[str], None],
) -> Path:
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


def run_glomap_calibrated(workspace: Path, images: Path, log: Callable[[str], None]) -> Path:
    """Feature match → view_graph_calibrator (copy DB) → global_mapper."""
    sparse = workspace / "sparse" / "0"
    sparse.mkdir(parents=True, exist_ok=True)
    if not _tool("colmap"):
        log("colmap not found — falling back to stub glomap path")
        return run_glomap(workspace, images, log)

    db = workspace / "database.db"
    db_cal = workspace / "database_global.db"
    if database_ready_for_mapper(workspace):
        log(
            "quality_calibrated: pre-matched database — skipping feature_extractor / "
            "exhaustive_matcher; view_graph_calibrator + global_mapper"
        )
    else:
        log("quality_calibrated: feature_extractor + matcher + view_graph_calibrator + global_mapper")
    _ensure_colmap_database(workspace, images, db, log)
    shutil.copy2(db, db_cal)
    _run(
        [
            "colmap",
            "view_graph_calibrator",
            "--database_path",
            str(db_cal),
        ],
        log,
        workspace,
    )
    return _global_mapper(workspace, images, db_cal, sparse, log)


def run_fastmap(workspace: Path, images: Path, log: Callable[[str], None]) -> Path:
    from pipeline.sfm.fastmap_run import run_fastmap as _run_fastmap_impl

    return _run_fastmap_impl(workspace, images, log)


def run_sfm(backend: str, workspace: Path, images: Path, log: Callable[[str], None]) -> Path:
    if backend == "glomap":
        return run_glomap(workspace, images, log)
    if backend == "glomap_calibrated":
        return run_glomap_calibrated(workspace, images, log)
    if backend == "colmap":
        return run_colmap(workspace, images, log)
    if backend == "fastmap":
        return run_fastmap(workspace, images, log)
    raise ValueError(f"unknown SfM backend: {backend}")


def run_colmap(workspace: Path, images: Path, log: Callable[[str], None]) -> Path:
    sparse = workspace / "sparse" / "0"
    sparse.mkdir(parents=True, exist_ok=True)
    db = workspace / "database.db"
    if not _tool("colmap"):
        log("colmap not found — stub sparse")
        (sparse / "cameras.txt").write_text("# stub\n")
        return sparse
    if not database_ready_for_mapper(workspace):
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
    else:
        log("COLMAP database already has features/matches — skipping extract+match")
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
