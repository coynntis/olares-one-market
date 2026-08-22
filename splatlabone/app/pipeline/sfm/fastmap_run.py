"""FastMap SfM backend (pals-ttic/fastmap)."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Callable

from pipeline.colmap_io import copy_colmap_sparse

_FASTMAP_INSTALL_HINT = (
    "FastMap not found. Clone https://github.com/pals-ttic/fastmap to /opt/fastmap, "
    "pip install dependencies (and compile CUDA kernels), or set FASTMAP_ROOT to the repo root."
)


def _find_fastmap_run_script() -> Path:
    candidates: list[Path] = []
    fastmap_root = os.environ.get("FASTMAP_ROOT")
    if fastmap_root:
        candidates.append(Path(fastmap_root) / "run.py")
    candidates.append(Path("/opt/fastmap/run.py"))

    try:
        import fastmap  # noqa: F401

        pkg_dir = Path(fastmap.__file__).resolve().parent
        candidates.extend([pkg_dir.parent / "run.py", pkg_dir / "run.py"])
    except ImportError:
        pass

    for path in candidates:
        if path.is_file():
            return path

    raise RuntimeError(_FASTMAP_INSTALL_HINT)


def run_fastmap(workspace: Path, images: Path, log: Callable[[str], None]) -> Path:
    # Lazy import — avoid circular import with pipeline.sfm.__init__
    from pipeline.sfm import _ensure_colmap_database, _run, _tool

    sparse = workspace / "sparse" / "0"
    sparse.mkdir(parents=True, exist_ok=True)
    db = workspace / "database.db"

    if not _tool("colmap"):
        raise RuntimeError("colmap required for FastMap feature extraction — install COLMAP")

    _ensure_colmap_database(workspace, images, db, log)

    run_py = _find_fastmap_run_script()
    fastmap_out = workspace / "fastmap_out"
    fastmap_out.mkdir(parents=True, exist_ok=True)

    log(f"FastMap: {run_py}")
    _run(
        [
            sys.executable,
            str(run_py),
            "--database",
            str(db),
            "--image_dir",
            str(images),
            "--output_dir",
            str(fastmap_out),
            "--headless",
        ],
        log,
        workspace,
    )

    src = fastmap_out / "sparse" / "0"
    if not src.is_dir():
        raise RuntimeError(f"FastMap did not produce sparse/0 under {fastmap_out}")

    copy_colmap_sparse(src, sparse)
    if not any(sparse.iterdir()):
        raise RuntimeError(f"FastMap sparse model empty after copy to {sparse}")

    log(f"FastMap sparse model → {sparse}")
    return sparse
