"""GlueMap geometry backend (colmap/gluemap) → COLMAP sparse/0."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Callable

from pipeline.colmap_io import copy_colmap_sparse, export_sparse_pointcloud_ply

_GLUEMAP_HINT = (
    "GlueMap not found. Install:\n"
    "  git clone https://github.com/colmap/gluemap /opt/gluemap\n"
    "  cd /opt/gluemap && git submodule update --init --recursive\n"
    "  pip install -e /opt/gluemap   # needs Ceres/Eigen/Boost/OpenMP\n"
    "  # download checkpoints → /opt/gluemap/checkpoints (INSTALL.md §4)\n"
    "Or set GLUEMAP_ROOT / GLUEMAP_CONFIG. Verify: gluemap-demo --help"
)


def _list_images(images: Path) -> list[Path]:
    exts = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff"}
    return sorted(p for p in images.iterdir() if p.suffix.lower() in exts)


def _find_gluemap_root() -> Path:
    env = os.environ.get("GLUEMAP_ROOT")
    if env:
        p = Path(env)
        if p.is_dir():
            return p.resolve()
    for candidate in (Path("/opt/gluemap"), Path("/opt/GlueMap")):
        if candidate.is_dir():
            return candidate.resolve()
    try:
        import gluemap  # noqa: F401

        return Path(gluemap.__file__).resolve().parent.parent
    except ImportError:
        pass
    raise RuntimeError(_GLUEMAP_HINT)


def _find_config(root: Path) -> Path:
    env = os.environ.get("GLUEMAP_CONFIG")
    if env:
        p = Path(env)
        if p.is_file():
            return p.resolve()
    for rel in (
        "configs/example.yaml",
        "configs/base.yaml",
        "config/example.yaml",
    ):
        hit = root / rel
        if hit.is_file():
            return hit
    raise RuntimeError(f"GlueMap config not found under {root}/configs/")


def _find_gluemap_bin(root: Path) -> list[str]:
    hit = shutil.which("gluemap-demo")
    if hit:
        return [hit]
    for script in (
        Path(sys.executable).resolve().parent / "gluemap-demo",
        root / ".venv" / "bin" / "gluemap-demo",
        root / "gluemap_demo.py",
        root / "scripts" / "gluemap_demo.py",
        root / "demo.py",
    ):
        if script.is_file() and os.access(script, os.X_OK):
            return [str(script)]
        if script.is_file() and script.suffix == ".py":
            return [sys.executable, str(script)]
    try:
        import importlib.metadata as im

        for ep in im.entry_points(group="console_scripts"):
            if ep.name == "gluemap-demo":
                # pip install -e . registers gluemap.cli:demo_main
                return [
                    sys.executable,
                    "-c",
                    "import sys; sys.argv[0]='gluemap-demo'; from gluemap.cli import demo_main; demo_main()",
                ]
    except Exception:
        pass
    raise RuntimeError(_GLUEMAP_HINT)


def _is_colmap_sparse(path: Path) -> bool:
    if not path.is_dir():
        return False
    has_cam = (path / "cameras.bin").is_file() or (path / "cameras.txt").is_file()
    has_img = (path / "images.bin").is_file() or (path / "images.txt").is_file()
    return has_cam and has_img


def _find_best_sparse(write_path: Path) -> Path | None:
    """Prefer refined ABA output, then coarse, then any sparse/0."""
    preferred_names = (
        "gluemap_aba",
        "refined",
        "coarse",
        "sparse",
    )
    ranked: list[tuple[int, Path]] = []
    for d in write_path.rglob("*"):
        if not d.is_dir():
            continue
        candidate = d
        if (d / "0").is_dir() and _is_colmap_sparse(d / "0"):
            candidate = d / "0"
        if not _is_colmap_sparse(candidate):
            continue
        score = 0
        name = candidate.parent.name.lower() if candidate.name == "0" else candidate.name.lower()
        for i, key in enumerate(preferred_names):
            if key in name:
                score = 100 - i * 10
                break
        if "aba" in name or "refine" in name:
            score = max(score, 95)
        score += min(candidate.stat().st_mtime_ns // 10**9, 10**6) % 1000
        ranked.append((score, candidate))
    if not ranked:
        return None
    ranked.sort(key=lambda x: x[0], reverse=True)
    return ranked[0][1]


def run_gluemap(
    workspace: Path,
    images: Path,
    log: Callable[[str], None],
    *,
    chunk_size: int = 8,
    extras: list[str] | None = None,
) -> tuple[Path, dict[str, str]]:
    extras = extras or []
    del chunk_size  # GlueMap processes full set; chunk unused
    log("GlueMap geometry (global SfM + feedforward)")
    out = workspace / "geometry" / "gluemap"
    out.mkdir(parents=True, exist_ok=True)
    img_list = _list_images(images)
    if not img_list:
        raise RuntimeError("no images for GlueMap")

    root = _find_gluemap_root()
    config = _find_config(root)
    write_path = out / "results"
    if write_path.exists():
        shutil.rmtree(write_path)
    write_path.mkdir(parents=True, exist_ok=True)

    cmd = _find_gluemap_bin(root) + [
        "--config",
        str(config),
        "--images_path",
        str(images),
        "--intrinsics_mode",
        os.environ.get("GLUEMAP_INTRINSICS_MODE", "SHARED"),
        "--write_path",
        str(write_path),
    ]
    if os.environ.get("GLUEMAP_SKIP_DOPPELGANGERS", "1") not in ("0", "false", "False"):
        cmd.append("--skip_doppelgangers")

    env = os.environ.copy()
    py_paths = [str(root), env.get("PYTHONPATH", "")]
    env["PYTHONPATH"] = ":".join(p for p in py_paths if p)
    # Prefer checkpoints under /opt/gluemap/checkpoints or model cache
    ckpt = Path(os.environ.get("GLUEMAP_CHECKPOINTS", str(root / "checkpoints")))
    if ckpt.is_dir():
        env.setdefault("GLUEMAP_CHECKPOINTS", str(ckpt))

    log(f"GlueMap root={root} config={config.name}")
    log(f"$ {' '.join(cmd)}")
    proc = subprocess.run(
        cmd,
        cwd=str(root),
        env=env,
        capture_output=True,
        text=True,
    )
    if proc.stdout:
        for line in proc.stdout.strip().splitlines()[-40:]:
            log(line)
    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "")[-2500:]
        raise RuntimeError(f"gluemap-demo failed ({proc.returncode}): {err}\n{_GLUEMAP_HINT}")

    sparse_src = _find_best_sparse(write_path)
    if sparse_src is None:
        raise RuntimeError(
            f"GlueMap finished but no COLMAP sparse model under {write_path}. "
            "Check checkpoints (path_feedforward / path_retrieval / path_tracker)."
        )

    sparse = workspace / "sparse" / "0"
    sparse.mkdir(parents=True, exist_ok=True)
    copy_colmap_sparse(sparse_src, sparse)
    log(f"GlueMap COLMAP → {sparse} (from {sparse_src})")

    artifacts: dict[str, str] = {}
    preview_ply = out / "sparse_preview.ply"
    if export_sparse_pointcloud_ply(sparse, preview_ply):
        artifacts["geometry_sparse_ply"] = str(preview_ply)

    meta = {
        "backend": "gluemap",
        "root": str(root),
        "config": str(config),
        "frames": len(img_list),
        "extras": extras,
        "status": "ok",
        "colmap": str(sparse),
        "source_sparse": str(sparse_src),
        "geometry_sparse_ply": artifacts.get("geometry_sparse_ply"),
    }
    meta_path = out / "preview.json"
    meta_path.write_text(json.dumps(meta, indent=2))
    artifacts["geometry_meta"] = str(meta_path)
    log("GlueMap done")
    return out, artifacts
