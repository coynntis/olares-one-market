"""Depth Anything 3 C++ (ggml) geometry backend via mudler/depth-anything.cpp CLI."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Callable

from pipeline.colmap_io import copy_colmap_sparse, export_sparse_pointcloud_ply
from pipeline.models.cache import CACHE_ROOT, resolve as resolve_model
from pipeline.models.download import download_model

_DEFAULT_GGUF = "depth-anything-base-q4_k.gguf"
_HF_REPO = "mudler/depth-anything.cpp-gguf"
_BUILD_HINT = (
    "Build mudler/depth-anything.cpp: "
    "git clone https://github.com/mudler/depth-anything.cpp && "
    "cmake -B build -DDA3_CLI=ON && cmake --build build -j. "
    "Set DA3_CPP_BIN to build/examples/cli/da3-cli or install da3-cli on PATH."
)


def _list_images(images: Path) -> list[Path]:
    exts = {".jpg", ".jpeg", ".png", ".webp"}
    return sorted(p for p in images.iterdir() if p.suffix.lower() in exts)


def _find_binary() -> Path:
    env_bin = os.environ.get("DA3_CPP_BIN")
    if env_bin:
        p = Path(env_bin)
        if p.is_file() and os.access(p, os.X_OK):
            return p.resolve()

    for name in ("da3-cli", "da3"):
        hit = shutil.which(name)
        if hit:
            return Path(hit).resolve()

    for candidate in (
        Path("/opt/depth-anything-cpp/build/examples/cli/da3-cli"),
        Path("/usr/local/bin/da3-cli"),
    ):
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate.resolve()

    raise RuntimeError(f"da3-cli binary not found. {_BUILD_HINT}")


def _score_gguf(path: Path) -> tuple[int, int]:
    """Higher is better: prefer q4_k / q8_0 / base quant names."""
    name = path.name.lower()
    score = 0
    if "q4_k" in name:
        score += 30
    elif "q8_0" in name:
        score += 25
    elif "base" in name:
        score += 20
    if "f32" in name or "f16" in name:
        score -= 5
    return score, path.stat().st_size


def _glob_gguf() -> list[Path]:
    hits: list[Path] = []
    models_dir = Path("/models")
    if models_dir.is_dir():
        hits.extend(models_dir.glob("depth-anything*.gguf"))
    da3_cache = CACHE_ROOT / "da3_cpp"
    if da3_cache.is_dir():
        hits.extend(da3_cache.rglob("*.gguf"))
    if CACHE_ROOT.is_dir():
        hits.extend(CACHE_ROOT.rglob("depth-anything*.gguf"))
    return hits


def _find_model_gguf(log: Callable[[str], None]) -> Path:
    env_model = os.environ.get("DA3_CPP_MODEL")
    if env_model:
        p = Path(env_model)
        if p.is_file():
            return p.resolve()
        raise RuntimeError(f"DA3_CPP_MODEL set but not found: {env_model}")

    candidates: list[Path] = [p for p in _glob_gguf() if p.is_file()]

    seen: set[Path] = set()
    unique: list[Path] = []
    for c in candidates:
        rp = c.resolve()
        if rp not in seen:
            seen.add(rp)
            unique.append(rp)

    if unique:
        best = max(unique, key=_score_gguf)
        log(f"DA3.cpp model: {best}")
        return best

    cached = resolve_model("da3_cpp")
    if cached is not None:
        if cached.is_file() and cached.suffix == ".gguf":
            return cached.resolve()
        ggufs = sorted(cached.rglob("*.gguf"), key=_score_gguf, reverse=True)
        if ggufs:
            log(f"DA3.cpp model from cache: {ggufs[0]}")
            return ggufs[0].resolve()

    log(f"downloading DA3.cpp GGUF ({_DEFAULT_GGUF}) from {_HF_REPO}...")
    downloaded = download_model("da3_cpp")
    if downloaded.is_file() and downloaded.suffix == ".gguf":
        return downloaded.resolve()
    ggufs = sorted(downloaded.rglob("*.gguf"), key=_score_gguf, reverse=True)
    if ggufs:
        return ggufs[0].resolve()
    raise RuntimeError(f"download_model(da3_cpp) did not produce a .gguf under {downloaded}")


def _find_colmap_sparse(root: Path) -> Path | None:
    """Locate COLMAP sparse model under CLI export tree."""
    if (root / "cameras.txt").is_file() or (root / "cameras.bin").is_file():
        return root
    if (root / "sparse" / "0").is_dir():
        return root / "sparse" / "0"
    for hit in root.rglob("cameras.txt"):
        return hit.parent
    for hit in root.rglob("cameras.bin"):
        return hit.parent
    return None


def _find_gs_ply(root: Path) -> Path | None:
    candidates: list[Path] = []
    for pat in ("**/*gs*.ply", "**/gaussians*.ply", "**/infer_gs*.ply", "**/*.ply"):
        candidates.extend(root.glob(pat))
    if not candidates:
        return None
    scored = sorted(
        candidates,
        key=lambda p: (
            "gs" in p.as_posix().lower() or "gaussian" in p.name.lower(),
            p.stat().st_size,
        ),
        reverse=True,
    )
    return scored[0]


def _run_cli(
    binary: Path,
    model: Path,
    image_paths: list[Path],
    export_dir: Path,
    log: Callable[[str], None],
    *,
    out_prefix: str = "scene",
) -> None:
    export_dir.mkdir(parents=True, exist_ok=True)
    colmap_dir = export_dir / "colmap"
    glb_path = export_dir / "preview.glb"
    ply_path = export_dir / "cloud.ply"

    cmd = [
        str(binary),
        "depth",
        "--model",
        str(model),
        "--colmap",
        str(colmap_dir),
        "--out-prefix",
        out_prefix,
        "--glb",
        str(glb_path),
        "--ply",
        str(ply_path),
    ]
    for img in image_paths:
        cmd.extend(["--input", str(img)])

    log(f"DA3.cpp: {' '.join(cmd[:8])} ... ({len(image_paths)} images)")
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.stdout.strip():
        for line in proc.stdout.strip().splitlines()[-20:]:
            log(f"  da3-cli: {line}")
    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "").strip()
        raise RuntimeError(f"da3-cli failed (exit {proc.returncode}): {err[-2000:]}")


def _maybe_infer_gs_python(
    workspace: Path,
    images: Path,
    log: Callable[[str], None],
    *,
    chunk_size: int,
    artifacts: dict[str, str],
) -> None:
    """Run Python DA3 infer_gs into temp dir; copy infer_gs.ply only."""
    import tempfile

    from pipeline.geometry.da3 import run_da3

    log("DA3.cpp hybrid: running Python DA3 infer_gs (extras infer_gs_python)")
    with tempfile.TemporaryDirectory(prefix="da3_infer_gs_", dir=workspace) as tmp:
        tmp_ws = Path(tmp)
        tmp_images = tmp_ws / "images"
        tmp_images.mkdir(parents=True, exist_ok=True)
        img_list = _list_images(images)
        # Light path: small chunk for infer_gs preview only
        subset = img_list[: min(len(img_list), chunk_size)]
        for p in subset:
            shutil.copy2(p, tmp_images / p.name)
        _, da3_artifacts = run_da3(tmp_ws, tmp_images, log, chunk_size=chunk_size)
        infer_src = da3_artifacts.get("infer_gs_ply")
        if not infer_src:
            hit = _find_gs_ply(tmp_ws)
            infer_src = str(hit) if hit else None
        if infer_src and Path(infer_src).is_file():
            dest = workspace / "geometry" / "da3_cpp" / "infer_gs.ply"
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(infer_src, dest)
            artifacts["infer_gs_ply"] = str(dest)
            log(f"DA3.cpp hybrid infer_gs -> {dest.name}")
        else:
            log("DA3.cpp hybrid: Python DA3 did not produce infer_gs.ply")


def run_da3_cpp(
    workspace: Path,
    images: Path,
    log: Callable[[str], None],
    *,
    chunk_size: int = 8,
    extras: list[str] | None = None,
) -> tuple[Path, dict[str, str]]:
    extras = extras or []
    log(f"DA3.cpp geometry (chunk_size={chunk_size})")
    out = workspace / "geometry" / "da3_cpp"
    out.mkdir(parents=True, exist_ok=True)
    img_list = _list_images(images)
    if not img_list:
        raise RuntimeError("no images for DA3.cpp")

    binary = _find_binary()
    model = _find_model_gguf(log)
    log(f"DA3.cpp binary={binary.name} model={model.name}")

    sparse = workspace / "sparse" / "0"
    export_root = out / "export"
    export_root.mkdir(parents=True, exist_ok=True)
    artifacts: dict[str, str] = {}

    for start in range(0, len(img_list), chunk_size):
        chunk = img_list[start : start + chunk_size]
        chunk_export = export_root / f"chunk_{start:04d}"
        log(f"DA3.cpp chunk {start // chunk_size + 1}: {len(chunk)} frames")
        _run_cli(
            binary,
            model,
            chunk,
            chunk_export,
            log,
            out_prefix=f"scene_{start:04d}",
        )
        colmap_src = _find_colmap_sparse(chunk_export)
        if colmap_src is None:
            colmap_src = _find_colmap_sparse(chunk_export / "colmap")
        if colmap_src is not None:
            copy_colmap_sparse(colmap_src, sparse)
            log(f"DA3.cpp COLMAP -> {sparse}")

    if not ((sparse / "images.txt").is_file() or (sparse / "images.bin").is_file()):
        raise RuntimeError(
            "DA3.cpp did not produce COLMAP sparse model. "
            "Check da3-cli logs and ensure multi-view COLMAP export is supported."
        )

    preview_ply = out / "sparse_preview.ply"
    if export_sparse_pointcloud_ply(sparse, preview_ply):
        artifacts["geometry_sparse_ply"] = str(preview_ply)
        log(f"DA3.cpp sparse preview PLY -> {preview_ply.name}")
    else:
        # Fall back to CLI-exported point cloud
        ply_hits = sorted(export_root.glob("**/*.ply"), key=lambda p: p.stat().st_size, reverse=True)
        if ply_hits:
            shutil.copy2(ply_hits[0], preview_ply)
            artifacts["geometry_sparse_ply"] = str(preview_ply)

    glb_hits = sorted(export_root.glob("**/*.glb"), key=lambda p: p.stat().st_size, reverse=True)
    if glb_hits:
        glb_dest = out / "preview.glb"
        shutil.copy2(glb_hits[0], glb_dest)
        artifacts["geometry_glb"] = str(glb_dest)

    gs_hit = _find_gs_ply(export_root)
    if gs_hit and gs_hit.is_file():
        gs_dest = out / "gaussians.ply"
        shutil.copy2(gs_hit, gs_dest)
        artifacts["infer_gs_ply"] = str(gs_dest)

    if "infer_gs_python" in extras and "infer_gs_ply" not in artifacts:
        _maybe_infer_gs_python(
            workspace, images, log, chunk_size=chunk_size, artifacts=artifacts
        )

    meta = {
        "backend": "da3_cpp",
        "binary": str(binary),
        "model": str(model),
        "frames": len(img_list),
        "chunk_size": chunk_size,
        "extras": extras,
        "status": "ok",
        "colmap": str(sparse),
        "geometry_sparse_ply": artifacts.get("geometry_sparse_ply"),
        "geometry_glb": artifacts.get("geometry_glb"),
        "infer_gs_ply": artifacts.get("infer_gs_ply"),
    }
    meta_path = out / "preview.json"
    meta_path.write_text(json.dumps(meta, indent=2))
    artifacts["geometry_meta"] = str(meta_path)

    log("DA3.cpp done")
    return out, artifacts
