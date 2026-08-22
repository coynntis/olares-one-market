"""Geometry front-end dispatch."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from pipeline.geometry.da3 import run_da3 as _run_da3
from pipeline.geometry.da3_cpp import run_da3_cpp as _run_da3_cpp
from pipeline.geometry.gluemap import run_gluemap as _run_gluemap
from pipeline.geometry.instant_splat import run_instant_splat as _run_instant_splat
from pipeline.geometry.lingbot_map import run_lingbot_map as _run_lingbot_map
from pipeline.geometry.vggt_omega import run_vggt_omega as _run_vggt_omega

# Backward-compat alias
run_vggt_x = _run_vggt_omega


def run_geometry(
    backend: str,
    workspace: Path,
    images: Path,
    log: Callable[[str], None],
    *,
    chunk_size: int = 8,
    extras: list[str] | None = None,
) -> tuple[Path, dict[str, str]]:
    extras = extras or []
    if backend in ("vggt_omega", "vggt_x"):
        return _run_vggt_omega(workspace, images, log, chunk_size=chunk_size)
    if backend == "da3":
        return _run_da3(workspace, images, log, chunk_size=chunk_size)
    if backend == "lingbot_map":
        return _run_lingbot_map(workspace, images, log, chunk_size=chunk_size)
    if backend == "instant_splat":
        return _run_instant_splat(workspace, images, log)
    if backend == "da3_cpp":
        return _run_da3_cpp(workspace, images, log, chunk_size=chunk_size, extras=extras)
    if backend == "gluemap":
        return _run_gluemap(workspace, images, log, chunk_size=chunk_size, extras=extras)
    raise ValueError(f"unknown geometry backend: {backend}")


def run_vggt_x_legacy(workspace: Path, images: Path, log: Callable[[str], None]) -> tuple[Path, dict[str, str]]:
    return _run_vggt_omega(workspace, images, log)


run_da3 = _run_da3
run_da3_cpp = _run_da3_cpp
run_gluemap = _run_gluemap
run_instant_splat = _run_instant_splat
run_lingbot_map = _run_lingbot_map
run_vggt_omega = _run_vggt_omega
