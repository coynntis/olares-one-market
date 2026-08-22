"""Matching front-ends (before / with SfM)."""

from __future__ import annotations

from pathlib import Path
from typing import Callable


def run_matching(
    backend: str,
    workspace: Path,
    images: Path,
    log: Callable[[str], None],
) -> None:
    """Populate COLMAP database with features/matches when not using default SIFT path."""
    if backend in ("none", "sift", "", None):
        # SIFT is handled inside glomap/colmap feature_extractor
        return
    if backend == "hloc":
        from pipeline.matching_hloc import run_hloc_matching

        run_hloc_matching(workspace, images, log)
        return
    if backend == "dense_sfm":
        from pipeline.matching_dense_sfm import run_dense_sfm_matching

        run_dense_sfm_matching(workspace, images, log)
        return
    raise ValueError(f"unknown matching backend: {backend}")
