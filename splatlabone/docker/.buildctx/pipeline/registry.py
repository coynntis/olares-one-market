"""Preset registry and stage matrix."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

BUILTIN_PRESETS: dict[str, dict[str, Any]] = {
    "quality": {
        "name": "quality",
        "title": "Quality (GLOMAP → 3DGS)",
        "summary": "Dense photos, best NVS. COLMAP-free matcher via GLOMAP.",
        "license": "Apache-2.0",
        "vram_gb": 18,
        "time_est_min": 45,
        "stages": {
            "sfm": "glomap",
            "geometry": "none",
            "representation": "3dgs",
            "densification": "default_adc",
            "pose_opt": "off",
            "repair": "none",
        },
        "iterations": 30000,
        "checkpoint_interval": 7000,
    },
    "robust": {
        "name": "robust",
        "title": "Robust (VGGT-X → 2DGS + MCMC)",
        "summary": "COLMAP-free feed-forward geometry, noisy init friendly.",
        "license": "Apache-2.0",
        "vram_gb": 20,
        "time_est_min": 60,
        "stages": {
            "sfm": "skip",
            "geometry": "vggt_x",
            "representation": "2dgs",
            "densification": "mcmc",
            "pose_opt": "joint",
            "repair": "none",
        },
        "iterations": 30000,
        "checkpoint_interval": 7000,
        "lambda_dist": 0.01,
        "mcmc_noise": 0.1,
    },
    "fast": {
        "name": "fast",
        "title": "Fast preview (DA3 infer_gs)",
        "summary": "Quick splat preview. DA3-Giant is CC BY-NC — commercial warning.",
        "license": "CC-BY-NC-4.0",
        "vram_gb": 16,
        "time_est_min": 10,
        "stages": {
            "sfm": "skip",
            "geometry": "da3",
            "representation": "3dgs",
            "densification": "default_adc",
            "pose_opt": "off",
            "repair": "none",
        },
        "iterations": 7000,
        "checkpoint_interval": 2000,
    },
    "sparse": {
        "name": "sparse",
        "title": "Sparse (InstantSplat)",
        "summary": "3–12 images, MASt3R-based init.",
        "license": "Apache-2.0",
        "vram_gb": 14,
        "time_est_min": 15,
        "stages": {
            "sfm": "skip",
            "geometry": "instant_splat",
            "representation": "3dgs",
            "densification": "default_adc",
            "pose_opt": "off",
            "repair": "none",
        },
        "iterations": 15000,
        "checkpoint_interval": 5000,
    },
    "stream": {
        "name": "stream",
        "title": "Stream geometry",
        "summary": "Video chunks → geometry preview, optional full train.",
        "license": "Apache-2.0",
        "vram_gb": 12,
        "time_est_min": 5,
        "stages": {
            "sfm": "skip",
            "geometry": "da3",
            "representation": "3dgs",
            "densification": "default_adc",
            "pose_opt": "off",
            "repair": "none",
        },
        "chunk_size": 16,
        "iterations": 7000,
        "checkpoint_interval": 2000,
    },
}

GUIDE_SECTIONS: dict[str, str] = {
    "overview": (
        "SplatLab pipelines: Ingest → SfM (GLOMAP/COLMAP) → optional geometry (VGGT-X, DA3) "
        "→ train (3DGS/2DGS+MCMC) → export .splat/.ply. Olares One: RTX 5090M 24GB, cu128."
    ),
    "glomap": "GLOMAP: fast global mapper, best NVS quality for dense photo sets. CPU matching + GPU BA.",
    "vggt_x": "VGGT-X: memory-efficient VGGT variant for COLMAP-free pose + depth.",
    "2dgs": "2DGS: surfel representation, better geometry than 3DGS on hard scenes.",
    "mcmc": "MCMC densification: handles noisy feed-forward initialization.",
    "da3": "Depth Anything 3 infer_gs: fast preview. Giant model NC license.",
    "realtime": (
        "geometry_preview: WS chunks, coarse point cloud. "
        "progressive_splat: gsplat checkpoints every N iters."
    ),
    "docker_build": (
        "Build image via dockerbuilderone MCP: zip splatlabone/docker/, upload_project, "
        "start_build ghcr.io/coynntis/splatlabone:TAG"
    ),
    "mcp": "SplatLab MCP at /mcp/mcp: ingest_*, create_job, get_job, get_scene_urls. Not image builds.",
}


def load_preset(name: str, presets_dir: Path | None = None) -> dict[str, Any]:
    if presets_dir and presets_dir.is_dir():
        for ext in (".yaml", ".yml"):
            p = presets_dir / f"{name}{ext}"
            if p.is_file():
                data = yaml.safe_load(p.read_text()) or {}
                return {**BUILTIN_PRESETS.get(name, {}), **data}
    if name in BUILTIN_PRESETS:
        return dict(BUILTIN_PRESETS[name])
    raise KeyError(f"Unknown preset: {name}")


def list_presets(presets_dir: Path | None = None) -> list[dict[str, Any]]:
    names = set(BUILTIN_PRESETS)
    if presets_dir and presets_dir.is_dir():
        for p in presets_dir.glob("*.yaml"):
            names.add(p.stem)
        for p in presets_dir.glob("*.yml"):
            names.add(p.stem)
    out = []
    for n in sorted(names):
        try:
            preset = load_preset(n, presets_dir)
            out.append(
                {
                    "name": n,
                    "title": preset.get("title", n),
                    "summary": preset.get("summary", ""),
                    "license": preset.get("license", ""),
                    "vram_gb": preset.get("vram_gb"),
                    "time_est_min": preset.get("time_est_min"),
                    "stages": preset.get("stages", {}),
                }
            )
        except Exception:
            continue
    return out


def merge_config(preset_name: str, overrides: dict[str, Any], presets_dir: Path | None) -> dict[str, Any]:
    base = load_preset(preset_name, presets_dir)
    stages = dict(base.get("stages", {}))
    for k, v in (overrides or {}).items():
        if k in stages or k in (
            "sfm",
            "geometry",
            "representation",
            "densification",
            "pose_opt",
            "repair",
        ):
            if v is not None:
                stages[k] = v
        elif v is not None:
            base[k] = v
    base["stages"] = stages
    return base


def guide_section(anchor: str) -> str:
    return GUIDE_SECTIONS.get(anchor, f"No guide section for: {anchor}")


get_guide_section = guide_section
