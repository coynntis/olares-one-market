"""Preset registry and stage matrix."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

BUILTIN_PRESETS: dict[str, dict[str, Any]] = {
    # --- live ---
    "quality": {
        "name": "quality",
        "title": "Quality (COLMAP global_mapper → 3DGS)",
        "summary": "Dense photos, best NVS. COLMAP 4 global_mapper + gsplat 3DGS.",
        "license": "Apache-2.0",
        "status": "live",
        "vram_gb": 18,
        "time_est_min": 45,
        "stages": {
            "sfm": "glomap",
            "geometry": "none",
            "matching": "sift",
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
        "title": "Robust (VGGT-Omega → 2DGS + MCMC)",
        "summary": "COLMAP-free feed-forward geometry via VGGT-Omega, noisy init friendly.",
        "license": "Apache-2.0",
        "status": "live",
        "vram_gb": 20,
        "time_est_min": 60,
        "stages": {
            "sfm": "skip",
            "geometry": "vggt_omega",
            "matching": "none",
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
        "status": "live",
        "vram_gb": 16,
        "time_est_min": 10,
        "stages": {
            "sfm": "skip",
            "geometry": "da3",
            "matching": "none",
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
        "status": "live",
        "vram_gb": 14,
        "time_est_min": 15,
        "stages": {
            "sfm": "skip",
            "geometry": "instant_splat",
            "matching": "none",
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
        "title": "Stream geometry (LingBot-Map)",
        "summary": "Video chunks → geometry map preview, optional full train.",
        "license": "Apache-2.0",
        "status": "live",
        "vram_gb": 12,
        "time_est_min": 5,
        "stages": {
            "sfm": "skip",
            "geometry": "lingbot_map",
            "matching": "none",
            "representation": "3dgs",
            "densification": "default_adc",
            "pose_opt": "off",
            "repair": "none",
        },
        "chunk_size": 16,
        "iterations": 7000,
        "checkpoint_interval": 2000,
    },
    # --- experimental / proposed ---
    "quality_hloc": {
        "name": "quality_hloc",
        "title": "Quality + HLoc (experimental)",
        "summary": "SuperPoint + LightGlue matching → GLOMAP. Better on hard texture.",
        "license": "Apache-2.0",
        "status": "experimental",
        "vram_gb": 20,
        "time_est_min": 55,
        "stages": {
            "sfm": "glomap",
            "geometry": "none",
            "matching": "hloc",
            "representation": "3dgs",
            "densification": "default_adc",
            "pose_opt": "off",
            "repair": "none",
        },
        "iterations": 30000,
        "checkpoint_interval": 7000,
    },
    "quality_calibrated": {
        "name": "quality_calibrated",
        "title": "Quality + view graph calibrator",
        "summary": "COLMAP view_graph_calibrator then global_mapper. Use when EXIF focals unreliable.",
        "license": "Apache-2.0",
        "status": "live",
        "vram_gb": 18,
        "time_est_min": 50,
        "stages": {
            "sfm": "glomap_calibrated",
            "geometry": "none",
            "matching": "sift",
            "representation": "3dgs",
            "densification": "default_adc",
            "pose_opt": "off",
            "repair": "none",
        },
        "iterations": 30000,
        "checkpoint_interval": 7000,
    },
    "scale_fastmap": {
        "name": "scale_fastmap",
        "title": "Scale FastMap (experimental)",
        "summary": "GPU first-order global SfM for large dense coverage (pals-ttic/fastmap).",
        "license": "MIT",
        "status": "experimental",
        "vram_gb": 14,
        "time_est_min": 25,
        "stages": {
            "sfm": "fastmap",
            "geometry": "none",
            "matching": "sift",
            "representation": "3dgs",
            "densification": "default_adc",
            "pose_opt": "off",
            "repair": "none",
        },
        "iterations": 20000,
        "checkpoint_interval": 5000,
    },
    "robust_gluemap": {
        "name": "robust_gluemap",
        "title": "Robust GlueMap (experimental)",
        "summary": "Global SfM + feedforward (VGGT/Pi3 via GlueMap). Hard scenes + global consistency.",
        "license": "Apache-2.0",
        "status": "experimental",
        "vram_gb": 22,
        "time_est_min": 90,
        "stages": {
            "sfm": "skip",
            "geometry": "gluemap",
            "matching": "none",
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
    "fast_dacpp": {
        "name": "fast_dacpp",
        "title": "Fast DA3.cpp (experimental)",
        "summary": "Low-VRAM ggml DA3.cpp → COLMAP poses. No infer_gs by default.",
        "license": "Apache-2.0",
        "status": "experimental",
        "vram_gb": 8,
        "time_est_min": 8,
        "stages": {
            "sfm": "skip",
            "geometry": "da3_cpp",
            "matching": "none",
            "representation": "3dgs",
            "densification": "default_adc",
            "pose_opt": "off",
            "repair": "none",
        },
        "iterations": 7000,
        "checkpoint_interval": 2000,
    },
    "fast_hybrid": {
        "name": "fast_hybrid",
        "title": "Fast hybrid dacpp + infer_gs (experimental)",
        "summary": "DA3.cpp poses + optional Python DA3 infer_gs for SuperSplat preview.",
        "license": "CC-BY-NC-4.0",
        "status": "experimental",
        "vram_gb": 14,
        "time_est_min": 12,
        "stages": {
            "sfm": "skip",
            "geometry": "da3_cpp",
            "matching": "none",
            "representation": "3dgs",
            "densification": "default_adc",
            "pose_opt": "off",
            "repair": "none",
            "extras": ["infer_gs_python"],
        },
        "iterations": 7000,
        "checkpoint_interval": 2000,
    },
    "dense_tracks": {
        "name": "dense_tracks",
        "title": "Dense-SfM tracks (experimental)",
        "summary": "LoFTR dense matching (hloc) + GLOMAP. Research density; high cost.",
        "license": "Apache-2.0",
        "status": "experimental",
        "vram_gb": 22,
        "time_est_min": 120,
        "stages": {
            "sfm": "glomap",
            "geometry": "none",
            "matching": "dense_sfm",
            "representation": "3dgs",
            "densification": "default_adc",
            "pose_opt": "off",
            "repair": "none",
        },
        "iterations": 30000,
        "checkpoint_interval": 7000,
    },
}

GUIDE_SECTIONS: dict[str, str] = {
    "overview": (
        "SplatLab: Import → SfM (COLMAP global_mapper) or feed-forward geometry "
        "(VGGT-Omega, DA3, LingBot-Map, InstantSplat) → COLMAP sparse/0 → gsplat train "
        "(3DGS or 2DGS+MCMC) → export .splat/.ply/.ply_compressed. COLMAP is the hub; "
        "geometry backends produce poses for gsplat --data-type colmap."
    ),
    "architecture": (
        "Helm: splatlabonesrv GPU server :7860 (admin) + splatlabone nginx client :8080. "
        "FastAPI + job queue + pipeline + MCP /mcp/mcp. Data: uploads/, workspaces/, outputs/, "
        "splatlab/cache/ under DATA_DIR=/data."
    ),
    "colmap": (
        "All SfM and geometry backends write workspace/sparse/0 (cameras, images, points3D). "
        "gsplat simple_trainer only reads --data-type colmap. DA3 also exports infer_gs.ply "
        "for SuperSplat preview separate from COLMAP refine path."
    ),
    "gsplat": (
        "nerfstudio-project/gsplat at /opt/gsplat: simple_trainer.py (default=3DGS, mcmc=2DGS+MCMC), "
        "simple_viewer.py (Viser), export_splats for ply/splat/ply_compressed."
    ),
    "artifacts": (
        "Stage keys: sfm_sparse_ply, geometry_sparse_ply, infer_gs_ply (DA3), geometry_glb, "
        "geometry_meta, splat/ply/ckpt after train. API: GET .../artifacts, .../artifact/{key}, "
        ".../infer_gs. SSE: stage_artifacts, geometry_preview."
    ),
    "libraries": (
        "Core: PyTorch cu128, gsplat, COLMAP 4, FastAPI, viser/nerfview, httpx/websockets. "
        "Geometry /opt: vggt-omega, da3, lingbot-map, instantsplat+dust3r+mast3r, "
        "hloc, fastmap, gluemap, depth-anything-cpp. "
        "Weights via ModelScope/HF prefetch initContainer, not image bake."
    ),
    "ingest": (
        "POST /api/v1/ingest/images|images-zip|video|colmap → uploads/ds_*/images. "
        "COLMAP zip includes sparse/0 to skip SfM. MCP: ingest_images_zip, ingest_video_base64, ingest_colmap_zip."
    ),
    "glomap": (
        "COLMAP 4 global_mapper (GLOMAP successor) or glomap binary: feature_extractor → "
        "exhaustive_matcher → global_mapper. Writes sparse/0 + sfm_sparse_ply. quality preset. "
        "quality_calibrated adds view_graph_calibrator first."
    ),
    "presets": (
        "Live: quality, quality_calibrated, robust, fast, sparse, stream. "
        "Experimental (implemented, need image deps): quality_hloc (HLoc), scale_fastmap, "
        "robust_gluemap, fast_dacpp / fast_hybrid (da3-cli + GGUF), dense_tracks (LoFTR)."
    ),
    "hloc": (
        "cvg/Hierarchical-Localization at /opt/hloc: SuperPoint + LightGlue → COLMAP database.db. "
        "quality_hloc preset; SfM skips feature_extractor when .matching_hloc_done present."
    ),
    "fastmap": (
        "pals-ttic/fastmap: COLMAP match DB → run.py --headless → sparse/0. scale_fastmap preset."
    ),
    "gluemap": (
        "colmap/gluemap: gluemap-demo feedforward+global SfM → COLMAP. robust_gluemap; "
        "needs checkpoints under /opt/gluemap/checkpoints."
    ),
    "da3_cpp": (
        "mudler/depth-anything.cpp da3-cli + GGUF. fast_dacpp / fast_hybrid (extras infer_gs_python)."
    ),
    "dense_sfm": (
        "hloc match_dense LoFTR → COLMAP DB (.matching_dense_done). dense_tracks preset."
    ),
    "vggt_omega": (
        "Meta VGGT-Omega: feed-forward pose_enc + depth unprojection → write_colmap_model. "
        "robust preset with 2DGS+MCMC+joint pose opt. ModelScope weights."
    ),
    "vggt_x": "Alias for vggt_omega.",
    "lingbot_map": (
        "LingBot-Map GCTStream: streaming video geometry → COLMAP + geometry_sparse_ply. "
        "stream preset, chunk_size 16."
    ),
    "instant_splat": (
        "NVlabs InstantSplat init_geo.py + MASt3R: 3-12 images → COLMAP sparse/0. sparse preset."
    ),
    "2dgs": "2DGS surfels via gsplat mcmc subcommand + --with_eval3d. Better geometry on hard scenes.",
    "mcmc": "MCMC densification (gsplat mcmc): handles noisy feed-forward init. Pair with VGGT-Omega.",
    "da3": (
        "Depth Anything 3: inference with colmap-gs_ply-glb export. COLMAP for gsplat train + "
        "infer_gs.ply feed-forward Gaussians for SuperSplat. Giant model CC BY-NC. fast preset."
    ),
    "realtime": (
        "none: full pipeline. geometry_preview: geometry only, skip train, infer_gs/sparse artifacts. "
        "progressive_splat: live Viser + checkpoint_ready SSE during train."
    ),
    "viewer": (
        "Viewer has Map and Splats tabs. Map: SfM/geometry sparse PLY → Three.js (#map). "
        "Splats: infer_gs → SuperSplat; trained ckpt → POST /viewer/start Viser (#splats). "
        "Viser is splat-only. Deep links: /viewer.html?job=…#map|#splats."
    ),
    "docker_build": (
        "Image ghcr.io/coynntis/splatlabone:TAG: PyTorch cu128 + gsplat + COLMAP copy + "
        "fetch_backends.py (/opt geometry repos). Build via dockerbuilderone MCP. "
        "Weights: initContainer download_models.py. App hotfix: ConfigMap inject."
    ),
    "mcp": (
        "SplatLab MCP /mcp/mcp: health_check, list_presets, ingest_*, create_job, get_job, "
        "subscribe_job_events, list_scenes, get_scene_urls, get_guide_section. "
        "Image builds: dockerbuilderone MCP."
    ),
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
                    "status": preset.get("status", "live"),
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
    stage_keys = (
        "sfm",
        "geometry",
        "matching",
        "representation",
        "densification",
        "pose_opt",
        "repair",
        "extras",
    )
    for k, v in (overrides or {}).items():
        if k in stages or k in stage_keys:
            if v is not None:
                stages[k] = v
        elif v is not None:
            base[k] = v
    base["stages"] = stages
    return base


def guide_section(anchor: str) -> str:
    return GUIDE_SECTIONS.get(anchor, f"No guide section for: {anchor}")


get_guide_section = guide_section
