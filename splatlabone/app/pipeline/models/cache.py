"""Model registry and path resolution."""

from __future__ import annotations

import os
from pathlib import Path

MODEL_REGISTRY: dict[str, dict] = {
    "vggt_omega": {
        "hub": "modelscope",
        "model_id": "facebook/VGGT-Omega",
        "files": ["vggt_omega_1b_512.pt"],
        "hf_fallback": "facebook/VGGT-Omega",
    },
    "lingbot": {
        "hub": "modelscope",
        "model_id": "Robbyant/lingbot-map",
        "files": [],
        "hf_fallback": "robbyant/lingbot-map",
    },
    "da3": {
        "hub": "huggingface",
        "model_id": "depth-anything/DA3NESTED-GIANT-LARGE",
        "files": [],
        "hf_fallback": None,
    },
    "da3_cpp": {
        "hub": "huggingface",
        "model_id": "mudler/depth-anything.cpp-gguf",
        "filename": "depth-anything-base-q4_k.gguf",
        "files": ["depth-anything-base-q4_k.gguf"],
        "hf_fallback": None,
    },
    "mast3r": {
        "hub": "url",
        "url": "https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth",
        "filename": "MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth",
    },
}

CACHE_ROOT = Path(os.environ.get("SPLATLAB_MODEL_CACHE", "/data/splatlab/cache/models"))
MODELSCOPE_CACHE = Path(os.environ.get("MODELSCOPE_CACHE", "/data/splatlab/cache/modelscope"))


def model_dir(key: str) -> Path:
    return CACHE_ROOT / key


def resolve(key: str) -> Path | None:
    """Return local path if model files exist."""
    entry = MODEL_REGISTRY.get(key)
    if not entry:
        return None
    base = model_dir(key)
    if entry.get("hub") == "url":
        p = base / entry["filename"]
        return p if p.is_file() else None
    if entry.get("filename"):
        candidate = base / entry["filename"]
        if candidate.is_file():
            return candidate
        for hit in base.rglob(Path(entry["filename"]).name):
            if hit.is_file():
                return hit
    if entry.get("files"):
        for fname in entry["files"]:
            for root in (base, MODELSCOPE_CACHE / entry["model_id"].replace("/", "--")):
                candidate = root / fname
                if candidate.is_file():
                    return candidate
                for hit in root.rglob(fname):
                    return hit
    if base.is_dir() and any(base.iterdir()):
        return base
    ms = MODELSCOPE_CACHE / entry["model_id"].replace("/", "--")
    if ms.is_dir() and any(ms.iterdir()):
        return ms
    return None


def is_cached(key: str) -> bool:
    return resolve(key) is not None
