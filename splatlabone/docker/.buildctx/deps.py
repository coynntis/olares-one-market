"""SplatLab One — paths, settings, JobManager singleton."""

from __future__ import annotations

import os
import uuid
from pathlib import Path

from jobs.worker import JobManager

DATA_DIR = Path(os.environ.get("DATA_DIR", "/data"))
UPLOADS_DIR = Path(os.environ.get("UPLOADS_DIR", str(DATA_DIR / "uploads")))
WORKSPACES_DIR = Path(os.environ.get("WORKSPACES_DIR", str(DATA_DIR / "workspaces")))
OUTPUTS_DIR = Path(os.environ.get("OUTPUTS_DIR", str(DATA_DIR / "outputs")))
MODELS_DIR = Path(os.environ.get("MODELS_DIR", "/models"))
PRESETS_DIR = Path(os.environ.get("PRESETS_DIR", "/app/presets"))
STATIC_DIR = Path(os.environ.get("STATIC_DIR", "/app/static"))
SERVER_PORT = int(os.environ.get("SERVER_PORT", "7860"))
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")

manager = JobManager(
    uploads_dir=UPLOADS_DIR,
    workspaces_dir=WORKSPACES_DIR,
    outputs_dir=OUTPUTS_DIR,
    models_dir=MODELS_DIR,
    presets_dir=PRESETS_DIR,
)


def new_id(prefix: str = "") -> str:
    uid = uuid.uuid4().hex[:12]
    return f"{prefix}{uid}" if prefix else uid


def ensure_dirs() -> None:
    for d in (UPLOADS_DIR, WORKSPACES_DIR, OUTPUTS_DIR, MODELS_DIR):
        d.mkdir(parents=True, exist_ok=True)
