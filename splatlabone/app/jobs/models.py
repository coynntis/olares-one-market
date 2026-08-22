"""Pydantic job / pipeline configuration."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class RealtimeMode(str, Enum):
    none = "none"
    geometry_preview = "geometry_preview"
    progressive_splat = "progressive_splat"


class IngestType(str, Enum):
    images = "images"
    video = "video"
    colmap = "colmap"
    poses = "poses"


class SfMBackend(str, Enum):
    glomap = "glomap"
    glomap_calibrated = "glomap_calibrated"
    colmap = "colmap"
    fastmap = "fastmap"
    skip = "skip"


class GeometryBackend(str, Enum):
    none = "none"
    vggt_x = "vggt_x"
    vggt_omega = "vggt_omega"
    da3 = "da3"
    da3_cpp = "da3_cpp"
    instant_splat = "instant_splat"
    lingbot_map = "lingbot_map"
    gluemap = "gluemap"


class MatchingBackend(str, Enum):
    none = "none"
    sift = "sift"
    hloc = "hloc"
    dense_sfm = "dense_sfm"


class Representation(str, Enum):
    gs3d = "3dgs"
    gs2d = "2dgs"
    gs3dgut = "3dgut"


class Densification(str, Enum):
    default_adc = "default_adc"
    mcmc = "mcmc"


class PoseOpt(str, Enum):
    off = "off"
    joint = "joint"


class RepairBackend(str, Enum):
    none = "none"
    fixer = "fixer"


class StageOverrides(BaseModel):
    ingest: IngestType | None = None
    sfm: SfMBackend | None = None
    geometry: GeometryBackend | None = None
    matching: MatchingBackend | None = None
    representation: Representation | None = None
    densification: Densification | None = None
    pose_opt: PoseOpt | None = None
    repair: RepairBackend | None = None
    iterations: int | None = Field(None, ge=1000, le=100000)
    checkpoint_interval: int | None = Field(None, ge=500, le=20000)
    max_frames: int | None = Field(None, ge=3, le=2000)
    fps: float | None = Field(None, ge=0.1, le=60)
    lambda_dist: float | None = None
    mcmc_noise: float | None = None
    chunk_size: int | None = None
    extra: dict[str, Any] = Field(default_factory=dict)


class JobConfig(BaseModel):
    dataset_id: str
    preset: str = "quality"
    overrides: StageOverrides = Field(default_factory=StageOverrides)
    realtime_mode: RealtimeMode = RealtimeMode.none
    name: str = ""


class JobState(str, Enum):
    queued = "queued"
    running = "running"
    completed = "completed"
    failed = "failed"
    cancelled = "cancelled"


class JobSnapshot(BaseModel):
    id: str
    state: JobState
    preset: str
    dataset_id: str
    stage: str = ""
    progress: float = 0.0
    realtime_mode: RealtimeMode = RealtimeMode.none
    error: str = ""
    metrics: dict[str, Any] = Field(default_factory=dict)
    artifacts: dict[str, str] = Field(default_factory=dict)
    created_at: float = 0.0
    updated_at: float = 0.0
    log_lines: int = 0
