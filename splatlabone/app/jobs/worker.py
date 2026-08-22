"""Background job queue with SSE event fan-out."""

from __future__ import annotations

import asyncio
import json
import logging
import queue
import threading
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncIterator

from jobs.models import JobConfig, JobSnapshot, JobState, RealtimeMode
from pipeline.runner import PipelineRunner

log = logging.getLogger("splatlab.jobs")


@dataclass
class Job:
    id: str
    config: JobConfig
    state: JobState = JobState.queued
    stage: str = ""
    progress: float = 0.0
    error: str = ""
    metrics: dict[str, Any] = field(default_factory=dict)
    artifacts: dict[str, str] = field(default_factory=dict)
    logs: list[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    _cancel: threading.Event = field(default_factory=threading.Event)
    _subscribers: list[queue.Queue] = field(default_factory=list)

    def append_log(self, line: str) -> None:
        ts = time.strftime("%H:%M:%S")
        msg = f"{ts} | {line}"
        self.logs.append(msg)
        self.updated_at = time.time()
        self._broadcast({"type": "log", "line": msg})

    def set_stage(self, stage: str, progress: float | None = None) -> None:
        self.stage = stage
        if progress is not None:
            self.progress = progress
        self.updated_at = time.time()
        self._broadcast({"type": "stage_start", "stage": stage, "progress": self.progress})

    def _broadcast(self, event: dict[str, Any]) -> None:
        event = {**event, "job_id": self.id, "ts": time.time()}
        dead: list[queue.Queue] = []
        for q in self._subscribers:
            try:
                q.put_nowait(event)
            except Exception:
                dead.append(q)
        for q in dead:
            if q in self._subscribers:
                self._subscribers.remove(q)

    def emit(self, event_type: str, **payload: Any) -> None:
        self._broadcast({"type": event_type, **payload})
        self.updated_at = time.time()

    def snapshot(self) -> JobSnapshot:
        return JobSnapshot(
            id=self.id,
            state=self.state,
            preset=self.config.preset,
            dataset_id=self.config.dataset_id,
            stage=self.stage,
            progress=self.progress,
            realtime_mode=self.config.realtime_mode,
            error=self.error,
            metrics=dict(self.metrics),
            artifacts=dict(self.artifacts),
            created_at=self.created_at,
            updated_at=self.updated_at,
            log_lines=len(self.logs),
        )


class JobManager:
    def __init__(
        self,
        uploads_dir: Path,
        workspaces_dir: Path,
        outputs_dir: Path,
        models_dir: Path,
        presets_dir: Path,
    ) -> None:
        self.uploads_dir = uploads_dir
        self.workspaces_dir = workspaces_dir
        self.outputs_dir = outputs_dir
        self.models_dir = models_dir
        self.presets_dir = presets_dir
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()
        self._work_q: queue.Queue[str] = queue.Queue()
        self._current: str | None = None
        self._worker_thread: threading.Thread | None = None
        self._live_sessions: dict[str, dict[str, Any]] = {}

    def ensure_dirs(self) -> None:
        for d in (self.uploads_dir, self.workspaces_dir, self.outputs_dir):
            d.mkdir(parents=True, exist_ok=True)

    def _start_queue_worker(self) -> None:
        if self._worker_thread and self._worker_thread.is_alive():
            return
        self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker_thread.start()

    def _worker_loop(self) -> None:
        while True:
            job_id = self._work_q.get()
            try:
                self._run_job(job_id)
            except Exception as exc:
                log.exception("worker error job=%s", job_id)
                job = self._jobs.get(job_id)
                if job:
                    job.state = JobState.failed
                    job.error = str(exc)
                    job.emit("error", message=str(exc))
            finally:
                with self._lock:
                    if self._current == job_id:
                        self._current = None
                self._work_q.task_done()

    def _run_job(self, job_id: str) -> None:
        job = self._jobs.get(job_id)
        if not job or job.state == JobState.cancelled:
            return
        with self._lock:
            self._current = job_id
        job.state = JobState.running
        job.emit("job_start")
        runner = PipelineRunner(self, job)
        try:
            runner.run()
            if job._cancel.is_set():
                job.state = JobState.cancelled
                job.emit("cancelled")
            else:
                job.state = JobState.completed
                job.progress = 1.0
                job.emit("job_complete", artifacts=job.artifacts)
        except Exception as exc:
            job.state = JobState.failed
            job.error = str(exc)
            job.append_log(f"ERROR: {exc}")
            job.emit("error", message=str(exc))
            raise

    def create_job(self, config: JobConfig) -> Job:
        self.ensure_dirs()
        job_id = uuid.uuid4().hex[:12]
        job = Job(id=job_id, config=config)
        with self._lock:
            self._jobs[job_id] = job
        self._work_q.put(job_id)
        return job

    def get_job(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id)

    def list_jobs(self) -> list[JobSnapshot]:
        return sorted(
            (j.snapshot() for j in self._jobs.values()),
            key=lambda s: s.created_at,
            reverse=True,
        )

    def cancel_job(self, job_id: str) -> Job:
        job = self._jobs.get(job_id)
        if not job:
            raise KeyError(job_id)
        if job.state in (JobState.completed, JobState.failed, JobState.cancelled):
            raise RuntimeError(f"Job {job_id} already {job.state}")
        job._cancel.set()
        if job.state == JobState.queued:
            job.state = JobState.cancelled
            job.emit("cancelled")
        return job

    def read_logs(self, job_id: str, offset: int = 0, limit: int = 200) -> dict[str, Any]:
        job = self._jobs.get(job_id)
        if not job:
            raise KeyError(job_id)
        lines = job.logs[offset : offset + limit]
        return {
            "job_id": job_id,
            "offset": offset,
            "lines": lines,
            "total": len(job.logs),
            "has_more": offset + len(lines) < len(job.logs),
        }

    def subscribe(self, job_id: str) -> queue.Queue:
        job = self._jobs.get(job_id)
        if not job:
            raise KeyError(job_id)
        q: queue.Queue = queue.Queue(maxsize=500)
        job._subscribers.append(q)
        return q

    async def stream_events(self, job_id: str) -> AsyncIterator[str]:
        job = self._jobs.get(job_id)
        if not job:
            raise KeyError(job_id)
        q = self.subscribe(job_id)
        snap = job.snapshot()
        yield f"data: {json.dumps({'type': 'snapshot', **snap.model_dump()})}\n\n"
        while True:
            try:
                event = await asyncio.get_event_loop().run_in_executor(
                    None, lambda: q.get(timeout=30)
                )
                yield f"data: {json.dumps(event, default=str)}\n\n"
                if event.get("type") in ("job_complete", "error", "cancelled"):
                    break
            except queue.Empty:
                yield ": keepalive\n\n"
                if job.state in (JobState.completed, JobState.failed, JobState.cancelled):
                    break

    def list_datasets(self) -> list[dict[str, Any]]:
        out = []
        if not self.uploads_dir.is_dir():
            return out
        for p in sorted(self.uploads_dir.iterdir()):
            if not p.is_dir():
                continue
            meta_path = p / "meta.json"
            meta: dict[str, Any] = {}
            if meta_path.is_file():
                try:
                    meta = json.loads(meta_path.read_text())
                except Exception:
                    pass
            images = list((p / "images").glob("*")) if (p / "images").is_dir() else []
            out.append(
                {
                    "dataset_id": p.name,
                    "path": str(p),
                    "type": meta.get("type", "unknown"),
                    "frame_count": len(images) or meta.get("frame_count", 0),
                    "meta": meta,
                }
            )
        return out

    def dataset_path(self, dataset_id: str) -> Path:
        p = self.uploads_dir / dataset_id
        if not p.is_dir():
            raise FileNotFoundError(f"Dataset not found: {dataset_id}")
        return p

    def workspace_path(self, job_id: str) -> Path:
        p = self.workspaces_dir / job_id
        p.mkdir(parents=True, exist_ok=True)
        return p

    def output_path(self, job_id: str) -> Path:
        p = self.outputs_dir / job_id
        p.mkdir(parents=True, exist_ok=True)
        return p

    def list_scenes(self) -> list[dict[str, Any]]:
        scenes = []
        for snap in self.list_jobs():
            if snap.state != JobState.completed:
                continue
            scenes.append(
                {
                    "job_id": snap.id,
                    "dataset_id": snap.dataset_id,
                    "preset": snap.preset,
                    "artifacts": snap.artifacts,
                }
            )
        return scenes

    def queue_status(self) -> dict[str, Any]:
        with self._lock:
            current = self._current
        pending = self._work_q.qsize()
        return {"running": current, "queued": pending}

    def create_live_session(self, mode: str = "geometry_preview") -> dict[str, Any]:
        sid = uuid.uuid4().hex[:12]
        self._live_sessions[sid] = {
            "id": sid,
            "mode": mode,
            "created_at": time.time(),
            "frames": 0,
        }
        return self._live_sessions[sid]

    def get_live_session(self, session_id: str) -> dict[str, Any] | None:
        return self._live_sessions.get(session_id)

    def gpu_info(self) -> dict[str, Any]:
        try:
            import torch

            cuda = torch.cuda.is_available()
            name = torch.cuda.get_device_name(0) if cuda else ""
            mem = torch.cuda.get_device_properties(0).total_memory if cuda else 0
            return {"cuda": cuda, "device": name, "vram_bytes": mem}
        except Exception as exc:
            return {"cuda": False, "error": str(exc)}
