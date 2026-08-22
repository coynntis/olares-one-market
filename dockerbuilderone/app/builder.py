"""Image build via Kaniko (default) or docker CLI when a daemon is available."""

from __future__ import annotations

import asyncio
import base64
import json
import os
import re
import shutil
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import AsyncIterator

IGNORED_TOP_NAMES = frozenset(
    {"__MACOSX", ".DS_Store", "Thumbs.db", "desktop.ini", ".git"}
)
DOCKERFILE_NAMES = ("Dockerfile", "dockerfile", "Dockerfile.dockerfile")


def _ghcr_username() -> str:
    user = os.environ.get("GHCR_USER", "").strip()
    if not user:
        raise RuntimeError(
            "GitHub username is not set. Set GHCR_USER in Application settings "
            "(mapped from your GitHub username), then restart the app."
        )
    return user


class BuildCancelled(Exception):
    """Raised when a build is cancelled by the user."""


def _is_ignored_top(name: str) -> bool:
    return name in IGNORED_TOP_NAMES or name.startswith(".")


def find_dockerfile(root: Path, max_depth: int = 6) -> Path | None:
    """Find Dockerfile under root (case variants, shallow tree)."""
    if not root.is_dir():
        return None
    for name in DOCKERFILE_NAMES:
        direct = root / name
        if direct.is_file():
            return direct
    best: Path | None = None
    best_depth = max_depth + 1
    for path in root.rglob("*"):
        if not path.is_file() or path.name not in DOCKERFILE_NAMES:
            continue
        try:
            depth = len(path.relative_to(root).parts)
        except ValueError:
            continue
        if depth < best_depth:
            best = path
            best_depth = depth
    return best


def dockerfile_relpath(project_dir: Path) -> str | None:
    df = find_dockerfile(project_dir)
    if not df:
        return None
    return df.relative_to(project_dir).as_posix()


class BuildState(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    CANCELLED = "cancelled"


TERMINAL_STATES = frozenset(
    {BuildState.SUCCESS, BuildState.FAILED, BuildState.CANCELLED}
)
ACTIVE_STATES = frozenset({BuildState.QUEUED, BuildState.RUNNING})

DEFAULT_LOG_PAGE = 500
MAX_LOG_PAGE = 2000
MEMORY_LOG_CAP = 4000
FAILURE_TAIL_LINES = 80
STREAM_CATCHUP_BATCH = 200
STREAM_SKIP_REPLAY_THRESHOLD = 3000
STREAM_TAIL_ON_SKIP = 500

# Dropped when verbose mode is off (Kaniko still at --verbosity=info).
_KANIKO_DEBUG_ONLY = re.compile(
    r"(?i)(^DEBU\[|^DEBU\s|Hash components for file|Taking snapshot of full filesystem|"
    r"snapshotting filesystem|using snapshotter|Storing source state|"
    r"Unmatched cache manifest|No cached layer found for|Checking for cached layer)"
)

EXIT_CODE_HINTS: dict[int, str] = {
    127: "command not found — missing binary in PATH, or a script (e.g. install.sh) does not exist",
    126: "command invoked but not executable",
    125: "command not runnable",
    1: "general error (see log tail below)",
}
# asyncio subprocess stdout defaults to 64 KiB per readline(); kaniko --verbosity=debug
# can emit much longer single lines.
SUBPROCESS_STREAM_LIMIT = max(
    65536,
    int(os.environ.get("BUILD_SUBPROCESS_STREAM_LIMIT", str(8 * 1024 * 1024))),
)
MAX_LOG_LINE_CHARS = max(
    4096,
    int(os.environ.get("BUILD_MAX_LOG_LINE_CHARS", str(512 * 1024))),
)


async def _read_subprocess_line(reader: asyncio.StreamReader) -> bytes:
    """Read one newline-terminated line; tolerate lines longer than stream limit."""
    sep = b"\n"
    parts: list[bytes] = []
    while True:
        try:
            parts.append(await reader.readuntil(sep))
            break
        except asyncio.IncompleteReadError as exc:
            if exc.partial:
                parts.append(exc.partial)
            break
        except asyncio.LimitOverrunError as exc:
            parts.append(await reader.readexactly(exc.consumed))
            continue
    return b"".join(parts)


async def _spawn_subprocess(*cmd: str, **kwargs) -> asyncio.subprocess.Process:
    kwargs.setdefault("limit", SUBPROCESS_STREAM_LIMIT)
    return await asyncio.create_subprocess_exec(*cmd, **kwargs)


@dataclass
class BuildJob:
    id: str
    project: str
    image: str
    dockerfile: str
    state: BuildState = BuildState.QUEUED
    logs: list[str] = field(default_factory=list)
    log_lines: int = 0
    exit_code: int | None = None
    queued_at: str | None = None
    started_at: str | None = None
    finished_at: str | None = None
    error: str | None = None
    backend: str | None = None
    cancel_requested: bool = False

    def snapshot(self, queue_position: int | None = None) -> dict:
        out = {
            "id": self.id,
            "project": self.project,
            "image": self.image,
            "dockerfile": self.dockerfile,
            "state": self.state.value,
            "exit_code": self.exit_code,
            "queued_at": self.queued_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "error": self.error,
            "backend": self.backend,
            "log_lines": self.log_lines,
            "cancel_requested": self.cancel_requested,
        }
        if queue_position is not None:
            out["queue_position"] = queue_position
        return out

    def meta_dict(self, queue_position: int | None = None) -> dict:
        return self.snapshot(queue_position=queue_position)


class BuildManager:
    def __init__(self, data_dir: Path) -> None:
        self.data_dir = data_dir
        self.projects_dir = data_dir / "projects"
        self.docker_config_dir = data_dir / ".docker"
        self.kaniko_cache_dir = data_dir / "kaniko-cache"
        self.jobs_dir = data_dir / "builds"
        self.active_path = self.jobs_dir / "active.json"
        self.queue_path = self.jobs_dir / "queue.json"
        self.jobs: dict[str, BuildJob] = {}
        self._queue_order: list[str] = []
        self._lock = asyncio.Lock()
        self._meta_save_interval = 25
        self._worker_task: asyncio.Task[None] | None = None
        self._running_proc: asyncio.subprocess.Process | None = None
        self._running_job_id: str | None = None
        self._settings_path = data_dir / "settings.json"
        self._settings: dict = self._load_settings()

    def _load_settings(self) -> dict:
        defaults = {"kaniko_verbose": False}
        if not self._settings_path.is_file():
            return defaults
        try:
            raw = json.loads(self._settings_path.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                return {**defaults, **raw}
        except (json.JSONDecodeError, OSError):
            pass
        return defaults

    def _save_settings(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._settings_path.write_text(
            json.dumps(self._settings, indent=2) + "\n",
            encoding="utf-8",
        )

    def get_settings(self) -> dict:
        return {
            "kaniko_verbose": bool(self._settings.get("kaniko_verbose", False)),
            "kaniko_verbosity": self.kaniko_verbosity_level(),
        }

    def set_kaniko_verbose(self, enabled: bool) -> dict:
        self._settings["kaniko_verbose"] = bool(enabled)
        self._save_settings()
        return self.get_settings()

    def kaniko_verbosity_level(self) -> str:
        return "debug" if self._settings.get("kaniko_verbose") else "info"

    def _filter_build_line(self, line: str) -> str | None:
        if self._settings.get("kaniko_verbose"):
            return line
        if _KANIKO_DEBUG_ONLY.search(line):
            return None
        return line

    def ensure_dirs(self) -> None:
        self.projects_dir.mkdir(parents=True, exist_ok=True)
        self.docker_config_dir.mkdir(parents=True, exist_ok=True)
        self.kaniko_cache_dir.mkdir(parents=True, exist_ok=True)
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        self._load_persisted_jobs()
        self._reconcile_queue_after_load()

    def _job_meta_path(self, job_id: str) -> Path:
        return self.jobs_dir / f"{job_id}.json"

    def _job_log_path(self, job_id: str) -> Path:
        return self.jobs_dir / f"{job_id}.log"

    def _read_queue_order(self) -> list[str]:
        if not self.queue_path.is_file():
            return []
        try:
            data = json.loads(self.queue_path.read_text(encoding="utf-8"))
            order = data.get("order", [])
            if isinstance(order, list):
                return [str(x) for x in order if x]
        except (json.JSONDecodeError, OSError):
            pass
        return []

    def _write_queue_order(self) -> None:
        payload = {
            "order": self._queue_order,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        self.queue_path.write_text(
            json.dumps(payload, indent=2) + "\n", encoding="utf-8"
        )

    def _load_persisted_jobs(self) -> None:
        if not self.jobs_dir.is_dir():
            return
        now = datetime.now(timezone.utc).isoformat()
        for meta_path in sorted(self.jobs_dir.glob("*.json")):
            if meta_path.name in ("active.json", "queue.json"):
                continue
            try:
                raw = json.loads(meta_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue
            job_id = raw.get("id") or meta_path.stem
            state = BuildState(raw.get("state", BuildState.FAILED.value))
            if state == BuildState.RUNNING:
                state = BuildState.FAILED
                raw["error"] = raw.get("error") or "Build interrupted (app restarted)"
                raw["finished_at"] = raw.get("finished_at") or now
                raw["state"] = state.value
                meta_path.write_text(
                    json.dumps(raw, indent=2) + "\n", encoding="utf-8"
                )
            job = BuildJob(
                id=job_id,
                project=raw.get("project", ""),
                image=raw.get("image", ""),
                dockerfile=raw.get("dockerfile", "Dockerfile"),
                state=state,
                log_lines=int(raw.get("log_lines", 0)),
                exit_code=raw.get("exit_code"),
                queued_at=raw.get("queued_at"),
                started_at=raw.get("started_at"),
                finished_at=raw.get("finished_at"),
                error=raw.get("error"),
                backend=raw.get("backend"),
            )
            log_path = self._job_log_path(job_id)
            if log_path.is_file() and job.log_lines > 0:
                try:
                    with log_path.open(encoding="utf-8", errors="replace") as fh:
                        tail = deque(fh, maxlen=MEMORY_LOG_CAP)
                    job.logs = [ln.rstrip("\n") for ln in tail]
                except OSError:
                    job.logs = []
            self.jobs[job_id] = job

        self._clear_active_job()

    def _reconcile_queue_after_load(self) -> None:
        order = self._read_queue_order()
        valid: list[str] = []
        seen: set[str] = set()
        for jid in order:
            job = self.jobs.get(jid)
            if job and job.state == BuildState.QUEUED and jid not in seen:
                valid.append(jid)
                seen.add(jid)
        orphans = sorted(
            (
                (j.queued_at or "", j.id, j)
                for j in self.jobs.values()
                if j.state == BuildState.QUEUED and j.id not in seen
            ),
            key=lambda t: t[0],
        )
        for _, jid, _ in orphans:
            valid.append(jid)
        self._queue_order = valid
        self._write_queue_order()

    def _read_active_job_id(self) -> str | None:
        if not self.active_path.is_file():
            return None
        try:
            data = json.loads(self.active_path.read_text(encoding="utf-8"))
            job_id = data.get("job_id")
            return job_id if isinstance(job_id, str) and job_id else None
        except (json.JSONDecodeError, OSError):
            return None

    def _set_active_job(self, job_id: str) -> None:
        payload = {
            "job_id": job_id,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        self.active_path.write_text(
            json.dumps(payload, indent=2) + "\n", encoding="utf-8"
        )

    def _clear_active_job(self) -> None:
        if self.active_path.is_file():
            self.active_path.unlink(missing_ok=True)

    def _save_job_meta(self, job: BuildJob) -> None:
        pos = self.queue_position(job.id)
        path = self._job_meta_path(job.id)
        path.write_text(
            json.dumps(job.meta_dict(queue_position=pos), indent=2) + "\n",
            encoding="utf-8",
        )

    def queue_position(self, job_id: str) -> int | None:
        if job_id not in self._queue_order:
            return None
        return self._queue_order.index(job_id) + 1

    def append_log(self, job: BuildJob, line: str) -> None:
        text = line.rstrip("\n")
        if len(text) > MAX_LOG_LINE_CHARS:
            text = (
                text[:MAX_LOG_LINE_CHARS]
                + f"… [truncated, {len(line)} chars total]"
            )
        job.log_lines += 1
        job.logs.append(text)
        if len(job.logs) > MEMORY_LOG_CAP:
            job.logs = job.logs[-MEMORY_LOG_CAP:]
        log_path = self._job_log_path(job.id)
        with log_path.open("a", encoding="utf-8") as fh:
            fh.write(text + "\n")
        if job.log_lines % self._meta_save_interval == 0:
            self._save_job_meta(job)

    async def append_log_async(self, job: BuildJob, line: str) -> None:
        # Sync open()/write on the event loop starves /health during kaniko floods
        # → kubelet liveness EOF → pod restart → "Build interrupted".
        await asyncio.to_thread(self.append_log, job, line)

    def _memory_log_start(self, job: BuildJob) -> int:
        return max(0, job.log_lines - len(job.logs))

    def _scan_log_file(
        self,
        path: Path,
        *,
        before: int | None = None,
        limit: int = DEFAULT_LOG_PAGE,
    ) -> tuple[list[str], int, int, int]:
        """Single-pass log read. Returns (lines, start, end, total_lines)."""
        if not path.is_file():
            return [], 0, 0, 0
        limit = max(1, min(limit, MAX_LOG_PAGE))
        if before is None:
            tail: deque[str] = deque(maxlen=limit)
            total = 0
            with path.open(encoding="utf-8", errors="replace") as fh:
                for raw in fh:
                    total += 1
                    tail.append(raw.rstrip("\n"))
            if total == 0:
                return [], 0, 0, 0
            lines = list(tail)
            end = total
            start = max(0, end - len(lines))
            return lines, start, end, total
        end = max(0, before)
        start = max(0, end - limit)
        lines: list[str] = []
        total = 0
        with path.open(encoding="utf-8", errors="replace") as fh:
            for i, raw in enumerate(fh):
                total = i + 1
                if i >= start and i < end:
                    lines.append(raw.rstrip("\n"))
        return lines, start, end, total

    def _ensure_log_line_count(self, job: BuildJob) -> int:
        log_path = self._job_log_path(job.id)
        if not log_path.is_file():
            job.log_lines = 0
            return 0
        if job.state in ACTIVE_STATES and job.log_lines > 0:
            return job.log_lines
        _, _, _, total = self._scan_log_file(log_path, before=None, limit=1)
        job.log_lines = total
        return total

    def _read_lines_range(self, path: Path, start: int, end: int) -> list[str]:
        if start >= end or not path.is_file():
            return []
        lines: list[str] = []
        with path.open(encoding="utf-8", errors="replace") as fh:
            for i, raw in enumerate(fh):
                if i >= end:
                    break
                if i >= start:
                    lines.append(raw.rstrip("\n"))
        return lines

    def _get_log_line(self, job: BuildJob, index: int) -> str:
        if index < 0 or index >= job.log_lines:
            return ""
        mem_start = self._memory_log_start(job)
        if index >= mem_start:
            return job.logs[index - mem_start]
        rows = self._read_lines_range(self._job_log_path(job.id), index, index + 1)
        return rows[0] if rows else ""

    def _failure_log_tail(self, job_id: str, n: int = FAILURE_TAIL_LINES) -> list[str]:
        data = self.read_logs(job_id, limit=min(n, MAX_LOG_PAGE))
        return data.get("lines") or []

    def _build_failure_message(
        self,
        job: BuildJob,
        exit_code: int,
        tool: str,
    ) -> str:
        hint = EXIT_CODE_HINTS.get(exit_code, "")
        head = f"{tool} failed with exit code {exit_code}"
        if hint:
            head = f"{head} — {hint}"
        tail = self._failure_log_tail(job.id)
        keywords = (
            "error",
            "fatal",
            "failed",
            "not found",
            "command not found",
            "no such file",
            "returned non-zero",
        )
        highlights = [
            ln
            for ln in tail
            if any(k in ln.lower() for k in keywords)
        ]
        parts = [head, "", "── log tail (scroll up for full build output) ──"]
        if highlights:
            parts.extend(highlights[-20:])
        elif tail:
            parts.extend(tail[-25:])
        else:
            parts.append("(no log lines captured)")
        return "\n".join(parts)

    def _log_build_failure(self, job: BuildJob, message: str) -> None:
        job.error = message
        self.append_log(job, "==> BUILD FAILED")
        for line in message.split("\n"):
            if line.strip():
                self.append_log(job, line)

    def read_logs(
        self,
        job_id: str,
        *,
        before: int | None = None,
        limit: int = DEFAULT_LOG_PAGE,
    ) -> dict:
        job = self.get_job(job_id)
        if not job:
            raise KeyError(job_id)
        limit = max(1, min(limit, MAX_LOG_PAGE))
        path = self._job_log_path(job_id)
        lines, start, end, total = self._scan_log_file(path, before=before, limit=limit)
        job.log_lines = total
        if total == 0:
            return {
                "lines": [],
                "start_line": 0,
                "end_line": 0,
                "total_lines": 0,
                "has_older": False,
            }
        return {
            "lines": lines,
            "start_line": start,
            "end_line": end,
            "total_lines": total,
            "has_older": start > 0,
        }

    def list_projects(self) -> list[dict]:
        if not self.projects_dir.is_dir():
            return []
        out: list[dict] = []
        for entry in sorted(self.projects_dir.iterdir()):
            if not entry.is_dir() or entry.name.startswith("."):
                continue
            rel = dockerfile_relpath(entry)
            out.append(
                {
                    "name": entry.name,
                    "path": str(entry),
                    "has_dockerfile": rel is not None,
                    "dockerfile": rel,
                    "modified": datetime.fromtimestamp(
                        entry.stat().st_mtime, tz=timezone.utc
                    ).isoformat(),
                }
            )
        return out

    def sanitize_name(self, name: str) -> str:
        cleaned = re.sub(r"[^a-zA-Z0-9._-]+", "-", name.strip()).strip("-")
        if not cleaned:
            raise ValueError("Project name is empty after sanitization")
        return cleaned[:120]

    def project_dir(self, name: str) -> Path:
        return self.projects_dir / self.sanitize_name(name)

    async def create_project_from_zip(self, name: str, zip_path: Path) -> Path:
        dest = self.project_dir(name)
        if dest.exists():
            shutil.rmtree(dest)
        dest.mkdir(parents=True, exist_ok=True)
        proc = await asyncio.create_subprocess_exec(
            "unzip",
            "-q",
            str(zip_path),
            "-d",
            str(dest),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await proc.communicate()
        if proc.returncode != 0:
            shutil.rmtree(dest, ignore_errors=True)
            raise RuntimeError(stderr.decode() or "unzip failed")
        self._normalize_zip_layout(dest)
        return dest

    def _normalize_zip_layout(self, dest: Path) -> None:
        for junk in ("__MACOSX",):
            junk_path = dest / junk
            if junk_path.is_dir():
                shutil.rmtree(junk_path, ignore_errors=True)

        for _ in range(8):
            if find_dockerfile(dest) and (dest / "Dockerfile").is_file():
                return

            entries = [p for p in dest.iterdir() if not _is_ignored_top(p.name)]
            dirs = [p for p in entries if p.is_dir()]
            files = [p for p in entries if p.is_file()]

            if len(dirs) == 1 and not files:
                nested = dirs[0]
                for item in list(nested.iterdir()):
                    target = dest / item.name
                    if target.exists():
                        if target.is_dir():
                            shutil.rmtree(target)
                        else:
                            target.unlink()
                    shutil.move(str(item), str(dest / item.name))
                nested.rmdir()
                continue

            break

    def resolve_dockerfile(self, project_dir: Path, dockerfile: str) -> Path:
        rel = dockerfile.strip().lstrip("/") or "Dockerfile"
        direct = project_dir / rel
        if direct.is_file():
            return direct
        found = find_dockerfile(project_dir)
        if found and rel in ("Dockerfile", "dockerfile"):
            return found
        if found:
            raise FileNotFoundError(
                f"Dockerfile not found at {rel!r}; detected {found.relative_to(project_dir).as_posix()!r} instead"
            )
        raise FileNotFoundError(
            f"No Dockerfile in {project_dir.name}. Zip the project folder (include Dockerfile at root or one level down)."
        )

    def _remove_from_queue(self, job_id: str) -> None:
        if job_id in self._queue_order:
            self._queue_order.remove(job_id)
            self._write_queue_order()

    def _start_queue_worker(self) -> None:
        """Start the FIFO worker when an event loop is running (startup / API)."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        if self._worker_task and not self._worker_task.done():
            return
        self._worker_task = loop.create_task(self._queue_worker())

    async def _queue_worker(self) -> None:
        try:
            while True:
                async with self._lock:
                    job_id = self._peek_next_queued_id()
                    if not job_id:
                        break
                    job = self.jobs[job_id]
                    if job.state != BuildState.QUEUED:
                        self._remove_from_queue(job_id)
                        continue
                    if job.cancel_requested:
                        self._finish_cancelled(job)
                        self._remove_from_queue(job_id)
                        continue
                    project_dir = self.project_dir(job.project)
                    self._remove_from_queue(job_id)

                await self._run_job(job, project_dir)
        finally:
            self._worker_task = None
            async with self._lock:
                if self._peek_next_queued_id():
                    self._start_queue_worker()

    def _peek_next_queued_id(self) -> str | None:
        for jid in self._queue_order:
            job = self.jobs.get(jid)
            if job and job.state == BuildState.QUEUED:
                return jid
        return None

    def _finish_cancelled(self, job: BuildJob) -> None:
        job.state = BuildState.CANCELLED
        job.finished_at = datetime.now(timezone.utc).isoformat()
        job.error = job.error or "Cancelled"
        self.append_log(job, "==> build cancelled")
        self._save_job_meta(job)

    async def enqueue_build(self, project: str, image: str, dockerfile: str) -> BuildJob:
        project_dir = self.project_dir(project)
        if not project_dir.is_dir():
            raise FileNotFoundError(f"Project not found: {project}")
        df = self.resolve_dockerfile(project_dir, dockerfile)
        dockerfile_rel = df.relative_to(project_dir).as_posix()

        now = datetime.now(timezone.utc).isoformat()
        job = BuildJob(
            id=uuid.uuid4().hex[:12],
            project=self.sanitize_name(project),
            image=image.strip(),
            dockerfile=dockerfile_rel,
            backend=build_backend(),
            state=BuildState.QUEUED,
            queued_at=now,
        )
        self._job_log_path(job.id).write_text("", encoding="utf-8")

        async with self._lock:
            self.jobs[job.id] = job
            self._queue_order.append(job.id)
            self._write_queue_order()
            pos = self.queue_position(job.id)
            self.append_log(job, f"==> queued (position {pos})")
            self._save_job_meta(job)

        self._start_queue_worker()
        return job

    async def cancel_build(self, job_id: str) -> BuildJob:
        async with self._lock:
            job = self.jobs.get(job_id)
            if not job:
                raise KeyError(job_id)
            if job.state in TERMINAL_STATES:
                raise RuntimeError(f"Build {job_id} already {job.state.value}")
            job.cancel_requested = True

            if job.state == BuildState.QUEUED:
                self._remove_from_queue(job_id)
                self._finish_cancelled(job)
                return job

            if job.state == BuildState.RUNNING:
                self.append_log(job, "==> cancel requested, stopping…")
                self._save_job_meta(job)

        await self._kill_running_proc()
        return job

    async def _kill_running_proc(self) -> None:
        proc = self._running_proc
        if proc is None or proc.returncode is not None:
            return
        proc.terminate()
        try:
            await asyncio.wait_for(proc.wait(), timeout=10.0)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()

    def get_job(self, job_id: str) -> BuildJob | None:
        return self.jobs.get(job_id)

    def running_job(self) -> BuildJob | None:
        active_id = self._read_active_job_id()
        if active_id:
            job = self.jobs.get(active_id)
            if job and job.state == BuildState.RUNNING:
                return job
        for job in self.jobs.values():
            if job.state == BuildState.RUNNING:
                return job
        return None

    def current_job(self) -> BuildJob | None:
        """Job to show in the UI: running build, else first queued."""
        running = self.running_job()
        if running:
            return running
        for jid in self._queue_order:
            job = self.jobs.get(jid)
            if job and job.state == BuildState.QUEUED:
                return job
        return None

    def queued_jobs(self) -> list[BuildJob]:
        out: list[BuildJob] = []
        for jid in self._queue_order:
            job = self.jobs.get(jid)
            if job and job.state == BuildState.QUEUED:
                out.append(job)
        return out

    def list_jobs(self) -> list[dict]:
        return [
            j.snapshot(queue_position=self.queue_position(j.id))
            for j in sorted(
                self.jobs.values(),
                key=lambda x: x.queued_at or x.started_at or x.finished_at or "",
                reverse=True,
            )
        ]

    def queue_status(self) -> dict:
        running = self.running_job()
        queued = self.queued_jobs()
        return {
            "running": running.snapshot() if running else None,
            "queued": [
                j.snapshot(queue_position=self.queue_position(j.id)) for j in queued
            ],
            "queue_length": len(queued),
        }

    async def stream_logs(self, job_id: str, from_line: int = 0) -> AsyncIterator[str]:
        job = self.get_job(job_id)
        if not job:
            return
        path = self._job_log_path(job_id)
        sent = max(0, from_line)
        if sent == 0 and path.is_file():
            _, _, _, total = self._scan_log_file(path, before=None, limit=1)
            job.log_lines = total
            if total > STREAM_SKIP_REPLAY_THRESHOLD:
                sent = max(0, total - STREAM_TAIL_ON_SKIP)
                yield (
                    f"==> log: skipped to line {sent + 1} of {total} "
                    f"(scroll up / Load older for full history)\n"
                )
        while True:
            job = self.get_job(job_id)
            if not job:
                return
            total = self._ensure_log_line_count(job)
            while sent < total:
                batch_end = min(sent + STREAM_CATCHUP_BATCH, total)
                for line in self._read_lines_range(path, sent, batch_end):
                    yield line + "\n"
                sent = batch_end
                if batch_end < total:
                    await asyncio.sleep(0)
            if job.state in TERMINAL_STATES:
                break
            await asyncio.sleep(0.4)

    def write_ghcr_auth(self) -> None:
        token = os.environ.get("GITHUB_TOKEN", "").strip()
        user = _ghcr_username()
        if not token:
            raise RuntimeError(
                "GITHUB_TOKEN is not set. Set GITHUB_TOKEN in Application settings "
                "(GitHub PAT with write:packages), then restart the app."
            )
        auth = base64.b64encode(f"{user}:{token}".encode()).decode()
        config = {"auths": {"ghcr.io": {"auth": auth}}}
        config_path = self.docker_config_dir / "config.json"
        config_path.write_text(json.dumps(config), encoding="utf-8")
        os.chmod(config_path, 0o600)

    def _check_cancelled(self, job: BuildJob) -> None:
        if job.cancel_requested:
            raise BuildCancelled("Build cancelled by user")

    async def _run_job(self, job: BuildJob, project_dir: Path) -> None:
        job.state = BuildState.RUNNING
        job.started_at = datetime.now(timezone.utc).isoformat()
        self._set_active_job(job.id)
        self._save_job_meta(job)
        self.append_log(job, "==> build started")
        try:
            self._check_cancelled(job)
            self.write_ghcr_auth()
            backend = job.backend or build_backend()
            job.backend = backend
            self.append_log(job, f"==> backend: {backend}")
            if backend == "kaniko":
                await self._kaniko_build_push(job, project_dir)
            elif backend == "docker":
                await self._wait_for_docker()
                self._check_cancelled(job)
                await self._docker_login(job)
                self._check_cancelled(job)
                await self._docker_build(job, project_dir)
                self._check_cancelled(job)
                await self._docker_push(job)
            else:
                raise RuntimeError(
                    "No build backend available (Kaniko binary missing and no docker daemon)"
                )
            self._check_cancelled(job)
            job.state = BuildState.SUCCESS
        except BuildCancelled:
            job.state = BuildState.CANCELLED
            job.error = "Cancelled by user"
            self.append_log(job, "==> build cancelled")
        except Exception as exc:
            if job.cancel_requested:
                job.state = BuildState.CANCELLED
                job.error = "Cancelled by user"
                self.append_log(job, "==> build cancelled")
            else:
                job.state = BuildState.FAILED
                self._log_build_failure(job, str(exc))
        finally:
            job.finished_at = datetime.now(timezone.utc).isoformat()
            self._running_proc = None
            self._running_job_id = None
            self._save_job_meta(job)
            self._clear_active_job()

    def _kaniko_cache_args(self) -> list[str]:
        """Layer cache on disk only — avoid ghcr.io/<destination>/cache manifest pulls (MANIFEST_UNKNOWN)."""
        if os.environ.get("KANIKO_CACHE", "true").strip().lower() in ("0", "false", "no", "off"):
            return ["--cache=false"]
        args = [
            "--cache=true",
            f"--cache-dir={self.kaniko_cache_dir}",
            "--no-push-cache",
        ]
        cache_repo = os.environ.get("KANIKO_CACHE_REPO", "").strip()
        if cache_repo:
            args.append(f"--cache-repo={cache_repo}")
        return args

    def _kaniko_args(self, job: BuildJob, project_dir: Path) -> list[str]:
        executor = kaniko_executor_path()
        if not executor:
            raise RuntimeError("Kaniko executor not found at KANIKO_EXECUTOR")
        context = f"dir://{project_dir.resolve()}"
        args = [
            str(executor),
            f"--context={context}",
            f"--dockerfile={job.dockerfile}",
            f"--destination={job.image}",
            *self._kaniko_cache_args(),
            f"--verbosity={self.kaniko_verbosity_level()}",
            f"--snapshot-mode={kaniko_snapshot_mode()}",
            "--log-format=text",
            f"--image-download-retry={kaniko_image_download_retry()}",
        ]
        mirror = os.environ.get("KANIKO_REGISTRY_MIRROR", "").strip()
        if mirror:
            args.append(f"--registry-mirror={mirror}")
        return args

    async def _run_subprocess(
        self,
        job: BuildJob,
        *cmd: str,
    ) -> asyncio.subprocess.Process:
        self._check_cancelled(job)
        proc = await _spawn_subprocess(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            env=self._build_env(),
        )
        self._running_proc = proc
        self._running_job_id = job.id
        await self._consume_proc(proc, job)
        return proc

    async def _kaniko_build_push(self, job: BuildJob, project_dir: Path) -> None:
        args = self._kaniko_args(job, project_dir)
        self.append_log(
            job,
            f"==> kaniko build & push {job.image} (context {project_dir}, dockerfile {job.dockerfile})",
        )
        self.append_log(job, "==> " + " ".join(args))
        proc = await self._run_subprocess(job, *args)
        job.exit_code = proc.returncode
        if job.cancel_requested:
            raise BuildCancelled("Build cancelled by user")
        if proc.returncode != 0:
            raise RuntimeError(
                self._build_failure_message(job, proc.returncode or 1, "kaniko")
            )

    async def _wait_for_docker(self, timeout: float = 120.0) -> None:
        deadline = asyncio.get_event_loop().time() + timeout
        while asyncio.get_event_loop().time() < deadline:
            proc = await asyncio.create_subprocess_exec(
                "docker",
                "info",
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await proc.wait()
            if proc.returncode == 0:
                return
            await asyncio.sleep(2)
        raise TimeoutError("Docker daemon not ready")

    async def _docker_login(self, job: BuildJob) -> None:
        token = os.environ.get("GITHUB_TOKEN", "").strip()
        user = _ghcr_username()
        self.append_log(job, f"==> docker login ghcr.io as {user}")
        proc = await _spawn_subprocess(
            "docker",
            "login",
            "ghcr.io",
            "-u",
            user,
            "--password-stdin",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            env=self._build_env(),
        )
        assert proc.stdin is not None
        proc.stdin.write(token.encode())
        await proc.stdin.drain()
        proc.stdin.close()
        self._running_proc = proc
        self._running_job_id = job.id
        await self._consume_proc(proc, job)
        if job.cancel_requested:
            raise BuildCancelled("Build cancelled by user")

    async def _docker_build(self, job: BuildJob, project_dir: Path) -> None:
        self.append_log(
            job,
            f"==> docker build --progress=plain -f {job.dockerfile} -t {job.image} {project_dir}",
        )
        proc = await self._run_subprocess(
            job,
            "docker",
            "build",
            "--progress=plain",
            "-f",
            job.dockerfile,
            "-t",
            job.image,
            str(project_dir),
        )
        job.exit_code = proc.returncode
        if job.cancel_requested:
            raise BuildCancelled("Build cancelled by user")
        if proc.returncode != 0:
            raise RuntimeError(
                self._build_failure_message(job, proc.returncode or 1, "docker build")
            )

    async def _docker_push(self, job: BuildJob) -> None:
        self.append_log(job, f"==> docker push {job.image}")
        proc = await self._run_subprocess(job, "docker", "push", job.image)
        if job.cancel_requested:
            raise BuildCancelled("Build cancelled by user")
        if proc.returncode != 0:
            raise RuntimeError(
                self._build_failure_message(job, proc.returncode or 1, "docker push")
            )

    def _build_env(self) -> dict[str, str]:
        env = os.environ.copy()
        env["DOCKER_CONFIG"] = str(self.docker_config_dir)
        # Optional: hide GPU from Kaniko child so smoke `import torch` skips HAMI init.
        # Pod may still have a GPU allocated (gpuCount); compile itself only needs stubs.
        if os.environ.get("KANIKO_HIDE_GPU", "1").strip() not in ("0", "false", "no"):
            env["CUDA_VISIBLE_DEVICES"] = ""
        return env

    async def _consume_proc(
        self,
        proc: asyncio.subprocess.Process,
        job: BuildJob,
    ) -> None:
        assert proc.stdout is not None
        while True:
            if job.cancel_requested:
                if proc.returncode is None:
                    proc.terminate()
                break
            try:
                line = await _read_subprocess_line(proc.stdout)
            except (asyncio.LimitOverrunError, ValueError) as exc:
                await self.append_log_async(
                    job,
                    f"[warn] skipped oversized log chunk ({exc}); build continues",
                )
                continue
            if not line:
                break
            text = line.decode(errors="replace").rstrip()
            filtered = self._filter_build_line(text)
            if filtered is None:
                continue
            await self.append_log_async(job, filtered)
        if proc.returncode is None:
            await proc.wait()


def kaniko_executor_path() -> Path | None:
    raw = os.environ.get("KANIKO_EXECUTOR", "/tools/kaniko/executor")
    path = Path(raw)
    if path.is_file() and os.access(path, os.X_OK):
        return path
    return None


def docker_cli_available() -> bool:
    return shutil.which("docker") is not None


def build_backend() -> str:
    if kaniko_executor_path():
        return "kaniko"
    if docker_cli_available():
        return "docker"
    return "none"


def kaniko_snapshot_mode() -> str:
    return os.environ.get("KANIKO_SNAPSHOT_MODE", "time").strip() or "time"


def kaniko_image_download_retry() -> str:
    return os.environ.get("KANIKO_IMAGE_DOWNLOAD_RETRY", "5").strip() or "5"
