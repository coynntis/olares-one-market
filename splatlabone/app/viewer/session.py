"""Manage gsplat Viser viewer subprocesses."""

from __future__ import annotations

import os
import signal
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from pipeline.export import find_latest_checkpoint

TRAINER_PATH = Path("/opt/gsplat/examples/simple_viewer.py")
BASE_PORT = int(os.environ.get("SPLATLAB_VISER_BASE_PORT", "8780"))
MAX_PORT = int(os.environ.get("SPLATLAB_VISER_MAX_PORT", "8799"))
IDLE_TIMEOUT_SEC = int(os.environ.get("SPLATLAB_VIEWER_IDLE_SEC", "900"))


@dataclass
class ViewerSession:
    job_id: str
    ckpt_path: Path
    port: int
    proc: subprocess.Popen
    last_access: float = field(default_factory=time.time)

    @property
    def url_path(self) -> str:
        return f"/api/v1/scenes/{self.job_id}/viewer/proxy/"


class ViewerManager:
    def __init__(self) -> None:
        self._sessions: dict[str, ViewerSession] = {}
        self._lock = threading.Lock()
        self._used_ports: set[int] = set()

    def _alloc_port(self) -> int:
        for port in range(BASE_PORT, MAX_PORT + 1):
            if port not in self._used_ports:
                self._used_ports.add(port)
                return port
        raise RuntimeError("no free Viser ports")

    def _release_port(self, port: int) -> None:
        self._used_ports.discard(port)

    def stop(self, job_id: str) -> None:
        with self._lock:
            sess = self._sessions.pop(job_id, None)
        if not sess:
            return
        try:
            sess.proc.terminate()
            sess.proc.wait(timeout=5)
        except Exception:
            try:
                sess.proc.kill()
            except Exception:
                pass
        self._release_port(sess.port)

    def start(self, job_id: str, result_dir: Path) -> ViewerSession:
        self.stop(job_id)
        if not TRAINER_PATH.is_file():
            raise RuntimeError("gsplat simple_viewer.py not found in image")

        ckpt = find_latest_checkpoint(result_dir)
        if not ckpt:
            raise RuntimeError("no checkpoint for viewer")

        port = self._alloc_port()
        env = os.environ.copy()
        py = env.get("PYTHONPATH", "")
        parts = [p for p in py.split(":") if p and p != "/opt/gsplat"]
        env["PYTHONPATH"] = ":".join(parts)
        cmd = [
            "python",
            str(TRAINER_PATH),
            "--ckpt",
            str(ckpt),
            "--port",
            str(port),
            "--output_dir",
            str(result_dir / "viewer_out"),
        ]
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env=env,
            cwd=str(TRAINER_PATH.parent),
            preexec_fn=os.setsid if hasattr(os, "setsid") else None,
        )
        sess = ViewerSession(job_id=job_id, ckpt_path=ckpt, port=port, proc=proc)
        with self._lock:
            self._sessions[job_id] = sess
        return sess

    def get(self, job_id: str) -> ViewerSession | None:
        with self._lock:
            sess = self._sessions.get(job_id)
            if sess:
                sess.last_access = time.time()
            return sess

    def touch(self, job_id: str) -> None:
        with self._lock:
            sess = self._sessions.get(job_id)
            if sess:
                sess.last_access = time.time()

    def cleanup_idle(self) -> None:
        now = time.time()
        stale = []
        with self._lock:
            for jid, sess in self._sessions.items():
                if now - sess.last_access > IDLE_TIMEOUT_SEC:
                    stale.append(jid)
        for jid in stale:
            self.stop(jid)


viewer_manager = ViewerManager()
