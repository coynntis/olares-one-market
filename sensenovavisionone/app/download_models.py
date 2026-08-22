#!/usr/bin/env python3
"""HF snapshot download with progress → bootstrap.log + phase file."""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path


def _append_log(path: Path, line: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    with path.open("a", encoding="utf-8") as fh:
        fh.write(f"{ts} {line}\n")
    print(f"[sensenova-dl] {line}", flush=True)


def _set_phase(phase_file: Path, status: str, detail: str, **extra) -> None:
    phase_file.parent.mkdir(parents=True, exist_ok=True)
    payload = {"status": status, "ready": False, "phase": status, "detail": detail, **extra}
    phase_file.write_text(json.dumps(payload), encoding="utf-8")


def main() -> int:
    repo_id = os.environ.get("HF_REPO_ID", "").strip()
    local_dir = os.environ.get("HF_LOCAL_DIR", "").strip()
    app = os.environ.get("APP_NAME", "sensenovavisionone")
    phase_file = Path(os.environ.get("BOOT_PHASE_FILE", f"/workspace/{app}/.boot-phase"))
    log_file = Path(os.environ.get("BOOT_ATTEMPTS_FILE", f"/workspace/{app}/bootstrap.log"))
    marker = Path(local_dir) / ".download-ok" if local_dir else None
    allow_patterns = os.environ.get("HF_ALLOW_PATTERNS", "").strip()
    revision = os.environ.get("HF_REVISION", "").strip() or None

    if not repo_id or not local_dir:
        _append_log(log_file, "FATAL missing HF_REPO_ID or HF_LOCAL_DIR")
        return 2

    local = Path(local_dir)
    local.mkdir(parents=True, exist_ok=True)

    required = [
        p.strip()
        for p in os.environ.get(
            "HF_REQUIRED_FILES", "ema.safetensors,ae.safetensors,config.json"
        ).split(",")
        if p.strip()
    ]

    def _has_required() -> bool:
        if not required:
            return any(local.iterdir())
        for name in required:
            if not any(local.rglob(name)):
                return False
        return True

    if marker and marker.is_file() and _has_required():
        _append_log(log_file, f"skip download — marker OK {marker} required={required or '*'}")
        _set_phase(phase_file, "installing", f"model_cached:{repo_id}")
        return 0
    if marker and marker.is_file() and not _has_required():
        _append_log(
            log_file,
            f"marker present but missing required files {required} — re-download",
        )
        try:
            marker.unlink()
        except OSError:
            pass

    attempt = 1
    max_attempts = int(os.environ.get("HF_DOWNLOAD_ATTEMPTS", "3"))
    last_err = ""

    while attempt <= max_attempts:
        t0 = time.time()
        _append_log(
            log_file,
            f"attempt={attempt}/{max_attempts} repo={repo_id} dir={local_dir}",
        )
        _set_phase(
            phase_file,
            "installing",
            f"hf_download:{repo_id}:attempt_{attempt}",
            repo_id=repo_id,
            attempt=attempt,
        )
        try:
            from huggingface_hub import snapshot_download

            kwargs = {
                "repo_id": repo_id,
                "local_dir": str(local),
                "local_dir_use_symlinks": False,
                "resume_download": True,
                "max_workers": int(os.environ.get("HF_DOWNLOAD_WORKERS", "8")),
            }
            if revision:
                kwargs["revision"] = revision
            if allow_patterns:
                kwargs["allow_patterns"] = [
                    p.strip() for p in allow_patterns.split(",") if p.strip()
                ]

            token = (
                os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or ""
            ).strip()
            if token:
                kwargs["token"] = token

            path = snapshot_download(**kwargs)
            elapsed = round(time.time() - t0, 1)
            total = 0
            for root, _, files in os.walk(path):
                for name in files:
                    try:
                        total += (Path(root) / name).stat().st_size
                    except OSError:
                        pass
            _append_log(
                log_file,
                f"OK repo={repo_id} bytes={total} elapsed_s={elapsed} path={path}",
            )
            if marker:
                marker.write_text(
                    json.dumps({"repo_id": repo_id, "bytes": total, "elapsed_s": elapsed}),
                    encoding="utf-8",
                )
            _set_phase(
                phase_file,
                "installing",
                f"hf_download_ok:{repo_id}:{total}B",
                repo_id=repo_id,
                bytes=total,
                elapsed_s=elapsed,
            )
            return 0
        except Exception as exc:
            last_err = str(exc)
            elapsed = round(time.time() - t0, 1)
            _append_log(
                log_file,
                f"FAIL attempt={attempt} repo={repo_id} elapsed_s={elapsed} err={last_err}",
            )
            _set_phase(
                phase_file,
                "installing",
                f"hf_download_fail:{repo_id}:attempt_{attempt}:{last_err[:180]}",
                attempt=attempt,
                error=last_err[:500],
            )
            attempt += 1
            time.sleep(min(30, 5 * attempt))

    _append_log(log_file, f"FATAL giving up repo={repo_id} last_err={last_err}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
