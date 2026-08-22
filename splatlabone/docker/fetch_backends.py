#!/usr/bin/env python3
"""Download geometry backend sources into /opt (python-builder stage).

Kaniko/Olares DNS is flaky — retry tarball fetch, then fall back to git clone.
"""

from __future__ import annotations

import io
import os
import shutil
import subprocess
import sys
import tarfile
import time
import urllib.error
import urllib.request

# (tarball_url, archive_top_dir, dest, git_url, git_ref)
REPOS = [
    (
        "https://github.com/nerfstudio-project/gsplat/archive/refs/heads/main.tar.gz",
        "gsplat-main",
        "/opt/gsplat",
        "https://github.com/nerfstudio-project/gsplat.git",
        "main",
    ),
    (
        "https://github.com/facebookresearch/vggt-omega/archive/refs/heads/main.tar.gz",
        "vggt-omega-main",
        "/opt/vggt-omega",
        "https://github.com/facebookresearch/vggt-omega.git",
        "main",
    ),
    (
        "https://github.com/ByteDance-Seed/Depth-Anything-3/archive/refs/heads/main.tar.gz",
        "Depth-Anything-3-main",
        "/opt/da3",
        "https://github.com/ByteDance-Seed/Depth-Anything-3.git",
        "main",
    ),
    (
        "https://github.com/Robbyant/lingbot-map/archive/refs/heads/main.tar.gz",
        "lingbot-map-main",
        "/opt/lingbot-map",
        "https://github.com/Robbyant/lingbot-map.git",
        "main",
    ),
    (
        "https://github.com/NVlabs/InstantSplat/archive/refs/heads/main.tar.gz",
        "InstantSplat-main",
        "/opt/instantsplat",
        "https://github.com/NVlabs/InstantSplat.git",
        "main",
    ),
    (
        "https://github.com/naver/dust3r/archive/refs/heads/main.tar.gz",
        "dust3r-main",
        "/opt/instantsplat/dust3r",
        "https://github.com/naver/dust3r.git",
        "main",
    ),
    (
        "https://github.com/naver/mast3r/archive/refs/heads/main.tar.gz",
        "mast3r-main",
        "/opt/instantsplat/mast3r",
        "https://github.com/naver/mast3r.git",
        "main",
    ),
    (
        "https://github.com/cvg/Hierarchical-Localization/archive/refs/heads/master.tar.gz",
        "Hierarchical-Localization-master",
        "/opt/hloc",
        "https://github.com/cvg/Hierarchical-Localization.git",
        "master",
    ),
    (
        "https://github.com/cvg/LightGlue/archive/refs/heads/main.tar.gz",
        "LightGlue-main",
        "/opt/lightglue",
        "https://github.com/cvg/LightGlue.git",
        "main",
    ),
    (
        "https://github.com/pals-ttic/fastmap/archive/refs/heads/main.tar.gz",
        "fastmap-main",
        "/opt/fastmap",
        "https://github.com/pals-ttic/fastmap.git",
        "main",
    ),
    # GlueMap + depth-anything.cpp: Dockerfile git clone + submodules.
]

MAX_ATTEMPTS = 6
BACKOFF_SEC = (5, 15, 30, 60, 90, 120)


def _place(src: str, dest: str) -> None:
    os.makedirs(os.path.dirname(dest) or "/", exist_ok=True)
    if os.path.exists(dest):
        shutil.rmtree(dest)
    os.rename(src, dest)
    print(f"ok {dest}")


def fetch_tarball(url: str, folder: str, dest: str) -> None:
    print(f"fetch tarball {url} -> {dest}")
    last: Exception | None = None
    for i in range(MAX_ATTEMPTS):
        try:
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "splatlabone-kaniko-fetch/1.0"},
            )
            with urllib.request.urlopen(req, timeout=600) as resp:
                data = resp.read()
            tarfile.open(fileobj=io.BytesIO(data)).extractall("/opt")
            _place(f"/opt/{folder}", dest)
            return
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            last = e
            wait = BACKOFF_SEC[min(i, len(BACKOFF_SEC) - 1)]
            print(f"  attempt {i + 1}/{MAX_ATTEMPTS} failed: {e}; sleep {wait}s", flush=True)
            time.sleep(wait)
    assert last is not None
    raise last


def fetch_git(git_url: str, ref: str, dest: str) -> None:
    print(f"fetch git {git_url}@{ref} -> {dest}")
    tmp = f"{dest}.git-tmp"
    if os.path.exists(tmp):
        shutil.rmtree(tmp)
    if os.path.exists(dest):
        shutil.rmtree(dest)
    last: Exception | None = None
    for i in range(MAX_ATTEMPTS):
        try:
            subprocess.check_call(
                [
                    "git",
                    "clone",
                    "--depth",
                    "1",
                    "--branch",
                    ref,
                    git_url,
                    tmp,
                ],
                stdout=sys.stdout,
                stderr=sys.stderr,
            )
            os.rename(tmp, dest)
            print(f"ok {dest} (git)")
            return
        except (subprocess.CalledProcessError, OSError) as e:
            last = e
            if os.path.exists(tmp):
                shutil.rmtree(tmp, ignore_errors=True)
            wait = BACKOFF_SEC[min(i, len(BACKOFF_SEC) - 1)]
            print(f"  git attempt {i + 1}/{MAX_ATTEMPTS} failed: {e}; sleep {wait}s", flush=True)
            time.sleep(wait)
    assert last is not None
    raise last


def fetch(url: str, folder: str, dest: str, git_url: str, ref: str) -> None:
    try:
        fetch_tarball(url, folder, dest)
    except Exception as e:
        print(f"tarball failed ({e}); falling back to git clone", flush=True)
        fetch_git(git_url, ref, dest)


def main() -> None:
    # Brief settle — DNS sometimes not ready right after prior pip layers.
    time.sleep(2)
    for url, folder, dest, git_url, ref in REPOS:
        fetch(url, folder, dest, git_url, ref)


if __name__ == "__main__":
    main()
