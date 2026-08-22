"""Install flash-attn FA2 for SenseNova on Olares One (Blackwell sm_120).

Order:
  1. Already importable → verify
  2. Prebuilt wheels that match THIS python/torch/cuda (skip wrong ABI tags)
  3. Optional controlled source build → cache wheel under /workspace/wheels (persist)
  4. Soft-fail → SDPA (FLASH_ATTN_SOFT_FAIL=1)

SenseNova uses flash_attn.flash_attn_func (FA2 API). FA3/FA4 are not drop-ins.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

# Community / Dao candidates — only used when tags match runtime (see _wheel_matches_runtime).
DAO_BASE = "https://github.com/Dao-AILab/flash-attention/releases/download"
ADITHYAXX_CU13_TORCH211_CP312 = (
    "https://github.com/adithyaxx/flash-attention/releases/download/v2.8.3/"
    "flash_attn-2.8.3+cu13torch2.11cxx11abiTRUE-cp312-cp312-linux_x86_64.whl"
)
JAHJEDI_SM120_CP312 = (
    "https://huggingface.co/JahJedi/sageattention-flashattn-blackwell-cu130-torch211-cp312"
    "/resolve/main/flash_attn-2.8.3.post1-cp312-cp312-linux_x86_64.whl"
)


def _soft() -> bool:
    return os.environ.get("FLASH_ATTN_SOFT_FAIL", "1").strip() not in ("0", "false", "False")


def _allow_source() -> bool:
    return os.environ.get("FLASH_ATTN_ALLOW_SOURCE", "1").strip() not in ("0", "false", "False")


def _fail(msg: str, code: int = 1) -> None:
    print(f"[flash-attn] ERROR: {msg}", flush=True)
    if _soft():
        print("[flash-attn] soft-fail → prefer ATTENTION_BACKEND=sdpa", flush=True)
        raise SystemExit(0)
    raise SystemExit(code)


def _runtime() -> dict:
    import torch

    ver = torch.__version__.split("+")[0]
    major, minor = (int(x) for x in ver.split(".")[:2])
    cuda = torch.version.cuda or ""
    cu_major = int(cuda.split(".")[0]) if cuda else 0
    py = f"cp{sys.version_info.major}{sys.version_info.minor}"
    prefer_true = bool(getattr(torch._C, "_GLIBCXX_USE_CXX11_ABI", True))
    info = {
        "torch": torch.__version__,
        "torch_mm": (major, minor),
        "torch_tag": f"{major}.{minor}",  # Dao uses torch2.9 not torch29
        "cuda": cuda,
        "cu": "cu13" if cu_major >= 13 else "cu12",
        "py": py,
        "abi_true": prefer_true,
    }
    print(
        f"[flash-attn] runtime torch={info['torch']} cuda={info['cuda']} "
        f"py={info['py']} exe={sys.executable}",
        flush=True,
    )
    return info


def _already_ok() -> bool:
    try:
        import flash_attn  # noqa: F401

        print(f"[flash-attn] already importable ({getattr(flash_attn, '__version__', '?')})", flush=True)
        return True
    except Exception as exc:
        print(f"[flash-attn] not installed yet ({exc})", flush=True)
        return False


def _wheel_matches_runtime(url_or_name: str, rt: dict) -> bool:
    """Skip wheels whose filename encodes a different CPython / obvious CUDA channel."""
    name = url_or_name.rstrip("/").split("/")[-1].split("?")[0]
    m = re.search(r"-cp(\d+)-cp\d+-", name)
    if m:
        got = f"cp{m.group(1)}"
        if got != rt["py"]:
            print(f"[flash-attn] skip ABI mismatch ({got} != {rt['py']}): {name}", flush=True)
            return False
    # Soft CUDA hint in filename
    if "cu13" in name and rt["cu"] != "cu13":
        print(f"[flash-attn] skip cu13 wheel on {rt['cu']}: {name}", flush=True)
        return False
    if "+cu12" in name and rt["cu"] == "cu13":
        # cu12 wheels sometimes work on cu13 runtime; allow but deprioritize
        pass
    return True


def _head_ok(url: str) -> bool:
    req = urllib.request.Request(url, method="HEAD")
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return 200 <= resp.status < 400
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return False
        raise
    except Exception:
        return False


def _dao_names(rt: dict) -> list[str]:
    """Build Dao release asset names (correct torch2.X tagging)."""
    cu, py, tag = rt["cu"], rt["py"], rt["torch_tag"]
    abis = ("TRUE", "FALSE") if rt["abi_true"] else ("FALSE", "TRUE")
    names: list[str] = []
    for abi in abis:
        # v2.8.3 primary
        names.append(f"flash_attn-2.8.3+{cu}torch{tag}cxx11abi{abi}-{py}-{py}-linux_x86_64.whl")
        # post1 variant naming
        names.append(
            f"flash_attn-2.8.3.post1+{cu}torch{tag}cxx11abi{abi}-{py}-{py}-linux_x86_64.whl"
        )
    return names


def _candidate_urls(rt: dict) -> list[str]:
    urls: list[str] = []
    spec = os.environ.get("FLASH_ATTN_PIP_SPEC", "").strip()
    if spec:
        urls.append(spec)

    # Prefer known good community wheels for cu130 + torch2.11 + cp312 (Olares One target)
    if rt["py"] == "cp312" and rt["cu"] == "cu13" and rt["torch_mm"] >= (2, 11):
        urls.append(ADITHYAXX_CU13_TORCH211_CP312)
        urls.append(JAHJEDI_SM120_CP312)
    elif rt["py"] == "cp312" and rt["cu"] == "cu13":
        urls.append(ADITHYAXX_CU13_TORCH211_CP312)
        urls.append(JAHJEDI_SM120_CP312)

    for release in ("v2.8.3", "v2.8.3.post1"):
        for name in _dao_names(rt):
            # post1 assets use post1-prefixed names already listed
            if release == "v2.8.3" and ".post1+" in name:
                continue
            if release == "v2.8.3.post1" and ".post1+" not in name and not name.startswith(
                "flash_attn-2.8.3.post1"
            ):
                # also try non-post1 name under post1 tag sometimes
                pass
            urls.append(f"{DAO_BASE}/{release}/{name}")

    # Also try cu12 Dao wheels as last prebuilt resort on cu13 (ABI match only)
    if rt["cu"] == "cu13":
        rt12 = dict(rt, cu="cu12")
        for name in _dao_names(rt12):
            if ".post1+" in name:
                urls.append(f"{DAO_BASE}/v2.8.3.post1/{name}")
            else:
                urls.append(f"{DAO_BASE}/v2.8.3/{name}")

    seen: set[str] = set()
    out: list[str] = []
    for u in urls:
        if u in seen:
            continue
        seen.add(u)
        if _wheel_matches_runtime(u, rt):
            out.append(u)
    return out


def _cache_dir() -> Path:
    p = Path(os.environ.get("FLASH_ATTN_WHEEL_DIR", "/workspace/wheels"))
    p.mkdir(parents=True, exist_ok=True)
    return p


def _download(url: str) -> Path:
    name = url.rstrip("/").split("/")[-1].split("?")[0] or "flash_attn.whl"
    dest = _cache_dir() / name
    if dest.is_file() and dest.stat().st_size > 1_000_000:
        print(f"[flash-attn] using cached wheel {dest} ({dest.stat().st_size} bytes)", flush=True)
        return dest

    tmp = dest.with_suffix(dest.suffix + ".partial")
    print(f"[flash-attn] downloading {url}", flush=True)
    if subprocess.call(["bash", "-lc", "command -v curl >/dev/null"], stdout=subprocess.DEVNULL) == 0:
        subprocess.check_call(
            [
                "curl",
                "-fL",
                "--retry",
                "20",
                "--retry-delay",
                "5",
                "--retry-all-errors",
                "-C",
                "-",
                "-o",
                str(tmp),
                url,
            ]
        )
    else:
        urllib.request.urlretrieve(url, str(tmp))
    tmp.replace(dest)
    print(f"[flash-attn] saved {dest} ({dest.stat().st_size} bytes)", flush=True)
    return dest


def _pip_install(path_or_url: str) -> None:
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--break-system-packages",
            "--no-cache-dir",
            path_or_url,
        ],
    )


def _find_cached_builtin_wheel() -> Path | None:
    """Wheels built by a previous source compile on this hostPath."""
    cdir = _cache_dir()
    cands = sorted(cdir.glob("flash_attn-*.whl"), key=lambda p: p.stat().st_mtime, reverse=True)
    for p in cands:
        if p.stat().st_size > 1_000_000:
            return p
    return None


def _source_build_wheel(rt: dict) -> Path:
    """Compile FA2 once; cache .whl under /workspace/wheels for restarts."""
    jobs = os.environ.get("FLASH_ATTN_MAX_JOBS", "1").strip() or "1"
    env = os.environ.copy()
    env["TORCH_CUDA_ARCH_LIST"] = os.environ.get("TORCH_CUDA_ARCH_LIST", "12.0")
    env["FLASH_ATTN_CUDA_ARCHS"] = os.environ.get("FLASH_ATTN_CUDA_ARCHS", "120")
    env["MAX_JOBS"] = jobs
    env["NVCC_THREADS"] = os.environ.get("NVCC_THREADS", "1")
    # Ensure pip does not silently force a bad binary
    env["FLASH_ATTENTION_FORCE_BUILD"] = "TRUE"

    print(
        f"[flash-attn] source-building FA2 for sm_120 "
        f"(TORCH_CUDA_ARCH_LIST={env['TORCH_CUDA_ARCH_LIST']} MAX_JOBS={jobs}) …",
        flush=True,
    )
    print(
        "[flash-attn] first build can take 20–60+ min; wheel caches under /workspace/wheels",
        flush=True,
    )

    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--break-system-packages",
            "packaging",
            "ninja",
            "wheel",
            "einops",
        ],
        env=env,
    )

    before = {p.resolve() for p in _cache_dir().glob("flash_attn-*.whl")}
    # Build wheel into persistent cache (survives pod restart; image FS does not)
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "wheel",
            "--no-deps",
            "--no-build-isolation",
            "-w",
            str(_cache_dir()),
            "flash-attn==2.8.3.post1",
        ],
        env=env,
    )
    after = {p.resolve() for p in _cache_dir().glob("flash_attn-*.whl")}
    new = after - before
    if not new:
        # pip may have reused existing name
        built = _find_cached_builtin_wheel()
        if not built:
            raise RuntimeError("pip wheel produced no flash_attn-*.whl")
        return built
    return max(new, key=lambda p: p.stat().st_mtime)


def _verify() -> None:
    backend = os.environ.get("ATTENTION_BACKEND", "flash").strip().lower()
    import flash_attn  # noqa: F401

    print(f"[flash-attn] import ok version={getattr(flash_attn, '__version__', '?')}", flush=True)
    if backend != "flash":
        return
    try:
        import sensenova_u1

        sensenova_u1.set_attn_backend("flash")
        eff = sensenova_u1.effective_attn_backend()
        if eff != "flash":
            _fail(f"ATTENTION_BACKEND=flash but effective_attn_backend()={eff!r}")
        print(f"[flash-attn] effective_attn_backend={eff}", flush=True)
    except ImportError:
        print("[flash-attn] sensenova_u1 not importable yet; skip effective backend check", flush=True)


def main() -> None:
    if _already_ok():
        try:
            _verify()
        except SystemExit:
            raise
        except Exception as exc:
            _fail(f"verify failed: {exc}")
        return

    try:
        rt = _runtime()
    except Exception as exc:
        _fail(f"torch probe failed: {exc}")
        return

    # Hint if chart image was upgraded but pod still on old 2.9/cp311
    if rt["py"] != "cp312" or rt["cu"] != "cu13" or rt["torch_mm"] < (2, 11):
        print(
            "[flash-attn] WARN: expected image pytorch/pytorch:2.11.0-cuda13.0-cudnn9-devel "
            f"(cp312 + cu13 + torch≥2.11); got py={rt['py']} cu={rt['cu']} torch={rt['torch']}. "
            "Upgrade app / pull new image if flash wheels keep failing.",
            flush=True,
        )

    last_err: Exception | None = None

    # Reuse previously built sm_120 wheel from hostPath
    cached = _find_cached_builtin_wheel()
    if cached and _wheel_matches_runtime(cached.name, rt):
        try:
            print(f"[flash-attn] trying cached built wheel {cached}", flush=True)
            _pip_install(str(cached))
            _verify()
            return
        except Exception as exc:
            last_err = exc
            print(f"[flash-attn] cached wheel failed ({exc})", flush=True)

    for url in _candidate_urls(rt):
        try:
            if url.startswith("http") and not _head_ok(url):
                print(f"[flash-attn] skip (404/unreachable): {url}", flush=True)
                continue
            local = _download(url) if url.startswith("http") else Path(url)
            print(f"[flash-attn] installing {local}", flush=True)
            _pip_install(str(local))
            _verify()
            return
        except SystemExit:
            raise
        except Exception as exc:
            last_err = exc
            print(f"[flash-attn] candidate failed ({exc}): {url}", flush=True)
            # Drop bad cached download so we don't keep reinstalling it
            try:
                name = url.rstrip("/").split("/")[-1].split("?")[0]
                bad = _cache_dir() / name
                if bad.is_file() and "not a supported wheel" in str(exc):
                    print(f"[flash-attn] removing incompatible cache {bad}", flush=True)
                    bad.unlink(missing_ok=True)
            except Exception:
                pass
            continue

    if _allow_source():
        try:
            built = _source_build_wheel(rt)
            print(f"[flash-attn] installing source-built {built}", flush=True)
            _pip_install(str(built))
            _verify()
            return
        except SystemExit:
            raise
        except Exception as exc:
            last_err = exc
            print(f"[flash-attn] source build failed ({exc})", flush=True)

    _fail(
        "no usable flash-attn FA2 wheel/build "
        f"(last={last_err!r}). Set ATTENTION_BACKEND=sdpa or FLASH_ATTN_PIP_SPEC, "
        "or ensure image is 2.11.0-cuda13.0-cudnn9-devel and allow source "
        "(FLASH_ATTN_ALLOW_SOURCE=1)."
    )


if __name__ == "__main__":
    main()
