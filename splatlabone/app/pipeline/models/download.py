"""Download geometry model weights (ModelScope-first for Meta/facebook)."""

from __future__ import annotations

import logging
import os
import urllib.request
from pathlib import Path

from pipeline.models.cache import CACHE_ROOT, MODEL_REGISTRY, MODELSCOPE_CACHE, model_dir

log = logging.getLogger("splatlab.models")


def _mark_ok(path: Path) -> None:
    (path.parent / f"{path.name}.ok").write_text("1")


def _skip_if_ok(path: Path) -> bool:
    ok = path.parent / f"{path.name}.ok"
    return path.is_file() and ok.is_file()


def _download_hf_file(model_id: str, filename: str, dest: Path, token: str | None) -> Path:
    from huggingface_hub import hf_hub_download

    dest.parent.mkdir(parents=True, exist_ok=True)
    kwargs: dict = {"repo_id": model_id, "filename": filename, "local_dir": str(dest.parent)}
    if token:
        kwargs["token"] = token
    endpoint = os.environ.get("HF_ENDPOINT")
    if endpoint:
        os.environ["HF_ENDPOINT"] = endpoint
    out = hf_hub_download(**kwargs)
    path = Path(out)
    if path.is_file():
        return path
    fallback = dest.parent / filename
    if fallback.is_file():
        return fallback
    return path


def _download_hf(model_id: str, dest: Path, token: str | None) -> Path:
    from huggingface_hub import snapshot_download

    dest.mkdir(parents=True, exist_ok=True)
    kwargs: dict = {}
    if token:
        kwargs["token"] = token
    endpoint = os.environ.get("HF_ENDPOINT")
    if endpoint:
        os.environ["HF_ENDPOINT"] = endpoint
    snapshot_download(repo_id=model_id, local_dir=str(dest), **kwargs)
    return dest


def _download_modelscope(model_id: str, dest: Path, token: str | None) -> Path:
    from modelscope import snapshot_download

    MODELSCOPE_CACHE.mkdir(parents=True, exist_ok=True)
    kwargs: dict = {"cache_dir": str(MODELSCOPE_CACHE)}
    if token:
        kwargs["token"] = token
    out = snapshot_download(model_id, **kwargs)
    return Path(out)


def _download_url(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if _skip_if_ok(dest):
        log.info("already present: %s", dest)
        return dest
    log.info("downloading %s -> %s", url, dest)
    urllib.request.urlretrieve(url, dest)  # noqa: S310
    _mark_ok(dest)
    return dest


def download_model(key: str, *, force: bool = False) -> Path:
    entry = MODEL_REGISTRY.get(key)
    if not entry:
        raise KeyError(f"unknown model key: {key}")

    dest = model_dir(key)
    dest.mkdir(parents=True, exist_ok=True)
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    ms_token = os.environ.get("MODELSCOPE_TOKEN")
    hub_mode = os.environ.get("SPLATLAB_MODEL_HUB", "auto")

    if entry.get("hub") == "url":
        path = dest / entry["filename"]
        if not force and _skip_if_ok(path):
            return path
        return _download_url(entry["url"], path)

    model_id = entry["model_id"]
    primary = entry.get("hub", "huggingface")
    if hub_mode == "auto" and entry.get("hf_fallback"):
        primary = entry["hub"]

    if entry.get("filename") and primary == "huggingface":
        path = dest / entry["filename"]
        if not force and _skip_if_ok(path):
            return path
        try:
            out = _download_hf_file(model_id, entry["filename"], path, hf_token)
            _mark_ok(out)
            log.info("downloaded %s via hf file -> %s", key, out)
            return out
        except Exception as exc:
            raise RuntimeError(f"download failed for {key}: hf file: {exc}") from exc

    errors: list[str] = []
    if primary == "modelscope":
        try:
            out = _download_modelscope(model_id, dest, ms_token)
            log.info("downloaded %s via modelscope -> %s", key, out)
            return Path(out)
        except Exception as exc:
            errors.append(f"modelscope: {exc}")
            fb = entry.get("hf_fallback")
            if fb:
                try:
                    out = _download_hf(fb, dest, hf_token)
                    log.info("downloaded %s via hf fallback -> %s", key, out)
                    return out
                except Exception as exc2:
                    errors.append(f"hf: {exc2}")
    else:
        try:
            out = _download_hf(model_id, dest, hf_token)
            log.info("downloaded %s via huggingface -> %s", key, out)
            return out
        except Exception as exc:
            errors.append(f"hf: {exc}")

    raise RuntimeError(f"download failed for {key}: {'; '.join(errors)}")


def prefetch_all(keys: list[str]) -> dict[str, str]:
    results: dict[str, str] = {}
    for key in keys:
        try:
            path = download_model(key)
            results[key] = str(path)
        except Exception as exc:
            log.warning("prefetch %s failed: %s", key, exc)
            results[key] = f"error: {exc}"
    return results
