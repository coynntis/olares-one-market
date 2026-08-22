"""Detect LLM backend capabilities from /v1/models and optional probe completion."""

from __future__ import annotations

import json
import logging
import re
from typing import Any

import httpx

from xiaozhi_bridge.config import BridgeSettings, load_settings
from xiaozhi_bridge.openai_endpoints import auth_headers, client_api_key, llm_api_key

logger = logging.getLogger(__name__)

_THINKING_MODEL_RE = re.compile(
    r"qwen3|qwen-?3|qwen3\.5|glm|deepseek-r1|nemotron.*reason",
    re.IGNORECASE,
)

_VISION_MODEL_HINTS = (
    "vision",
    "-vl",
    "_vl",
    "llava",
    "pixtral",
    "gemini",
    "gpt-4o",
    "gpt-4.1",
    "qwen2-vl",
    "qwen3-vl",
    "qvq",
    "internvl",
    "minicpm-v",
    "glm-4v",
    "cogvlm",
)


def model_likely_supports_vision(model_id: str) -> bool:
    hay = (model_id or "").lower()
    return any(h in hay for h in _VISION_MODEL_HINTS)


def _model_entry(models_payload: dict[str, Any], model_id: str) -> dict[str, Any] | None:
    data = models_payload.get("data")
    if not isinstance(data, list):
        return None
    for item in data:
        if isinstance(item, dict) and str(item.get("id", "")) == model_id:
            return item
    if data and isinstance(data[0], dict):
        return data[0]
    return None


def _detect_backend(models_payload: dict[str, Any], model_entry: dict[str, Any] | None) -> str:
    owned = str((model_entry or {}).get("owned_by") or "").lower()
    root = str((model_entry or {}).get("root") or "").lower()
    blob = json.dumps(models_payload).lower()
    if "vllm" in owned or "vllm" in blob:
        return "vllm"
    if "ggml" in owned or "llama.cpp" in blob or "llamacpp" in blob:
        return "llamacpp"
    if owned in ("openai", "system"):
        return "openai"
    if "llama" in root or "gguf" in root:
        return "llamacpp"
    return "openai_compat"


def _thinking_capable(model_id: str, root: str) -> bool:
    hay = f"{model_id} {root}"
    return bool(_THINKING_MODEL_RE.search(hay))


def capabilities_from_models(
    models_payload: dict[str, Any],
    *,
    model_id: str,
    probe_top_k: bool | None = None,
) -> dict[str, Any]:
    entry = _model_entry(models_payload, model_id)
    backend = _detect_backend(models_payload, entry)
    root = str((entry or {}).get("root") or "")
    think = _thinking_capable(model_id, root)

    # OpenAI-compatible servers generally accept temperature/top_p/max_tokens.
    supports: dict[str, bool] = {
        "temperature": True,
        "top_p": True,
        "top_k": False,
        "max_tokens": True,
        "think_mode": think,
    }
    if probe_top_k is True:
        supports["top_k"] = True
    elif probe_top_k is None and backend == "llamacpp":
        supports["top_k"] = True

    return {
        "backend": backend,
        "model": model_id,
        "root": root or None,
        "owned_by": (entry or {}).get("owned_by"),
        "supports": supports,
    }


async def _probe_top_k(base: str, model_id: str, headers: dict[str, str], timeout: float) -> bool:
    """Send tiny completion with top_k; some vLLM builds reject it."""
    url = f"{base.rstrip('/')}/chat/completions"
    body = {
        "model": model_id,
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 1,
        "temperature": 0,
        "top_k": 20,
    }
    try:
        async with httpx.AsyncClient(timeout=min(timeout, 15.0), follow_redirects=True) as client:
            resp = await client.post(url, json=body, headers=headers)
        if resp.status_code < 400:
            return True
        err = (resp.text or "").lower()
        return "top_k" not in err and "unknown" not in err
    except Exception:
        return False


async def fetch_llm_capabilities(cfg: BridgeSettings | None = None) -> dict[str, Any]:
    s = cfg or load_settings()
    base = s.llm_url()
    model_id = s.llm_model.strip()
    if not base:
        return {"ok": False, "error": "LLM base URL not set"}
    if not model_id:
        return {"ok": False, "error": "LLM model name not set"}

    url = f"{base.rstrip('/')}/models"
    headers = auth_headers(llm_api_key())
    if not headers.get("Authorization"):
        headers["Authorization"] = f"Bearer {client_api_key(None)}"

    try:
        async with httpx.AsyncClient(timeout=min(s.http_timeout, 30.0), follow_redirects=True) as client:
            resp = await client.get(url, headers=headers)
        if resp.status_code >= 400:
            return {"ok": False, "error": (resp.text or "")[:240], "url": url}
        payload = resp.json()
        if not isinstance(payload, dict):
            return {"ok": False, "error": "invalid models payload"}
    except Exception as e:
        return {"ok": False, "error": str(e), "url": url}

    probe_top_k: bool | None = None
    caps = capabilities_from_models(payload, model_id=model_id)
    backend = str(caps.get("backend") or "openai_compat")
    if backend in ("vllm", "llamacpp", "openai_compat"):
        probe_top_k = await _probe_top_k(base, model_id, headers, s.http_timeout)
        caps = capabilities_from_models(payload, model_id=model_id, probe_top_k=probe_top_k)

    caps["ok"] = True
    caps["url"] = url
    return caps
