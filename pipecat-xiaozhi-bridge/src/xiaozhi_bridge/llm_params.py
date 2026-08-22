"""Build OpenAI chat completion request kwargs from bridge settings."""

from __future__ import annotations

import re
from typing import Any

from xiaozhi_bridge.config import BridgeSettings, load_settings

# OpenAI Python SDK accepts these at the top level of chat.completions.create().
# Anything else must go in extra_body or the SDK raises TypeError before HTTP.
_SDK_NATIVE_PARAMS = frozenset(
    {
        "model",
        "messages",
        "stream",
        "temperature",
        "top_p",
        "max_tokens",
        "max_completion_tokens",
        "n",
        "stop",
        "presence_penalty",
        "frequency_penalty",
        "logit_bias",
        "user",
        "tools",
        "tool_choice",
        "response_format",
        "seed",
        "stream_options",
        "parallel_tool_calls",
        "service_tier",
        "store",
        "metadata",
        "reasoning_effort",
        "web_search_options",
        "audio",
        "modalities",
        "prediction",
        "extra_body",
        "extra_headers",
        "extra_query",
        "timeout",
    }
)

# llama.cpp / vLLM / compat servers — never pass these as top-level SDK kwargs.
_EXTENSION_BODY_KEYS = frozenset(
    {
        "top_k",
        "min_p",
        "repetition_penalty",
        "repeat_penalty",
        "tfs_z",
        "typical_p",
        "mirostat",
        "mirostat_tau",
        "mirostat_eta",
    }
)

_UNEXPECTED_KWARG_RE = re.compile(r"unexpected keyword argument ['\"](\w+)['\"]")


def _f(value: float, default: float | None = None) -> float | None:
    if value < 0:
        return default
    return value


def _i(value: int) -> int | None:
    return value if value > 0 else None


def _extra_body(kwargs: dict[str, Any]) -> dict[str, Any]:
    extra = kwargs.get("extra_body")
    if isinstance(extra, dict):
        return extra
    extra = {}
    kwargs["extra_body"] = extra
    return extra


def build_completion_kwargs(
    messages: list[dict[str, Any]],
    cfg: BridgeSettings | None = None,
    *,
    backend: str = "",
    tools: list[dict[str, Any]] | None = None,
    supports: dict[str, bool] | None = None,
) -> dict[str, Any]:
    s = cfg or load_settings()
    sup = supports or {}
    msgs = list(messages)
    kwargs: dict[str, Any] = {
        "model": s.llm_model.strip() or "gpt-4.1-mini",
        "messages": msgs,
    }

    temp = _f(s.llm_temperature, None)
    if temp is not None and sup.get("temperature", True):
        kwargs["temperature"] = temp
    top_p = _f(s.llm_top_p, None)
    if top_p is not None and sup.get("top_p", True):
        kwargs["top_p"] = top_p
    top_k = _i(s.llm_top_k)
    if top_k is not None and sup.get("top_k", True):
        _extra_body(kwargs)["top_k"] = top_k
    max_tok = _i(s.llm_max_tokens)
    if max_tok is not None and sup.get("max_tokens", True):
        kwargs["max_tokens"] = max_tok

    think = (s.llm_think_mode or "auto").strip().lower()
    think_ok = sup.get("think_mode", True)
    if think_ok and think in ("think", "no_think") and backend in ("vllm", "llamacpp", "openai_compat"):
        enable = think == "think"
        ctk = _extra_body(kwargs).setdefault("chat_template_kwargs", {})
        if isinstance(ctk, dict):
            ctk["enable_thinking"] = enable
        if backend == "llamacpp":
            prefix = "/think" if enable else "/no_think"
            msgs = _prefix_last_user(msgs, prefix)
            kwargs["messages"] = msgs

    if tools:
        kwargs["tools"] = tools
        kwargs["tool_choice"] = "auto"

    return kwargs


def finalize_sdk_completion_kwargs(
    kwargs: dict[str, Any],
    supports: dict[str, bool],
) -> dict[str, Any]:
    """Ensure only SDK-native keys are top-level; strip unsupported extension fields."""
    out = dict(kwargs)
    extra: dict[str, Any] = {}
    raw_extra = out.pop("extra_body", None)
    if isinstance(raw_extra, dict):
        extra.update(raw_extra)

    # Defensive: relocate any extension keys that ended up top-level (regression guard).
    for key in list(out.keys()):
        if key in _EXTENSION_BODY_KEYS:
            val = out.pop(key)
            if val is not None:
                extra[key] = val

    if not supports.get("top_k", True):
        extra.pop("top_k", None)
    if not supports.get("think_mode", True):
        extra.pop("chat_template_kwargs", None)

    if extra:
        out["extra_body"] = extra
    elif "extra_body" in out:
        out.pop("extra_body", None)
    return out


def parse_unexpected_sdk_kwarg(exc: BaseException) -> str | None:
    """Extract param name from AsyncOpenAI TypeError ('unexpected keyword argument')."""
    m = _UNEXPECTED_KWARG_RE.search(str(exc))
    return m.group(1) if m else None


def relocate_sdk_kwarg_to_extra_body(kwargs: dict[str, Any], param: str) -> dict[str, Any] | None:
    """Move one top-level kwarg into extra_body for a single retry."""
    if param not in kwargs:
        return None
    out = dict(kwargs)
    val = out.pop(param)
    extra = dict(out.get("extra_body") or {}) if isinstance(out.get("extra_body"), dict) else {}
    extra[param] = val
    out["extra_body"] = extra
    return out


def strip_sdk_kwarg(kwargs: dict[str, Any], param: str) -> dict[str, Any]:
    """Remove param from top-level or extra_body."""
    out = dict(kwargs)
    out.pop(param, None)
    extra = out.get("extra_body")
    if isinstance(extra, dict):
        extra = dict(extra)
        extra.pop(param, None)
        if extra:
            out["extra_body"] = extra
        else:
            out.pop("extra_body", None)
    return out


def _prefix_last_user(messages: list[dict[str, Any]], prefix: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    last_user = -1
    for i, m in enumerate(messages):
        if m.get("role") == "user":
            last_user = i
    for i, m in enumerate(messages):
        if i != last_user:
            out.append(m)
            continue
        content = m.get("content")
        if isinstance(content, str) and not content.lstrip().startswith(prefix):
            out.append({**m, "content": f"{prefix}\n{content}"})
        else:
            out.append(m)
    return out
