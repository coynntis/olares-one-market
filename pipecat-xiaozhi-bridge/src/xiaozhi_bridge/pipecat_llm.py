"""OpenAI-compatible LLM — batch, streaming, and MCP agent harness."""

from __future__ import annotations

import logging
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass
from typing import Any

import httpx

from xiaozhi_bridge.agent_harness import (
    MAX_TOOL_ROUNDS,
    AgentHarnessCallbacks,
    AgentStep,
    run_agent_loop,
)
from xiaozhi_bridge.config import load_settings
from xiaozhi_bridge.llm_capabilities import fetch_llm_capabilities
from xiaozhi_bridge.llm_client import sdk_create_completion
from xiaozhi_bridge.llm_params import build_completion_kwargs, finalize_sdk_completion_kwargs
from xiaozhi_bridge.mcp_tools import get_mcp_openai_tools, tools_configured
from xiaozhi_bridge.openai_endpoints import client_api_key, llm_api_key, llm_base_url
from xiaozhi_bridge.pipeline_types import ChatResult, LlmUsage

logger = logging.getLogger(__name__)

ToolEventFn = Callable[[str, str, str], Awaitable[None]]

_caps_cache: dict[str, Any] | None = None


@dataclass
class StreamChatResult:
    text: str
    elapsed_ms: int = 0
    first_token_ms: int | None = None
    usage: LlmUsage | None = None
    tokens_per_sec: float = 0.0
    backend: str = ""


async def _cached_capabilities() -> dict[str, Any]:
    global _caps_cache
    if _caps_cache is None:
        _caps_cache = await fetch_llm_capabilities()
    return _caps_cache


def invalidate_capabilities_cache() -> None:
    global _caps_cache
    _caps_cache = None


def _llm_timeout_seconds() -> float:
    cfg = load_settings()
    return max(float(cfg.http_timeout or 120.0), 30.0)


def _build_client_and_kwargs(
    messages: list[dict[str, Any]],
    *,
    backend: str,
    supports: dict[str, bool],
    stream: bool,
    tools: list[dict[str, Any]] | None = None,
) -> tuple[Any, dict[str, Any], float]:
    from openai import AsyncOpenAI

    cfg = load_settings()
    timeout_s = _llm_timeout_seconds()
    kwargs = build_completion_kwargs(
        messages, cfg, backend=backend, tools=tools, supports=supports
    )
    kwargs = finalize_sdk_completion_kwargs(kwargs, supports)
    kwargs["stream"] = stream
    if stream:
        kwargs["stream_options"] = {"include_usage": True}
    client = AsyncOpenAI(
        api_key=client_api_key(llm_api_key()),
        base_url=llm_base_url(),
        timeout=httpx.Timeout(timeout=timeout_s, connect=10.0),
        max_retries=0,
    )
    return client, kwargs, timeout_s


def _legacy_tool_adapter(on_tool_event: ToolEventFn) -> AgentHarnessCallbacks:
    async def on_step(step: AgentStep) -> None:
        if step.phase == "running":
            await on_tool_event("start", step.tool_name, step.message)
        elif step.phase == "done":
            await on_tool_event("done", step.tool_name, step.detail)

    return AgentHarnessCallbacks(on_step=on_step)


async def complete_agent_chat(
    messages: list[dict[str, Any]],
    *,
    on_tool_event: ToolEventFn | None = None,
    callbacks: AgentHarnessCallbacks | None = None,
    device_id: str | None = None,
) -> ChatResult:
    """Run LLM with MCP tools until final natural-language reply."""
    cb = callbacks
    if cb is None and on_tool_event:
        cb = _legacy_tool_adapter(on_tool_event)
    return await run_agent_loop(messages, callbacks=cb, timeout_s=_llm_timeout_seconds(), device_id=device_id)


class LlmStreamSession:
    """Single streaming LLM request: async-iterate deltas, then call result()."""

    def __init__(self, messages: list[dict[str, Any]]) -> None:
        self._messages = messages
        self._started = time.perf_counter()
        self._first_token_ms: int | None = None
        self._parts: list[str] = []
        self._usage = LlmUsage()
        self._backend = ""
        self._client: Any | None = None
        self._stream = None
        self._result: StreamChatResult | None = None
        self._closed = False

    async def _ensure_stream(self) -> None:
        if self._stream is not None:
            return
        caps = await _cached_capabilities()
        self._backend = str(caps.get("backend") or "openai_compat")
        supports = caps.get("supports") if isinstance(caps.get("supports"), dict) else {}
        client, kwargs, timeout_s = _build_client_and_kwargs(
            self._messages, backend=self._backend, supports=supports, stream=True
        )
        self._client = client
        logger.info(
            "llm stream start backend=%s timeout=%.0fs messages=%d",
            self._backend,
            timeout_s,
            len(self._messages),
        )
        self._stream = await sdk_create_completion(client, kwargs)

    def __aiter__(self) -> AsyncIterator[str]:
        return self._iter_deltas()

    async def _iter_deltas(self) -> AsyncIterator[str]:
        await self._ensure_stream()
        assert self._stream is not None
        try:
            async for chunk in self._stream:
                if chunk.usage:
                    self._usage = LlmUsage(
                        prompt_tokens=int(chunk.usage.prompt_tokens or 0),
                        completion_tokens=int(chunk.usage.completion_tokens or 0),
                        total_tokens=int(chunk.usage.total_tokens or 0),
                    )
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta.content or ""
                if delta:
                    if self._first_token_ms is None:
                        self._first_token_ms = int((time.perf_counter() - self._started) * 1000)
                    self._parts.append(delta)
                    yield delta
        finally:
            # Drop TCP/stream ASAP so llama.cpp / HAMI can release GPU before OmniVoice.
            await self.aclose()

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = True
        stream = self._stream
        client = self._client
        self._stream = None
        self._client = None
        if stream is not None:
            close = getattr(stream, "close", None) or getattr(stream, "aclose", None)
            if close is not None:
                try:
                    out = close()
                    if hasattr(out, "__await__"):
                        await out
                except Exception:
                    pass
        if client is not None:
            try:
                await client.close()
            except Exception:
                pass

    def result(self) -> StreamChatResult:
        if self._result is not None:
            return self._result
        elapsed_ms = int((time.perf_counter() - self._started) * 1000)
        text = "".join(self._parts).strip()
        # Generation throughput: tokens after first token → last token (not wall including waits).
        if self._first_token_ms is not None and self._usage.completion_tokens:
            gen_secs = max((elapsed_ms - self._first_token_ms) / 1000.0, 0.001)
            tps = self._usage.completion_tokens / gen_secs
        else:
            secs = max(elapsed_ms / 1000.0, 0.001)
            tps = self._usage.completion_tokens / secs if self._usage.completion_tokens else 0.0
        self._result = StreamChatResult(
            text=text,
            elapsed_ms=elapsed_ms,
            first_token_ms=self._first_token_ms,
            usage=self._usage,
            tokens_per_sec=round(tps, 2),
            backend=self._backend,
        )
        logger.info(
            "llm stream done ms=%d first_token_ms=%s chars=%d tps=%.2f backend=%s",
            elapsed_ms,
            self._first_token_ms,
            len(text),
            tps,
            self._backend,
        )
        return self._result


async def open_llm_stream(messages: list[dict[str, Any]]) -> LlmStreamSession:
    return LlmStreamSession(messages)


async def complete_chat(messages: list[dict[str, Any]]) -> ChatResult:
    caps = await _cached_capabilities()
    backend = str(caps.get("backend") or "openai_compat")
    supports = caps.get("supports") if isinstance(caps.get("supports"), dict) else {}
    return await _openai_complete(messages, backend=backend, supports=supports)


async def mcp_tools_enabled() -> bool:
    if tools_configured():
        tools, router = await get_mcp_openai_tools()
        return bool(tools and router)
    return False


async def _openai_complete(
    messages: list[dict[str, Any]],
    *,
    backend: str,
    supports: dict[str, bool],
) -> ChatResult:
    timeout_s = _llm_timeout_seconds()
    started = time.perf_counter()
    client, kwargs, _ = _build_client_and_kwargs(
        messages, backend=backend, supports=supports, stream=False
    )
    logger.info("llm request backend=%s timeout=%.0fs messages=%d", backend, timeout_s, len(messages))
    try:
        resp = await sdk_create_completion(client, kwargs)
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        choice = resp.choices[0].message
        text = (choice.content or "").strip()
        usage_raw = resp.usage
        usage = LlmUsage(
            prompt_tokens=int(getattr(usage_raw, "prompt_tokens", 0) or 0),
            completion_tokens=int(getattr(usage_raw, "completion_tokens", 0) or 0),
            total_tokens=int(getattr(usage_raw, "total_tokens", 0) or 0),
        )
        secs = max(elapsed_ms / 1000.0, 0.001)
        tps = usage.completion_tokens / secs if usage.completion_tokens else 0.0
        logger.info(
            "llm done ms=%d chars=%d tps=%.2f backend=%s",
            elapsed_ms,
            len(text),
            tps,
            backend,
        )
        return ChatResult(
            text=text,
            elapsed_ms=elapsed_ms,
            usage=usage,
            tokens_per_sec=round(tps, 2),
            backend=backend,
        )
    finally:
        try:
            await client.close()
        except Exception:
            pass
