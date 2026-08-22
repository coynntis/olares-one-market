"""Agent loop harness — tool announcements, step events, human labels."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from xiaozhi_bridge.agent_tool_utils import (
    clip_tool_content,
    ensure_tool_call_ids,
    generated_image_caption,
    human_tool_label,
    light_generated_image_meta,
    process_mcp_tool_result,
    tool_announce_message,
    tool_done_message,
    tool_running_message,
)
from xiaozhi_bridge.llm_capabilities import fetch_llm_capabilities
from xiaozhi_bridge.llm_client import sdk_create_completion
from xiaozhi_bridge.llm_params import build_completion_kwargs, finalize_sdk_completion_kwargs
from xiaozhi_bridge.mcp_tools import McpToolRouter, get_mcp_openai_tools
from xiaozhi_bridge.openai_endpoints import client_api_key, llm_api_key, llm_base_url
from xiaozhi_bridge.pipeline_types import AgentStep, ChatResult, LlmUsage

logger = logging.getLogger(__name__)

MAX_TOOL_ROUNDS = 8
MAX_ANNOUNCE_SPEAK_CHARS = 160
AgentStepCallback = Callable[[AgentStep], Awaitable[None]]

def humanize_tool_name(openai_name: str) -> str:
    """Turn namespaced OpenAI tool id into short UI label (no URLs)."""
    return human_tool_label(openai_name)


@dataclass
class AgentHarnessCallbacks:
    """Hooks for voice TTS, WebSocket UI, REST step logs."""

    on_step: AgentStepCallback | None = None
    on_announce_spoken: Callable[[str], Awaitable[None]] | None = None

    async def emit(self, step: AgentStep) -> None:
        if self.on_step:
            await self.on_step(step)


def _merge_usage(total: LlmUsage, add: LlmUsage) -> LlmUsage:
    return LlmUsage(
        prompt_tokens=total.prompt_tokens + add.prompt_tokens,
        completion_tokens=total.completion_tokens + add.completion_tokens,
        total_tokens=total.total_tokens + add.total_tokens,
    )


def _ensure_tool_call_ids(tool_calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return ensure_tool_call_ids(tool_calls)


def _serialize_tool_calls(message: Any) -> list[dict[str, Any]]:
    raw = getattr(message, "tool_calls", None) or []
    out: list[dict[str, Any]] = []
    for tc in raw:
        fn = getattr(tc, "function", None)
        if not fn:
            continue
        out.append(
            {
                "id": tc.id,
                "type": getattr(tc, "type", None) or "function",
                "function": {
                    "name": fn.name,
                    "arguments": fn.arguments or "{}",
                },
            }
        )
    return _ensure_tool_call_ids(out)


async def _completion_step(
    messages: list[dict[str, Any]],
    *,
    backend: str,
    supports: dict[str, bool],
    tools: list[dict[str, Any]] | None,
    timeout_s: float,
) -> tuple[str, list[dict[str, Any]], LlmUsage, int]:
    from openai import AsyncOpenAI
    import httpx

    from xiaozhi_bridge.config import load_settings

    started = time.perf_counter()
    cfg = load_settings()
    kwargs = build_completion_kwargs(
        messages, cfg, backend=backend, tools=tools, supports=supports
    )
    kwargs = finalize_sdk_completion_kwargs(kwargs, supports)
    kwargs["stream"] = False
    client = AsyncOpenAI(
        api_key=client_api_key(llm_api_key()),
        base_url=llm_base_url(),
        timeout=httpx.Timeout(timeout=timeout_s, connect=10.0),
        max_retries=0,
    )
    try:
        resp = await sdk_create_completion(client, kwargs)
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        choice = resp.choices[0].message
        text = (choice.content or "").strip()
        tool_calls = _serialize_tool_calls(choice)
        usage_raw = resp.usage
        usage = LlmUsage(
            prompt_tokens=int(getattr(usage_raw, "prompt_tokens", 0) or 0),
            completion_tokens=int(getattr(usage_raw, "completion_tokens", 0) or 0),
            total_tokens=int(getattr(usage_raw, "total_tokens", 0) or 0),
        )
        return text, tool_calls, usage, elapsed_ms
    finally:
        try:
            await client.close()
        except Exception:
            pass


async def _completion_step_timed(
    messages: list[dict[str, Any]],
    *,
    backend: str,
    supports: dict[str, bool],
    tools: list[dict[str, Any]] | None,
    timeout_s: float,
) -> tuple[str, list[dict[str, Any]], LlmUsage, int]:
    return await asyncio.wait_for(
        _completion_step(
            messages,
            backend=backend,
            supports=supports,
            tools=tools,
            timeout_s=timeout_s,
        ),
        timeout=timeout_s,
    )


async def _openai_plain(
    messages: list[dict[str, Any]],
    *,
    backend: str,
    supports: dict[str, bool],
    timeout_s: float,
) -> ChatResult:
    text, _, usage, elapsed_ms = await _completion_step_timed(
        messages, backend=backend, supports=supports, tools=None, timeout_s=timeout_s
    )
    secs = max(elapsed_ms / 1000.0, 0.001)
    tps = usage.completion_tokens / secs if usage.completion_tokens else 0.0
    return ChatResult(
        text=text,
        elapsed_ms=elapsed_ms,
        usage=usage,
        tokens_per_sec=round(tps, 2),
        backend=backend,
    )


async def _emit_error_step(
    cb: AgentHarnessCallbacks,
    steps: list[AgentStep],
    *,
    global_step: int,
    tool_rounds: int,
    message: str,
) -> None:
    step = AgentStep(
        phase="error",
        round_index=tool_rounds,
        step_index=global_step,
        tool_name="",
        label="Agent",
        message=message,
    )
    steps.append(step)
    await cb.emit(step)


async def _chat_result(
    *,
    text: str,
    started: float,
    total_usage: LlmUsage,
    total_ms: int,
    backend: str,
    tool_rounds: int,
    trace: list[dict[str, Any]],
    steps: list[AgentStep],
    generated_images: list[dict[str, Any]],
    cb: AgentHarnessCallbacks,
    global_step: int,
) -> ChatResult:
    secs = max((time.perf_counter() - started), 0.001)
    tps = total_usage.completion_tokens / secs if total_usage.completion_tokens else 0.0
    if text:
        trace.append({"role": "assistant", "text": text, "meta": {}})
        final_step = AgentStep(
            phase="final",
            round_index=tool_rounds,
            step_index=global_step,
            tool_name="",
            label="Reply",
            message=text,
        )
        steps.append(final_step)
        await cb.emit(final_step)
    return ChatResult(
        text=text,
        elapsed_ms=total_ms or int((time.perf_counter() - started) * 1000),
        usage=total_usage,
        tokens_per_sec=round(tps, 2),
        backend=backend,
        tool_rounds=tool_rounds,
        tool_trace=trace,
        agent_steps=steps,
        generated_images=generated_images or [],
    )


async def run_agent_loop(
    messages: list[dict[str, Any]],
    *,
    callbacks: AgentHarnessCallbacks | None = None,
    timeout_s: float = 120.0,
    device_id: str | None = None,
) -> ChatResult:
    """LLM + MCP tools until final reply. Emits announce → running → done per tool."""
    tools, router = await get_mcp_openai_tools(device_id=device_id)
    if not tools or router is None:
        caps = await fetch_llm_capabilities()
        backend = str(caps.get("backend") or "openai_compat")
        supports = caps.get("supports") if isinstance(caps.get("supports"), dict) else {}
        return await _openai_plain(messages, backend=backend, supports=supports, timeout_s=timeout_s)

    caps = await fetch_llm_capabilities()
    backend = str(caps.get("backend") or "openai_compat")
    supports = caps.get("supports") if isinstance(caps.get("supports"), dict) else {}

    working = list(messages)
    total_usage = LlmUsage()
    total_ms = 0
    trace: list[dict[str, Any]] = []
    steps: list[AgentStep] = []
    generated_images: list[dict[str, Any]] = []
    tool_rounds = 0
    started = time.perf_counter()
    cb = callbacks or AgentHarnessCallbacks()
    global_step = 0
    preamble_spoken = False

    async def _llm_with_retries(*, use_tools: list[dict[str, Any]] | None) -> tuple[str, list[dict[str, Any]], LlmUsage, int]:
        nonlocal supports
        try:
            return await _completion_step_timed(
                working,
                backend=backend,
                supports=supports,
                tools=use_tools,
                timeout_s=timeout_s,
            )
        except asyncio.TimeoutError:
            raise
        except Exception as e:
            err = str(e).lower()
            if "top_k" in err and supports.get("top_k", True):
                logger.warning("LLM rejected top_k, retrying without: %s", e)
                supports = {**supports, "top_k": False}
                return await _completion_step_timed(
                    working,
                    backend=backend,
                    supports=supports,
                    tools=use_tools,
                    timeout_s=timeout_s,
                )
            if use_tools and ("tool" in err or "function" in err):
                logger.warning("LLM rejected tools, falling back: %s", e)
                return await _completion_step_timed(
                    working,
                    backend=backend,
                    supports=supports,
                    tools=None,
                    timeout_s=timeout_s,
                )
            raise

    while tool_rounds < MAX_TOOL_ROUNDS:
        try:
            text, tool_calls, usage, elapsed_ms = await _llm_with_retries(use_tools=tools)
        except asyncio.TimeoutError:
            await _emit_error_step(
                cb,
                steps,
                global_step=global_step,
                tool_rounds=tool_rounds,
                message="LLM timed out — try a shorter question or check LLM load",
            )
            raise RuntimeError("LLM timed out during agent step") from None

        total_usage = _merge_usage(total_usage, usage)
        total_ms += elapsed_ms

        if not tool_calls:
            return await _chat_result(
                text=text,
                started=started,
                total_usage=total_usage,
                total_ms=total_ms,
                backend=backend,
                tool_rounds=tool_rounds,
                trace=trace,
                steps=steps,
                generated_images=generated_images,
                cb=cb,
                global_step=global_step,
            )

        tool_calls = _ensure_tool_call_ids(tool_calls)
        working.append(
            {
                "role": "assistant",
                "content": text or None,
                "tool_calls": tool_calls,
            }
        )
        trace.append(
            {
                "role": "assistant",
                "text": text or "",
                "meta": {"tool_calls": tool_calls},
            }
        )

        round_preamble = (text or "").strip()
        if round_preamble and not preamble_spoken:
            announce = AgentStep(
                phase="announce",
                round_index=tool_rounds,
                step_index=global_step,
                tool_name="",
                label="Assistant",
                message=round_preamble,
            )
            steps.append(announce)
            await cb.emit(announce)
            preamble_spoken = True
            global_step += 1

        for tc in tool_calls:
            fn = tc.get("function") or {}
            tool_name = str(fn.get("name") or "")
            label = humanize_tool_name(tool_name)
            raw_args = str(fn.get("arguments") or "{}")
            try:
                args = json.loads(raw_args) if raw_args.strip() else {}
            except json.JSONDecodeError:
                args = {}
            if not isinstance(args, dict):
                args = {}

            if not round_preamble:
                per_tool_msg = tool_announce_message(tool_name)
                announce = AgentStep(
                    phase="announce",
                    round_index=tool_rounds,
                    step_index=global_step,
                    tool_name=tool_name,
                    label=label,
                    message=per_tool_msg,
                )
                steps.append(announce)
                await cb.emit(announce)
                if cb.on_announce_spoken and len(per_tool_msg) <= MAX_ANNOUNCE_SPEAK_CHARS:
                    await cb.on_announce_spoken(per_tool_msg)
                global_step += 1

            running = AgentStep(
                phase="running",
                round_index=tool_rounds,
                step_index=global_step,
                tool_name=tool_name,
                label=label,
                message=tool_running_message(tool_name),
            )
            steps.append(running)
            await cb.emit(running)
            global_step += 1

            raw_tool_text = await router.call(tool_name, args)
            tool_text, gen_image = process_mcp_tool_result(tool_name, raw_tool_text)
            call_id = str(tc.get("id") or "")
            tool_meta: dict[str, Any] = {"tool_call_id": call_id, "tool_name": tool_name}
            if gen_image:
                # Keep full data_url only on ChatResult.generated_images (in-memory).
                # Trace/SQLite meta must stay small or inserts / WS blow up.
                generated_images.append(gen_image)
                tool_meta["generated_image"] = light_generated_image_meta(gen_image)
            working.append({"role": "tool", "tool_call_id": call_id, "content": tool_text})
            trace.append(
                {
                    "role": "tool",
                    "text": tool_text,
                    "meta": tool_meta,
                }
            )

            done_detail = tool_text[:240]
            if gen_image:
                w, h = gen_image.get("width"), gen_image.get("height")
                if w and h:
                    done_detail = f"{w}×{h} seed={gen_image.get('seed', '?')}"

            done = AgentStep(
                phase="done",
                round_index=tool_rounds,
                step_index=global_step,
                tool_name=tool_name,
                label=label,
                message=tool_done_message(tool_name),
                detail=done_detail,
            )
            steps.append(done)
            await cb.emit(done)
            global_step += 1

        preamble_spoken = False
        tool_rounds += 1

        synth = AgentStep(
            phase="running",
            round_index=tool_rounds,
            step_index=global_step,
            tool_name="",
            label="Assistant",
            message="Composing answer from results…",
        )
        steps.append(synth)
        await cb.emit(synth)
        global_step += 1

        try:
            synth_text, synth_tools, usage, elapsed_ms = await _llm_with_retries(use_tools=None)
        except asyncio.TimeoutError:
            await _emit_error_step(
                cb,
                steps,
                global_step=global_step,
                tool_rounds=tool_rounds,
                message="Timed out composing answer after tool use",
            )
            raise RuntimeError("Timed out composing answer after tool use") from None

        total_usage = _merge_usage(total_usage, usage)
        total_ms += elapsed_ms

        if synth_text.strip() and not synth_tools:
            return await _chat_result(
                text=synth_text,
                started=started,
                total_usage=total_usage,
                total_ms=total_ms,
                backend=backend,
                tool_rounds=tool_rounds,
                trace=trace,
                steps=steps,
                generated_images=generated_images,
                cb=cb,
                global_step=global_step,
            )

        if synth_tools:
            logger.warning("synthesis pass returned tool_calls — continuing agent loop")
            working.append(
                {
                    "role": "assistant",
                    "content": synth_text or None,
                    "tool_calls": synth_tools,
                }
            )
            continue

        logger.info("synthesis empty after tool round %d — allowing another tool round", tool_rounds)

    logger.warning("max tool rounds (%d) reached", MAX_TOOL_ROUNDS)
    final = await _openai_plain(working, backend=backend, supports=supports, timeout_s=timeout_s)
    total_usage = _merge_usage(total_usage, final.usage)
    return await _chat_result(
        text=final.text,
        started=started,
        total_usage=total_usage,
        total_ms=total_ms + final.elapsed_ms,
        backend=final.backend,
        tool_rounds=tool_rounds,
        trace=trace,
        steps=steps,
        generated_images=generated_images,
        cb=cb,
        global_step=global_step,
    )
