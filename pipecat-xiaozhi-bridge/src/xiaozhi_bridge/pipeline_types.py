"""Shared pipeline timing / LLM usage types."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class LlmUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


@dataclass
class AgentStep:
    phase: str  # announce | running | done | final | error
    round_index: int = 0
    step_index: int = 0
    tool_name: str = ""
    label: str = ""
    message: str = ""
    detail: str = ""
    image_url: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ChatResult:
    text: str
    elapsed_ms: int = 0
    usage: LlmUsage = field(default_factory=LlmUsage)
    tokens_per_sec: float = 0.0
    backend: str = ""
    tool_rounds: int = 0
    tool_trace: list[dict[str, Any]] = field(default_factory=list)
    agent_steps: list[AgentStep] = field(default_factory=list)
    generated_images: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class PipelineStats:
    stt_ms: int | None = None
    llm_ms: int = 0
    tts_ms: int = 0
    tts_http_ms: int = 0
    tts_decode_ms: int = 0
    tts_audio_ms: int = 0
    tts_rtf: float = 0.0
    tts_via: str = ""
    tts_warmup_ms: int = 0
    opus_ms: int = 0
    total_ms: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    tokens_per_sec: float = 0.0
    backend: str = ""
    first_token_ms: int | None = None
    first_audio_ms: int | None = None
    segments: int = 0
    tool_rounds: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None and v != 0 and v != ""}
