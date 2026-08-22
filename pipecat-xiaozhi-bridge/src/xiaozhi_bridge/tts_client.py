"""TTS client — MeloTTS (CPU) + OmniVoice (GPU) via OpenAI-compatible HTTP."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from urllib.parse import urlparse

from xiaozhi_bridge.audio import speech_bytes_to_pcm
from xiaozhi_bridge.config import BridgeSettings, load_settings
from xiaozhi_bridge.openai_endpoints import auth_headers, tts_api_key, tts_base_url
from xiaozhi_bridge.tts_params import (
    AUDIO8_CLUSTER_URL,
    KOKORO_CLUSTER_URL,
    MELO_CLUSTER_URL,
    SHERPA_CLUSTER_URL,
    build_audio8_speech_json,
    build_clone_form,
    build_kokoro_speech_json,
    build_melo_speech_json,
    build_sherpa_speech_json,
    build_speech_json,
    default_cluster_tts_url,
    effective_tts_provider,
    resolve_clone_ref,
    segmenter_min_chars,
)

logger = logging.getLogger(__name__)

__all__ = [
    "synthesize_speech_pcm",
    "synthesize_speech_pcm_timed",
    "segmenter_min_chars",
    "resolve_clone_ref",
    "TtsTiming",
    "MELO_CLUSTER_URL",
    "SHERPA_CLUSTER_URL",
    "KOKORO_CLUSTER_URL",
    "AUDIO8_CLUSTER_URL",
]


@dataclass
class TtsTiming:
    """Breakdown of one TTS call."""

    http_ms: int = 0
    decode_ms: int = 0
    audio_ms: int = 0
    rtf: float = 0.0
    via: str = ""
    bytes_in: int = 0
    num_step: int = 0
    endpoint: str = ""
    provider: str = ""


def _classify_tts_via(base_url: str, provider: str) -> str:
    if provider in ("melo", "sherpa", "kokoro", "audio8"):
        host = (urlparse(base_url).hostname or "").lower()
        if any(
            x in host
            for x in (
                "pipecatxiaozhimelo",
                "pipecatxiaozhisherpa",
                "pipecatxiaozhikokoro",
                "pipecatxiaozhiaudio8",
            )
        ):
            return provider
        if "shared.olares.com" in host or "shared.olares.cn" in host:
            return f"{provider}-gateway"
        return provider
    host = (urlparse(base_url).hostname or "").lower()
    if "shared.olares.com" in host or "shared.olares.cn" in host:
        return "gateway"
    if host.endswith(".svc") or host.endswith(".svc.cluster.local") or "." not in host:
        return "cluster"
    if host.endswith(".local") or host.startswith("10.") or host.startswith("192.168."):
        return "cluster"
    return "other"


async def synthesize_speech_pcm(
    text: str,
    *,
    output_sample_rate: int,
    cfg: BridgeSettings | None = None,
    voice_id: str | None = None,
) -> bytes:
    """Synthesize via Melo or OmniVoice. Returns PCM only."""
    pcm, _timing = await synthesize_speech_pcm_timed(
        text,
        output_sample_rate=output_sample_rate,
        cfg=cfg,
        voice_id=voice_id,
    )
    return pcm


async def synthesize_speech_pcm_timed(
    text: str,
    *,
    output_sample_rate: int,
    cfg: BridgeSettings | None = None,
    voice_id: str | None = None,
) -> tuple[bytes, TtsTiming]:
    """Like synthesize_speech_pcm, plus timing breakdown for telemetry."""
    import time

    import httpx

    s = cfg or load_settings()
    provider = effective_tts_provider(s)
    base = tts_base_url()
    if not base and provider in ("melo", "sherpa", "kokoro", "audio8"):
        base = default_cluster_tts_url(provider)
    if not base:
        raise RuntimeError("TTS base URL not configured — open Settings in the web UI")

    base = base.rstrip("/")
    headers = auth_headers(tts_api_key())
    response_format = s.tts_response_format.strip().lower() or "wav"
    via = _classify_tts_via(base, provider)
    timing = TtsTiming(via=via, provider=provider)

    http_started = time.perf_counter()
    async with httpx.AsyncClient(timeout=s.http_timeout) as client:
        if provider in ("melo", "sherpa", "kokoro", "audio8"):
            url = f"{base}/audio/speech"
            if provider == "sherpa":
                body = build_sherpa_speech_json(text, s)
            elif provider == "kokoro":
                body = build_kokoro_speech_json(text, s)
            elif provider == "audio8":
                body = build_audio8_speech_json(text, s)
            else:
                body = build_melo_speech_json(text, s)
            timing.endpoint = "speech"
            logger.info(
                "tts POST start provider=%s endpoint=speech via=%s chars=%d voice=%r url=%s",
                provider,
                via,
                len(text),
                body.get("voice"),
                url,
            )
            resp = await client.post(url, json=body, headers=headers)
        else:
            clone = resolve_clone_ref(s, voice_id=voice_id)
            mode = (s.tts_voice_mode or "instruct").strip().lower()
            use_clone = bool(
                clone
                and (
                    voice_id
                    or (s.tts_active_voice_id.strip() and mode == "clone")
                )
            )
            num_step = max(8, int(s.tts_num_step or 16))
            timing.num_step = num_step
            if use_clone and clone:
                ref_b64, ref_text, lang = clone
                url = f"{base}/audio/clone"
                form = build_clone_form(text, s, ref_b64, ref_text, lang)
                timing.endpoint = "clone"
                logger.info(
                    "tts POST start provider=omnivoice endpoint=clone via=%s chars=%d num_step=%d url=%s",
                    via,
                    len(text),
                    num_step,
                    url,
                )
                resp = await client.post(url, data=form, headers=headers)
            else:
                url = f"{base}/audio/speech"
                body = build_speech_json(text, s)
                timing.endpoint = "speech"
                timing.num_step = int(body.get("num_step") or num_step)
                logger.info(
                    "tts POST start provider=omnivoice endpoint=speech via=%s chars=%d voice=%r "
                    "num_step=%d lang=%s url=%s",
                    via,
                    len(text),
                    body.get("voice"),
                    body.get("num_step"),
                    body.get("language_id"),
                    url,
                )
                resp = await client.post(url, json=body, headers=headers)

        resp.raise_for_status()
        raw = resp.content
    timing.http_ms = int((time.perf_counter() - http_started) * 1000)
    timing.bytes_in = len(raw)

    decode_started = time.perf_counter()
    pcm, _ = speech_bytes_to_pcm(
        raw,
        response_format=response_format,
        target_rate=output_sample_rate,
    )
    timing.decode_ms = int((time.perf_counter() - decode_started) * 1000)
    if output_sample_rate > 0 and pcm:
        timing.audio_ms = int(len(pcm) / 2 / output_sample_rate * 1000)
        if timing.audio_ms > 0:
            timing.rtf = round(timing.http_ms / timing.audio_ms, 4)
    logger.info(
        "tts POST done provider=%s via=%s endpoint=%s http_ms=%d decode_ms=%d audio_ms=%d "
        "rtf=%.4f bytes_in=%d",
        timing.provider,
        timing.via,
        timing.endpoint,
        timing.http_ms,
        timing.decode_ms,
        timing.audio_ms,
        timing.rtf,
        timing.bytes_in,
    )
    return pcm, timing
