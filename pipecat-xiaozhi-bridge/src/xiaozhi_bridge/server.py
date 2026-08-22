"""
Xiaozhi-compatible WebSocket server (path /xiaozhi/v1/).

Text messages follow xiaozhi JSON types: hello, listen, abort, ping, …
Binary frames are Opus packets (or MQTT-gateway framed payloads).

Voice pipeline per utterance: Opus → WAV → STT → LLM → TTS (WAV/PCM) → Opus.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from collections.abc import AsyncIterator
from typing import Any, Callable
from urllib.parse import parse_qs, urlparse

import httpx

from xiaozhi_bridge.audio import (
    opus_packets_to_wav_bytes,
    pcm16_to_opus_packets,
    pcm_s16le_mono_to_wav,
    speech_bytes_to_pcm,
    strip_mqtt_gateway_audio_frame,
)
from xiaozhi_bridge.config import load_settings
from xiaozhi_bridge import chat_store
from xiaozhi_bridge import session_registry
from xiaozhi_bridge import device_registry
from xiaozhi_bridge.openai_endpoints import auth_headers, stt_api_key, stt_base_url, tts_api_key, tts_base_url
from xiaozhi_bridge.agent_harness import AgentHarnessCallbacks, AgentStep
from xiaozhi_bridge.pipecat_llm import MAX_TOOL_ROUNDS, complete_agent_chat, mcp_tools_enabled, open_llm_stream
from xiaozhi_bridge.text_segment import (
    SentenceSegmenter,
    clean_for_tts,
    display_subtitle,
    expand_segments_for_streaming,
)
from xiaozhi_bridge.pipeline_types import LlmUsage, PipelineStats
from xiaozhi_bridge.vad import FrameVad
from xiaozhi_bridge.ws_port import XiaozhiWsPort

logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))
logger = logging.getLogger("xiaozhi_bridge")

FRAME_DURATION_MS = int(os.environ.get("AUDIO_FRAME_DURATION_MS", "60"))


def _settings():
    return load_settings()


@dataclass
class Session:
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    device_id: str = ""
    conversation_id: str = ""
    uplink_sample_rate: int = 16000
    downlink_sample_rate: int = 24000
    channels: int = 1
    audio_format: str = "opus"
    """Uplink from device: opus packets (xiaozhi default) or raw pcm_s16le mono frames."""
    uplink_encoding: str = "opus"
    mqtt_gateway: bool = False
    opus_buffer: list[bytes] = field(default_factory=list)
    listen_started: bool = False
    listen_mode: str = "manual"
    client_aec: bool = False
    speaking: bool = False
    client_abort: bool = False
    vad: FrameVad = field(default_factory=FrameVad)
    dialogue: list[dict[str, Any]] = field(default_factory=list)
    _turn_lock: asyncio.Lock | None = field(default=None, repr=False)

    @property
    def turn_lock(self) -> asyncio.Lock:
        if self._turn_lock is None:
            self._turn_lock = asyncio.Lock()
        return self._turn_lock

    @property
    def system_prompt(self) -> str:
        return _settings().system_prompt

    def allow_uplink_while_speaking(self) -> bool:
        return self.listen_mode == "realtime" and self.client_aec

    def vad_enabled(self) -> bool:
        return self.listen_mode in ("auto", "realtime")


def _auth_headers(api_key: str | None) -> dict[str, str]:
    from xiaozhi_bridge.openai_endpoints import auth_headers as _ah

    return _ah(api_key)


def _welcome_blob(sess: Session) -> dict[str, Any]:
    ap: dict[str, Any] = {
        "format": sess.audio_format,
        "sample_rate": sess.downlink_sample_rate,
        "channels": sess.channels,
        "frame_duration": FRAME_DURATION_MS,
    }
    if sess.uplink_encoding != "opus":
        ap["uplink_encoding"] = sess.uplink_encoding
    blob: dict[str, Any] = {
        "type": "hello",
        "version": 1,
        "transport": "websocket",
        "audio_params": ap,
        "session_id": sess.session_id,
    }
    from xiaozhi_bridge.udp_audio import udp_port

    port = udp_port()
    if port:
        blob["udp_audio"] = {"port": port, "magic": "XZ01", "session_id": sess.session_id}
    return blob


async def transcribe_openai(wav_bytes: bytes) -> tuple[str, dict[str, Any] | None]:
    from xiaozhi_bridge.stt_text import MIN_STT_WAV_BYTES, normalize_stt_transcript

    if not wav_bytes or len(wav_bytes) < MIN_STT_WAV_BYTES:
        return "", None

    cfg = _settings()
    model = cfg.stt_model
    language = cfg.stt_language.strip()
    base = stt_base_url()
    if not base:
        raise RuntimeError("STT base URL not configured — open Settings in the web UI")

    url = f"{base.rstrip('/')}/audio/transcriptions"
    data: dict[str, str] = {"model": model}
    if language:
        data["language"] = language

    async with httpx.AsyncClient(timeout=cfg.http_timeout) as client:
        resp = await client.post(
            url,
            data=data,
            files={"file": ("audio.wav", wav_bytes, "audio/wav")},
            headers=_auth_headers(stt_api_key()),
        )
        resp.raise_for_status()
        payload = resp.json()
    raw = str(payload.get("text") or "").strip()
    text, sensevoice = normalize_stt_transcript(raw)
    return text, sensevoice


async def synthesize_speech_pcm(text: str, *, output_sample_rate: int) -> bytes:
    from xiaozhi_bridge.tts_client import synthesize_speech_pcm as _client_tts

    return await _client_tts(text, output_sample_rate=output_sample_rate)


async def synthesize_speech_pcm_timed(text: str, *, output_sample_rate: int):
    from xiaozhi_bridge.tts_client import synthesize_speech_pcm_timed as _client_tts_timed

    return await _client_tts_timed(text, output_sample_rate=output_sample_rate)


def _silence_pcm(ms: int, sample_rate: int) -> bytes:
    frames = max(0, int(sample_rate * ms / 1000))
    return b"\x00\x00" * frames


def _new_segmenter() -> SentenceSegmenter:
    from xiaozhi_bridge.tts_client import segmenter_min_chars

    return SentenceSegmenter(min_chars=segmenter_min_chars(_settings()))


def _buffer_to_wav(sess: Session) -> bytes:
    if sess.uplink_encoding == "pcm_s16le":
        raw = b"".join(sess.opus_buffer)
        return pcm_s16le_mono_to_wav(raw, sess.uplink_sample_rate)
    packets = list(sess.opus_buffer)
    return opus_packets_to_wav_bytes(packets, sess.uplink_sample_rate, sess.channels)


async def _finalize_listen(
    sess: Session,
    port: XiaozhiWsPort,
    device_header: str,
) -> None:
    if not sess.opus_buffer:
        sess.vad.reset()
        return
    async with sess.turn_lock:
        if not sess.opus_buffer:
            sess.vad.reset()
            return
        sess.listen_started = False
        stt_started = time.perf_counter()
        wav = _buffer_to_wav(sess)
        frames = len(sess.opus_buffer)
        sess.opus_buffer.clear()
        sess.vad.reset()
        logger.info(
            "listen finalize device=%s mode=%s frames=%d wav_bytes=%d",
            device_header,
            sess.listen_mode,
            frames,
            len(wav),
        )
        if not wav:
            return
        try:
            text, sensevoice = await transcribe_openai(wav)
            stt_ms = int((time.perf_counter() - stt_started) * 1000)
            logger.info("stt done device=%s ms=%d text=%r", device_header, stt_ms, text[:120])
        except Exception as e:
            logger.exception("stt failed")
            await _send_pipeline_error(port, sess, f"STT failed: {e}")
            return
        if not text:
            logger.info("stt empty — skip turn device=%s", device_header)
            return
        spawn_handle_turn(sess, port, text, stt_ms=stt_ms, sensevoice=sensevoice)


async def _ingest_uplink(
    sess: Session,
    port: XiaozhiWsPort,
    device_header: str,
    raw: bytes,
) -> None:
    if not raw:
        return
    if sess.speaking and not sess.allow_uplink_while_speaking():
        return
    if not sess.listen_started:
        return
    sess.opus_buffer.append(raw)
    if sess.vad_enabled() and sess.uplink_encoding == "pcm_s16le":
        if sess.vad.feed_pcm(raw):
            await _finalize_listen(sess, port, device_header)


async def _stream_opus_to_port(
    port: XiaozhiWsPort,
    sess: Session,
    pcm: bytes,
) -> tuple[int, int]:
    """Encode PCM to Opus and send with frame pacing. Returns (opus_ms, packet_count)."""
    opus_started = time.perf_counter()
    packets = pcm16_to_opus_packets(
        pcm,
        sample_rate=sess.downlink_sample_rate,
        channels=sess.channels,
        frame_duration_ms=FRAME_DURATION_MS,
    )
    delay = FRAME_DURATION_MS / 1000.0
    sent = 0
    for pkt in packets:
        if sess.client_abort:
            logger.info("turn[%s] aborted during opus stream", sess.session_id)
            break
        await port.send_bytes(pkt)
        sent += 1
        await asyncio.sleep(delay)
    opus_ms = int((time.perf_counter() - opus_started) * 1000)
    return opus_ms, sent


async def _segment_pcm(
    segment: str,
    *,
    sample_rate: int,
    pad_leading_ms: int = 0,
):
    """Synthesize one segment to PCM. Returns (pcm, TtsTiming)."""
    pcm, timing = await synthesize_speech_pcm_timed(
        text=segment, output_sample_rate=sample_rate
    )
    if pad_leading_ms > 0:
        pcm = _silence_pcm(pad_leading_ms, sample_rate) + pcm
    return pcm, timing


async def _announce_sentence(port: XiaozhiWsPort, sess: Session, text: str) -> None:
    await port.send_text(
        json.dumps(
            {
                "type": "tts",
                "state": "sentence_start",
                "text": text,
                "session_id": sess.session_id,
            },
            ensure_ascii=False,
        )
    )


async def _speak_segment(
    port: XiaozhiWsPort,
    sess: Session,
    segment: str,
    *,
    subtitle: str | None = None,
    pad_leading_ms: int = 0,
) -> tuple[int, int]:
    """TTS one segment + Opus downlink. Returns (tts_ms, opus_ms)."""
    show = subtitle if subtitle is not None else segment
    await _announce_sentence(port, sess, show)
    pcm, timing = await _segment_pcm(
        segment, sample_rate=sess.downlink_sample_rate, pad_leading_ms=pad_leading_ms
    )
    opus_ms, _ = await _stream_opus_to_port(port, sess, pcm)
    return timing.http_ms + timing.decode_ms, opus_ms


@dataclass
class _SpeakPipelineResult:
    tts_ms: int = 0
    tts_http_ms: int = 0
    tts_decode_ms: int = 0
    tts_audio_ms: int = 0
    tts_rtf: float = 0.0
    tts_via: str = ""
    tts_warmup_ms: int = 0
    opus_ms: int = 0
    segments: int = 0
    first_segment_ms: int | None = None
    first_audio_ms: int | None = None


async def _tts_warmup_discard(*, sample_rate: int, session_id: str = "") -> int:
    """
    Tiny OmniVoice request; audio discarded.

    Fire at LLM start (serial path): request waits on HAMI while llama holds GPU,
    then pays cold/handoff tax so the first real segment stays warm (~RTF 0.2).
    Returns wall-clock ms (includes any lock wait).
    """
    cfg = _settings()
    text = (cfg.tts_warmup_text or "嗯").strip() or "嗯"
    started = time.perf_counter()
    try:
        _pcm, timing = await synthesize_speech_pcm_timed(
            text=text, output_sample_rate=sample_rate
        )
        wall_ms = int((time.perf_counter() - started) * 1000)
        logger.info(
            "tts warmup discard session=%s wall_ms=%d http_ms=%d audio_ms=%d rtf=%.4f via=%s text=%r",
            session_id or "-",
            wall_ms,
            timing.http_ms,
            timing.audio_ms,
            timing.rtf,
            timing.via,
            text,
        )
        return wall_ms
    except Exception as e:
        wall_ms = int((time.perf_counter() - started) * 1000)
        logger.warning(
            "tts warmup failed session=%s wall_ms=%d err=%s",
            session_id or "-",
            wall_ms,
            e,
        )
        return wall_ms


async def _run_speak_queue(
    port: XiaozhiWsPort,
    sess: Session,
    segment_q: asyncio.Queue[str | None],
    *,
    llm_started: float,
    pad_ms: int,
    announce: bool = True,
) -> _SpeakPipelineResult:
    """
    Consume speakable segments; synthesize next while pacing current Opus.

    On single-GPU setups, call this AFTER the LLM stream finishes (see _stream_reply_with_tts).
    """
    out = _SpeakPipelineResult()
    seg_index = 0

    async def _prepare(next_index: int):
        seg = await segment_q.get()
        if seg is None:
            return None
        subtitle = display_subtitle(seg)
        if announce:
            await _announce_sentence(port, sess, subtitle)
        if out.first_segment_ms is None:
            out.first_segment_ms = int((time.perf_counter() - llm_started) * 1000)
        pad = pad_ms if next_index > 0 else 0
        pcm, timing = await _segment_pcm(
            seg, sample_rate=sess.downlink_sample_rate, pad_leading_ms=pad
        )
        return pcm, timing, subtitle

    pending = asyncio.create_task(_prepare(0))
    try:
        while not sess.client_abort:
            item = await pending
            if item is None:
                break
            pcm, timing, subtitle = item
            out.tts_ms += timing.http_ms + timing.decode_ms
            out.tts_http_ms += timing.http_ms
            out.tts_decode_ms += timing.decode_ms
            out.tts_audio_ms += timing.audio_ms
            if timing.via:
                out.tts_via = timing.via
            seg_index += 1
            # Kick next TTS/wait while we real-time pace this segment's Opus.
            pending = asyncio.create_task(_prepare(seg_index))
            if out.first_audio_ms is None:
                out.first_audio_ms = int((time.perf_counter() - llm_started) * 1000)
            opus_ms, _ = await _stream_opus_to_port(port, sess, pcm)
            out.opus_ms += opus_ms
            out.segments += 1
            logger.info(
                "turn[%s] segment[%d] http_ms=%d decode_ms=%d audio_ms=%d rtf=%.4f via=%s opus_ms=%d text=%r",
                sess.session_id,
                out.segments,
                timing.http_ms,
                timing.decode_ms,
                timing.audio_ms,
                timing.rtf,
                timing.via,
                opus_ms,
                subtitle[:48],
            )
    finally:
        if not pending.done():
            pending.cancel()
            try:
                await pending
            except (asyncio.CancelledError, Exception):
                pass
        if sess.client_abort:
            while True:
                try:
                    item = segment_q.get_nowait()
                except asyncio.QueueEmpty:
                    break
                if item is None:
                    break
    return out


async def _speak_chunked_reply(
    port: XiaozhiWsPort,
    sess: Session,
    reply: str,
    *,
    llm_started: float,
) -> _SpeakPipelineResult:
    """After full LLM: expand into short TTS chunks; play N while synth N+1."""
    cfg = _settings()
    text = clean_for_tts(reply)
    if not text or sess.client_abort:
        return _SpeakPipelineResult()

    from xiaozhi_bridge.tts_params import (
        chunk_limits_for_provider,
        cpu_tts_single_utterance,
        effective_tts_provider,
    )

    provider = effective_tts_provider(cfg)
    delay_ms = max(0, int(getattr(cfg, "tts_post_llm_delay_ms", 0) or 0))
    # Settle only helps OmniVoice GPU handoff; Melo/Sherpa/Kokoro are CPU — skip.
    if delay_ms and provider == "omnivoice":
        logger.info(
            "turn[%s] post-LLM settle %dms before OmniVoice",
            sess.session_id,
            delay_ms,
        )
        await asyncio.sleep(delay_ms / 1000.0)
        if sess.client_abort:
            return _SpeakPipelineResult()

    first_chars, max_chars = chunk_limits_for_provider(cfg)
    pad_ms = max(0, int(cfg.tts_segment_pad_ms or 0))
    if cpu_tts_single_utterance(cfg):
        chunks = [text]
        logger.info(
            "turn[%s] TTS single-utterance provider=%s chars=%d (no bridge chunk split)",
            sess.session_id,
            provider,
            len(text),
        )
    else:
        chunks = expand_segments_for_streaming(
            [text],
            max_chars=max_chars,
            first_max_chars=first_chars,
        )
    if not chunks:
        return _SpeakPipelineResult()

    logger.info(
        "turn[%s] TTS chunked provider=%s chunks=%d chars=%d first=%d max=%d",
        sess.session_id,
        provider,
        len(chunks),
        len(text),
        first_chars,
        max_chars,
    )
    # Subtitles already streamed during LLM; announce only if single chunk path needs it.
    return await _speak_segments_serial(
        port,
        sess,
        chunks,
        llm_started=llm_started,
        pad_ms=pad_ms,
        announce=True,
    )


async def _speak_full_reply(
    port: XiaozhiWsPort,
    sess: Session,
    reply: str,
    *,
    llm_started: float,
) -> _SpeakPipelineResult:
    """Speak full reply with post-LLM TTS chunking (play-while-synth)."""
    cfg = _settings()
    warmup_ms = 0
    from xiaozhi_bridge.tts_params import effective_tts_provider

    if (
        bool(getattr(cfg, "tts_warmup", False))
        and not bool(cfg.tts_overlap_llm)
        and effective_tts_provider(cfg) == "omnivoice"
    ):
        warmup_ms = await _tts_warmup_discard(
            sample_rate=sess.downlink_sample_rate, session_id=sess.session_id
        )
    spoken = await _speak_chunked_reply(port, sess, reply, llm_started=llm_started)
    spoken.tts_warmup_ms = warmup_ms
    if spoken.tts_audio_ms > 0 and spoken.tts_http_ms > 0:
        spoken.tts_rtf = round(spoken.tts_http_ms / spoken.tts_audio_ms, 4)
    return spoken


async def _stream_reply_with_tts(
    port: XiaozhiWsPort,
    sess: Session,
    messages: list[dict[str, Any]],
    *,
    llm_started: float,
) -> tuple[Any, _SpeakPipelineResult]:
    """
    Stream LLM to completion, then chunked TTS (play N while synth N+1).

    Optional overlap: sentence-level TTS while LLM streams (experimental on shared GPU).
    """
    cfg = _settings()
    pad_ms = max(0, int(cfg.tts_segment_pad_ms or 0))
    from xiaozhi_bridge.tts_params import effective_tts_provider, is_cpu_tts_provider

    provider = effective_tts_provider(cfg)
    # Overlap sentence TTS while LLM streams cuts mid-phrase on CPU engines.
    overlap = bool(cfg.tts_overlap_llm) and not is_cpu_tts_provider(cfg)
    warmup_on = (
        bool(getattr(cfg, "tts_warmup", False))
        and not overlap
        and provider == "omnivoice"
    )

    warmup_task: asyncio.Task[int] | None = None
    segment_q: asyncio.Queue[str | None] | None = None
    speak_task: asyncio.Task[_SpeakPipelineResult] | None = None
    chunks_queued = 0

    if overlap:
        segment_q = asyncio.Queue()
        speak_task = asyncio.create_task(
            _run_speak_queue(
                port,
                sess,
                segment_q,
                llm_started=llm_started,
                pad_ms=pad_ms,
                announce=False,
            )
        )
        logger.info(
            "turn[%s] TTS overlap ON — sentence TTS while LLM streams (experimental)",
            sess.session_id,
        )
    else:
        logger.info(
            "turn[%s] TTS serial — full LLM then chunked TTS (provider=%s)",
            sess.session_id,
            provider,
        )

    llm_stream = await open_llm_stream(messages)
    segmenter = _new_segmenter()
    collected: list[str] = []

    async for delta in llm_stream:
        if sess.client_abort:
            break
        if warmup_on and warmup_task is None:
            warmup_task = asyncio.create_task(
                _tts_warmup_discard(
                    sample_rate=sess.downlink_sample_rate, session_id=sess.session_id
                )
            )
        segmenter.feed(delta)
        while True:
            segment = segmenter.pop_segment()
            if not segment:
                break
            collected.append(segment)
            if overlap:
                assert segment_q is not None
                await _announce_sentence(port, sess, display_subtitle(segment))
                await segment_q.put(segment)
                chunks_queued += 1
            else:
                # Live subtitle only — TTS waits for full reply then chunks.
                await _announce_sentence(port, sess, display_subtitle(segment))

    if not sess.client_abort:
        segmenter.mark_end()
        while True:
            segment = segmenter.pop_segment()
            if not segment:
                break
            collected.append(segment)
            if overlap:
                assert segment_q is not None
                await _announce_sentence(port, sess, display_subtitle(segment))
                await segment_q.put(segment)
                chunks_queued += 1
            else:
                await _announce_sentence(port, sess, display_subtitle(segment))

    if warmup_on and warmup_task is None and not sess.client_abort:
        warmup_task = asyncio.create_task(
            _tts_warmup_discard(
                sample_rate=sess.downlink_sample_rate, session_id=sess.session_id
            )
        )

    llm_result = llm_stream.result()
    # Ensure stream/client closed before settle+TTS (HAMI GPU handoff).
    await llm_stream.aclose()
    handoff_started = time.perf_counter()

    warmup_ms = 0
    if warmup_task is not None:
        if sess.client_abort:
            warmup_task.cancel()
            try:
                await warmup_task
            except (asyncio.CancelledError, Exception):
                pass
        else:
            try:
                await_started = time.perf_counter()
                await warmup_task
                warmup_ms = int((time.perf_counter() - await_started) * 1000)
            except Exception as e:
                logger.warning("turn[%s] tts warmup await failed: %s", sess.session_id, e)

    if sess.client_abort:
        if segment_q is not None:
            await segment_q.put(None)
        if speak_task is not None:
            try:
                spoken = await speak_task
            except Exception:
                spoken = _SpeakPipelineResult()
            spoken.tts_warmup_ms = warmup_ms
            return llm_result, spoken
        return llm_result, _SpeakPipelineResult(tts_warmup_ms=warmup_ms)

    if overlap:
        assert segment_q is not None and speak_task is not None
        await segment_q.put(None)
        logger.info(
            "turn[%s] llm done ms=%d overlap_tts=True sentences=%d",
            sess.session_id,
            llm_result.elapsed_ms,
            chunks_queued,
        )
        spoken = await speak_task
        spoken.tts_warmup_ms = warmup_ms
        return llm_result, spoken

    full = clean_for_tts(llm_result.text or "") or clean_for_tts("".join(collected))
    handoff_ms = int((time.perf_counter() - handoff_started) * 1000)
    logger.info(
        "turn[%s] llm done ms=%d overlap_tts=False handoff_ms=%d reply_chars=%d max_tokens=%d provider=%s",
        sess.session_id,
        llm_result.elapsed_ms,
        handoff_ms,
        len(full or ""),
        int(cfg.llm_max_tokens or 0),
        provider,
    )
    spoken = await _speak_chunked_reply(port, sess, full or "", llm_started=llm_started)
    spoken.tts_warmup_ms = warmup_ms
    if spoken.tts_audio_ms > 0 and spoken.tts_http_ms > 0:
        spoken.tts_rtf = round(spoken.tts_http_ms / spoken.tts_audio_ms, 4)
    return llm_result, spoken


async def _speak_segments_serial(
    port: XiaozhiWsPort,
    sess: Session,
    segments: list[str],
    *,
    llm_started: float,
    pad_ms: int,
    announce: bool = True,
) -> _SpeakPipelineResult:
    """TTS each segment with next-segment prefetch; optional sentence_start."""
    q: asyncio.Queue[str | None] = asyncio.Queue()
    for seg in segments:
        await q.put(seg)
    await q.put(None)
    # Temporarily mute announce by wrapping? Cleaner: flag on speak queue.
    return await _run_speak_queue(
        port,
        sess,
        q,
        llm_started=llm_started,
        pad_ms=pad_ms,
        announce=announce,
    )


async def handle_turn(
    sess: Session,
    port: XiaozhiWsPort,
    user_text: str,
    *,
    stt_ms: int | None = None,
    sensevoice: dict[str, Any] | None = None,
) -> None:
    if not user_text:
        return
    turn_started = time.perf_counter()
    sess.client_abort = False
    sess.session_id = str(uuid.uuid4())
    logger.info("turn[%s] start text=%r", sess.session_id, user_text[:120])
    stt_payload: dict[str, Any] = {
        "type": "stt",
        "text": user_text,
        "session_id": sess.session_id,
    }
    if sensevoice:
        stt_payload["sensevoice"] = sensevoice
    await port.send_text(json.dumps(stt_payload, ensure_ascii=False))
    await port.send_text(json.dumps({"type": "tts", "state": "start", "session_id": sess.session_id}))
    sess.speaking = True
    if not sess.conversation_id:
        conv = chat_store.resolve_device_conversation(sess.device_id)
        sess.conversation_id = conv["id"]
    chat_store.append_message(
        sess.conversation_id,
        role="user",
        text=user_text,
        source="voice",
        meta={"sensevoice": sensevoice} if sensevoice else None,
    )
    messages: list[dict[str, Any]] = chat_store.llm_messages_for_conversation(
        sess.conversation_id, sess.system_prompt
    )
    llm_started = time.perf_counter()
    segment_count = 0
    tts_ms_total = 0
    tts_http_total = 0
    tts_decode_total = 0
    tts_audio_total = 0
    tts_via = ""
    tts_warmup_total = 0
    opus_ms_total = 0
    first_segment_ms: int | None = None
    tool_rounds = 0
    reply = ""
    llm_ms = 0
    llm_result_usage = LlmUsage()
    llm_backend = ""
    first_token_ms: int | None = None
    first_audio_ms: int | None = None
    tokens_per_sec = 0.0
    tts_pad_ms = max(0, int(_settings().tts_segment_pad_ms or 0))

    async def _emit_agent_step(step: AgentStep) -> None:
        await port.send_text(
            json.dumps(
                {
                    "type": "agent",
                    "phase": step.phase,
                    "round": step.round_index,
                    "step": step.step_index,
                    "tool": step.tool_name,
                    "label": step.label,
                    "message": step.message,
                    "detail": step.detail,
                    "image_url": step.image_url,
                    "session_id": sess.session_id,
                },
                ensure_ascii=False,
            )
        )

    async def _speak_announcement(text: str) -> None:
        nonlocal segment_count, tts_ms_total, opus_ms_total, first_segment_ms, first_audio_ms
        spoken = clean_for_tts(text)
        if not spoken or sess.client_abort:
            return
        if first_segment_ms is None:
            first_segment_ms = int((time.perf_counter() - llm_started) * 1000)
        seg_tts, seg_opus = await _speak_segment(
            port,
            sess,
            spoken,
            subtitle=display_subtitle(spoken),
            pad_leading_ms=tts_pad_ms if segment_count > 0 else 0,
        )
        # Announcement path is sequential; audio began ~ after this TTS started (TTS ms before opus).
        if first_audio_ms is None:
            first_audio_ms = int((time.perf_counter() - llm_started) * 1000) - seg_opus
        tts_ms_total += seg_tts
        opus_ms_total += seg_opus
        segment_count += 1

    agent_callbacks = AgentHarnessCallbacks(
        on_step=_emit_agent_step,
        on_announce_spoken=_speak_announcement,
    )

    use_agent = await mcp_tools_enabled()
    if use_agent:
        cfg = _settings()
        agent_timeout = max(float(cfg.http_timeout or 120.0), 30.0) * min(MAX_TOOL_ROUNDS, 4)
        try:
            agent_result = await asyncio.wait_for(
                complete_agent_chat(messages, callbacks=agent_callbacks, device_id=sess.device_id),
                timeout=agent_timeout,
            )
        except asyncio.TimeoutError:
            await _send_pipeline_error(port, sess, f"LLM/tools timeout after {agent_timeout:.0f}s")
            await port.send_text(json.dumps({"type": "tts", "state": "stop", "session_id": sess.session_id}))
            sess.speaking = False
            return
        except RuntimeError as e:
            await _send_pipeline_error(port, sess, str(e))
            await port.send_text(json.dumps({"type": "tts", "state": "stop", "session_id": sess.session_id}))
            sess.speaking = False
            return
        reply = agent_result.text
        llm_ms = agent_result.elapsed_ms
        llm_result_usage = agent_result.usage
        llm_backend = agent_result.backend
        tokens_per_sec = agent_result.tokens_per_sec
        tool_rounds = agent_result.tool_rounds
        if agent_result.tool_trace or agent_result.generated_images:
            image_msgs = chat_store.append_agent_trace(
                sess.conversation_id,
                agent_result.tool_trace or [],
                source="voice",
                generated_images=agent_result.generated_images or [],
            )
            for img_msg in image_msgs:
                await port.send_text(
                    json.dumps(
                        {
                            "type": "generated_image",
                            "message": img_msg,
                            "session_id": sess.session_id,
                        },
                        ensure_ascii=False,
                    )
                )
        elif reply:
            chat_store.append_message(
                sess.conversation_id,
                role="assistant",
                text=reply,
                source="voice",
            )
        if not reply.strip() and agent_result.tool_trace:
            await _send_pipeline_error(port, sess, "No final reply after tool use — check LLM or try again")
            await port.send_text(json.dumps({"type": "tts", "state": "stop", "session_id": sess.session_id}))
            sess.speaking = False
            return
        if reply and not sess.client_abort:
            spoken = await _speak_full_reply(port, sess, reply, llm_started=llm_started)
            tts_ms_total += spoken.tts_ms
            tts_http_total += spoken.tts_http_ms
            tts_decode_total += spoken.tts_decode_ms
            tts_audio_total += spoken.tts_audio_ms
            tts_warmup_total += spoken.tts_warmup_ms
            if spoken.tts_via:
                tts_via = spoken.tts_via
            opus_ms_total += spoken.opus_ms
            segment_count += spoken.segments
            if first_segment_ms is None:
                first_segment_ms = spoken.first_segment_ms
            if first_audio_ms is None:
                first_audio_ms = spoken.first_audio_ms
    else:
        llm_result, spoken = await _stream_reply_with_tts(
            port, sess, messages, llm_started=llm_started
        )
        llm_ms = llm_result.elapsed_ms or int((time.perf_counter() - llm_started) * 1000)
        reply = llm_result.text
        llm_result_usage = llm_result.usage or LlmUsage()
        llm_backend = llm_result.backend
        first_token_ms = llm_result.first_token_ms
        tokens_per_sec = llm_result.tokens_per_sec
        tts_ms_total += spoken.tts_ms
        tts_http_total += spoken.tts_http_ms
        tts_decode_total += spoken.tts_decode_ms
        tts_audio_total += spoken.tts_audio_ms
        tts_warmup_total += spoken.tts_warmup_ms
        if spoken.tts_via:
            tts_via = spoken.tts_via
        opus_ms_total += spoken.opus_ms
        segment_count += spoken.segments
        if first_segment_ms is None:
            first_segment_ms = spoken.first_segment_ms
        if first_audio_ms is None:
            first_audio_ms = spoken.first_audio_ms

        if reply:
            chat_store.append_message(
                sess.conversation_id,
                role="assistant",
                text=reply,
                source="voice",
            )

    logger.info(
        "turn[%s] llm done ms=%d first_token_ms=%s first_audio_ms=%s reply_chars=%d segments=%d tps=%.2f tools=%d warmup_ms=%d",
        sess.session_id,
        llm_ms,
        first_token_ms,
        first_audio_ms or first_segment_ms,
        len(reply or ""),
        segment_count,
        tokens_per_sec,
        tool_rounds,
        tts_warmup_total,
    )

    total_ms = int((time.perf_counter() - turn_started) * 1000)
    tts_rtf = round(tts_http_total / tts_audio_total, 4) if tts_audio_total > 0 else 0.0
    stats = PipelineStats(
        stt_ms=stt_ms,
        llm_ms=llm_ms,
        tts_ms=tts_ms_total,
        tts_http_ms=tts_http_total,
        tts_decode_ms=tts_decode_total,
        tts_audio_ms=tts_audio_total,
        tts_rtf=tts_rtf,
        tts_via=tts_via,
        tts_warmup_ms=tts_warmup_total,
        opus_ms=opus_ms_total,
        total_ms=total_ms,
        prompt_tokens=llm_result_usage.prompt_tokens,
        completion_tokens=llm_result_usage.completion_tokens,
        tokens_per_sec=tokens_per_sec,
        backend=llm_backend,
        first_token_ms=first_token_ms,
        first_audio_ms=first_audio_ms or first_segment_ms,
        segments=segment_count,
        tool_rounds=tool_rounds,
    )
    await port.send_text(
        json.dumps(
            {
                "type": "stats",
                "stats": stats.to_dict(),
                "conversation_id": sess.conversation_id,
                "session_id": sess.session_id,
            },
            ensure_ascii=False,
        )
    )
    await port.send_text(json.dumps({"type": "tts", "state": "stop", "session_id": sess.session_id}))
    sess.speaking = False
    logger.info(
        "turn[%s] done total_ms=%d segments=%d tts_ms=%d",
        sess.session_id,
        total_ms,
        segment_count,
        tts_ms_total,
    )


async def _send_pipeline_error(port: XiaozhiWsPort, sess: Session, message: str) -> None:
    sid = sess.session_id or str(uuid.uuid4())
    await port.send_text(
        json.dumps({"type": "error", "message": message, "session_id": sid}, ensure_ascii=False)
    )
    await port.send_text(json.dumps({"type": "tts", "state": "stop", "session_id": sid}))


async def safe_handle_turn(
    sess: Session,
    port: XiaozhiWsPort,
    user_text: str,
    *,
    stt_ms: int | None = None,
    sensevoice: dict[str, Any] | None = None,
) -> None:
    from xiaozhi_bridge.stt_text import normalize_stt_transcript

    user_text, sv_extra = normalize_stt_transcript(user_text)
    if sv_extra and not sensevoice:
        sensevoice = sv_extra
    if not user_text:
        return
    async with sess.turn_lock:
        try:
            await handle_turn(sess, port, user_text, stt_ms=stt_ms, sensevoice=sensevoice)
        except Exception as e:
            logger.exception("voice turn failed")
            await _send_pipeline_error(port, sess, str(e))


def spawn_handle_turn(
    sess: Session,
    port: XiaozhiWsPort,
    user_text: str,
    *,
    stt_ms: int | None = None,
    sensevoice: dict[str, Any] | None = None,
) -> None:
    """Run voice turn in background so the WS loop can still answer pings."""
    asyncio.create_task(
        safe_handle_turn(sess, port, user_text, stt_ms=stt_ms, sensevoice=sensevoice)
    )


async def run_xiaozhi_session(
    req_path: str,
    header_get: Callable[[str], str | None],
    messages: AsyncIterator[str | bytes],
    port: XiaozhiWsPort,
) -> None:
    parsed = urlparse(req_path)
    norm = (parsed.path or "").rstrip("/")
    if norm != "/xiaozhi/v1":
        await port.close(1008, "invalid path")
        return

    qs = parse_qs(parsed.query or "")
    device_header = header_get("device-id")
    if device_header is None and "device-id" in qs:
        device_header = qs["device-id"][0]
    if not device_header:
        await port.send_text("端口正常，如需测试连接，请使用test_page.html")
        await port.close()
        return

    sess = Session()
    sess.device_id = device_header
    sess.downlink_sample_rate = _settings().downlink_sample_rate
    sess.mqtt_gateway = qs.get("from", [""])[0] == "mqtt_gateway"
    conv_q = qs.get("conversation_id", [""])[0].strip() or None
    try:
        conv = chat_store.resolve_device_conversation(device_header, conv_q)
        sess.conversation_id = conv["id"]
    except (PermissionError, LookupError, ValueError) as e:
        logger.warning("conversation resolve failed device=%s: %s", device_header, e)
        await port.close(1008, str(e))
        return

    logger.info(
        "connect path=%s device=%s mqtt=%s conversation=%s",
        req_path,
        device_header,
        sess.mqtt_gateway,
        sess.conversation_id,
    )

    async for message in messages:
        if isinstance(message, str):
            try:
                data = json.loads(message)
            except json.JSONDecodeError:
                logger.warning("non-json text message")
                continue
            mtype = data.get("type")
            if mtype == "hello":
                ap = data.get("audio_params") or {}
                features = data.get("features") or {}
                sess.client_aec = bool(features.get("aec"))
                sess.audio_format = ap.get("format", sess.audio_format)
                sess.uplink_sample_rate = int(ap.get("sample_rate", sess.uplink_sample_rate))
                sess.channels = int(ap.get("channels", sess.channels))
                up = ap.get("uplink_encoding", "opus")
                sess.uplink_encoding = up if up in ("opus", "pcm_s16le") else "opus"
                sess.downlink_sample_rate = _settings().downlink_sample_rate
                sess.vad = FrameVad(sample_rate=sess.uplink_sample_rate, frame_ms=FRAME_DURATION_MS)
                logger.info(
                    "hello device=%s up_enc=%s up_rate=%s down_rate=%s aec=%s",
                    device_header,
                    sess.uplink_encoding,
                    sess.uplink_sample_rate,
                    sess.downlink_sample_rate,
                    sess.client_aec,
                )
                await port.send_text(json.dumps(_welcome_blob(sess), ensure_ascii=False))
                await port.send_text(
                    json.dumps(
                        {
                            "type": "conversation",
                            "conversation_id": sess.conversation_id,
                            "session_id": sess.session_id,
                        },
                        ensure_ascii=False,
                    )
                )

                async def _on_udp_audio(payload: bytes) -> None:
                    await _ingest_uplink(sess, port, device_header, payload)

                session_registry.register(sess.session_id, sess, _on_udp_audio)
            elif mtype == "ping":
                await port.send_text(
                    json.dumps(
                        {"type": "pong", "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")},
                        ensure_ascii=False,
                    )
                )
            elif mtype == "abort":
                sess.client_abort = True
                if str(data.get("reason") or "") == "wake_word_detected":
                    sess.speaking = False
            elif mtype == "listen":
                state = data.get("state")
                if state == "start":
                    mode = str(data.get("mode") or "manual").strip().lower()
                    if mode not in ("manual", "auto", "realtime"):
                        mode = "manual"
                    sess.listen_mode = mode
                    sess.opus_buffer.clear()
                    sess.vad.reset()
                    sess.listen_started = True
                    sess.client_abort = False
                    logger.info("listen start device=%s mode=%s", device_header, sess.listen_mode)
                elif state == "stop":
                    await _finalize_listen(sess, port, device_header)
                elif state == "detect":
                    wake = (data.get("text") or "").strip()
                    had_audio = bool(sess.opus_buffer) and sess.listen_started
                    sv_for_turn: dict[str, Any] | None = None
                    if had_audio:
                        sess.listen_started = False
                        wav = _buffer_to_wav(sess)
                        sess.opus_buffer.clear()
                        sess.vad.reset()
                        try:
                            heard, sv_meta = await transcribe_openai(wav)
                        except Exception as e:
                            logger.exception("stt failed on wake audio")
                            await _send_pipeline_error(port, sess, f"STT failed: {e}")
                            continue
                        text = wake or heard
                        sv_for_turn = sv_meta if not wake else None
                    elif wake:
                        text = wake
                    else:
                        continue
                    if text.strip():
                        logger.info("listen detect device=%s text=%r", device_header, text[:120])
                        spawn_handle_turn(sess, port, text.strip(), sensevoice=sv_for_turn)
            elif mtype == "builtin_tool_result":
                from xiaozhi_bridge.browser_tool_bridge import complete_browser_tool

                rid = str(data.get("request_id") or "")
                if rid:
                    err = data.get("error")
                    result = data.get("result")
                    if err:
                        complete_browser_tool(rid, error=str(err))
                    else:
                        complete_browser_tool(rid, result=str(result or "{}"))
            elif mtype in ("mcp", "iot", "server"):
                logger.debug("stub type=%s payload keys=%s", mtype, list(data.keys()))
            else:
                logger.warning("unknown message type=%s", mtype)
        else:
            if not sess.listen_started:
                continue
            raw = message
            if sess.mqtt_gateway:
                audio, handled = strip_mqtt_gateway_audio_frame(raw)
                if handled and audio:
                    raw = audio
                elif handled:
                    continue
            if raw:
                await _ingest_uplink(sess, port, device_header, raw)
    session_registry.unregister(sess.session_id)
    logger.info("disconnect device=%s", device_header)
