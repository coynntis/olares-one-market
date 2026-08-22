"""PCM ↔ Opus helpers for xiaozhi-style 60 ms frames."""

from __future__ import annotations

import io
import os
import wave

import numpy as np


def opus_packets_to_wav_bytes(packets: list[bytes], sample_rate: int, channels: int = 1) -> bytes:
    """Decode a sequence of Opus packets to a mono WAV (PCM16) byte blob."""
    import opuslib

    decoder = opuslib.Decoder(sample_rate, channels)
    pcm_chunks: list[bytes] = []
    frame_samples = int(sample_rate * 60 / 1000)
    for pkt in packets:
        if not pkt:
            continue
        try:
            pcm = decoder.decode(pkt, frame_samples * channels * 2)
        except opuslib.OpusError:
            continue
        pcm_chunks.append(pcm)
    if not pcm_chunks:
        return b""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(b"".join(pcm_chunks))
    return buf.getvalue()


def pcm16_to_opus_packets(
    pcm_bytes: bytes,
    sample_rate: int,
    channels: int = 1,
    frame_duration_ms: int = 60,
) -> list[bytes]:
    """Encode linear PCM16 mono audio into Opus packets (fixed frame size)."""
    import opuslib

    if not pcm_bytes:
        return []
    enc = opuslib.Encoder(sample_rate, channels, opuslib.APPLICATION_AUDIO)
    samples_per_frame = int(sample_rate * frame_duration_ms / 1000)
    frame_bytes = samples_per_frame * channels * 2
    packets: list[bytes] = []
    for i in range(0, len(pcm_bytes), frame_bytes):
        chunk = pcm_bytes[i : i + frame_bytes]
        if len(chunk) < frame_bytes:
            chunk = chunk + b"\x00" * (frame_bytes - len(chunk))
        arr = np.frombuffer(chunk, dtype=np.int16)
        pkt = enc.encode(arr.tobytes(), samples_per_frame)
        packets.append(pkt)
    return packets


def pcm_s16le_mono_to_wav(pcm: bytes, sample_rate: int) -> bytes:
    """Wrap raw little-endian mono s16 PCM in a WAV container for STT upload."""
    if not pcm:
        return b""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm)
    return buf.getvalue()


def wav_bytes_to_pcm_s16le(wav_bytes: bytes) -> tuple[bytes, int, int]:
    """Parse WAV container → (pcm_bytes, sample_rate, channels)."""
    if not wav_bytes:
        return b"", 24000, 1
    with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
        channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        sample_rate = wf.getframerate()
        frames = wf.readframes(wf.getnframes())
    if sample_width != 2:
        raise ValueError(f"unsupported WAV sample width: {sample_width}")
    return frames, sample_rate, channels


def pcm16_stereo_to_mono(pcm: bytes, channels: int) -> bytes:
    if channels <= 1:
        return pcm
    arr = np.frombuffer(pcm, dtype=np.int16).reshape(-1, channels)
    mono = arr.mean(axis=1).astype(np.int16)
    return mono.tobytes()


def resample_pcm16_mono(pcm: bytes, src_rate: int, dst_rate: int) -> bytes:
    if src_rate == dst_rate or not pcm:
        return pcm
    arr = np.frombuffer(pcm, dtype=np.int16).astype(np.float64)
    if len(arr) == 0:
        return pcm
    out_len = max(1, int(round(len(arr) * dst_rate / src_rate)))
    x_old = np.arange(len(arr), dtype=np.float64)
    x_new = np.linspace(0, len(arr) - 1, out_len)
    out = np.interp(x_new, x_old, arr).astype(np.int16)
    return out.tobytes()


def speech_bytes_to_pcm(
    audio_bytes: bytes,
    *,
    response_format: str,
    target_rate: int | None = None,
) -> tuple[bytes, int]:
    """Decode TTS response (WAV or raw PCM16) to mono PCM at target_rate."""
    fmt = (response_format or "wav").strip().lower()
    if fmt == "pcm":
        rate = target_rate or int(os.environ.get("TTS_PCM_SAMPLE_RATE", "24000"))
        return audio_bytes, rate
    if fmt == "wav" or audio_bytes[:4] == b"RIFF":
        pcm, rate, channels = wav_bytes_to_pcm_s16le(audio_bytes)
        pcm = pcm16_stereo_to_mono(pcm, channels)
        if target_rate and rate != target_rate:
            pcm = resample_pcm16_mono(pcm, rate, target_rate)
            rate = target_rate
        return pcm, rate
    raise ValueError(f"unsupported TTS response format: {response_format}")


def strip_mqtt_gateway_audio_frame(message: bytes) -> tuple[bytes | None, bool]:
    """
    If payload matches MQTT-gateway framing (16-byte header), return audio bytes.
    Returns (audio_or_none, handled).
    """
    if len(message) < 16:
        return None, False
    audio_length = int.from_bytes(message[12:16], "big")
    if audio_length > 0 and len(message) >= 16 + audio_length:
        return message[16 : 16 + audio_length], True
    if len(message) > 16:
        return message[16:], True
    return None, False
