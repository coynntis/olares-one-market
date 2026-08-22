"""Lightweight energy VAD for xiaozhi auto / realtime listen modes."""

from __future__ import annotations

import audioop
import os


class FrameVad:
    """Detect end-of-utterance from 16-bit mono PCM frames (typically 60 ms @ 16 kHz)."""

    def __init__(
        self,
        *,
        sample_rate: int = 16000,
        frame_ms: int = 60,
        energy_threshold: int | None = None,
        silence_ms: int | None = None,
        min_speech_ms: int | None = None,
    ) -> None:
        self.sample_rate = sample_rate
        self.frame_ms = frame_ms
        self.energy_threshold = energy_threshold or int(os.environ.get("VAD_ENERGY_THRESHOLD", "350"))
        self.silence_ms = silence_ms or int(os.environ.get("VAD_SILENCE_MS", "900"))
        self.min_speech_ms = min_speech_ms or int(os.environ.get("VAD_MIN_SPEECH_MS", "280"))
        self._in_speech = False
        self._speech_ms = 0
        self._silence_ms = 0

    def reset(self) -> None:
        self._in_speech = False
        self._speech_ms = 0
        self._silence_ms = 0

    def _rms(self, pcm: bytes) -> int:
        if len(pcm) < 2:
            return 0
        try:
            return audioop.rms(pcm, 2)
        except audioop.error:
            return 0

    def feed_pcm(self, pcm: bytes) -> bool:
        """Return True when utterance should end (silence after speech)."""
        rms = self._rms(pcm)
        if rms >= self.energy_threshold:
            self._in_speech = True
            self._speech_ms += self.frame_ms
            self._silence_ms = 0
            return False
        if not self._in_speech:
            return False
        self._silence_ms += self.frame_ms
        if self._speech_ms >= self.min_speech_ms and self._silence_ms >= self.silence_ms:
            self.reset()
            return True
        return False

    def force_end(self) -> bool:
        """Flush when client sends listen stop."""
        had = self._in_speech and self._speech_ms >= self.min_speech_ms
        self.reset()
        return had


def pcm_frame_rms(pcm: bytes) -> float:
    """Normalized RMS 0..1 for UI meters."""
    rms = FrameVad()._rms(pcm)
    return min(1.0, rms / 4000.0)
