"""Tests for STT transcript normalization."""

from __future__ import annotations

import unittest

from xiaozhi_bridge.stt_text import is_funasr_metadata_blob, normalize_stt_transcript


class SttTextTests(unittest.TestCase):
    def test_empty_funasr_segment_blob(self) -> None:
        raw = "{'key': 'tmporabev6k', 'text': '', 'timestamp': []}"
        self.assertTrue(is_funasr_metadata_blob(raw))
        text, meta = normalize_stt_transcript(raw)
        self.assertEqual(text, "")
        self.assertIsNone(meta)

    def test_funasr_segment_with_text(self) -> None:
        raw = "{'key': 'x', 'text': 'hello', 'timestamp': [0, 1]}"
        text, _ = normalize_stt_transcript(raw)
        self.assertEqual(text, "hello")

    def test_sensevoice_tags_stripped(self) -> None:
        raw = "<|yue|><|EMO_UNKNOWN|><|Speech|>你好"
        text, meta = normalize_stt_transcript(raw)
        self.assertEqual(text, "你好")
        self.assertEqual(meta.get("language"), "yue")

    def test_nospeech_returns_empty(self) -> None:
        raw = "<|nospeech|>"
        text, meta = normalize_stt_transcript(raw)
        self.assertEqual(text, "")
        self.assertEqual(meta.get("language"), "nospeech")


if __name__ == "__main__":
    unittest.main()
