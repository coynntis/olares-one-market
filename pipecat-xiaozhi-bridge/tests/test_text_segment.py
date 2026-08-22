"""Tests for TTS sentence segmentation."""

from __future__ import annotations

import unittest

from xiaozhi_bridge.config import BridgeSettings
from xiaozhi_bridge.text_segment import SentenceSegmenter, expand_segments_for_streaming, split_long_segment
from xiaozhi_bridge.tts_params import chunk_limits_for_provider, cpu_tts_single_utterance


class TextSegmentTests(unittest.TestCase):
    def test_no_tiny_comma_only_chunk(self) -> None:
        seg = SentenceSegmenter(min_chars=12)
        seg.feed("好，后面这句足够长了可以读。")
        out = seg.pop_segment()
        self.assertIsNotNone(out)
        self.assertGreater(len(out or ""), 8)
        self.assertIn("足够长", out or "")

    def test_short_opener_merged_with_next(self) -> None:
        seg = SentenceSegmenter(min_chars=8)
        seg.feed("短句。后面是最长的第二部分内容。")
        out = seg.pop_segment()
        self.assertIn("短句", out or "")
        self.assertIn("第二部分", out or "")

    def test_no_mid_word_english_cut(self) -> None:
        text = (
            "Hello there this is a reasonably long English sentence without commas "
            "that previously got chopped mid-word by tiny max_chars."
        )
        parts = split_long_segment(text, max_chars=40)
        joined = " ".join(parts)
        self.assertIn("reasonably", joined)
        for p in parts:
            self.assertFalse(p.startswith("sonably"))
            self.assertFalse(p.endswith("reason"))

    def test_cpu_sized_expand_keeps_phrases(self) -> None:
        text = "The quick brown fox jumps over the lazy dog near the river bank today."
        chunks = expand_segments_for_streaming([text], max_chars=160, first_max_chars=96)
        self.assertGreaterEqual(len(chunks), 1)

    def test_cpu_provider_single_utterance_even_with_tiny_saved_settings(self) -> None:
        cfg = BridgeSettings(
            tts_provider="sherpa",
            tts_first_chunk_chars=12,
            tts_max_chunk_chars=40,
        )
        self.assertTrue(cpu_tts_single_utterance(cfg))
        first, maxc = chunk_limits_for_provider(cfg)
        self.assertGreaterEqual(maxc, 10_000)
        self.assertEqual(first, maxc)

    def test_omnivoice_keeps_short_chunks(self) -> None:
        cfg = BridgeSettings(
            tts_provider="omnivoice",
            tts_first_chunk_chars=12,
            tts_max_chunk_chars=40,
        )
        self.assertFalse(cpu_tts_single_utterance(cfg))
        first, maxc = chunk_limits_for_provider(cfg)
        self.assertEqual(first, 12)
        self.assertEqual(maxc, 40)


if __name__ == "__main__":
    unittest.main()
