"""Tests for OmniVoice TTS routing."""

from __future__ import annotations

import unittest

from xiaozhi_bridge.config import BridgeSettings
from xiaozhi_bridge.tts_params import build_clone_form, build_speech_json


class TtsClientTests(unittest.TestCase):
    def test_speech_uses_voice_not_instruct(self) -> None:
        cfg = BridgeSettings(tts_instruct="female, low pitch", tts_voice_mode="instruct")
        body = build_speech_json("hello", cfg)
        self.assertEqual(body["voice"], "female, low pitch")
        self.assertNotIn("instruct", body)
        self.assertEqual(body["class_temperature"], 0.0)

    def test_clone_form_uses_ref_audio_base64(self) -> None:
        cfg = BridgeSettings()
        form = build_clone_form("hi", cfg, "abc123", "reference words", "yue")
        self.assertEqual(form["ref_audio_base64"], "abc123")
        self.assertEqual(form["ref_text"], "reference words")
        self.assertEqual(form["language_id"], "yue")
        self.assertNotIn("ref_audio_b64", form)

    def test_preset_voice_mode(self) -> None:
        cfg = BridgeSettings(tts_voice_mode="default", tts_voice="female_br")
        body = build_speech_json("test", cfg)
        self.assertEqual(body["voice"], "female_br")

    def test_speech_cantonese_when_stt_yue_and_tts_zh(self) -> None:
        cfg = BridgeSettings(stt_language="yue", tts_language_id="zh", tts_voice_mode="default")
        body = build_speech_json("你好", cfg)
        self.assertEqual(body["language_id"], "yue")

    def test_speech_respects_explicit_tts_language(self) -> None:
        cfg = BridgeSettings(stt_language="yue", tts_language_id="en", tts_voice_mode="default")
        body = build_speech_json("hello", cfg)
        self.assertEqual(body["language_id"], "en")


if __name__ == "__main__":
    unittest.main()
