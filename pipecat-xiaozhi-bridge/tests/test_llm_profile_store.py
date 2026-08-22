"""Tests for LLM profile store."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from xiaozhi_bridge import llm_profile_store
from xiaozhi_bridge.config import load_settings, save_settings, BridgeSettings


class LlmProfileStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.config_path = Path(self.tmp.name) / "config.json"
        self.env = mock.patch.dict(os.environ, {"CONFIG_PATH": str(self.config_path)}, clear=False)
        self.env.start()
        save_settings(
            BridgeSettings(
                llm_base_url="https://llm.example/v1",
                llm_model="test-model",
                system_prompt="You are test.",
            )
        )
        load_settings(force=True)

    def tearDown(self) -> None:
        self.env.stop()
        self.tmp.cleanup()

    def test_create_and_activate_profile(self) -> None:
        profile = llm_profile_store.create_profile(
            name="Coding",
            llm_base_url="https://coder.example/v1",
            llm_model="coder-7b",
            system_prompt="Code only.",
            set_active=True,
        )
        self.assertEqual(profile["name"], "Coding")
        cfg = load_settings(force=True)
        self.assertEqual(cfg.llm_model, "coder-7b")
        self.assertEqual(cfg.active_llm_profile_id, profile["id"])

        profiles, active = llm_profile_store.list_profiles()
        self.assertEqual(len(profiles), 1)
        self.assertEqual(active, profile["id"])

    def test_save_current_as_profile(self) -> None:
        profile = llm_profile_store.save_current_as_profile("Current")
        cfg = load_settings(force=True)
        self.assertEqual(profile["llm_model"], cfg.llm_model)
        self.assertEqual(profile["system_prompt"], cfg.system_prompt)
        self.assertEqual(cfg.active_llm_profile_id, profile["id"])

    def test_delete_profile_clears_active(self) -> None:
        profile = llm_profile_store.create_profile(name="Tmp", set_active=True)
        ok = llm_profile_store.delete_profile(profile["id"])
        self.assertTrue(ok)
        cfg = load_settings(force=True)
        self.assertEqual(cfg.active_llm_profile_id, "")
        data = json.loads(self.config_path.read_text(encoding="utf-8"))
        self.assertEqual(data.get("llm_profiles"), [])


if __name__ == "__main__":
    unittest.main()
