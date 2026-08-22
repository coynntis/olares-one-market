"""Agent harness helpers."""

from __future__ import annotations

import unittest

from xiaozhi_bridge.agent_tool_utils import (
    clip_tool_content,
    ensure_tool_call_ids,
    human_tool_label,
    light_generated_image_meta,
    process_mcp_tool_result,
    tool_announce_message,
    tool_done_message,
)


class AgentHarnessTests(unittest.TestCase):
    def test_clip_tool_content_short(self) -> None:
        self.assertEqual(clip_tool_content("hello"), "hello")

    def test_clip_tool_content_long(self) -> None:
        raw = "x" * 20000
        clipped = clip_tool_content(raw)
        self.assertIn("truncated", clipped)
        self.assertLess(len(clipped), 20000)

    def test_ensure_tool_call_ids(self) -> None:
        out = ensure_tool_call_ids([{"function": {"name": "search", "arguments": "{}"}}])
        self.assertTrue(str(out[0].get("id", "")).startswith("call_"))

    def test_tool_announce_no_url(self) -> None:
        msg = tool_announce_message("openwebsearch__fetchWebContent")
        self.assertNotIn("http", msg.lower())
        self.assertIn("fetch", msg.lower())

    def test_human_tool_label(self) -> None:
        self.assertEqual(human_tool_label("browser__take_picture"), "camera")

    def test_process_mcp_tool_result_strips_image_b64(self) -> None:
        raw = '{"seed": 42, "width": 512, "height": 512, "prompt": "a fox", "image_b64": "aGVsbG8="}'
        llm_text, gen = process_mcp_tool_result("krea__generate_image", raw)
        self.assertIsNotNone(gen)
        assert gen is not None
        self.assertIn("data:image/png;base64,aGVsbG8=", gen["data_url"])
        self.assertNotIn("aGVsbG8=", llm_text)
        self.assertIn("generated image shown in chat", llm_text)
        light = light_generated_image_meta(gen)
        self.assertTrue(light["has_image"])
        self.assertNotIn("data_url", light)


if __name__ == "__main__":
    unittest.main()
