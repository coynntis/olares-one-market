"""Tests for LLM request kwargs shaping."""

from __future__ import annotations

import unittest

from xiaozhi_bridge.llm_params import (
    build_completion_kwargs,
    finalize_sdk_completion_kwargs,
    parse_unexpected_sdk_kwarg,
    relocate_sdk_kwarg_to_extra_body,
    strip_sdk_kwarg,
)


class LlmParamsTests(unittest.TestCase):
    def test_top_k_goes_to_extra_body(self) -> None:
        raw = build_completion_kwargs(
            [{"role": "user", "content": "hi"}],
            supports={"top_k": True},
        )
        self.assertNotIn("top_k", raw)
        self.assertEqual(raw["extra_body"]["top_k"], 20)
        final = finalize_sdk_completion_kwargs(raw, {"top_k": True})
        self.assertEqual(final["extra_body"]["top_k"], 20)

    def test_top_k_stripped_when_unsupported(self) -> None:
        raw = build_completion_kwargs(
            [{"role": "user", "content": "hi"}],
            supports={"top_k": False},
        )
        self.assertNotIn("extra_body", raw)
        final = finalize_sdk_completion_kwargs(raw, {"top_k": False})
        self.assertNotIn("extra_body", final)

    def test_think_mode_strip_preserves_top_k(self) -> None:
        raw = {
            "model": "m",
            "messages": [],
            "extra_body": {
                "top_k": 20,
                "chat_template_kwargs": {"enable_thinking": True},
            },
        }
        final = finalize_sdk_completion_kwargs(raw, {"top_k": True, "think_mode": False})
        self.assertEqual(final["extra_body"]["top_k"], 20)
        self.assertNotIn("chat_template_kwargs", final["extra_body"])

    def test_relocate_top_level_extension(self) -> None:
        raw = {"model": "m", "messages": [], "top_k": 40}
        final = finalize_sdk_completion_kwargs(raw, {"top_k": True})
        self.assertNotIn("top_k", final)
        self.assertEqual(final["extra_body"]["top_k"], 40)

    def test_parse_unexpected_kwarg(self) -> None:
        err = TypeError("AsyncCompletions.create() got an unexpected keyword argument 'top_k'")
        self.assertEqual(parse_unexpected_sdk_kwarg(err), "top_k")

    def test_relocate_sdk_kwarg(self) -> None:
        raw = {"model": "m", "messages": [], "top_k": 5}
        moved = relocate_sdk_kwarg_to_extra_body(raw, "top_k")
        assert moved is not None
        self.assertEqual(moved["extra_body"]["top_k"], 5)

    def test_strip_sdk_kwarg_from_extra(self) -> None:
        raw = {"model": "m", "messages": [], "extra_body": {"top_k": 5}}
        out = strip_sdk_kwarg(raw, "top_k")
        self.assertNotIn("extra_body", out)


if __name__ == "__main__":
    unittest.main()
