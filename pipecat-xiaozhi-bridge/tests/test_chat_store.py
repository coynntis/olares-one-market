"""Chat history ordering tests."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from xiaozhi_bridge import chat_store


class ChatStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        base = Path(self._tmpdir.name)
        os.environ["CHAT_DB_PATH"] = str(base / "chat.db")
        os.environ["CHAT_IMAGES_DIR"] = str(base / "images")
        chat_store._INITIALIZED = False

    def tearDown(self) -> None:
        self._tmpdir.cleanup()

    def test_list_messages_returns_most_recent(self) -> None:
        conv = chat_store.create_conversation(device_id="dev-1")
        cid = conv["id"]
        for i in range(50):
            chat_store.append_message(cid, role="user", text=f"msg-{i}")
        rows = chat_store.list_messages(cid, limit=10)
        self.assertEqual(len(rows), 10)
        self.assertEqual(rows[0]["text"], "msg-40")
        self.assertEqual(rows[-1]["text"], "msg-49")

    def test_llm_messages_use_recent_history(self) -> None:
        conv = chat_store.create_conversation(device_id="dev-2")
        cid = conv["id"]
        for i in range(50):
            chat_store.append_message(cid, role="user", text=f"turn-{i}")
        msgs = chat_store.llm_messages_for_conversation(cid, "sys")
        user_texts = [m["content"] for m in msgs if m["role"] == "user"]
        self.assertGreaterEqual(len(user_texts), 10)
        self.assertEqual(user_texts[-1], "turn-49")
        self.assertNotIn("turn-0", user_texts[-5:])


if __name__ == "__main__":
    unittest.main()
