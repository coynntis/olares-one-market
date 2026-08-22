"""SQLite chat history on app data volume (/data/chat.db by default)."""

from __future__ import annotations

import base64
import json
import logging
import os
import sqlite3
import threading
import time
import uuid
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_LOCK = threading.Lock()
_INITIALIZED = False
MAX_HISTORY = 80


def db_path() -> Path:
    raw = os.environ.get("CHAT_DB_PATH", "/data/chat.db")
    return Path(raw)


def images_dir() -> Path:
    raw = os.environ.get("CHAT_IMAGES_DIR", "/data/chat_images")
    return Path(raw)


def _connect() -> sqlite3.Connection:
    path = db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def _init_db() -> None:
    global _INITIALIZED
    with _LOCK:
        if _INITIALIZED:
            return
        images_dir().mkdir(parents=True, exist_ok=True)
        conn = _connect()
        try:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS conversations (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL DEFAULT 'Chat',
                    device_id TEXT,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS messages (
                    id TEXT PRIMARY KEY,
                    conversation_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    text TEXT NOT NULL,
                    image_path TEXT,
                    source TEXT NOT NULL DEFAULT 'text',
                    created_at REAL NOT NULL,
                    FOREIGN KEY (conversation_id) REFERENCES conversations(id)
                );
                CREATE TABLE IF NOT EXISTS devices (
                    device_id TEXT PRIMARY KEY,
                    current_conversation_id TEXT,
                    updated_at REAL NOT NULL,
                    FOREIGN KEY (current_conversation_id) REFERENCES conversations(id)
                );
                CREATE INDEX IF NOT EXISTS idx_messages_conv
                    ON messages(conversation_id, created_at);
                CREATE INDEX IF NOT EXISTS idx_conversations_device
                    ON conversations(device_id, updated_at DESC);
                """
            )
            _migrate_schema(conn)
            conn.commit()
        finally:
            conn.close()
        _INITIALIZED = True


def _now() -> float:
    return time.time()


def _migrate_schema(conn: sqlite3.Connection) -> None:
    cols = {row[1] for row in conn.execute("PRAGMA table_info(conversations)").fetchall()}
    if "device_id" not in cols:
        conn.execute("ALTER TABLE conversations ADD COLUMN device_id TEXT")
    msg_cols = {row[1] for row in conn.execute("PRAGMA table_info(messages)").fetchall()}
    if "meta_json" not in msg_cols:
        conn.execute("ALTER TABLE messages ADD COLUMN meta_json TEXT")


def _normalize_device_id(device_id: str) -> str:
    did = device_id.strip()
    if not did:
        raise ValueError("device_id required")
    return did


def _conversation_row(row: sqlite3.Row) -> dict[str, Any]:
    item = dict(row)
    for key in ("created_at", "updated_at"):
        if key in item and isinstance(item[key], float):
            item[key] = int(item[key] * 1000)
    return item


def _set_device_current(conn: sqlite3.Connection, device_id: str, conversation_id: str) -> None:
    ts = _now()
    conn.execute(
        """
        INSERT INTO devices (device_id, current_conversation_id, updated_at)
        VALUES (?, ?, ?)
        ON CONFLICT(device_id) DO UPDATE SET
            current_conversation_id = excluded.current_conversation_id,
            updated_at = excluded.updated_at
        """,
        (device_id, conversation_id, ts),
    )


def _assign_conversation_device(conn: sqlite3.Connection, conversation_id: str, device_id: str) -> None:
    conn.execute(
        "UPDATE conversations SET device_id = ? WHERE id = ? AND (device_id IS NULL OR device_id = '')",
        (device_id, conversation_id),
    )


def get_device_current_conversation_id(device_id: str) -> str | None:
    _init_db()
    device_id = _normalize_device_id(device_id)
    with _LOCK:
        conn = _connect()
        try:
            row = conn.execute(
                "SELECT current_conversation_id FROM devices WHERE device_id = ?",
                (device_id,),
            ).fetchone()
        finally:
            conn.close()
    if not row or not row["current_conversation_id"]:
        return None
    return str(row["current_conversation_id"])


def create_conversation(title: str = "Chat", device_id: str = "") -> dict[str, Any]:
    _init_db()
    device_id = _normalize_device_id(device_id)
    cid = str(uuid.uuid4())
    ts = _now()
    with _LOCK:
        conn = _connect()
        try:
            conn.execute(
                """
                INSERT INTO conversations (id, title, device_id, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (cid, title.strip() or "Chat", device_id, ts, ts),
            )
            _set_device_current(conn, device_id, cid)
            conn.commit()
        finally:
            conn.close()
    return {
        "id": cid,
        "title": title.strip() or "Chat",
        "device_id": device_id,
        "created_at": int(ts * 1000),
        "updated_at": int(ts * 1000),
    }


def activate_device_conversation(device_id: str, conversation_id: str) -> dict[str, Any]:
    """Set current conversation for device (continue past thread)."""
    _init_db()
    device_id = _normalize_device_id(device_id)
    conversation_id = conversation_id.strip()
    if not conversation_id:
        raise ValueError("conversation_id required")
    with _LOCK:
        conn = _connect()
        try:
            row = conn.execute(
                "SELECT id, title, device_id, created_at, updated_at FROM conversations WHERE id = ?",
                (conversation_id,),
            ).fetchone()
            if not row:
                raise LookupError("conversation not found")
            owner = row["device_id"]
            if owner and owner != device_id:
                raise PermissionError("conversation belongs to another device")
            if not owner:
                _assign_conversation_device(conn, conversation_id, device_id)
            _set_device_current(conn, device_id, conversation_id)
            conn.commit()
            refreshed = conn.execute(
                "SELECT id, title, device_id, created_at, updated_at FROM conversations WHERE id = ?",
                (conversation_id,),
            ).fetchone()
        finally:
            conn.close()
    return _conversation_row(refreshed)


def resolve_device_conversation(
    device_id: str,
    conversation_id: str | None = None,
) -> dict[str, Any]:
    """Current conversation for device; create if missing. Optional id switches current."""
    _init_db()
    device_id = _normalize_device_id(device_id)
    if conversation_id:
        return activate_device_conversation(device_id, conversation_id)

    current_id = get_device_current_conversation_id(device_id)
    if current_id:
        conv = get_conversation(current_id)
        if conv and (not conv.get("device_id") or conv.get("device_id") == device_id):
            return conv

    existing = list_conversations_for_device(device_id, limit=1)
    if existing:
        return activate_device_conversation(device_id, existing[0]["id"])
    return create_conversation(device_id=device_id)


def get_conversation(conversation_id: str) -> dict[str, Any] | None:
    _init_db()
    with _LOCK:
        conn = _connect()
        try:
            row = conn.execute(
                "SELECT id, title, device_id, created_at, updated_at FROM conversations WHERE id = ?",
                (conversation_id,),
            ).fetchone()
        finally:
            conn.close()
    if not row:
        return None
    return _conversation_row(row)


def list_conversations_for_device(device_id: str, limit: int = 50) -> list[dict[str, Any]]:
    _init_db()
    device_id = _normalize_device_id(device_id)
    with _LOCK:
        conn = _connect()
        try:
            rows = conn.execute(
                """
                SELECT c.id, c.title, c.device_id, c.created_at, c.updated_at,
                       (SELECT COUNT(*) FROM messages m WHERE m.conversation_id = c.id) AS message_count
                FROM conversations c
                WHERE c.device_id = ?
                ORDER BY c.updated_at DESC
                LIMIT ?
                """,
                (device_id, limit),
            ).fetchall()
        finally:
            conn.close()
    return [_conversation_row(r) for r in rows]


def list_conversations(limit: int = 50) -> list[dict[str, Any]]:
    """Legacy global list (admin/debug). Prefer list_conversations_for_device."""
    _init_db()
    with _LOCK:
        conn = _connect()
        try:
            rows = conn.execute(
                """
                SELECT c.id, c.title, c.device_id, c.created_at, c.updated_at,
                       (SELECT COUNT(*) FROM messages m WHERE m.conversation_id = c.id) AS message_count
                FROM conversations c
                ORDER BY c.updated_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        finally:
            conn.close()
    return [_conversation_row(r) for r in rows]


def get_or_create_active_conversation(conversation_id: str | None = None) -> dict[str, Any]:
    """Deprecated: use resolve_device_conversation with device_id."""
    _init_db()
    if conversation_id:
        conv = get_conversation(conversation_id)
        if conv:
            return conv
    existing = list_conversations(limit=1)
    if existing:
        return existing[0]
    return create_conversation(device_id="legacy-unscoped")


def list_messages(conversation_id: str, *, limit: int = 200) -> list[dict[str, Any]]:
    _init_db()
    with _LOCK:
        conn = _connect()
        try:
            rows = conn.execute(
                """
                SELECT id, conversation_id, role, text, image_path, source, created_at, meta_json
                FROM messages
                WHERE conversation_id = ?
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (conversation_id, limit),
            ).fetchall()
            rows = list(reversed(rows))
        finally:
            conn.close()
    out: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        item["created_at"] = int(item["created_at"] * 1000)
        if item.get("image_path"):
            item["image_url"] = f"/api/messages/{item['id']}/image"
        raw_meta = item.pop("meta_json", None)
        if raw_meta:
            try:
                item["meta"] = json.loads(raw_meta)
            except json.JSONDecodeError:
                pass
        out.append(item)
    return out


def _save_image(message_id: str, image_data_url: str) -> str | None:
    raw = image_data_url.strip()
    if not raw:
        return None
    mime = "image/jpeg"
    b64 = raw
    if raw.startswith("data:"):
        import re

        m = re.match(r"^data:(image/[a-zA-Z0-9.+-]+);base64,(.+)$", raw, re.DOTALL)
        if not m:
            logger.warning("bad data URL for message %s", message_id)
            return None
        mime, b64 = m.group(1), m.group(2)
    b64 = "".join(b64.split())
    try:
        data = base64.b64decode(b64, validate=False)
    except Exception:
        logger.warning("base64 decode failed for message %s", message_id)
        return None
    if not data:
        return None
    ext = "png" if "png" in mime else "jpg" if "jpeg" in mime or "jpg" in mime else "bin"
    path = images_dir() / f"{message_id}.{ext}"
    path.write_bytes(data)
    return str(path)


def append_message(
    conversation_id: str,
    *,
    role: str,
    text: str,
    image_data_url: str | None = None,
    source: str = "text",
    meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    _init_db()
    mid = str(uuid.uuid4())
    ts = _now()
    image_path: str | None = None
    if image_data_url:
        image_path = _save_image(mid, image_data_url)
    meta_json = json.dumps(meta) if meta else None
    with _LOCK:
        conn = _connect()
        try:
            conn.execute(
                """
                INSERT INTO messages (id, conversation_id, role, text, image_path, source, created_at, meta_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (mid, conversation_id, role, text, image_path, source, ts, meta_json),
            )
            conn.execute(
                "UPDATE conversations SET updated_at = ? WHERE id = ?",
                (ts, conversation_id),
            )
            conn.commit()
        finally:
            conn.close()
    item: dict[str, Any] = {
        "id": mid,
        "conversation_id": conversation_id,
        "role": role,
        "text": text,
        "source": source,
        "created_at": int(ts * 1000),
    }
    if image_path:
        item["image_url"] = f"/api/messages/{mid}/image"
    if meta:
        item["meta"] = meta
    return item


def append_generated_images(
    conversation_id: str,
    images: list[dict[str, Any]],
    *,
    source: str = "text",
) -> list[dict[str, Any]]:
    """Persist MCP/tool-generated images (data_url) as assistant bubbles with /api/.../image links."""
    from xiaozhi_bridge.agent_tool_utils import generated_image_caption

    created: list[dict[str, Any]] = []
    for gen in images:
        if not isinstance(gen, dict):
            continue
        data_url = str(gen.get("data_url") or "").strip()
        if not data_url.startswith("data:"):
            logger.warning("skip generated image — missing data_url")
            continue
        try:
            msg = append_message(
                conversation_id,
                role="assistant",
                text=generated_image_caption(gen),
                image_data_url=data_url,
                source=source,
                meta={
                    "generated_image": True,
                    "tool_name": gen.get("tool_name"),
                    "seed": gen.get("seed"),
                    "width": gen.get("width"),
                    "height": gen.get("height"),
                },
            )
        except Exception:
            logger.exception("failed to persist generated image")
            continue
        if msg.get("image_url"):
            created.append(msg)
            logger.info(
                "persisted generated image id=%s url=%s",
                msg.get("id"),
                msg.get("image_url"),
            )
        else:
            logger.warning("generated image saved without image_url id=%s", msg.get("id"))
            created.append(msg)
    return created


def append_agent_trace(
    conversation_id: str,
    trace: list[dict[str, Any]],
    *,
    source: str = "text",
    generated_images: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Persist tool-call rounds from agent loop. Returns assistant image messages created."""
    created_images: list[dict[str, Any]] = []
    pending_gens = list(generated_images or [])
    gen_idx = 0

    for row in trace:
        role = str(row.get("role") or "")
        if role not in ("assistant", "tool"):
            continue
        text = str(row.get("text") or "")
        meta = row.get("meta") if isinstance(row.get("meta"), dict) else {}
        if role == "assistant" and meta.get("tool_calls"):
            append_message(
                conversation_id,
                role="assistant",
                text=text,
                source=source,
                meta={"tool_calls": meta["tool_calls"]},
            )
        elif role == "tool":
            # Never persist multi-MB data_url in tool meta.
            meta_store = dict(meta)
            gen = meta_store.pop("generated_image", None)
            if isinstance(gen, dict):
                light = {k: v for k, v in gen.items() if k != "data_url"}
                light["has_image"] = True
                meta_store["generated_image"] = light
            append_message(
                conversation_id,
                role="tool",
                text=text,
                source="tool",
                meta=meta_store,
            )
            # Match image tools → in-memory generated_images (has data_url).
            wants_image = isinstance(gen, dict) and (
                gen.get("has_image") or str(gen.get("data_url") or "").startswith("data:")
            )
            if wants_image:
                gen_full: dict[str, Any] | None = None
                if gen_idx < len(pending_gens):
                    gen_full = pending_gens[gen_idx]
                    gen_idx += 1
                elif isinstance(gen, dict) and str(gen.get("data_url") or "").startswith("data:"):
                    gen_full = gen
                if gen_full is not None:
                    created_images.extend(
                        append_generated_images(conversation_id, [gen_full], source=source)
                    )
        elif role == "assistant" and text.strip():
            append_message(
                conversation_id,
                role="assistant",
                text=text,
                source=source,
            )

    # Any leftover images (tool row missing) still persist.
    if gen_idx < len(pending_gens):
        created_images.extend(
            append_generated_images(conversation_id, pending_gens[gen_idx:], source=source)
        )
    return created_images


def read_message_image(message_id: str) -> tuple[bytes, str] | None:
    _init_db()
    with _LOCK:
        conn = _connect()
        try:
            row = conn.execute(
                "SELECT image_path FROM messages WHERE id = ?",
                (message_id,),
            ).fetchone()
        finally:
            conn.close()
    if not row or not row["image_path"]:
        return None
    path = Path(row["image_path"])
    if not path.is_file():
        return None
    ext = path.suffix.lower()
    mime = "image/png" if ext == ".png" else "image/jpeg"
    return path.read_bytes(), mime


def delete_conversation(conversation_id: str, *, device_id: str | None = None) -> bool:
    _init_db()
    with _LOCK:
        conn = _connect()
        try:
            row = conn.execute(
                "SELECT device_id FROM conversations WHERE id = ?",
                (conversation_id,),
            ).fetchone()
            if not row:
                return False
            owner = row["device_id"]
            if device_id and owner and owner != device_id:
                raise PermissionError("conversation belongs to another device")
            rows = conn.execute(
                "SELECT image_path FROM messages WHERE conversation_id = ? AND image_path IS NOT NULL",
                (conversation_id,),
            ).fetchall()
            conn.execute("DELETE FROM messages WHERE conversation_id = ?", (conversation_id,))
            cur = conn.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))
            deleted = cur.rowcount > 0
            if deleted and device_id:
                current = conn.execute(
                    "SELECT current_conversation_id FROM devices WHERE device_id = ?",
                    (device_id,),
                ).fetchone()
                if current and current["current_conversation_id"] == conversation_id:
                    nxt = conn.execute(
                        """
                        SELECT id FROM conversations
                        WHERE device_id = ?
                        ORDER BY updated_at DESC
                        LIMIT 1
                        """,
                        (device_id,),
                    ).fetchone()
                    if nxt:
                        _set_device_current(conn, device_id, nxt["id"])
                    else:
                        conn.execute("DELETE FROM devices WHERE device_id = ?", (device_id,))
            conn.commit()
        finally:
            conn.close()
    for row in rows:
        try:
            Path(row["image_path"]).unlink(missing_ok=True)
        except OSError:
            pass
    return deleted


def clear_conversation_messages(conversation_id: str) -> None:
    _init_db()
    with _LOCK:
        conn = _connect()
        try:
            rows = conn.execute(
                "SELECT image_path FROM messages WHERE conversation_id = ? AND image_path IS NOT NULL",
                (conversation_id,),
            ).fetchall()
            conn.execute("DELETE FROM messages WHERE conversation_id = ?", (conversation_id,))
            conn.execute(
                "UPDATE conversations SET updated_at = ? WHERE id = ?",
                (_now(), conversation_id),
            )
            conn.commit()
        finally:
            conn.close()
    for row in rows:
        try:
            Path(row["image_path"]).unlink(missing_ok=True)
        except OSError:
            pass


def llm_messages_for_conversation(conversation_id: str, system_prompt: str) -> list[dict[str, Any]]:
    """Build OpenAI chat messages from stored history."""
    from xiaozhi_bridge.stt_text import is_funasr_metadata_blob, normalize_stt_transcript

    stored = list_messages(conversation_id, limit=MAX_HISTORY)
    messages: list[dict[str, Any]] = [{"role": "system", "content": system_prompt}]
    for item in stored[-MAX_HISTORY:]:
        role = item.get("role")
        text = str(item.get("text") or "")
        meta = item.get("meta") if isinstance(item.get("meta"), dict) else {}

        if role == "user" and (is_funasr_metadata_blob(text) or not normalize_stt_transcript(text)[0]):
            if not item.get("image_path"):
                continue
            text = text if not is_funasr_metadata_blob(text) else "(image)"

        if role == "tool":
            call_id = str(meta.get("tool_call_id") or "")
            if call_id:
                messages.append({"role": "tool", "tool_call_id": call_id, "content": text})
            continue

        if role == "assistant" and meta.get("tool_calls"):
            messages.append(
                {
                    "role": "assistant",
                    "content": text or None,
                    "tool_calls": meta["tool_calls"],
                }
            )
            continue

        if role not in ("user", "assistant"):
            continue

        image_path = item.get("image_path")
        if role == "user" and image_path:
            path = Path(image_path)
            if path.is_file():
                data = path.read_bytes()
                ext = path.suffix.lower()
                mime = "image/png" if ext == ".png" else "image/jpeg"
                b64 = base64.b64encode(data).decode("ascii")
                content: str | list[dict[str, Any]] = [
                    {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
                    {"type": "text", "text": text or "Describe this image."},
                ]
                messages.append({"role": "user", "content": content})
                continue
        messages.append({"role": role, "content": text})
    return messages
