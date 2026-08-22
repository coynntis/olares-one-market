import { useCallback, useEffect, useState } from "react";
import { activateConversation, fetchConversations } from "../api";
import type { Conversation } from "../types";

function formatWhen(ts: number): string {
  const d = new Date(ts);
  return d.toLocaleString(undefined, {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

export function ConversationDrawer({
  deviceId,
  currentId,
  open,
  onClose,
  onSwitch,
}: {
  deviceId: string;
  currentId: string | null;
  open: boolean;
  onClose: () => void;
  onSwitch: (conversationId: string, messages: import("../types").StoredMessage[]) => void;
}) {
  const [items, setItems] = useState<Conversation[]>([]);
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  const reload = useCallback(async () => {
    setLoading(true);
    setErr(null);
    try {
      const data = await fetchConversations(deviceId);
      setItems(data.conversations);
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }, [deviceId]);

  useEffect(() => {
    if (open) void reload();
  }, [open, reload]);

  const pick = async (id: string) => {
    if (id === currentId) {
      onClose();
      return;
    }
    try {
      const data = await activateConversation(deviceId, id);
      onSwitch(data.conversation.id, data.messages);
      onClose();
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    }
  };

  if (!open) return null;

  return (
    <div className="conv-drawer-backdrop" role="presentation" onClick={onClose}>
      <aside
        className="conv-drawer hud-panel"
        role="dialog"
        aria-label="Past conversations"
        onClick={(e) => e.stopPropagation()}
      >
        <header className="conv-drawer-head">
          <h2>Past threads</h2>
          <button type="button" className="btn btn-ghost btn-sm" onClick={onClose}>
            Close
          </button>
        </header>
        <p className="muted conv-drawer-sub">Device {deviceId}</p>
        {loading && <p className="muted">Loading…</p>}
        {err && <div className="banner banner-error">{err}</div>}
        <ul className="conv-list">
          {items.map((c) => (
            <li key={c.id}>
              <button
                type="button"
                className={`conv-item ${c.id === currentId ? "active" : ""}`}
                onClick={() => void pick(c.id)}
              >
                <span className="conv-item-title">{c.title || "Chat"}</span>
                <span className="conv-item-meta">
                  {formatWhen(c.updated_at)}
                  {c.message_count != null ? `, ${c.message_count} msgs` : ""}
                </span>
              </button>
            </li>
          ))}
          {!loading && items.length === 0 && (
            <li className="muted conv-empty">No past conversations for this device.</li>
          )}
        </ul>
      </aside>
    </div>
  );
}
