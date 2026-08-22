import { useCallback, useEffect, useState } from "react";
import { fetchConfig, saveConfig } from "../api";
import type { BuiltinToolsConfig } from "../types";

const TOOL_ROWS: { key: keyof BuiltinToolsConfig; label: string; hint: string }[] = [
  {
    key: "camera",
    label: "Camera",
    hint: "Agent can capture a photo from this browser tab (browser__take_picture).",
  },
  {
    key: "bluetooth",
    label: "Bluetooth",
    hint: "List Bluetooth devices known to this browser (browser__list_bluetooth_devices).",
  },
  {
    key: "geolocation",
    label: "Location",
    hint: "Read GPS coordinates when permitted (browser__get_geolocation).",
  },
];

interface Props {
  adminToken: string;
}

export function BuiltinToolsSection({ adminToken }: Props) {
  const [tools, setTools] = useState<BuiltinToolsConfig>({
    camera: false,
    bluetooth: false,
    geolocation: false,
  });
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [msg, setMsg] = useState<string | null>(null);
  const [err, setErr] = useState<string | null>(null);

  const load = useCallback(async () => {
    setLoading(true);
    setErr(null);
    try {
      const cfg = await fetchConfig();
      setTools({
        camera: cfg.builtin_tools?.camera ?? false,
        bluetooth: cfg.builtin_tools?.bluetooth ?? false,
        geolocation: cfg.builtin_tools?.geolocation ?? false,
      });
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void load();
  }, [load]);

  const save = async () => {
    setSaving(true);
    setMsg(null);
    setErr(null);
    try {
      await saveConfig({ builtin_tools: tools } as Record<string, unknown>, adminToken);
      setMsg("Built-in browser tools saved.");
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    } finally {
      setSaving(false);
    }
  };

  return (
    <section className="settings-section">
      <h2>Browser tools (this tab)</h2>
      <p className="muted settings-lead">
        Xiaozhi-style capabilities that run in your browser — not remote MCP servers. Keep Agent R
        open while the agent uses them. Off by default.
      </p>
      {loading && <p className="muted">Loading…</p>}
      {err && <div className="banner banner-error">{err}</div>}
      {msg && <div className="banner banner-ok">{msg}</div>}
      <ul className="builtin-tools-list">
        {TOOL_ROWS.map((row) => (
          <li key={row.key} className="builtin-tool-row">
            <label className="composer-opt">
              <input
                type="checkbox"
                checked={tools[row.key]}
                onChange={(e) => setTools((t) => ({ ...t, [row.key]: e.target.checked }))}
              />
              <span>{row.label}</span>
            </label>
            <p className="muted builtin-tool-hint">{row.hint}</p>
          </li>
        ))}
      </ul>
      <button type="button" className="btn btn-primary btn-sm" disabled={saving || loading} onClick={() => void save()}>
        {saving ? "Saving…" : "Save browser tools"}
      </button>
    </section>
  );
}
