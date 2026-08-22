import { useCallback, useEffect, useState } from "react";
import {
  addMcpFromSuggestion,
  fetchMcpServers,
  fetchMcpSuggestions,
  saveMcpServers,
  testMcpServer,
} from "../api";
import type { McpServer, McpSuggestion, McpTestResult } from "../types";

function formatMcpTest(r: McpTestResult): string {
  if (r.ok) {
    if (r.kind === "browser") {
      return r.detail || `HTTP ${r.status ?? "OK"}`;
    }
    const n = r.tool_count ?? r.tools?.length ?? 0;
    return `${n} tool${n === 1 ? "" : "s"}${r.latency_ms ? ` · ${r.latency_ms}ms` : ""}`;
  }
  return r.error || r.detail || "Failed";
}

function McpTestLine({ result }: { result: McpTestResult | null }) {
  if (!result) return null;
  return (
    <div className={`test-result ${result.ok ? "test-ok" : "test-fail"}`}>
      <span className="test-status">{result.ok ? "PASS" : "FAIL"}</span>
      {result.name && <strong>{result.name}</strong>} {formatMcpTest(result)}
      {result.url && <span className="muted test-url">{result.url}</span>}
      {result.tools && result.tools.length > 0 && (
        <ul className="mcp-tool-list">
          {result.tools.slice(0, 8).map((t) => (
            <li key={t.name}>
              <code>{t.name}</code>
              {t.description ? <span className="muted"> — {t.description}</span> : null}
            </li>
          ))}
          {result.tools.length > 8 && (
            <li className="muted">+{result.tools.length - 8} more</li>
          )}
        </ul>
      )}
    </div>
  );
}

function newCustomServer(): McpServer {
  return {
    id: crypto.randomUUID(),
    name: "Remote MCP",
    enabled: true,
    transport: "http",
    url: "",
    headers: {},
    catalog_id: null,
  };
}

interface Props {
  adminToken: string;
}

export function McpServersSection({ adminToken }: Props) {
  const [servers, setServers] = useState<McpServer[]>([]);
  const [suggestions, setSuggestions] = useState<McpSuggestion[]>([]);
  const [sharedUrls, setSharedUrls] = useState<Record<string, string>>({});
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [testingId, setTestingId] = useState<string | "all" | null>(null);
  const [testResults, setTestResults] = useState<Record<string, McpTestResult>>({});
  const [msg, setMsg] = useState<string | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [showCustom, setShowCustom] = useState(false);
  const [draft, setDraft] = useState<McpServer>(newCustomServer());

  const load = useCallback(async () => {
    setLoading(true);
    setErr(null);
    try {
      const [sug, srv] = await Promise.all([fetchMcpSuggestions(), fetchMcpServers()]);
      setSuggestions(sug);
      setServers(srv);
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void load();
  }, [load]);

  const persist = async (next: McpServer[]) => {
    setSaving(true);
    setErr(null);
    setMsg(null);
    try {
      const saved = await saveMcpServers(next, adminToken);
      setServers(saved);
      setMsg("MCP servers saved");
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    } finally {
      setSaving(false);
    }
  };

  const addSuggestion = async (sug: McpSuggestion) => {
    setErr(null);
    setMsg(null);
    try {
      const shared = (sharedUrls[sug.id] || "").trim();
      const saved = await addMcpFromSuggestion({
        catalog_id: sug.id,
        shared_base_url: shared || undefined,
        token: adminToken,
      });
      setServers(saved);
      setMsg(`Added ${sug.name}`);
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    }
  };

  const updateServer = (id: string, patch: Partial<McpServer>) => {
    setServers((prev) => prev.map((s) => (s.id === id ? { ...s, ...patch } : s)));
  };

  const removeServer = (id: string) => {
    setServers((prev) => prev.filter((s) => s.id !== id));
    setTestResults((prev) => {
      const next = { ...prev };
      delete next[id];
      return next;
    });
  };

  const runTest = async (server: McpServer) => {
    setTestingId(server.id);
    setErr(null);
    try {
      const result = (await testMcpServer({ server })) as McpTestResult;
      setTestResults((prev) => ({ ...prev, [server.id]: result }));
    } catch (e) {
      setTestResults((prev) => ({
        ...prev,
        [server.id]: { ok: false, name: server.name, error: e instanceof Error ? e.message : String(e) },
      }));
    } finally {
      setTestingId(null);
    }
  };

  const runTestAll = async () => {
    setTestingId("all");
    setErr(null);
    try {
      const data = await testMcpServer({ service: "all" });
      if ("results" in data && Array.isArray(data.results)) {
        const map: Record<string, McpTestResult> = {};
        for (const r of data.results) {
          const match = servers.find((s) => s.name === r.name);
          if (match) map[match.id] = r;
        }
        setTestResults(map);
      }
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    } finally {
      setTestingId(null);
    }
  };

  const addCustom = () => {
    if (!draft.name.trim() || !draft.url?.trim()) {
      setErr("Name and URL required for custom MCP server");
      return;
    }
    setServers((prev) => [...prev, { ...draft, id: crypto.randomUUID() }]);
    setDraft(newCustomServer());
    setShowCustom(false);
    setMsg("Custom server added — click Save MCP to persist");
  };

  if (loading) {
    return (
      <section className="settings-section">
        <h2>MCP servers</h2>
        <p className="muted">Loading…</p>
      </section>
    );
  }

  return (
    <section className="settings-section mcp-section">
      <h2>MCP servers</h2>
      <p className="muted mcp-intro">
        Connect Olares shared entrances or remote MCP endpoints. Test runs <code>list_tools</code> (or HTTP
        health for browserless). Enabled MCP tools are passed to the LLM automatically when supported.
      </p>

      <h3 className="mcp-subhead">Olares market suggestions</h3>
      <div className="mcp-suggestion-grid">
        {suggestions.map((sug) => (
          <article key={sug.id} className={`mcp-card ${sug.kind === "browser" ? "mcp-card-browser" : ""}`}>
            <div className="mcp-card-head">
              <strong>{sug.name}</strong>
              <span className="tag">{sug.kind === "browser" ? "browser" : sug.transport}</span>
            </div>
            <p className="muted mcp-card-desc">{sug.description}</p>
            <p className="muted mcp-card-hint">
              <code>{sug.market_app}</code> · path <code>{sug.path}</code>
            </p>
            <label>Shared entrance base URL</label>
            <input
              type="url"
              placeholder={`http://xxxxxxxx.shared.olares.com`}
              value={sharedUrls[sug.id] || ""}
              onChange={(e) => setSharedUrls((prev) => ({ ...prev, [sug.id]: e.target.value }))}
            />
            <p className="muted mcp-card-foot">{sug.install_hint}</p>
            <button
              type="button"
              className="btn btn-ghost btn-sm"
              disabled={saving}
              onClick={() => void addSuggestion(sug)}
            >
              Add{sug.kind === "browser" ? " (health check)" : ""}
            </button>
          </article>
        ))}
      </div>

      <h3 className="mcp-subhead">Configured servers</h3>
      {servers.length === 0 ? (
        <p className="muted">No MCP servers yet. Add a suggestion or custom server below.</p>
      ) : (
        <div className="mcp-server-list">
          {servers.map((srv) => (
            <div key={srv.id} className="mcp-server-row">
              <div className="mcp-server-row-head">
                <label className="mcp-enable">
                  <input
                    type="checkbox"
                    checked={srv.enabled}
                    onChange={(e) => updateServer(srv.id, { enabled: e.target.checked })}
                  />
                  <span>{srv.name}</span>
                </label>
                <span className="tag">{srv.transport}</span>
                {srv.catalog_id && <span className="tag tag-dim">{srv.catalog_id}</span>}
              </div>
              {srv.transport !== "stdio" ? (
                <input
                  type="url"
                  value={srv.url || ""}
                  onChange={(e) => updateServer(srv.id, { url: e.target.value })}
                  placeholder="MCP URL"
                />
              ) : (
                <div className="row">
                  <input
                    value={srv.command || ""}
                    onChange={(e) => updateServer(srv.id, { command: e.target.value })}
                    placeholder="command"
                  />
                  <input
                    value={(srv.args || []).join(" ")}
                    onChange={(e) =>
                      updateServer(srv.id, {
                        args: e.target.value.split(/\s+/).filter(Boolean),
                      })
                    }
                    placeholder="args"
                  />
                </div>
              )}
              <div className="toolbar toolbar-tight">
                <button
                  type="button"
                  className="btn btn-ghost btn-sm"
                  disabled={testingId !== null}
                  onClick={() => void runTest(srv)}
                >
                  {testingId === srv.id ? "Testing…" : "Test tools"}
                </button>
                <button type="button" className="btn btn-ghost btn-sm" onClick={() => removeServer(srv.id)}>
                  Remove
                </button>
              </div>
              <McpTestLine result={testResults[srv.id] ?? null} />
            </div>
          ))}
        </div>
      )}

      <div className="toolbar">
        <button
          type="button"
          className="btn btn-primary"
          disabled={saving}
          onClick={() => void persist(servers)}
        >
          {saving ? "Saving…" : "Save MCP servers"}
        </button>
        <button
          type="button"
          className="btn btn-ghost"
          disabled={testingId !== null || servers.length === 0}
          onClick={() => void runTestAll()}
        >
          {testingId === "all" ? "Testing all…" : "Test all enabled"}
        </button>
        <button type="button" className="btn btn-ghost" onClick={() => setShowCustom((v) => !v)}>
          {showCustom ? "Cancel custom" : "Add custom server"}
        </button>
        <button type="button" className="btn btn-ghost" onClick={() => void load()}>
          Reload
        </button>
      </div>

      {showCustom && (
        <div className="mcp-custom-form">
          <h3 className="mcp-subhead">Custom MCP server</h3>
          <label>Name</label>
          <input value={draft.name} onChange={(e) => setDraft((d) => ({ ...d, name: e.target.value }))} />
          <label>Transport</label>
          <select
            value={draft.transport}
            onChange={(e) =>
              setDraft((d) => ({
                ...d,
                transport: e.target.value as McpServer["transport"],
              }))
            }
          >
            <option value="http">HTTP (streamable)</option>
            <option value="sse">SSE</option>
            <option value="stdio">stdio</option>
          </select>
          {draft.transport === "stdio" ? (
            <>
              <label>Command</label>
              <input
                value={draft.command || ""}
                onChange={(e) => setDraft((d) => ({ ...d, command: e.target.value }))}
              />
              <label>Args (space-separated)</label>
              <input
                value={(draft.args || []).join(" ")}
                onChange={(e) =>
                  setDraft((d) => ({ ...d, args: e.target.value.split(/\s+/).filter(Boolean) }))
                }
              />
            </>
          ) : (
            <>
              <label>URL</label>
              <input
                type="url"
                value={draft.url || ""}
                onChange={(e) => setDraft((d) => ({ ...d, url: e.target.value }))}
                placeholder="https://example.com/mcp"
              />
            </>
          )}
          <button type="button" className="btn btn-ghost" onClick={addCustom}>
            Add to list
          </button>
        </div>
      )}

      {msg && <div className="banner banner-ok">{msg}</div>}
      {err && <div className="banner banner-error">{err}</div>}
    </section>
  );
}
