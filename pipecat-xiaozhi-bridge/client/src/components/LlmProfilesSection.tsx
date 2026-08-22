import { useCallback, useEffect, useState } from "react";
import {
  activateLlmProfile,
  createLlmProfile,
  deleteLlmProfile,
  fetchLlmProfiles,
} from "../api";
import type { BridgeSettings, LlmProfile } from "../types";

interface Props {
  form: BridgeSettings;
  onFormChange: (patch: Partial<BridgeSettings>) => void;
  onSettingsSaved: (settings: BridgeSettings) => void;
  adminToken?: string;
}

export function LlmProfilesSection({ form, onFormChange, onSettingsSaved, adminToken }: Props) {
  const [profiles, setProfiles] = useState<LlmProfile[]>([]);
  const [activeId, setActiveId] = useState("");
  const [newName, setNewName] = useState("");
  const [busy, setBusy] = useState(false);
  const [msg, setMsg] = useState<string | null>(null);
  const [err, setErr] = useState<string | null>(null);

  const load = useCallback(async () => {
    try {
      const data = await fetchLlmProfiles();
      setProfiles(data.profiles);
      setActiveId(data.active_profile_id);
      setErr(null);
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    }
  }, []);

  useEffect(() => {
    void load();
  }, [load]);

  const applySettings = (settings: BridgeSettings) => {
    onSettingsSaved(settings);
    onFormChange({
      llm_base_url: settings.llm_base_url,
      llm_model: settings.llm_model,
      system_prompt: settings.system_prompt,
      active_llm_profile_id: settings.active_llm_profile_id,
    });
  };

  const saveCurrent = async () => {
    const name = newName.trim();
    if (!name) {
      setErr("Profile name required");
      return;
    }
    setBusy(true);
    setMsg(null);
    setErr(null);
    try {
      const { profile, settings } = await createLlmProfile(
        { name, from_current: true },
        adminToken
      );
      setProfiles((prev) => [profile, ...prev.filter((p) => p.id !== profile.id)]);
      setActiveId(profile.id);
      applySettings(settings);
      setNewName("");
      setMsg(`Saved profile “${profile.name}” and activated it.`);
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  const activate = async (profileId: string) => {
    setBusy(true);
    setMsg(null);
    setErr(null);
    try {
      const { profile, settings } = await activateLlmProfile(profileId, adminToken);
      setActiveId(profile.id);
      applySettings(settings);
      setMsg(`Activated profile “${profile.name}”.`);
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  const remove = async (profileId: string) => {
    setBusy(true);
    setMsg(null);
    setErr(null);
    try {
      const settings = await deleteLlmProfile(profileId, adminToken);
      setProfiles((prev) => prev.filter((p) => p.id !== profileId));
      if (activeId === profileId) setActiveId("");
      applySettings(settings);
      setMsg("Profile deleted.");
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  return (
    <section className="settings-section">
      <div className="section-head">
        <h2>LLM profiles</h2>
        <button type="button" className="btn btn-ghost btn-sm" disabled={busy} onClick={() => void load()}>
          Refresh
        </button>
      </div>
      <p className="muted">
        Save endpoint + model + system prompt presets. Activating a profile updates the live LLM settings
        for both text chat and voice pipeline.
      </p>

      <div className="row">
        <div style={{ flex: "2 1 220px" }}>
          <label htmlFor="llm-profile-name">Save current LLM as profile</label>
          <input
            id="llm-profile-name"
            value={newName}
            onChange={(e) => setNewName(e.target.value)}
            placeholder="e.g. Coding agent / Cantonese chat"
          />
        </div>
        <div style={{ alignSelf: "end" }}>
          <button type="button" className="btn btn-primary btn-sm" disabled={busy} onClick={() => void saveCurrent()}>
            Save & activate
          </button>
        </div>
      </div>

      {profiles.length === 0 ? (
        <p className="muted">No saved profiles yet.</p>
      ) : (
        <ul className="profile-list">
          {profiles.map((p) => (
            <li key={p.id} className={`profile-card ${p.id === activeId ? "active" : ""}`}>
              <div className="profile-card-head">
                <strong>{p.name}</strong>
                {p.id === activeId && <span className="tag">active</span>}
              </div>
              <p className="muted profile-card-meta">
                {p.llm_model || "(no model)"} · {(p.llm_base_url || "no URL").replace(/^https?:\/\//, "")}
              </p>
              {p.system_prompt && (
                <p className="profile-card-prompt">{p.system_prompt.slice(0, 120)}{p.system_prompt.length > 120 ? "…" : ""}</p>
              )}
              <div className="toolbar">
                <button
                  type="button"
                  className="btn btn-ghost btn-sm"
                  disabled={busy || p.id === activeId}
                  onClick={() => void activate(p.id)}
                >
                  Use
                </button>
                <button
                  type="button"
                  className="btn btn-ghost btn-sm"
                  disabled={busy}
                  onClick={() => void remove(p.id)}
                >
                  Delete
                </button>
              </div>
            </li>
          ))}
        </ul>
      )}

      {(form.active_llm_profile_id || activeId) && (
        <p className="muted">
          Active profile id: <code>{form.active_llm_profile_id || activeId}</code>
        </p>
      )}
      {msg && <div className="banner banner-ok">{msg}</div>}
      {err && <div className="banner banner-error">{err}</div>}
    </section>
  );
}
