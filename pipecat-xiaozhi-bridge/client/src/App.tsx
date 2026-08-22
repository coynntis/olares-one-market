import { useState, useEffect } from "react";
import { AgentGlobe } from "./components/AgentGlobe";
import { ChatPanel } from "./components/ChatPanel";
import { SettingsPanel } from "./components/SettingsPanel";
import { useVoiceStatus } from "./hooks/voiceStatusContext";
import { unlockSharedAudio } from "./audio/sharedAudioSession";
import type { AppTab } from "./types";

const TABS: { id: AppTab; label: string; code: string }[] = [
  { id: "chat", label: "Interface", code: "IF-01" },
  { id: "settings", label: "Systems", code: "CFG" },
];

export function App() {
  const [tab, setTab] = useState<AppTab>("chat");
  const { status, level } = useVoiceStatus();

  useEffect(() => {
    const unlock = () => {
      void unlockSharedAudio();
    };
    window.addEventListener("pointerdown", unlock, { passive: true });
    window.addEventListener("touchstart", unlock, { passive: true });
    return () => {
      window.removeEventListener("pointerdown", unlock);
      window.removeEventListener("touchstart", unlock);
    };
  }, []);

  return (
    <div className="app-shell">
      <div className="hud-grid" aria-hidden />
      <header className="app-header app-header-compact hud-panel">
        <div className="brand brand-compact">
          <AgentGlobe status={status} level={level} compact />
          <h1>AGENT R</h1>
        </div>
        <nav className="tab-nav tab-nav-inline" role="tablist" aria-label="Main modules">
          {TABS.map((t) => (
            <button
              key={t.id}
              type="button"
              role="tab"
              aria-selected={tab === t.id}
              className={`tab-btn tab-btn-compact ${tab === t.id ? "active" : ""}`}
              onClick={() => setTab(t.id)}
            >
              <span className="tab-code">{t.code}</span>
              <span className="tab-label">{t.label}</span>
            </button>
          ))}
        </nav>
      </header>

      <main className="app-main" role="tabpanel">
        {tab === "chat" && <ChatPanel />}
        {tab === "settings" && <SettingsPanel />}
      </main>

      <footer className="hud-footer">
        <span className="hud-footer-line">AGENT R · XIAOZHI BRIDGE</span>
      </footer>
    </div>
  );
}
