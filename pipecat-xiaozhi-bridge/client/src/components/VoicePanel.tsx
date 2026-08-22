import { useCallback, useEffect, useState } from "react";
import { useXiaozhiWs } from "../hooks/useXiaozhiWs";
import { PipelineStatsBar } from "./PipelineStatsBar";

const STATUS_LABEL: Record<string, string> = {
  idle: "Ready",
  listening: "Listening…",
  thinking: "Thinking…",
  speaking: "Speaking…",
};

export function VoicePanel() {
  const [deviceId, setDeviceId] = useState("web-agent-001");
  const [wsBase, setWsBase] = useState("");
  const [textLine, setTextLine] = useState("");

  const {
    connected,
    status,
    level,
    lines,
    lastStats,
    connect,
    disconnect,
    startListen,
    stopListen,
    abort,
    sendText,
  } = useXiaozhiWs(deviceId, wsBase);

  const pttDown = useCallback(() => {
    if (!connected || status === "speaking") return;
    void startListen();
  }, [connected, startListen, status]);

  const pttUp = useCallback(() => {
    if (status === "listening") stopListen();
  }, [status, stopListen]);

  useEffect(() => {
    const down = (e: KeyboardEvent) => {
      if (e.code !== "Space" || !connected || e.repeat) return;
      const tag = (e.target as HTMLElement)?.tagName;
      if (tag === "INPUT" || tag === "TEXTAREA") return;
      e.preventDefault();
      pttDown();
    };
    const up = (e: KeyboardEvent) => {
      if (e.code === "Space" && connected) {
        e.preventDefault();
        pttUp();
      }
    };
    window.addEventListener("keydown", down);
    window.addEventListener("keyup", up);
    return () => {
      window.removeEventListener("keydown", down);
      window.removeEventListener("keyup", up);
    };
  }, [connected, pttDown, pttUp]);

  return (
    <div className="panel voice-panel">
      <div className="voice-hero">
        <div className={`orb orb-${status}`} aria-hidden>
          <div className="orb-ring" style={{ transform: `scale(${1 + level * 0.35})` }} />
        </div>
        <p className="voice-status-label">{STATUS_LABEL[status] ?? status}</p>
        <p className="muted voice-hint">
          Hold <kbd>Space</kbd> or the button below. Uplink PCM → SenseVoice STT → LLM → OmniVoice → Opus
          playback.
        </p>
      </div>

      <div className="card">
        <div className="row">
          <div style={{ flex: "2 1 200px" }}>
            <label htmlFor="ws">WebSocket override (empty = same host)</label>
            <input
              id="ws"
              type="url"
              placeholder="wss://…/xiaozhi/v1"
              value={wsBase}
              onChange={(e) => setWsBase(e.target.value)}
              disabled={connected}
            />
          </div>
          <div>
            <label htmlFor="dev">device-id</label>
            <input
              id="dev"
              type="text"
              value={deviceId}
              onChange={(e) => setDeviceId(e.target.value)}
              disabled={connected}
            />
          </div>
        </div>
        <div className="toolbar">
          {!connected ? (
            <button type="button" className="btn btn-primary" onClick={() => void connect()}>
              Connect
            </button>
          ) : (
            <button type="button" className="btn btn-ghost" onClick={disconnect}>
              Disconnect
            </button>
          )}
          <button type="button" className="btn btn-danger" onClick={abort} disabled={!connected}>
            Abort
          </button>
          <span className="status-pill">{connected ? "● online" : "○ offline"}</span>
        </div>
      </div>

      <button
        type="button"
        className={`btn ptt ${status === "listening" ? "listening" : ""}`}
        disabled={!connected || status === "speaking"}
        onPointerDown={(e) => {
          e.preventDefault();
          pttDown();
        }}
        onPointerUp={(e) => {
          e.preventDefault();
          pttUp();
        }}
        onPointerLeave={() => status === "listening" && pttUp()}
      >
        {status === "listening" ? "Release to send" : "Hold to talk"}
      </button>

      <div className="meter">
        <i style={{ width: `${Math.round(level * 100)}%` }} />
      </div>

      <div className="card">
        <label htmlFor="vtx">Quick text (listen detect)</label>
        <div className="row composer-row">
          <input
            id="vtx"
            type="text"
            value={textLine}
            onChange={(e) => setTextLine(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter") {
                sendText(textLine);
                setTextLine("");
              }
            }}
            placeholder="Type without mic"
            disabled={!connected}
          />
          <button
            type="button"
            className="btn btn-primary"
            style={{ flex: "0 0 auto" }}
            disabled={!connected}
            onClick={() => {
              sendText(textLine);
              setTextLine("");
            }}
          >
            Send
          </button>
        </div>
      </div>

      <div className="card">
        <label>Live transcript</label>
        <PipelineStatsBar stats={lastStats} />
        <div className="transcript">
          {lines.map((l) => (
            <div key={l.id} className={`tx-line tx-${l.role}`}>
              <span className="tx-ts">[{l.ts}]</span> {l.text}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
