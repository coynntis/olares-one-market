import { useCallback, useEffect, useRef, useState } from "react";
import {
  clearConversation,
  createConversation,
  fetchActiveConversation,
  fetchConfigStatus,
  getDeviceId,
  postChat,
  setStoredConversationId,
  getStoredConversationId,
  storedToChatMessage,
} from "../api";
import { unlockSharedAudio } from "../audio/sharedAudioSession";
import { playWavBase64 } from "../audio/wavPlayback";
import { useVoiceStatus } from "../hooks/voiceStatusContext";
import { useXiaozhiWs } from "../hooks/useXiaozhiWs";
import { AgentStepRail } from "./AgentStepRail";
import { ConversationDrawer } from "./ConversationDrawer";
import { ImageAttachMenu } from "./ImageAttachMenu";
import { IconMic } from "./Icons";
import { VoiceActivityBar } from "./VoiceActivityBar";
import { PipelineStatsBar } from "./PipelineStatsBar";
import { SenseVoiceSttMeta } from "./SenseVoiceSttMeta";
import type { AgentStep, ChatMessage, ListenMode, PipelineStats, StoredMessage } from "../types";

const STATUS_LABEL: Record<string, string> = {
  idle: "",
  listening: "Listening…",
  thinking: "Thinking…",
  speaking: "Speaking…",
};

export function ChatPanel() {
  const { setVoiceStatus } = useVoiceStatus();
  const deviceId = getDeviceId();
  const [conversationId, setConversationId] = useState<string | null>(getStoredConversationId());
  const [historyOpen, setHistoryOpen] = useState(false);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [draft, setDraft] = useState("");
  const [imagePreview, setImagePreview] = useState<string | null>(null);
  const [imageData, setImageData] = useState<string | null>(null);
  const [speakReplies, setSpeakReplies] = useState(true);
  const [listenMode, setListenMode] = useState<ListenMode>("auto");
  const [autoListen, setAutoListen] = useState(false);
  const [wakeWord, setWakeWord] = useState("Hi Xiaozhi");
  const [busy, setBusy] = useState(false);
  const [ready, setReady] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [lastStats, setLastStats] = useState<PipelineStats | null>(null);
  const [loadingHistory, setLoadingHistory] = useState(true);
  const [streamingAssistantId, setStreamingAssistantId] = useState<string | null>(null);
  const [agentSteps, setAgentSteps] = useState<AgentStep[]>([]);
  const streamingAssistantIdRef = useRef<string | null>(null);
  const bottomRef = useRef<HTMLDivElement>(null);

  const reloadHistory = useCallback(async (cid?: string | null) => {
    try {
      const data = await fetchActiveConversation(deviceId, cid ?? conversationId);
      setConversationId(data.conversation.id);
      setStoredConversationId(data.conversation.id);
      setMessages(data.messages.map(storedToChatMessage).filter((m): m is ChatMessage => m !== null));
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }, [conversationId, deviceId]);

  const {
    connected,
    status: voiceStatus,
    level,
    lastStats: voiceStats,
    lastError: voiceError,
    startListen,
    stopListen,
    abort,
    sendText: sendVoiceText,
    sendWakeWord,
  } = useXiaozhiWs({
    deviceId,
    conversationId,
    listenMode,
    autoListen,
    wakeWord,
    autoConnect: true,
    onConversationId: (id) => {
      setConversationId(id);
      setStoredConversationId(id);
    },
    onTurnStart: () => {
      streamingAssistantIdRef.current = null;
      setStreamingAssistantId(null);
      setAgentSteps([]);
    },
    onUserTranscript: (text, sensevoice) => {
      setMessages((prev) => {
        const last = prev[prev.length - 1];
        if (last?.role === "user" && last.source === "voice" && last.text === text) {
          return prev;
        }
        return [
          ...prev,
          {
            id: `stt-${Date.now()}`,
            role: "user",
            text,
            source: "voice",
            sensevoice,
            ts: Date.now(),
          },
        ];
      });
    },
    onTurnComplete: () => {
      streamingAssistantIdRef.current = null;
      setStreamingAssistantId(null);
      setAgentSteps([]);
      void reloadHistory();
    },
    onAgentStep: (step) => {
      setAgentSteps((prev) => [...prev, step]);
    },
    onGeneratedImage: (stored) => {
      const chatMsg = storedToChatMessage(stored);
      if (!chatMsg) return;
      setMessages((prev) => {
        if (prev.some((m) => m.id === chatMsg.id)) return prev;
        return [...prev, chatMsg];
      });
    },
    onAssistantSegment: (segText) => {
      setMessages((prev) => {
        const sid = streamingAssistantIdRef.current;
        if (sid) {
          return prev.map((m) =>
            m.id === sid ? { ...m, text: m.text ? `${m.text} ${segText}` : segText } : m
          );
        }
        const id = `stream-${Date.now()}`;
        streamingAssistantIdRef.current = id;
        setStreamingAssistantId(id);
        return [...prev, { id, role: "assistant", text: segText, source: "voice", ts: Date.now() }];
      });
    },
  });

  useEffect(() => {
    void fetchConfigStatus()
      .then((s) => setReady(s.ready))
      .catch(() => setReady(false));
  }, []);

  useEffect(() => {
    setLoadingHistory(true);
    void reloadHistory(getStoredConversationId()).finally(() => setLoadingHistory(false));
  }, [deviceId]);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, busy, voiceStatus]);

  useEffect(() => {
    if (voiceStats) setLastStats(voiceStats);
  }, [voiceStats]);

  useEffect(() => {
    if (voiceError) setError(voiceError);
  }, [voiceError]);

  useEffect(() => {
    setVoiceStatus({ status: voiceStatus, level, connected });
  }, [voiceStatus, level, connected, setVoiceStatus]);

  const clearImage = () => {
    setImagePreview(null);
    setImageData(null);
  };

  const onImageAttached = (dataUrl: string) => {
    setImagePreview(dataUrl);
    setImageData(dataUrl);
    setError(null);
  };

  const sendText = useCallback(async () => {
    const text = draft.trim();
    if (!text && !imageData) return;
    if (busy || voiceStatus === "thinking" || voiceStatus === "speaking") return;

    setDraft("");
    clearImage();
    setBusy(true);
    setError(null);
    setLastStats(null);
    setAgentSteps([]);

    try {
      await unlockSharedAudio();
      const res = await postChat({
        text,
        image: imageData ?? undefined,
        speak: speakReplies,
        conversation_id: conversationId,
        device_id: deviceId,
      });
      setConversationId(res.conversation_id);
      setStoredConversationId(res.conversation_id);
      const userMsg = storedToChatMessage(res.user_message);
      const assistantMsg = storedToChatMessage(res.assistant_message);
      const genMsgs = (res.generated_images ?? [])
        .map(storedToChatMessage)
        .filter((x): x is ChatMessage => x !== null);
      const freshIds = new Set(
        [userMsg?.id, assistantMsg?.id, ...genMsgs.map((m) => m.id)].filter(Boolean) as string[]
      );
      setMessages((m) =>
        [
          ...m.filter((x) => !freshIds.has(x.id)),
          ...(userMsg ? [userMsg] : []),
          ...genMsgs,
          ...(assistantMsg ? [assistantMsg] : []),
        ]
      );
      if (res.audio_wav_b64) {
        try {
          await playWavBase64(res.audio_wav_b64);
        } catch (e) {
          setError(`TTS playback: ${e instanceof Error ? e.message : String(e)}`);
        }
      }
      if (res.tts_error) setError(`TTS: ${res.tts_error}`);
      if (res.vision_warning) setError(res.vision_warning);
      if (res.stats) setLastStats(res.stats);
      if (res.agent_steps?.length) setAgentSteps(res.agent_steps);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }, [busy, conversationId, deviceId, draft, imageData, speakReplies, voiceStatus]);

  const sendViaVoicePipeline = useCallback(() => {
    const text = draft.trim();
    if (!text) return;
    if (!connected) {
      setError("Voice WebSocket not connected");
      return;
    }
    sendVoiceText(text);
    setDraft("");
    clearImage();
  }, [connected, draft, sendVoiceText]);

  const startNewChat = async () => {
    try {
      const conv = await createConversation(deviceId);
      setConversationId(conv.id);
      setStoredConversationId(conv.id);
      setMessages([]);
      setLastStats(null);
      setError(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  };

  const switchConversation = (id: string, stored: StoredMessage[]) => {
    setConversationId(id);
    setStoredConversationId(id);
    setMessages(stored.map(storedToChatMessage).filter((m): m is ChatMessage => m !== null));
    setLastStats(null);
    setError(null);
  };

  const clearHistory = async () => {
    if (!conversationId) return;
    try {
      await clearConversation(deviceId, conversationId);
      setMessages([]);
      setLastStats(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  };

  const pttDown = () => {
    if (!connected || voiceStatus === "speaking" || busy) return;
    void unlockSharedAudio().then(() => startListen());
  };

  const pttUp = () => {
    if (voiceStatus === "listening") stopListen();
  };

  const voiceBusy = voiceStatus === "thinking" || voiceStatus === "speaking" || voiceStatus === "listening";
  const showAgentRail = agentSteps.length > 0;
  const showThinking = (busy || voiceStatus === "thinking") && !showAgentRail;

  return (
    <div className="panel chat-panel hud-panel">
      {!ready && (
        <div className="banner banner-warn">
          Pipeline not ready. Open <strong>Systems</strong> and set STT, TTS, and LLM URLs plus model.
        </div>
      )}

      <div className="chat-toolbar">
        <VoiceActivityBar status={voiceStatus} level={level} />
        <span className={`status-pill ${connected ? "online" : "offline"}`}>
          {connected ? "ONLINE" : "OFFLINE"}
        </span>
        {STATUS_LABEL[voiceStatus] && (
          <span className={`voice-inline-status voice-inline-${voiceStatus}`}>
            {STATUS_LABEL[voiceStatus]}
          </span>
        )}
        <div className="chat-toolbar-actions">
          <button type="button" className="btn btn-ghost btn-sm" onClick={() => setHistoryOpen(true)}>
            History
          </button>
          <button type="button" className="btn btn-ghost btn-sm" onClick={() => void startNewChat()}>
            New chat
          </button>
          <button
            type="button"
            className="btn btn-ghost btn-sm"
            disabled={!conversationId || messages.length === 0}
            onClick={() => void clearHistory()}
          >
            Clear
          </button>
          <button type="button" className="btn btn-ghost btn-sm" onClick={abort} disabled={!connected}>
            Abort
          </button>
        </div>
      </div>

      <div className="chat-thread" role="log" aria-live="polite">
        {loadingHistory && (
          <div className="chat-empty">
            <p className="muted">Loading history…</p>
          </div>
        )}
        {!loadingHistory && messages.length === 0 && (
          <div className="chat-empty">
            <p className="chat-empty-title">Agent R online</p>
            <p className="muted">
              Text, voice, and vision in one thread. Hold the mic button for manual mode,
              or use auto VAD. Camera button: take a photo or attach an image.
            </p>
          </div>
        )}
        {messages.map((m) => (
          <div key={m.id} className={`bubble-row ${m.role}`}>
            <div className={`bubble ${m.role}`}>
              {m.imagePreview && (
                <div className="bubble-image-wrap">
                  <a href={m.imagePreview} target="_blank" rel="noreferrer">
                    <img src={m.imagePreview} alt={m.text || "generated"} className="bubble-image" />
                  </a>
                  <div className="bubble-image-actions">
                    <a
                      className="bubble-image-link"
                      href={m.imagePreview}
                      target="_blank"
                      rel="noreferrer"
                    >
                      Open
                    </a>
                    <a
                      className="bubble-image-link"
                      href={m.imagePreview}
                      download={`image-${m.id}.png`}
                    >
                      Download
                    </a>
                  </div>
                </div>
              )}
              {m.role === "user" && m.sensevoice && <SenseVoiceSttMeta meta={m.sensevoice} />}
              <p>{m.text}</p>
              {m.source === "voice" && <span className="msg-source">voice</span>}
            </div>
          </div>
        ))}
        <AgentStepRail steps={agentSteps} />
        {showThinking && (
          <div className="bubble-row assistant">
            <div className="bubble assistant thinking">
              <span className="dots" aria-hidden />
              Thinking…
            </div>
          </div>
        )}
        <div ref={bottomRef} />
      </div>

      {error && <div className="banner banner-error">{error}</div>}
      <PipelineStatsBar stats={lastStats} />

      {imagePreview && (
        <div className="attach-preview">
          <img src={imagePreview} alt="attachment" />
          <button type="button" className="btn btn-ghost btn-sm" onClick={clearImage}>
            Remove
          </button>
        </div>
      )}

      <div className="voice-controls card">
        <div className="row">
          <div>
            <label htmlFor="listen-mode">Listen mode</label>
            <select
              id="listen-mode"
              value={listenMode}
              onChange={(e) => setListenMode(e.target.value as ListenMode)}
            >
              <option value="manual">Manual (hold mic)</option>
              <option value="auto">Auto (VAD, stop on silence)</option>
              <option value="realtime">Realtime (AEC, talk while playing)</option>
            </select>
          </div>
          <div>
            <label htmlFor="wake-word">Wake phrase</label>
            <input
              id="wake-word"
              value={wakeWord}
              onChange={(e) => setWakeWord(e.target.value)}
              placeholder="Hi Xiaozhi"
            />
          </div>
        </div>
        <div className="toolbar">
          <label className="composer-opt">
            <input
              type="checkbox"
              checked={autoListen}
              onChange={(e) => setAutoListen(e.target.checked)}
            />
            Auto-start mic (auto / realtime)
          </label>
          <button
            type="button"
            className="btn btn-ghost btn-sm"
            disabled={!connected}
            onClick={() => sendWakeWord()}
          >
            Wake word
          </button>
        </div>
      </div>

      {voiceStatus === "listening" && (
        <div className="meter" aria-label="Microphone level">
          <i style={{ width: `${Math.round(level * 100)}%` }} />
        </div>
      )}

      <div className="composer composer-unified">
        <ImageAttachMenu
          disabled={busy || voiceBusy}
          onImage={onImageAttached}
          onError={setError}
        />
        <button
          type="button"
          className={`btn btn-ghost icon-btn ptt-inline ${voiceStatus === "listening" ? "listening" : ""}`}
          title="Hold to talk"
          aria-label="Microphone — hold to talk"
          disabled={!connected || voiceStatus === "speaking" || busy}
          onPointerDown={(e) => {
            e.preventDefault();
            pttDown();
          }}
          onPointerUp={(e) => {
            e.preventDefault();
            pttUp();
          }}
          onPointerLeave={() => voiceStatus === "listening" && pttUp()}
        >
          <IconMic />
        </button>
        <textarea
          className="composer-input"
          rows={1}
          placeholder="Message… Enter send · hold mic for voice"
          value={draft}
          disabled={busy || voiceBusy}
          onChange={(e) => setDraft(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault();
              if (imageData) void sendText();
              else if (e.altKey) sendViaVoicePipeline();
              else void sendText();
            }
          }}
        />
        <button
          type="button"
          className="btn btn-primary"
          disabled={busy || voiceBusy || (!draft.trim() && !imageData)}
          onClick={() => void sendText()}
        >
          Send
        </button>
      </div>

      <label className="composer-opt">
        <input
          type="checkbox"
          checked={speakReplies}
          onChange={(e) => setSpeakReplies(e.target.checked)}
        />
        Speak text replies (TTS)
      </label>

      <ConversationDrawer
        deviceId={deviceId}
        currentId={conversationId}
        open={historyOpen}
        onClose={() => setHistoryOpen(false)}
        onSwitch={switchConversation}
      />
    </div>
  );
}
