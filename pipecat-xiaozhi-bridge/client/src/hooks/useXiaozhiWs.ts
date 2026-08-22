import { useCallback, useEffect, useRef, useState } from "react";
import { BrowserAudioSession } from "../audio/audioSession";
import { getSharedAudioSession } from "../audio/sharedAudioSession";
import { executeBuiltinTool } from "../browserTools";
import type { AgentStep, ListenMode, PipelineStats, SenseVoiceMeta, StoredMessage, VoiceStatus } from "../types";

export interface UseXiaozhiWsOptions {
  deviceId?: string;
  wsBase?: string;
  conversationId?: string | null;
  listenMode?: ListenMode;
  wakeWord?: string;
  autoListen?: boolean;
  autoConnect?: boolean;
  onConversationId?: (id: string) => void;
  onTurnComplete?: () => void;
  onTurnStart?: () => void;
  onUserTranscript?: (text: string, sensevoice?: SenseVoiceMeta) => void;
  onAssistantSegment?: (text: string) => void;
  onAgentStep?: (step: AgentStep) => void;
  onGeneratedImage?: (message: StoredMessage) => void;
}

function buildWsUrl(base: string, deviceId: string, conversationId?: string | null): string {
  const trimmed = base.trim();
  let url: URL;
  if (!trimmed) {
    url = new URL("/xiaozhi/v1", window.location.href);
    url.protocol = url.protocol === "https:" ? "wss:" : "ws:";
  } else {
    url = new URL(trimmed);
    const path = url.pathname.replace(/\/$/, "");
    if (!path.endsWith("/xiaozhi/v1")) {
      url.pathname = (path === "" ? "" : path) + "/xiaozhi/v1";
    }
  }
  url.searchParams.set("device-id", deviceId);
  if (conversationId) url.searchParams.set("conversation_id", conversationId);
  return url.toString();
}

export function useXiaozhiWs(opts: UseXiaozhiWsOptions = {}) {
  const deviceId = opts.deviceId ?? "web-agent-001";
  const wsBase = opts.wsBase ?? "";
  const conversationId = opts.conversationId ?? null;
  const autoConnect = opts.autoConnect ?? true;

  const [connected, setConnected] = useState(false);
  const [status, setStatus] = useState<VoiceStatus>("idle");
  const [level, setLevel] = useState(0);
  const [lastStats, setLastStats] = useState<PipelineStats | null>(null);
  const [lastError, setLastError] = useState<string | null>(null);
  const [listening, setListening] = useState(false);

  const wsRef = useRef<WebSocket | null>(null);
  const connectGenRef = useRef(0);
  const intentionalCloseRef = useRef(false);
  const conversationIdRef = useRef(conversationId);
  const serverConversationIdRef = useRef<string | null>(null);
  const receivingOpusRef = useRef(false);
  const optsRef = useRef(opts);
  const listenModeRef = useRef<ListenMode>(opts.listenMode ?? "manual");
  const sessionIdRef = useRef<string>("");
  const udpPortRef = useRef<number | null>(null);
  const beginMicRef = useRef<() => Promise<void>>(async () => undefined);
  const maybeResumeAutoListenRef = useRef<() => void>(() => undefined);
  optsRef.current = opts;
  listenModeRef.current = opts.listenMode ?? "manual";
  conversationIdRef.current = conversationId;

  const detachWs = (ws: WebSocket) => {
    ws.onopen = null;
    ws.onerror = null;
    ws.onclose = null;
    ws.onmessage = null;
  };

  const getAudioSession = useCallback((): BrowserAudioSession => {
    return getSharedAudioSession();
  }, []);

  const sendListenStart = useCallback((mode: ListenMode) => {
    const ws = wsRef.current;
    if (!ws || ws.readyState !== WebSocket.OPEN) return;
    ws.send(JSON.stringify({ type: "listen", state: "start", mode }));
  }, []);

  const sendListenStop = useCallback(() => {
    const ws = wsRef.current;
    if (!ws || ws.readyState !== WebSocket.OPEN) return;
    ws.send(JSON.stringify({ type: "listen", state: "stop" }));
  }, []);

  const stopMicInternal = useCallback(() => {
    getSharedAudioSession().stopMic();
    setListening(false);
    setLevel(0);
  }, []);

  const beginMic = useCallback(async () => {
    const gen = connectGenRef.current;
    const ws = wsRef.current;
    if (!ws || ws.readyState !== WebSocket.OPEN || listening) return;
    const audio = getAudioSession();
    const mode = listenModeRef.current;
    audio.setRealtimeAec(mode === "realtime");
    try {
      await audio.ensure();
    } catch (e) {
      if (gen !== connectGenRef.current) return;
      setLastError(`Audio: ${e instanceof Error ? e.message : String(e)}`);
      return;
    }
    if (gen !== connectGenRef.current) return;
    sendListenStart(mode);
    setStatus("listening");
    setListening(true);
    setLastError(null);

    const sendPcm = (pcm: Int16Array) => {
      if (gen !== connectGenRef.current) return;
      if (ws.readyState !== WebSocket.OPEN) return;
      ws.send(pcm.buffer.slice(pcm.byteOffset, pcm.byteOffset + pcm.byteLength));
    };

    try {
      await audio.startMic(sendPcm, (rms) => {
        if (gen === connectGenRef.current) setLevel(rms);
      });
      if (gen !== connectGenRef.current) {
        stopMicInternal();
      }
    } catch (e) {
      if (gen !== connectGenRef.current) return;
      setLastError(`Mic: ${e instanceof Error ? e.message : String(e)}`);
      sendListenStop();
      stopMicInternal();
      setStatus("idle");
    }
  }, [getAudioSession, listening, sendListenStart, sendListenStop, stopMicInternal]);

  const maybeResumeAutoListen = useCallback(() => {
    const mode = listenModeRef.current;
    if (mode === "manual") return;
    if (!optsRef.current.autoListen && mode === "auto") return;
    void beginMicRef.current();
  }, []);

  beginMicRef.current = beginMic;
  maybeResumeAutoListenRef.current = maybeResumeAutoListen;

  const disconnect = useCallback((_disposeAudio = true) => {
    stopMicInternal();
    receivingOpusRef.current = false;
    const ws = wsRef.current;
    if (ws) {
      intentionalCloseRef.current = true;
      detachWs(ws);
      try {
        ws.close();
      } catch {
        /* ignore */
      }
    }
    wsRef.current = null;
    setConnected(false);
    setStatus("idle");
  }, [stopMicInternal]);

  const connect = useCallback(() => {
    connectGenRef.current += 1;
    const gen = connectGenRef.current;
    if (wsRef.current) disconnect(false);
    intentionalCloseRef.current = false;
    const url = buildWsUrl(wsBase, deviceId, conversationIdRef.current);
    const ws = new WebSocket(url);
    ws.binaryType = "arraybuffer";
    wsRef.current = ws;

    ws.onopen = () => {
      if (gen !== connectGenRef.current) return;
      intentionalCloseRef.current = false;
      setConnected(true);
      setLastError(null);
      const mode = listenModeRef.current;
      ws.send(
        JSON.stringify({
          type: "hello",
          version: 1,
          transport: "websocket",
          features: { mcp: false, aec: mode === "realtime" },
          audio_params: {
            format: "opus",
            sample_rate: 16000,
            channels: 1,
            frame_duration: 60,
            uplink_encoding: "pcm_s16le",
          },
        })
      );
    };

    ws.onerror = () => {
      if (gen !== connectGenRef.current || intentionalCloseRef.current) return;
      setLastError("WebSocket error");
    };
    ws.onclose = (ev) => {
      if (gen !== connectGenRef.current) return;
      stopMicInternal();
      setConnected(false);
      setStatus("idle");
      if (!intentionalCloseRef.current && ev.code !== 1000) {
        setLastError(ev.reason?.trim() || `Connection closed (${ev.code})`);
      }
      intentionalCloseRef.current = false;
    };

    ws.onmessage = async (ev) => {
      if (gen !== connectGenRef.current) return;
      if (typeof ev.data === "string") {
        let msg: Record<string, unknown>;
        try {
          msg = JSON.parse(ev.data) as Record<string, unknown>;
        } catch {
          return;
        }
        const typ = msg.type as string;
        if (typ === "hello") {
          sessionIdRef.current = String(msg.session_id ?? "");
          const udp = msg.udp_audio as { port?: number } | undefined;
          udpPortRef.current = udp?.port ?? null;
          return;
        }
        if (typ === "conversation") {
          const cid = String(msg.conversation_id ?? "");
          if (cid) {
            serverConversationIdRef.current = cid;
            optsRef.current.onConversationId?.(cid);
          }
          return;
        }
        if (typ === "error") {
          setLastError(String(msg.message ?? "pipeline failed"));
          setStatus("idle");
          stopMicInternal();
          return;
        }
        if (typ === "stt") {
          const userText = String(msg.text ?? "").trim();
          const sv = msg.sensevoice as SenseVoiceMeta | undefined;
          if (userText) {
            optsRef.current.onUserTranscript?.(userText, sv);
          }
          optsRef.current.onTurnStart?.();
          stopMicInternal();
          setStatus("thinking");
          return;
        }
        if (typ === "builtin_tool") {
          const requestId = String(msg.request_id ?? "");
          const tool = String(msg.tool ?? "");
          const args = (msg.arguments as Record<string, unknown>) ?? {};
          const wsOut = wsRef.current;
          if (requestId && wsOut?.readyState === WebSocket.OPEN) {
            void (async () => {
              try {
                const result = await executeBuiltinTool(tool, args);
                wsOut.send(JSON.stringify({ type: "builtin_tool_result", request_id: requestId, result }));
              } catch (e) {
                wsOut.send(
                  JSON.stringify({
                    type: "builtin_tool_result",
                    request_id: requestId,
                    error: e instanceof Error ? e.message : String(e),
                  })
                );
              }
            })();
          }
          return;
        }
        if (typ === "agent") {
          const step: AgentStep = {
            phase: String(msg.phase ?? "running") as AgentStep["phase"],
            round_index: Number(msg.round ?? 0),
            step_index: Number(msg.step ?? 0),
            tool_name: String(msg.tool ?? ""),
            label: String(msg.label ?? ""),
            message: String(msg.message ?? ""),
            detail: String(msg.detail ?? ""),
            image_url: String(msg.image_url ?? ""),
          };
          optsRef.current.onAgentStep?.(step);
          if (step.phase === "running") {
            setStatus("thinking");
          }
          return;
        }
        if (typ === "generated_image") {
          const raw = msg.message as StoredMessage | undefined;
          if (raw?.id) {
            optsRef.current.onGeneratedImage?.(raw);
          }
          return;
        }
        if (typ === "stats") {
          const stats = msg.stats as PipelineStats | undefined;
          if (stats) setLastStats(stats);
          optsRef.current.onTurnComplete?.();
          return;
        }
        if (typ === "tts") {
          const st = msg.state as string;
          if (st === "start") {
            stopMicInternal();
            setStatus("thinking");
            receivingOpusRef.current = false;
            return;
          }
          if (st === "sentence_start") {
            const segText = String(msg.text ?? "").trim();
            if (segText) optsRef.current.onAssistantSegment?.(segText);
            setStatus("speaking");
            receivingOpusRef.current = true;
            try {
              await getAudioSession().ensure();
              getAudioSession().alignPlaybackSchedule();
            } catch {
              /* playback will retry on next packet */
            }
            return;
          }
          if (st === "stop") {
            receivingOpusRef.current = false;
            setStatus(listening ? "listening" : "idle");
            maybeResumeAutoListenRef.current();
          }
          return;
        }
        if (typ === "listen" && msg.state === "start") {
          if (!listening) void beginMicRef.current();
          return;
        }
        return;
      }
      if (ev.data instanceof ArrayBuffer && receivingOpusRef.current) {
        try {
          getAudioSession().enqueueOpus(ev.data);
        } catch {
          /* ignore decode errors */
        }
      }
    };
  }, [deviceId, disconnect, getAudioSession, stopMicInternal, wsBase]);

  useEffect(() => {
    if (!autoConnect) return;
    let cancelled = false;
    const timer = window.setTimeout(() => {
      if (!cancelled) connect();
    }, 50);
    return () => {
      cancelled = true;
      window.clearTimeout(timer);
      disconnect(true);
    };
  }, [autoConnect, deviceId, wsBase, connect, disconnect]);

  // Application-level keepalive so proxies don't drop WS during long LLM/TTS turns.
  useEffect(() => {
    if (!connected) return;
    const sendPing = () => {
      const ws = wsRef.current;
      if (!ws || ws.readyState !== WebSocket.OPEN) return;
      ws.send(JSON.stringify({ type: "ping", timestamp: Date.now() }));
    };
    sendPing();
    const interval = window.setInterval(sendPing, 15000);
    return () => window.clearInterval(interval);
  }, [connected]);

  // Reconnect when user switches conversation — not when server syncs the same thread id.
  useEffect(() => {
    if (!autoConnect || !connected) return;
    const desired = conversationIdRef.current;
    const bound = serverConversationIdRef.current;
    if (!desired || !bound || desired === bound) return;
    const timer = window.setTimeout(() => connect(), 100);
    return () => window.clearTimeout(timer);
  }, [autoConnect, connected, connect, conversationId]);

  useEffect(() => {
    listenModeRef.current = opts.listenMode ?? "manual";
    getSharedAudioSession().setRealtimeAec(listenModeRef.current === "realtime");
  }, [opts.listenMode]);

  useEffect(() => {
    if (!connected || !opts.autoListen) return;
    if (listenModeRef.current === "manual") return;
    if (status !== "idle") return;
    void beginMic();
  }, [beginMic, connected, opts.autoListen, status]);

  const startListen = useCallback(async () => {
    listenModeRef.current = "manual";
    await beginMic();
  }, [beginMic]);

  const stopListen = useCallback(() => {
    sendListenStop();
    stopMicInternal();
    setStatus("idle");
  }, [sendListenStop, stopMicInternal]);

  const abort = useCallback(() => {
    wsRef.current?.send(JSON.stringify({ type: "abort" }));
    stopMicInternal();
    receivingOpusRef.current = false;
    setStatus("idle");
  }, [stopMicInternal]);

  const sendWakeWord = useCallback(
    (text?: string) => {
      const ws = wsRef.current;
      if (!ws || ws.readyState !== WebSocket.OPEN) return;
      const phrase = (text ?? optsRef.current.wakeWord ?? "").trim();
      ws.send(
        JSON.stringify({
          type: "abort",
          reason: "wake_word_detected",
        })
      );
      ws.send(
        JSON.stringify({
          type: "listen",
          state: "detect",
          text: phrase || "wake",
        })
      );
      setStatus("thinking");
      setLastError(null);
    },
    []
  );

  const sendText = useCallback((text: string) => {
    const t = text.trim();
    if (!t) return;
    const ws = wsRef.current;
    if (!ws || ws.readyState !== WebSocket.OPEN) return;
    stopMicInternal();
    ws.send(JSON.stringify({ type: "listen", state: "detect", text: t }));
    setStatus("thinking");
    setLastError(null);
  }, [stopMicInternal]);

  return {
    connected,
    status,
    level,
    listening,
    lastStats,
    lastError,
    sessionId: sessionIdRef.current,
    udpPort: udpPortRef.current,
    connect,
    disconnect,
    startListen,
    stopListen,
    abort,
    sendText,
    sendWakeWord,
    beginAutoListen: beginMic,
    stopMicInternal,
  };
}
