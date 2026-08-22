import type {
  AgentStep,
  BridgeSettings,
  ChatMessage,
  ConfigStatus,
  Conversation,
  LlmCapabilities,
  McpServer,
  McpSuggestion,
  McpTestResult,
  PipelineStats,
  StoredMessage,
  TestConnectionResult,
  TestService,
  VoiceProfile,
  LlmProfile,
} from "./types";
import { normalizeStoredSttText } from "./sensevoiceStt";

const CONV_STORAGE_KEY = "xiaozhi-active-conversation";
const DEVICE_STORAGE_KEY = "xiaozhi-device-id";
export const DEFAULT_DEVICE_ID = "web-agent-001";

export function getDeviceId(): string {
  return localStorage.getItem(DEVICE_STORAGE_KEY) || DEFAULT_DEVICE_ID;
}

export function setDeviceId(id: string): void {
  localStorage.setItem(DEVICE_STORAGE_KEY, id.trim() || DEFAULT_DEVICE_ID);
}

export function getStoredConversationId(): string | null {
  return localStorage.getItem(CONV_STORAGE_KEY);
}

export function setStoredConversationId(id: string): void {
  localStorage.setItem(CONV_STORAGE_KEY, id);
}

function deviceQuery(deviceId: string): string {
  return `device_id=${encodeURIComponent(deviceId)}`;
}

export async function fetchConfig(): Promise<BridgeSettings> {
  const r = await fetch("/api/config");
  if (!r.ok) throw new Error(`config GET ${r.status}`);
  return r.json() as Promise<BridgeSettings>;
}

export async function saveConfig(
  patch: Record<string, unknown>,
  token?: string
): Promise<BridgeSettings> {
  const headers: Record<string, string> = { "Content-Type": "application/json" };
  if (token?.trim()) headers.Authorization = `Bearer ${token.trim()}`;
  const r = await fetch("/api/config", {
    method: "PUT",
    headers,
    body: JSON.stringify(patch),
  });
  if (!r.ok) {
    const err = await r.json().catch(() => ({}));
    throw new Error((err as { error?: string }).error || `config PUT ${r.status}`);
  }
  const data = (await r.json()) as { settings: BridgeSettings };
  return data.settings;
}

export async function fetchConfigStatus(): Promise<ConfigStatus> {
  const r = await fetch("/api/config/status");
  if (!r.ok) throw new Error(`status ${r.status}`);
  return r.json() as Promise<ConfigStatus>;
}

export async function testConnection(service: TestService): Promise<TestConnectionResult> {
  const r = await fetch("/api/config/test", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ service }),
  });
  const data = (await r.json().catch(() => ({}))) as TestConnectionResult;
  if (!r.ok) {
    throw new Error(data.error || `test ${r.status}`);
  }
  return data;
}

export async function fetchLlmCapabilities(): Promise<LlmCapabilities> {
  const r = await fetch("/api/config/llm-capabilities");
  const data = (await r.json().catch(() => ({}))) as LlmCapabilities;
  if (!r.ok) throw new Error(data.error || `capabilities ${r.status}`);
  return data;
}

export async function fetchConversations(deviceId: string): Promise<{
  conversations: Conversation[];
  current_conversation_id: string | null;
}> {
  const r = await fetch(`/api/conversations?${deviceQuery(deviceId)}`);
  const data = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error((data as { error?: string }).error || `list conv ${r.status}`);
  return data as { conversations: Conversation[]; current_conversation_id: string | null };
}

export async function fetchActiveConversation(
  deviceId: string,
  conversationId?: string | null
): Promise<{ conversation: Conversation; messages: StoredMessage[]; current_conversation_id: string | null }> {
  const parts = [deviceQuery(deviceId)];
  if (conversationId) parts.push(`id=${encodeURIComponent(conversationId)}`);
  const r = await fetch(`/api/conversations/active?${parts.join("&")}`);
  const data = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error((data as { error?: string }).error || `active conv ${r.status}`);
  return data as {
    conversation: Conversation;
    messages: StoredMessage[];
    current_conversation_id: string | null;
  };
}

export async function activateConversation(
  deviceId: string,
  conversationId: string
): Promise<{ conversation: Conversation; messages: StoredMessage[] }> {
  const r = await fetch(
    `/api/conversations/${encodeURIComponent(conversationId)}/activate?${deviceQuery(deviceId)}`,
    { method: "POST" }
  );
  const data = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error((data as { error?: string }).error || `activate ${r.status}`);
  return data as { conversation: Conversation; messages: StoredMessage[] };
}

export async function createConversation(
  deviceId: string,
  title = "Chat"
): Promise<Conversation> {
  const r = await fetch("/api/conversations", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ title, device_id: deviceId }),
  });
  const data = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error((data as { error?: string }).error || `create conv ${r.status}`);
  return data as Conversation;
}

export async function clearConversation(deviceId: string, conversationId: string): Promise<void> {
  const r = await fetch(
    `/api/conversations/${encodeURIComponent(conversationId)}/clear?${deviceQuery(deviceId)}`,
    { method: "POST" }
  );
  if (!r.ok) {
    const data = await r.json().catch(() => ({}));
    throw new Error((data as { error?: string }).error || `clear ${r.status}`);
  }
}

export function storedToChatMessage(m: StoredMessage): ChatMessage | null {
  if (m.role === "tool") return null;
  if (m.role === "assistant" && !m.text?.trim() && m.meta && "tool_calls" in m.meta) return null;
  const normalized = normalizeStoredSttText(m.text, m.meta);
  return {
    id: m.id,
    role: m.role,
    text: normalized.text,
    imagePreview: m.image_url,
    source: m.source === "voice" ? "voice" : "text",
    sensevoice: normalized.sensevoice,
    ts: m.created_at,
  };
}

export async function postChat(opts: {
  text: string;
  image?: string;
  speak?: boolean;
  conversation_id?: string | null;
  device_id: string;
}): Promise<{
  text: string;
  conversation_id: string;
  user_message: StoredMessage;
  assistant_message: StoredMessage;
  audio_wav_b64?: string;
  tts_error?: string;
  stats?: PipelineStats;
  agent_steps?: AgentStep[];
  vision_warning?: string;
  generated_images?: StoredMessage[];
}> {
  const r = await fetch("/api/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      text: opts.text,
      image: opts.image,
      speak: opts.speak ?? false,
      conversation_id: opts.conversation_id ?? undefined,
      device_id: opts.device_id,
    }),
  });
  const data = await r.json().catch(() => ({}));
  if (!r.ok) {
    throw new Error((data as { error?: string }).error || `chat ${r.status}`);
  }
  return data as {
    text: string;
    conversation_id: string;
    user_message: StoredMessage;
    assistant_message: StoredMessage;
    audio_wav_b64?: string;
    tts_error?: string;
    stats?: PipelineStats;
    agent_steps?: AgentStep[];
    vision_warning?: string;
    generated_images?: StoredMessage[];
  };
}

export async function fetchLlmProfiles(): Promise<{
  profiles: LlmProfile[];
  active_profile_id: string;
}> {
  const r = await fetch("/api/llm-profiles");
  const data = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error((data as { error?: string }).error || `llm profiles ${r.status}`);
  return data as { profiles: LlmProfile[]; active_profile_id: string };
}

export async function createLlmProfile(
  opts: {
    name: string;
    from_current?: boolean;
    llm_base_url?: string;
    llm_model?: string;
    system_prompt?: string;
    set_active?: boolean;
  },
  token?: string
): Promise<{ profile: LlmProfile; settings: BridgeSettings }> {
  const headers: Record<string, string> = { "Content-Type": "application/json" };
  if (token?.trim()) headers.Authorization = `Bearer ${token.trim()}`;
  const r = await fetch("/api/llm-profiles", {
    method: "POST",
    headers,
    body: JSON.stringify(opts),
  });
  const data = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error((data as { error?: string }).error || `create llm profile ${r.status}`);
  return data as { profile: LlmProfile; settings: BridgeSettings };
}

export async function activateLlmProfile(
  profileId: string,
  token?: string
): Promise<{ profile: LlmProfile; settings: BridgeSettings }> {
  const headers: Record<string, string> = {};
  if (token?.trim()) headers.Authorization = `Bearer ${token.trim()}`;
  const r = await fetch(`/api/llm-profiles/${encodeURIComponent(profileId)}/activate`, {
    method: "POST",
    headers,
  });
  const data = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error((data as { error?: string }).error || `activate llm profile ${r.status}`);
  return data as { profile: LlmProfile; settings: BridgeSettings };
}

export async function deleteLlmProfile(profileId: string, token?: string): Promise<BridgeSettings> {
  const headers: Record<string, string> = {};
  if (token?.trim()) headers.Authorization = `Bearer ${token.trim()}`;
  const r = await fetch(`/api/llm-profiles/${encodeURIComponent(profileId)}`, {
    method: "DELETE",
    headers,
  });
  const data = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error((data as { error?: string }).error || `delete llm profile ${r.status}`);
  return (data as { settings: BridgeSettings }).settings;
}

export async function fetchVoices(): Promise<{
  voices: VoiceProfile[];
  active_voice_id: string;
  voice_mode: string;
}> {
  const r = await fetch("/api/voices");
  const data = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error((data as { error?: string }).error || `voices ${r.status}`);
  return data as {
    voices: VoiceProfile[];
    active_voice_id: string;
    voice_mode: string;
  };
}

export async function createVoice(opts: {
  name: string;
  ref_text: string;
  audio: string;
  language_id?: string;
  instruct?: string;
  set_active?: boolean;
}): Promise<{ voice: VoiceProfile }> {
  const r = await fetch("/api/voices", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(opts),
  });
  const data = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error((data as { error?: string }).error || `create voice ${r.status}`);
  return data as { voice: VoiceProfile };
}

export async function deleteVoice(voiceId: string): Promise<void> {
  const r = await fetch(`/api/voices/${encodeURIComponent(voiceId)}`, { method: "DELETE" });
  const data = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error((data as { error?: string }).error || `delete voice ${r.status}`);
}

export async function activateVoice(voiceId: string): Promise<void> {
  const r = await fetch(`/api/voices/${encodeURIComponent(voiceId)}/activate`, { method: "POST" });
  const data = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error((data as { error?: string }).error || `activate voice ${r.status}`);
}

export async function previewVoice(
  voiceId: string,
  text?: string
): Promise<{ audio_wav_b64: string; text: string; tts_ms: number }> {
  const r = await fetch(`/api/voices/${encodeURIComponent(voiceId)}/preview`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(text ? { text } : {}),
  });
  const data = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error((data as { error?: string }).error || `preview voice ${r.status}`);
  return data as { audio_wav_b64: string; text: string; tts_ms: number };
}

export async function fetchMcpSuggestions(): Promise<McpSuggestion[]> {
  const r = await fetch("/api/mcp/suggestions");
  const data = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error((data as { error?: string }).error || `mcp suggestions ${r.status}`);
  return (data as { suggestions: McpSuggestion[] }).suggestions;
}

export async function fetchMcpServers(): Promise<McpServer[]> {
  const r = await fetch("/api/mcp/servers");
  const data = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error((data as { error?: string }).error || `mcp servers ${r.status}`);
  return (data as { servers: McpServer[] }).servers;
}

export async function saveMcpServers(
  servers: McpServer[],
  token?: string
): Promise<McpServer[]> {
  const headers: Record<string, string> = { "Content-Type": "application/json" };
  if (token?.trim()) headers.Authorization = `Bearer ${token.trim()}`;
  const r = await fetch("/api/mcp/servers", {
    method: "PUT",
    headers,
    body: JSON.stringify({ servers }),
  });
  const data = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error((data as { error?: string }).error || `mcp save ${r.status}`);
  return (data as { servers: McpServer[] }).servers;
}

export async function addMcpFromSuggestion(opts: {
  catalog_id: string;
  shared_base_url?: string;
  name?: string;
  token?: string;
}): Promise<McpServer[]> {
  const headers: Record<string, string> = { "Content-Type": "application/json" };
  if (opts.token?.trim()) headers.Authorization = `Bearer ${opts.token.trim()}`;
  const r = await fetch("/api/mcp/servers/from-suggestion", {
    method: "POST",
    headers,
    body: JSON.stringify({
      catalog_id: opts.catalog_id,
      shared_base_url: opts.shared_base_url,
      name: opts.name,
    }),
  });
  const data = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error((data as { error?: string }).error || `mcp add ${r.status}`);
  return (data as { servers: McpServer[] }).servers;
}

export async function testMcpServer(opts: {
  server_id?: string;
  server?: Partial<McpServer>;
  service?: "one" | "all";
}): Promise<McpTestResult | { ok: boolean; results: McpTestResult[]; count: number }> {
  const r = await fetch("/api/mcp/test", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(opts),
  });
  const data = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error((data as { error?: string }).error || `mcp test ${r.status}`);
  return data as McpTestResult | { ok: boolean; results: McpTestResult[]; count: number };
}
