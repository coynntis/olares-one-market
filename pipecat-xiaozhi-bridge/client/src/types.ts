export type AppTab = "chat" | "settings";

export type VoiceStatus = "idle" | "listening" | "thinking" | "speaking";

export type ListenMode = "manual" | "auto" | "realtime";

export interface SenseVoiceMeta {
  language?: string;
  emotion?: string;
  event?: string;
  itn?: boolean;
  tags?: string[];
}

export type AgentStepPhase = "announce" | "running" | "done" | "final" | "error";

export interface AgentStep {
  phase: AgentStepPhase;
  round_index: number;
  step_index: number;
  tool_name: string;
  label: string;
  message: string;
  detail?: string;
  image_url?: string;
}

export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  text: string;
  imagePreview?: string;
  source?: "text" | "voice";
  sensevoice?: SenseVoiceMeta;
  ts: number;
}

export interface Conversation {
  id: string;
  title: string;
  device_id?: string;
  created_at: number;
  updated_at: number;
  message_count?: number;
}

export interface StoredMessage {
  id: string;
  conversation_id: string;
  role: "user" | "assistant" | "tool";
  text: string;
  image_url?: string;
  source?: string;
  created_at: number;
  meta?: Record<string, unknown>;
}

export type McpTransport = "http" | "sse" | "stdio";

export interface McpServer {
  id: string;
  name: string;
  enabled: boolean;
  transport: McpTransport;
  url?: string;
  command?: string;
  args?: string[];
  headers?: Record<string, string>;
  catalog_id?: string | null;
}

export interface McpSuggestion {
  id: string;
  name: string;
  kind: "mcp" | "browser";
  transport: string;
  path: string;
  shared_entrance_name: string;
  in_cluster_url: string;
  ws_url?: string;
  market_app: string;
  description: string;
  install_hint: string;
}

export interface McpTestResult {
  ok: boolean;
  name?: string;
  transport?: string;
  url?: string;
  tool_count?: number;
  tools?: { name: string; description: string }[];
  error?: string;
  latency_ms?: number;
  status?: number;
  detail?: string;
  kind?: string;
  catalog_id?: string;
}

export interface VoiceProfile {
  id: string;
  name: string;
  ref_text: string;
  language_id?: string;
  instruct?: string;
  created_at: number;
  audio_url: string;
}

export interface LlmProfile {
  id: string;
  name: string;
  llm_base_url: string;
  llm_model: string;
  system_prompt: string;
  created_at: number;
}

export interface BuiltinToolsConfig {
  camera: boolean;
  bluetooth: boolean;
  geolocation: boolean;
}

export interface BridgeSettings {
  openai_base_url: string;
  stt_base_url: string;
  tts_base_url: string;
  llm_base_url: string;
  openai_api_key_set: boolean;
  stt_api_key_set: boolean;
  tts_api_key_set: boolean;
  llm_api_key_set: boolean;
  stt_model: string;
  stt_language: string;
  tts_model: string;
  tts_response_format: string;
  tts_language_id: string;
  tts_instruct: string;
  tts_voice: string;
  tts_voice_mode: string;
  tts_active_voice_id: string;
  tts_ref_text: string;
  tts_num_step: number;
  tts_speed: number;
  tts_provider?: string;
  tts_overlap_llm?: boolean;
  tts_warmup?: boolean;
  tts_warmup_text?: string;
  tts_first_chunk_chars?: number;
  tts_min_segment_chars: number;
  tts_segment_pad_ms: number;
  tts_post_llm_delay_ms?: number;
  llm_model: string;
  llm_temperature: number;
  llm_top_p: number;
  llm_top_k: number;
  llm_max_tokens: number;
  llm_think_mode: string;
  downlink_sample_rate: number;
  system_prompt: string;
  http_timeout: number;
  llm_profiles?: LlmProfile[];
  active_llm_profile_id?: string;
  mcp_servers?: McpServer[];
  builtin_tools?: BuiltinToolsConfig;
}

export interface ConfigStatus {
  ready: boolean;
  missing?: string[];
  stt_base_url: string;
  tts_base_url: string;
  llm_base_url: string;
  llm_model: string;
  stt_language: string;
}

export type TestService = "stt" | "tts" | "llm" | "all";

export interface TestConnectionResult {
  ok: boolean;
  service?: string;
  url?: string;
  status?: number;
  detail?: string;
  error?: string | null;
  bytes?: number;
  model?: string;
  provider?: string;
  speed?: number;
  http_ms?: number;
  audio_ms?: number;
  rtf?: number;
  results?: Record<string, TestConnectionResult>;
  capabilities?: LlmCapabilities;
}

export interface LlmParamSupport {
  temperature: boolean;
  top_p: boolean;
  top_k: boolean;
  max_tokens: boolean;
  think_mode: boolean;
}

export interface LlmCapabilities {
  ok?: boolean;
  backend?: string;
  model?: string;
  root?: string | null;
  owned_by?: string;
  supports?: LlmParamSupport;
  error?: string;
}

export interface PipelineStats {
  stt_ms?: number;
  llm_ms?: number;
  tts_ms?: number;
  tts_http_ms?: number;
  tts_decode_ms?: number;
  tts_audio_ms?: number;
  tts_rtf?: number;
  tts_via?: string;
  tts_warmup_ms?: number;
  opus_ms?: number;
  total_ms?: number;
  prompt_tokens?: number;
  completion_tokens?: number;
  tokens_per_sec?: number;
  backend?: string;
  first_token_ms?: number;
  first_audio_ms?: number;
  segments?: number;
  tool_rounds?: number;
}
