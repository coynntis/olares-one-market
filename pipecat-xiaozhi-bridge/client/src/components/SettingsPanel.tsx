import { useCallback, useEffect, useState } from "react";
import { fetchConfig, fetchConfigStatus, fetchLlmCapabilities, saveConfig, testConnection } from "../api";
import type { BridgeSettings, ConfigStatus, LlmCapabilities, TestConnectionResult, TestService } from "../types";
import { BuiltinToolsSection } from "./BuiltinToolsSection";
import { LlmProfilesSection } from "./LlmProfilesSection";
import { McpServersSection } from "./McpServersSection";
import { VoiceClonesSection } from "./VoiceClonesSection";
import { TtsPipelineSection } from "./TtsPipelineSection";

/** Sherpa kokoro-int8-multi-lang-v1_1 = Kokoro v1.1-zh (sid → name). Favorites first. */
const SHERPA_SPEAKER_BY_SID: string[] = [
  "af_maple", "af_sol", "bf_vale", "zf_001", "zf_002", "zf_003", "zf_004", "zf_005", "zf_006",
  "zf_007", "zf_008", "zf_017", "zf_018", "zf_019", "zf_021", "zf_022", "zf_023", "zf_024",
  "zf_026", "zf_027", "zf_028", "zf_032", "zf_036", "zf_038", "zf_039", "zf_040", "zf_042",
  "zf_043", "zf_044", "zf_046", "zf_047", "zf_048", "zf_049", "zf_051", "zf_059", "zf_060",
  "zf_067", "zf_070", "zf_071", "zf_072", "zf_073", "zf_074", "zf_075", "zf_076", "zf_077",
  "zf_078", "zf_079", "zf_083", "zf_084", "zf_085", "zf_086", "zf_087", "zf_088", "zf_090",
  "zf_092", "zf_093", "zf_094", "zf_099", "zm_009", "zm_010", "zm_011", "zm_012", "zm_013",
  "zm_014", "zm_015", "zm_016", "zm_020", "zm_025", "zm_029", "zm_030", "zm_031", "zm_033",
  "zm_034", "zm_035", "zm_037", "zm_041", "zm_045", "zm_050", "zm_052", "zm_053", "zm_054",
  "zm_055", "zm_056", "zm_057", "zm_058", "zm_061", "zm_062", "zm_063", "zm_064", "zm_065",
  "zm_066", "zm_068", "zm_069", "zm_080", "zm_081", "zm_082", "zm_089", "zm_091", "zm_095",
  "zm_096", "zm_097", "zm_098", "zm_100",
];

const SHERPA_VOICE_FAVORITES = new Set(["bf_vale", "af_maple", "af_sol", "zm_009", "zm_010", "zf_001", "zf_002"]);

function sherpaVoiceLabel(name: string, sid: number): string {
  if (name === "bf_vale") return `${name} — British female (best EN in Sherpa pack)`;
  if (name === "af_maple" || name === "af_sol") return `${name} — American female`;
  if (name.startsWith("zf_")) return `${name} — Mandarin female (sid ${sid})`;
  if (name.startsWith("zm_")) return `${name} — Mandarin male (sid ${sid})`;
  return `${name} (sid ${sid})`;
}

const SHERPA_VOICES: { value: string; label: string }[] = (() => {
  const fav: { value: string; label: string }[] = [];
  const rest: { value: string; label: string }[] = [];
  SHERPA_SPEAKER_BY_SID.forEach((name, sid) => {
    const item = { value: name, label: sherpaVoiceLabel(name, sid) };
    if (SHERPA_VOICE_FAVORITES.has(name)) fav.push(item);
    else rest.push(item);
  });
  // Keep favorite order as declared in the Set iteration is unstable — use explicit order.
  const favOrder = ["bf_vale", "af_maple", "af_sol", "zm_009", "zm_010", "zf_001", "zf_002"];
  const favSorted = favOrder
    .map((n) => fav.find((v) => v.value === n))
    .filter((v): v is { value: string; label: string } => !!v);
  return [...favSorted, ...rest];
})();

/** Classic Kokoro-82M names — Kokoro-FastAPI sidecar only. */
const KOKORO_FASTAPI_VOICES: { value: string; label: string }[] = [
  { value: "bm_lewis", label: "bm_lewis — Jarvis-like British male" },
  { value: "bm_george", label: "bm_george — British male" },
  { value: "bm_daniel", label: "bm_daniel — British male" },
  { value: "am_michael", label: "am_michael — American male" },
  { value: "am_adam", label: "am_adam — American male" },
  { value: "af_heart", label: "af_heart — American female" },
  { value: "af_bella", label: "af_bella — American female" },
  { value: "af_sky", label: "af_sky — American female" },
  { value: "bf_emma", label: "bf_emma — British female" },
  { value: "zf_xiaoxiao", label: "zf_xiaoxiao — Mandarin female" },
  { value: "zf_xiaoyi", label: "zf_xiaoyi — Mandarin female" },
  { value: "zm_yunxi", label: "zm_yunxi — Mandarin male" },
  { value: "zm_yunyang", label: "zm_yunyang — Mandarin male" },
];

const EMPTY: BridgeSettings = {
  openai_base_url: "",
  stt_base_url: "",
  tts_base_url: "",
  llm_base_url: "",
  openai_api_key_set: false,
  stt_api_key_set: false,
  tts_api_key_set: false,
  llm_api_key_set: false,
  stt_model: "sensevoice",
  stt_language: "yue",
  tts_provider: "sherpa",
  tts_model: "kokoro-int8-multi-lang-v1_1",
  tts_response_format: "wav",
  tts_language_id: "en",
  tts_instruct: "",
  tts_voice: "bf_vale",
  tts_voice_mode: "default",
  tts_active_voice_id: "",
  tts_ref_text: "",
  tts_num_step: 16,
  tts_speed: 1,
  tts_overlap_llm: false,
  tts_warmup: false,
  tts_warmup_text: "嗯",
  tts_first_chunk_chars: 12,
  tts_min_segment_chars: 12,
  tts_segment_pad_ms: 40,
  tts_post_llm_delay_ms: 0,
  llm_model: "",
  llm_temperature: 0.7,
  llm_top_p: 0.8,
  llm_top_k: 20,
  llm_max_tokens: 128,
  llm_think_mode: "auto",
  downlink_sample_rate: 24000,
  system_prompt:
    "You are a helpful voice assistant. Keep replies concise for speech (1-3 sentences).",
  http_timeout: 120,
};

const FIELD_LABELS: Record<string, string> = {
  stt_base_url: "STT base URL",
  tts_base_url: "TTS base URL",
  llm_base_url: "LLM base URL",
  llm_model: "LLM model name",
};

function formatTestResult(r: TestConnectionResult): string {
  if (r.ok) {
    const bits: string[] = [];
    if (r.status) bits.push(`HTTP ${r.status}`);
    if (r.bytes) bits.push(`${r.bytes} bytes`);
    if (r.http_ms != null) bits.push(`http ${r.http_ms}ms`);
    if (r.audio_ms != null && r.audio_ms > 0) bits.push(`audio ${r.audio_ms}ms`);
    if (r.rtf != null && r.rtf > 0) bits.push(`RTF ${r.rtf.toFixed(3)}`);
    if (r.speed != null && r.speed > 0) bits.push(`speed ${r.speed}`);
    if (!bits.length && r.detail) bits.push(r.detail);
    return bits.join(" | ") || r.detail || "OK";
  }
  return r.error || r.detail || "Failed";
}

function TestResultLine({ result }: { result: TestConnectionResult | null }) {
  if (!result) return null;
  return (
    <div className={`test-result ${result.ok ? "test-ok" : "test-fail"}`}>
      <span className="test-status">{result.ok ? "PASS" : "FAIL"}</span>
      {formatTestResult(result)}
      {result.url && <span className="muted test-url">{result.url}</span>}
    </div>
  );
}

export function SettingsPanel() {
  const [form, setForm] = useState<BridgeSettings>(EMPTY);
  const [status, setStatus] = useState<ConfigStatus | null>(null);
  const [adminToken, setAdminToken] = useState("");
  const [saving, setSaving] = useState(false);
  const [msg, setMsg] = useState<string | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [testing, setTesting] = useState<TestService | null>(null);
  const [testStt, setTestStt] = useState<TestConnectionResult | null>(null);
  const [testTts, setTestTts] = useState<TestConnectionResult | null>(null);
  const [testLlm, setTestLlm] = useState<TestConnectionResult | null>(null);
  const [testAll, setTestAll] = useState<TestConnectionResult | null>(null);
  const [llmCaps, setLlmCaps] = useState<LlmCapabilities | null>(null);

  const supports = llmCaps?.supports;
  const capLabel = llmCaps?.backend ? `Backend: ${llmCaps.backend}` : null;

  const loadCaps = useCallback(async () => {
    try {
      setLlmCaps(await fetchLlmCapabilities());
    } catch {
      setLlmCaps(null);
    }
  }, []);

  const load = useCallback(async () => {
    try {
      const [cfg, st] = await Promise.all([fetchConfig(), fetchConfigStatus()]);
      setForm(cfg);
      setStatus(st);
      setErr(null);
      void loadCaps();
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    }
  }, [loadCaps]);

  useEffect(() => {
    void load();
  }, [load]);

  const set = (key: keyof BridgeSettings, value: string | number) => {
    setForm((f) => ({ ...f, [key]: value }));
  };

  const save = async () => {
    setSaving(true);
    setMsg(null);
    setErr(null);
    try {
      const patch: Record<string, string | number> = {
        openai_base_url: form.openai_base_url,
        stt_base_url: form.stt_base_url,
        tts_base_url: form.tts_base_url,
        llm_base_url: form.llm_base_url,
        stt_model: form.stt_model,
        stt_language: form.stt_language,
        tts_model: form.tts_model,
        tts_response_format: form.tts_response_format,
        tts_language_id: form.tts_language_id,
        tts_instruct: form.tts_instruct,
        tts_voice: form.tts_voice,
        tts_voice_mode: form.tts_voice_mode,
        tts_active_voice_id: form.tts_active_voice_id,
        tts_ref_text: form.tts_ref_text,
        tts_num_step: form.tts_num_step,
        tts_speed: form.tts_speed,
        tts_provider: form.tts_provider || "melo",
        tts_overlap_llm: form.tts_overlap_llm,
        tts_warmup: form.tts_warmup,
        tts_warmup_text: form.tts_warmup_text,
        tts_first_chunk_chars: form.tts_first_chunk_chars,
        tts_min_segment_chars: form.tts_min_segment_chars,
        tts_segment_pad_ms: form.tts_segment_pad_ms,
        tts_post_llm_delay_ms: form.tts_post_llm_delay_ms,
        llm_model: form.llm_model,
        llm_temperature: form.llm_temperature,
        llm_top_p: form.llm_top_p,
        llm_top_k: form.llm_top_k,
        llm_max_tokens: form.llm_max_tokens,
        llm_think_mode: form.llm_think_mode,
        downlink_sample_rate: form.downlink_sample_rate,
        system_prompt: form.system_prompt,
        http_timeout: form.http_timeout,
      };
      const keys = document.querySelectorAll<HTMLInputElement>("[data-secret-key]");
      keys.forEach((el) => {
        const k = el.dataset.secretKey;
        const v = el.value.trim();
        if (k && v) patch[k] = v;
      });
      const updated = await saveConfig(patch, adminToken || undefined);
      setForm(updated);
      setStatus(await fetchConfigStatus());
      void loadCaps();
      setMsg("Saved. Settings persist in app data.");
      keys.forEach((el) => {
        el.value = "";
      });
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    } finally {
      setSaving(false);
    }
  };

  const runTest = async (service: TestService) => {
    setTesting(service);
    setErr(null);
    try {
      const result = await testConnection(service);
      if (service === "stt") setTestStt(result);
      else if (service === "tts") setTestTts(result);
      else if (service === "llm") {
        setTestLlm(result);
        if (result.capabilities) setLlmCaps(result.capabilities);
        else void loadCaps();
      }
      else {
        setTestAll(result);
        if (result.results) {
          setTestStt(result.results.stt ?? null);
          setTestTts(result.results.tts ?? null);
          setTestLlm(result.results.llm ?? null);
          if (result.results.llm?.capabilities) setLlmCaps(result.results.llm.capabilities);
        }
      }
    } catch (e) {
      const fail: TestConnectionResult = {
        ok: false,
        service,
        error: e instanceof Error ? e.message : String(e),
      };
      if (service === "stt") setTestStt(fail);
      else if (service === "tts") setTestTts(fail);
      else if (service === "llm") setTestLlm(fail);
      else setTestAll(fail);
    } finally {
      setTesting(null);
    }
  };

  const missingLabels =
    status?.missing?.map((k) => FIELD_LABELS[k] || k).join(", ") ?? "";

  return (
    <div className="panel settings-panel hud-panel">
      {status && (
        <div className={`banner ${status.ready ? "banner-ok" : "banner-warn"}`}>
          {status.ready
            ? "Pipeline ready. STT, TTS, and LLM configured."
            : missingLabels
              ? `Not ready. Missing: ${missingLabels}.`
              : "Not ready. Fill in shared entrance URLs below (include /v1 suffix)."}
        </div>
      )}

      <div className="settings-test-all">
        <button
          type="button"
          className="btn btn-ghost"
          disabled={testing !== null}
          onClick={() => void runTest("all")}
        >
          {testing === "all" ? "Testing all…" : "Test all from bridge"}
        </button>
        <span className="muted">HTTP from bridge pod → your shared entrances (save first)</span>
        {testAll && <TestResultLine result={testAll} />}
      </div>

      <section className="settings-section">
        <div className="section-head">
          <h2>Speech-to-text</h2>
          <button
            type="button"
            className="btn btn-ghost btn-sm"
            disabled={testing !== null}
            onClick={() => void runTest("stt")}
          >
            {testing === "stt" ? "Testing…" : "Test connection"}
          </button>
        </div>
        <label>STT base URL</label>
        <input
          type="url"
          placeholder="https://xxxx.shared.olares.com/v1"
          value={form.stt_base_url}
          onChange={(e) => set("stt_base_url", e.target.value)}
        />
        <div className="row">
          <div>
            <label>Model</label>
            <input value={form.stt_model} onChange={(e) => set("stt_model", e.target.value)} />
          </div>
          <div>
            <label>Language</label>
            <input value={form.stt_language} onChange={(e) => set("stt_language", e.target.value)} />
          </div>
        </div>
        <label>
          STT API key <span className="muted">(optional)</span>{" "}
          {form.stt_api_key_set && <span className="tag">set</span>}
        </label>
        <input type="password" data-secret-key="stt_api_key" placeholder="leave blank to keep" autoComplete="off" />
        <TestResultLine result={testStt} />
      </section>

      <section className="settings-section">
        <div className="section-head">
          <h2>Text-to-speech</h2>
          <button
            type="button"
            className="btn btn-ghost btn-sm"
            disabled={testing !== null}
            onClick={() => void runTest("tts")}
          >
            {testing === "tts" ? "Testing…" : "Test connection"}
          </button>
        </div>
        <label>TTS provider</label>
        <select
          value={form.tts_provider || "sherpa"}
          onChange={(e) => {
            const provider = e.target.value;
            const cpu =
              provider === "melo" ||
              provider === "sherpa" ||
              provider === "kokoro" ||
              provider === "audio8";
            setForm((f) => ({
              ...f,
              tts_provider: provider,
              tts_model:
                provider === "melo"
                  ? "melo"
                  : provider === "sherpa"
                    ? "kokoro-int8-multi-lang-v1_1"
                    : provider === "kokoro"
                      ? "kokoro"
                      : provider === "audio8"
                        ? "arktts"
                        : f.tts_model === "melo" ||
                            (f.tts_model || "").startsWith("kokoro") ||
                            f.tts_model === "arktts"
                          ? "omnivoice"
                          : f.tts_model,
              tts_voice:
                provider === "melo"
                  ? "EN-US"
                  : provider === "sherpa"
                    ? "bf_vale"
                    : provider === "kokoro"
                      ? "bm_lewis"
                      : provider === "audio8"
                        ? "en_default"
                        : f.tts_voice,
              tts_base_url:
                provider === "melo"
                  ? "http://pipecatxiaozhimelo:8000/v1"
                  : provider === "sherpa"
                    ? "http://pipecatxiaozhisherpa:10500/v1"
                    : provider === "kokoro"
                      ? "http://pipecatxiaozhikokoro:8880/v1"
                      : provider === "audio8"
                        ? "http://pipecatxiaozhiaudio8:8024/v1"
                        : f.tts_base_url.includes("pipecatxiaozhi")
                          ? ""
                          : f.tts_base_url,
              tts_voice_mode: provider === "omnivoice" ? f.tts_voice_mode || "instruct" : "default",
              tts_post_llm_delay_ms: cpu ? 0 : f.tts_post_llm_delay_ms || 200,
              tts_overlap_llm: cpu ? false : f.tts_overlap_llm,
              // <2000 → bridge single-utterance path (ignores tiny leftover settings).
              tts_first_chunk_chars: cpu ? 100000 : 12,
              tts_max_chunk_chars: cpu ? 100000 : 40,
              tts_min_segment_chars: cpu ? 24 : 12,
            }));
          }}
        >
          <option value="sherpa">Sherpa-ONNX (Kokoro v1.1-zh — 103 voices)</option>
          <option value="kokoro">Kokoro-FastAPI CPU (classic voices / Jarvis)</option>
          <option value="audio8">Audio8 TTS 0.6B ONNX INT4 (11 langs + Yue)</option>
          <option value="melo">MeloTTS CPU (EN+ZH, slower)</option>
          <option value="omnivoice">OmniVoice GPU (HQ / clone / Cantonese)</option>
        </select>
        <p className="muted">
          Sherpa = multilingual <code>kokoro-int8-multi-lang-v1_1</code>. Audio8 = DualAR 0.6B
          ONNX INT4 (~1 GiB first download). Want Jarvis → <strong>Kokoro-FastAPI</strong> +{" "}
          <code>bm_lewis</code>.
        </p>
        <div className="row">
          <button
            type="button"
            className="btn btn-ghost btn-sm"
            onClick={() =>
              setForm((f) => ({
                ...f,
                tts_provider: "sherpa",
                tts_base_url: "http://pipecatxiaozhisherpa:10500/v1",
                tts_model: "kokoro-int8-multi-lang-v1_1",
                tts_voice: "bf_vale",
                tts_post_llm_delay_ms: 0,
              }))
            }
          >
            Use Sherpa + bf_vale
          </button>
          <button
            type="button"
            className="btn btn-ghost btn-sm"
            onClick={() =>
              setForm((f) => ({
                ...f,
                tts_provider: "kokoro",
                tts_base_url: "http://pipecatxiaozhikokoro:8880/v1",
                tts_model: "kokoro",
                tts_voice: "bm_lewis",
                tts_post_llm_delay_ms: 0,
              }))
            }
          >
            Use Kokoro-FastAPI + Jarvis
          </button>
          <button
            type="button"
            className="btn btn-ghost btn-sm"
            onClick={() =>
              setForm((f) => ({
                ...f,
                tts_provider: "audio8",
                tts_base_url: "http://pipecatxiaozhiaudio8:8024/v1",
                tts_model: "arktts",
                tts_voice: "en_default",
                tts_post_llm_delay_ms: 0,
                tts_first_chunk_chars: 100000,
                tts_max_chunk_chars: 100000,
              }))
            }
          >
            Use Audio8 0.6B ONNX
          </button>
          <button
            type="button"
            className="btn btn-ghost btn-sm"
            onClick={() =>
              setForm((f) => ({
                ...f,
                tts_provider: "melo",
                tts_base_url: "http://pipecatxiaozhimelo:8000/v1",
                tts_model: "melo",
                tts_voice: "EN-US",
                tts_post_llm_delay_ms: 0,
              }))
            }
          >
            Use in-cluster Melo
          </button>
        </div>
        <label>TTS base URL</label>
        <input
          type="url"
          placeholder="http://pipecatxiaozhisherpa:10500/v1"
          value={form.tts_base_url}
          onChange={(e) => set("tts_base_url", e.target.value)}
        />
        {/shared\.olares\.com/i.test(form.tts_base_url) && (
          <p className="muted">
            Shared-entrance URL hairpins via gateway. Prefer in-cluster Sherpa/Kokoro/Audio8/Melo
            URLs.
          </p>
        )}
        <div className="row">
          <div>
            <label>Model</label>
            <input value={form.tts_model} onChange={(e) => set("tts_model", e.target.value)} />
          </div>
          <div>
            <label>Format</label>
            <input
              value={form.tts_response_format}
              onChange={(e) => set("tts_response_format", e.target.value)}
            />
          </div>
        </div>
        <label>Language ID</label>
        <input value={form.tts_language_id} onChange={(e) => set("tts_language_id", e.target.value)} />
        <p className="muted">
          OmniVoice: use <code>yue</code> for Cantonese. Audio8 speaks 11 langs + Yue from text
          (voice = registered clone profile).
        </p>
        {(form.tts_provider || "sherpa") === "melo" ? (
          <>
            <label>Melo voice</label>
            <select value={form.tts_voice || "EN-US"} onChange={(e) => set("tts_voice", e.target.value)}>
              <option value="EN-US">EN-US</option>
              <option value="EN-Default">EN-Default</option>
              <option value="EN-BR">EN-BR</option>
              <option value="EN-AU">EN-AU</option>
              <option value="EN_INDIA">EN_INDIA</option>
              <option value="ZH">ZH (Mandarin)</option>
            </select>
          </>
        ) : (form.tts_provider || "") === "sherpa" ? (
          <>
            <label>Voice (Sherpa Kokoro v1.1-zh)</label>
            <select
              value={
                SHERPA_VOICES.some((v) => v.value === form.tts_voice)
                  ? form.tts_voice
                  : "bf_vale"
              }
              onChange={(e) => set("tts_voice", e.target.value)}
            >
              {SHERPA_VOICES.map((v) => (
                <option key={v.value} value={v.value}>
                  {v.label}
                </option>
              ))}
            </select>
            <p className="muted">
              These names match the model log (<code>af_maple</code>, <code>zf_*</code>,{" "}
              <code>zm_*</code>). No <code>bm_lewis</code> here — use Kokoro-FastAPI for Jarvis.
              Or type <code>speakerN</code> (0–102) in a saved override if needed.
            </p>
          </>
        ) : (form.tts_provider || "") === "kokoro" ? (
          <>
            <label>Voice (Kokoro-FastAPI)</label>
            <select
              value={
                KOKORO_FASTAPI_VOICES.some((v) => v.value === form.tts_voice)
                  ? form.tts_voice
                  : "bm_lewis"
              }
              onChange={(e) => set("tts_voice", e.target.value)}
            >
              {KOKORO_FASTAPI_VOICES.map((v) => (
                <option key={v.value} value={v.value}>
                  {v.label}
                </option>
              ))}
            </select>
            <p className="muted">
              Classic Kokoro-82M voices. <code>bm_lewis</code> ≈ OpenJarvis-style British butler.
            </p>
          </>
        ) : (form.tts_provider || "") === "audio8" ? (
          <>
            <label>Voice (Audio8 registered profile)</label>
            <select
              value={form.tts_voice || "en_default"}
              onChange={(e) => set("tts_voice", e.target.value)}
            >
              <option value="en_default">en_default — English demo</option>
              <option value="zh_default">zh_default — Mandarin-style (EN ref)</option>
              <option value="yue_default">yue_default — Cantonese demo</option>
            </select>
            <p className="muted">
              Zero-shot clone profiles. Register more via Audio8 shared entrance web UI (
              <code>/api/voices/register</code>). No speaking-speed API on ONNX OpenAI endpoint.
            </p>
          </>
        ) : (
          <>
            <label>Voice instruct</label>
            <input
              value={form.tts_instruct}
              onChange={(e) => set("tts_instruct", e.target.value)}
              placeholder="female, low pitch, british accent"
            />
            <label>Voice preset (OmniVoice /v1/audio/speech)</label>
            <select value={form.tts_voice || "auto"} onChange={(e) => set("tts_voice", e.target.value)}>
              <option value="auto">auto</option>
              <option value="female">female</option>
              <option value="male">male</option>
              <option value="female_en">female_en (American)</option>
              <option value="male_en">male_en (American)</option>
              <option value="female_br">female_br (British)</option>
              <option value="male_br">male_br (British)</option>
              <option value="child">child</option>
              <option value="elderly">elderly</option>
              <option value="whisper">whisper</option>
            </select>
            <p className="muted">
              Used when mode is Default or Instruct (instruct overrides preset). Clone mode ignores
              this.
            </p>
          </>
        )}
        {(form.tts_provider || "") !== "audio8" ? (
          <div className="row">
            <div>
              <label>Speaking speed</label>
              <input
                type="number"
                min={0.5}
                max={2}
                step={0.05}
                value={form.tts_speed ?? 1}
                onChange={(e) => set("tts_speed", Number(e.target.value))}
              />
              <p className="muted">
                1.0 = normal. Melo / Sherpa / Kokoro / OmniVoice send <code>speed</code>. Save, then
                Test connection — result shows RTF at this speed. Audio8 ONNX has no speed field.
              </p>
            </div>
          </div>
        ) : null}
        <VoiceClonesSection
          form={form}
          onFormChange={(patch) => setForm((f) => ({ ...f, ...patch }))}
        />
        <TtsPipelineSection
          form={form}
          onFormChange={(patch) => setForm((f) => ({ ...f, ...patch }))}
        />
        <label>
          TTS API key <span className="muted">(optional)</span>{" "}
          {form.tts_api_key_set && <span className="tag">set</span>}
        </label>
        <input type="password" data-secret-key="tts_api_key" placeholder="leave blank to keep" autoComplete="off" />
        <TestResultLine result={testTts} />
        {testTts?.ok && testTts.rtf != null && testTts.rtf > 0 && (
          <p className="muted">
            RTF = HTTP time ÷ audio duration. &lt;1 faster than realtime; CPU TTS often &gt;1. Speed{" "}
            {testTts.speed ?? form.tts_speed} applied on this probe.
          </p>
        )}
      </section>

      <section className="settings-section">
        <div className="section-head">
          <h2>LLM (chat + voice brain)</h2>
          <button
            type="button"
            className="btn btn-ghost btn-sm"
            disabled={testing !== null}
            onClick={() => void runTest("llm")}
          >
            {testing === "llm" ? "Testing…" : "Test connection"}
          </button>
        </div>
        <label>LLM base URL</label>
        <input
          type="url"
          placeholder="https://xxxx.shared.olares.com/v1"
          value={form.llm_base_url}
          onChange={(e) => set("llm_base_url", e.target.value)}
        />
        <label>Model name</label>
        <input
          value={form.llm_model}
          onChange={(e) => set("llm_model", e.target.value)}
          placeholder="gemma-4-e4b-mtp"
        />
        {capLabel && <p className="muted cap-hint">{capLabel}. Unsupported fields are grayed out.</p>}
        <div className="row">
          <div className={supports?.temperature === false ? "field-disabled" : ""}>
            <label>Temperature</label>
            <input
              type="number"
              step="0.05"
              min="0"
              max="2"
              disabled={supports?.temperature === false}
              value={form.llm_temperature}
              onChange={(e) => set("llm_temperature", Number(e.target.value))}
            />
          </div>
          <div className={supports?.top_p === false ? "field-disabled" : ""}>
            <label>Top P</label>
            <input
              type="number"
              step="0.05"
              min="0"
              max="1"
              disabled={supports?.top_p === false}
              value={form.llm_top_p}
              onChange={(e) => set("llm_top_p", Number(e.target.value))}
            />
          </div>
        </div>
        <div className="row">
          <div className={supports?.top_k === false ? "field-disabled" : ""}>
            <label>Top K</label>
            <input
              type="number"
              min="0"
              disabled={supports?.top_k === false}
              value={form.llm_top_k}
              onChange={(e) => set("llm_top_k", Number(e.target.value))}
            />
          </div>
          <div className={supports?.max_tokens === false ? "field-disabled" : ""}>
            <label>Max tokens</label>
            <input
              type="number"
              min="1"
              disabled={supports?.max_tokens === false}
              value={form.llm_max_tokens}
              onChange={(e) => set("llm_max_tokens", Number(e.target.value))}
            />
            <p className="muted" style={{ marginTop: "0.35rem" }}>
              Voice: keep low (~64–128) so GEN finishes fast → OmniVoice RTF ~0.1–0.2. More
              reasoning = another turn, not a huge cap.
            </p>
          </div>
        </div>
        <label className={supports?.think_mode === false ? "field-disabled" : ""}>
          Think mode
        </label>
        <select
          value={form.llm_think_mode}
          disabled={supports?.think_mode === false}
          onChange={(e) => set("llm_think_mode", e.target.value)}
        >
          <option value="auto">Auto (server default)</option>
          <option value="no_think">No think</option>
          <option value="think">Think</option>
        </select>
        <label>
          LLM API key <span className="muted">(optional)</span>{" "}
          {form.llm_api_key_set && <span className="tag">set</span>}
        </label>
        <input type="password" data-secret-key="llm_api_key" placeholder="leave blank to keep" autoComplete="off" />
        <TestResultLine result={testLlm} />
      </section>

      <LlmProfilesSection
        form={form}
        adminToken={adminToken}
        onFormChange={(patch) => setForm((f) => ({ ...f, ...patch }))}
        onSettingsSaved={(settings) => setForm(settings)}
      />

      <McpServersSection adminToken={adminToken} />

      <BuiltinToolsSection adminToken={adminToken} />

      <section className="settings-section">
        <h2>General</h2>
        <label>Fallback OpenAI base URL (optional single gateway)</label>
        <input
          type="url"
          value={form.openai_base_url}
          onChange={(e) => set("openai_base_url", e.target.value)}
        />
        <label>System prompt</label>
        <textarea
          className="settings-textarea"
          rows={4}
          value={form.system_prompt}
          onChange={(e) => set("system_prompt", e.target.value)}
        />
        <div className="row">
          <div>
            <label>Downlink sample rate</label>
            <input
              type="number"
              value={form.downlink_sample_rate}
              onChange={(e) => set("downlink_sample_rate", Number(e.target.value))}
            />
          </div>
          <div>
            <label>HTTP timeout (s)</label>
            <input
              type="number"
              value={form.http_timeout}
              onChange={(e) => set("http_timeout", Number(e.target.value))}
            />
          </div>
        </div>
        <label>Settings save token (only if SETTINGS_TOKEN set on server)</label>
        <input
          type="password"
          value={adminToken}
          onChange={(e) => setAdminToken(e.target.value)}
          placeholder="optional"
          autoComplete="off"
        />
      </section>

      {msg && <div className="banner banner-ok">{msg}</div>}
      {err && <div className="banner banner-error">{err}</div>}

      <div className="toolbar">
        <button type="button" className="btn btn-primary" disabled={saving} onClick={() => void save()}>
          {saving ? "Saving…" : "Save settings"}
        </button>
        <button type="button" className="btn btn-ghost" onClick={() => void load()}>
          Reload
        </button>
      </div>

      <p className="muted settings-foot">
        URLs are stored in app data (<code>/data/config.json</code>), not in the chart. Paste your Olares{" "}
        <strong>shared entrance</strong> URLs with <code>/v1</code> suffix. API keys can stay empty. Olares
        shared entrances use internal auth; the bridge only sends <code>Authorization</code> when you set a key.
      </p>
    </div>
  );
}
