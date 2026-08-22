import type { PipelineStats } from "../types";

function fmtMs(ms?: number): string {
  if (ms == null || ms <= 0) return "n/a";
  if (ms < 1000) return `${ms}ms`;
  return `${(ms / 1000).toFixed(2)}s`;
}

export function PipelineStatsBar({ stats }: { stats: PipelineStats | null }) {
  if (!stats) return null;

  const chips: { label: string; value: string; hint?: string }[] = [];
  if (stats.stt_ms) chips.push({ label: "STT", value: fmtMs(stats.stt_ms) });
  if (stats.llm_ms) chips.push({ label: "LLM", value: fmtMs(stats.llm_ms) });
  if (stats.tts_ms) chips.push({ label: "TTS", value: fmtMs(stats.tts_ms), hint: "http+decode" });
  if (stats.tts_http_ms)
    chips.push({
      label: "TTS-HTTP",
      value: fmtMs(stats.tts_http_ms),
      hint: "wait on OmniVoice (network + model)",
    });
  if (stats.tts_decode_ms)
    chips.push({
      label: "DECODE",
      value: fmtMs(stats.tts_decode_ms),
      hint: "WAV→PCM+resample in bridge",
    });
  if (stats.tts_audio_ms)
    chips.push({
      label: "AUDLEN",
      value: fmtMs(stats.tts_audio_ms),
      hint: "generated speech duration",
    });
  if (stats.tts_rtf)
    chips.push({
      label: "TTS-RTF",
      value: stats.tts_rtf.toFixed(3),
      hint: "http_ms / audio_ms (compare to OmniVoice UI RTF)",
    });
  if (stats.tts_via)
    chips.push({
      label: "VIA",
      value: stats.tts_via,
      hint: "gateway=shared.olares.com hairpin; cluster=in-cluster DNS",
    });
  if (stats.tts_warmup_ms)
    chips.push({
      label: "WARM",
      value: fmtMs(stats.tts_warmup_ms),
      hint: "discard TTS after LLM (pays GPU handoff tax so seg1 stays warm)",
    });
  if (stats.opus_ms)
    chips.push({
      label: "OPUS",
      value: fmtMs(stats.opus_ms),
      hint: "realtime frame pacing ≈ play length",
    });
  if (stats.total_ms) chips.push({ label: "TOTAL", value: fmtMs(stats.total_ms) });
  if (stats.first_token_ms) chips.push({ label: "TTFT", value: fmtMs(stats.first_token_ms) });
  if (stats.first_audio_ms)
    chips.push({
      label: "AUDIO",
      value: fmtMs(stats.first_audio_ms),
      hint: "ms from LLM start to first Opus packet",
    });
  if (stats.tokens_per_sec) chips.push({ label: "GEN", value: `${stats.tokens_per_sec.toFixed(1)} t/s` });
  if (stats.segments) chips.push({ label: "SEG", value: String(stats.segments) });
  if (stats.completion_tokens) chips.push({ label: "OUT", value: String(stats.completion_tokens) });
  if (stats.backend) chips.push({ label: "ENG", value: stats.backend });

  if (!chips.length) return null;

  return (
    <div className="pipeline-stats hud-panel" aria-live="polite">
      <span className="pipeline-stats-title">Telemetry</span>
      <div className="telemetry-row">
        {chips.map((c) => (
          <div key={c.label} className="telemetry-chip" title={c.hint || c.label}>
            <span className="telemetry-label">{c.label}</span>
            <span className="telemetry-value">{c.value}</span>
          </div>
        ))}
      </div>
      {(stats.tts_http_ms || stats.tts_via) && (
        <p className="muted telemetry-hint">
          Flow: finish LLM → chunked TTS (play N while synth N+1). Melo CPU = default (no GPU
          cold tax). OmniVoice = HQ/clone/yue. Overlap = experimental TTS during GEN.{" "}
          <strong>VIA=gateway</strong> = hairpin — prefer in-cluster Melo URL.
        </p>
      )}
    </div>
  );
}
