import type { BridgeSettings } from "../types";

type Props = {
  form: BridgeSettings;
  onFormChange: (patch: Partial<BridgeSettings>) => void;
};

/** Chunk / settle knobs — OmniVoice only for size; CPU = whole-reply utterance. */
export function TtsPipelineSection({ form, onFormChange }: Props) {
  const provider = form.tts_provider || "sherpa";
  const cpu =
    provider === "melo" ||
    provider === "sherpa" ||
    provider === "kokoro" ||
    provider === "audio8";

  return (
    <section className="settings-section">
      <h2>TTS chunking</h2>
      <p className="muted">
        {cpu
          ? "CPU TTS: bridge sends whole reply as one request (no mid-phrase split). Kokoro-FastAPI may still sentence-split internally at high token targets."
          : "OmniVoice: short first chunk so you hear audio ASAP while GPU warms; later chunks stay warm."}
      </p>
      <div className="row">
        {provider === "omnivoice" && (
          <div>
            <label>TTS num_step</label>
            <input
              type="number"
              min={8}
              max={64}
              value={form.tts_num_step ?? 16}
              onChange={(e) => onFormChange({ tts_num_step: Number(e.target.value) })}
            />
            <p className="muted">Diffusion steps. 16 ≈ fast; 32 ≈ higher quality.</p>
          </div>
        )}
        <div>
          <label>Overlap TTS with LLM</label>
          <select
            value={form.tts_overlap_llm ? "on" : "off"}
            onChange={(e) => onFormChange({ tts_overlap_llm: e.target.value === "on" })}
            disabled={cpu}
          >
            <option value="off">Off (recommended)</option>
            <option value="on">On (experimental)</option>
          </select>
          {cpu && (
            <p className="muted">Forced off for CPU TTS — overlap cuts mid-phrase.</p>
          )}
        </div>
        {!cpu && (
          <>
            <div>
              <label>First chunk chars</label>
              <input
                type="number"
                min={4}
                max={200}
                value={form.tts_first_chunk_chars ?? 12}
                onChange={(e) =>
                  onFormChange({ tts_first_chunk_chars: Number(e.target.value) })
                }
              />
            </div>
            <div>
              <label>Max chunk chars</label>
              <input
                type="number"
                min={8}
                max={400}
                value={form.tts_max_chunk_chars ?? 40}
                onChange={(e) =>
                  onFormChange({ tts_max_chunk_chars: Number(e.target.value) })
                }
              />
            </div>
            <div>
              <label>Min chars per chunk</label>
              <input
                type="number"
                min={4}
                max={120}
                value={form.tts_min_segment_chars ?? 12}
                onChange={(e) =>
                  onFormChange({ tts_min_segment_chars: Number(e.target.value) })
                }
              />
            </div>
          </>
        )}
        <div>
          <label>Chunk pad (ms)</label>
          <input
            type="number"
            min={0}
            max={200}
            value={form.tts_segment_pad_ms ?? 40}
            onChange={(e) => onFormChange({ tts_segment_pad_ms: Number(e.target.value) })}
          />
        </div>
        {provider === "omnivoice" && (
          <div>
            <label>Post-LLM settle (ms)</label>
            <input
              type="number"
              min={0}
              max={2000}
              step={50}
              value={form.tts_post_llm_delay_ms ?? 200}
              onChange={(e) =>
                onFormChange({ tts_post_llm_delay_ms: Number(e.target.value) })
              }
            />
          </div>
        )}
      </div>
    </section>
  );
}
