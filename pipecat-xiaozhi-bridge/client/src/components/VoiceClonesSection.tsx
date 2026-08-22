import { useCallback, useEffect, useRef, useState } from "react";
import {
  activateVoice,
  createVoice,
  deleteVoice,
  fetchVoices,
  previewVoice,
} from "../api";
import { playWavBase64 } from "../audio/wavPlayback";
import { audioBlobToWavBase64, wavBase64DataUrl } from "../audio/wavEncode";
import type { BridgeSettings, VoiceProfile } from "../types";

type Props = {
  form: BridgeSettings;
  onFormChange: (patch: Partial<BridgeSettings>) => void;
};

function supportsClone(provider: string | undefined): boolean {
  return (provider || "").trim().toLowerCase() === "omnivoice";
}

/** Voice clone UI — only for OmniVoice (GPU). Hidden for Melo/Sherpa/Kokoro. */
export function VoiceClonesSection({ form, onFormChange }: Props) {
  const [voices, setVoices] = useState<VoiceProfile[]>([]);
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [name, setName] = useState("Agent R voice");
  const [refText, setRefText] = useState("");
  const [draftAudioB64, setDraftAudioB64] = useState<string | null>(null);
  const [draftPreviewUrl, setDraftPreviewUrl] = useState<string | null>(null);
  const [recording, setRecording] = useState(false);
  const [busy, setBusy] = useState<string | null>(null);
  const mediaRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);

  const canClone = supportsClone(form.tts_provider);

  const reload = useCallback(async () => {
    if (!canClone) return;
    setLoading(true);
    try {
      const data = await fetchVoices();
      setVoices(data.voices);
      setErr(null);
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }, [canClone]);

  useEffect(() => {
    void reload();
  }, [reload]);

  useEffect(() => {
    if (!canClone) return;
    void (async () => {
      try {
        const data = await fetchVoices();
        if (data.active_voice_id && !form.tts_active_voice_id) {
          onFormChange({ tts_active_voice_id: data.active_voice_id });
        }
      } catch {
        /* ignore */
      }
    })();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [canClone]);

  if (!canClone) {
    return (
      <p className="muted">
        Voice clone / design disabled for <code>{form.tts_provider || "sherpa"}</code>. Switch
        provider to <strong>OmniVoice</strong> (or CosyVoice3 / MOSS via base URL) for clone and
        Cantonese.
      </p>
    );
  }

  const setDraftFromB64 = (b64: string) => {
    setDraftAudioB64(b64);
    setDraftPreviewUrl(wavBase64DataUrl(b64));
  };

  const startRecord = async () => {
    setErr(null);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const rec = new MediaRecorder(stream);
      chunksRef.current = [];
      rec.ondataavailable = (e) => {
        if (e.data.size) chunksRef.current.push(e.data);
      };
      rec.onstop = () => {
        stream.getTracks().forEach((t) => t.stop());
        void (async () => {
          const blob = new Blob(chunksRef.current, { type: rec.mimeType || "audio/webm" });
          try {
            const b64 = await audioBlobToWavBase64(blob);
            setDraftFromB64(b64);
          } catch (e) {
            setErr(e instanceof Error ? e.message : String(e));
          }
        })();
      };
      mediaRef.current = rec;
      rec.start();
      setRecording(true);
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    }
  };

  const stopRecord = () => {
    mediaRef.current?.stop();
    mediaRef.current = null;
    setRecording(false);
  };

  const onUpload = (file: File | null) => {
    if (!file) return;
    const reader = new FileReader();
    reader.onload = () => {
      const raw = String(reader.result || "");
      if (raw.startsWith("data:")) {
        const b64 = raw.split(",", 2)[1] || "";
        if (b64) setDraftFromB64(b64);
      } else {
        setErr("Upload WAV or record in browser");
      }
    };
    reader.readAsDataURL(file);
  };

  const saveVoice = async () => {
    if (!draftAudioB64) {
      setErr("Record or upload reference audio first");
      return;
    }
    if (!refText.trim()) {
      setErr("Reference transcript (ref_text) required — what was said in the recording");
      return;
    }
    setBusy("save");
    setErr(null);
    try {
      const { voice } = await createVoice({
        name: name.trim() || "Agent R voice",
        ref_text: refText.trim(),
        audio: wavBase64DataUrl(draftAudioB64),
        language_id: form.tts_language_id,
        instruct: form.tts_instruct,
        set_active: true,
      });
      onFormChange({ tts_active_voice_id: voice.id, tts_voice_mode: "clone" });
      setDraftAudioB64(null);
      setDraftPreviewUrl(null);
      setRefText("");
      await reload();
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(null);
    }
  };

  const selectVoice = async (id: string) => {
    setBusy(`sel-${id}`);
    try {
      await activateVoice(id);
      onFormChange({ tts_active_voice_id: id });
      await reload();
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(null);
    }
  };

  const runPreview = async (id: string) => {
    setBusy(`prev-${id}`);
    try {
      const res = await previewVoice(id);
      playWavBase64(res.audio_wav_b64).catch((e) => setErr(e instanceof Error ? e.message : String(e)));
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(null);
    }
  };

  const removeVoice = async (id: string) => {
    if (!confirm("Delete this voice profile?")) return;
    setBusy(`del-${id}`);
    try {
      await deleteVoice(id);
      if (form.tts_active_voice_id === id) {
        onFormChange({ tts_active_voice_id: "" });
      }
      await reload();
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(null);
    }
  };

  const activeId = form.tts_active_voice_id;

  return (
    <div className="voice-clones">
      <h3>Voice clone (OmniVoice)</h3>
      <p className="muted">
        Clone uses <code>POST /v1/audio/clone</code> with reference WAV + ref_text. Preset voices use
        the <code>voice</code> field on <code>/v1/audio/speech</code>.
      </p>

      <label>Voice mode</label>
      <select
        value={form.tts_voice_mode || "instruct"}
        onChange={(e) => onFormChange({ tts_voice_mode: e.target.value })}
      >
        <option value="default">Default (no instruct / clone)</option>
        <option value="instruct">Voice design (instruct text)</option>
        <option value="clone">Saved clone profile</option>
      </select>

      {form.tts_voice_mode === "instruct" && (
        <p className="muted">Set Voice instruct above — applied to every chunk.</p>
      )}

      <div className="voice-list">
        {loading && <p className="muted">Loading voices…</p>}
        {!loading && voices.length === 0 && (
          <p className="muted">No saved voices yet. Record one below.</p>
        )}
        {voices.map((v) => (
          <div
            key={v.id}
            className={`voice-row ${activeId === v.id && form.tts_voice_mode === "clone" ? "voice-row-active" : ""}`}
          >
            <label className="voice-row-main">
              <input
                type="radio"
                name="active-voice"
                checked={activeId === v.id}
                onChange={() => void selectVoice(v.id)}
              />
              <span className="voice-name">{v.name}</span>
              <span className="muted voice-ref">{v.ref_text.slice(0, 48)}</span>
            </label>
            <div className="voice-row-actions">
              <button
                type="button"
                className="btn btn-ghost btn-sm"
                disabled={busy !== null}
                onClick={() => void runPreview(v.id)}
              >
                {busy === `prev-${v.id}` ? "…" : "Preview"}
              </button>
              <button
                type="button"
                className="btn btn-ghost btn-sm"
                disabled={busy !== null}
                onClick={() => void removeVoice(v.id)}
              >
                Delete
              </button>
            </div>
          </div>
        ))}
      </div>

      <h4>New voice clone</h4>
      <label>Name</label>
      <input value={name} onChange={(e) => setName(e.target.value)} />
      <label>Reference transcript</label>
      <input
        value={refText}
        onChange={(e) => setRefText(e.target.value)}
        placeholder="Exact words spoken in the reference recording"
      />
      <div className="row">
        {!recording ? (
          <button type="button" className="btn btn-ghost" onClick={() => void startRecord()}>
            Record
          </button>
        ) : (
          <button type="button" className="btn btn-ghost" onClick={stopRecord}>
            Stop
          </button>
        )}
        <label className="btn btn-ghost file-btn">
          Upload WAV
          <input
            type="file"
            accept="audio/wav,audio/*"
            hidden
            onChange={(e) => onUpload(e.target.files?.[0] ?? null)}
          />
        </label>
      </div>

      {draftPreviewUrl && (
        <div className="voice-draft-preview">
          <audio controls src={draftPreviewUrl} />
        </div>
      )}

      <button
        type="button"
        className="btn btn-primary"
        disabled={busy === "save" || !draftAudioB64}
        onClick={() => void saveVoice()}
      >
        {busy === "save" ? "Saving…" : "Save & use for Agent R"}
      </button>

      {err && <div className="banner banner-error">{err}</div>}
    </div>
  );
}
