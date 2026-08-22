import { formatSenseVoiceMeta } from "../sensevoiceStt";
import type { SenseVoiceMeta } from "../types";

export function SenseVoiceSttMeta({ meta }: { meta?: SenseVoiceMeta }) {
  const chips = formatSenseVoiceMeta(meta);
  if (!chips.length) return null;
  return (
    <div className="stt-meta" aria-label="SenseVoice STT metadata">
      {chips.map((chip) => (
        <span key={chip} className="stt-meta-chip">
          {chip}
        </span>
      ))}
    </div>
  );
}
