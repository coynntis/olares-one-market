import { useMemo } from "react";
import type { VoiceStatus } from "../types";

const BAR_COUNT = 9;

type Props = {
  status: VoiceStatus;
  level?: number;
};

export function VoiceActivityBar({ status, level = 0 }: Props) {
  const active = status === "listening" || status === "speaking" || status === "thinking";

  const barHeights = useMemo(() => {
    if (!active) return Array(BAR_COUNT).fill(18);
    if (status === "speaking") return null;
    return Array.from({ length: BAR_COUNT }, (_, i) => {
      const wobble = 0.45 + 0.55 * Math.sin((i / BAR_COUNT) * Math.PI * 1.6);
      return Math.round(14 + level * 72 * wobble);
    });
  }, [active, status, level]);

  return (
    <div
      className={`voice-activity voice-activity-${status}${active ? " voice-activity-live" : ""}`}
      aria-hidden={!active}
      role={active ? "img" : undefined}
      aria-label={active ? status : undefined}
    >
      {Array.from({ length: BAR_COUNT }, (_, i) => (
        <span
          key={i}
          className="voice-activity-bar"
          style={
            barHeights
              ? { height: `${barHeights[i]}%`, animationDelay: `${i * 0.05}s` }
              : { animationDelay: `${i * 0.06}s` }
          }
        />
      ))}
    </div>
  );
}
