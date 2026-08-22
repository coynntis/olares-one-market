import type { VoiceStatus } from "../types";

export interface AgentGlobeProps {
  status: VoiceStatus;
  level?: number;
  compact?: boolean;
}

export function AgentGlobe({ status, level = 0, compact = false }: AgentGlobeProps) {
  const pulse =
    status === "listening" ? 1 + level * 0.28 : status === "speaking" ? 1.1 : 1;

  return (
    <div
      className={`agent-globe agent-globe-${status}${compact ? " agent-globe-compact" : ""}`}
      aria-hidden
      style={{ transform: `scale(${pulse})` }}
    >
      <div className="agent-globe-halo" />
      <div className="agent-globe-ring agent-globe-ring-a" />
      <div className="agent-globe-ring agent-globe-ring-b" />
      <div className="agent-globe-core" />
      <div className="agent-globe-flare" />
    </div>
  );
}
