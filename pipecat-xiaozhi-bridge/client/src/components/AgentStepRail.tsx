import type { AgentStep } from "../types";

const PHASE_ICON: Record<string, string> = {
  announce: "◈",
  running: "◎",
  done: "✓",
  error: "✕",
};

interface Props {
  steps: AgentStep[];
}

export function AgentStepRail({ steps }: Props) {
  const visible = steps.filter((s) => s.phase !== "final");
  if (visible.length === 0) return null;

  return (
    <div className="agent-rail" aria-live="polite">
      {visible.map((s) => (
        <div
          key={`${s.step_index}-${s.phase}-${s.tool_name}`}
          className={`agent-step agent-step-${s.phase}`}
        >
          <span className="agent-step-icon">{PHASE_ICON[s.phase] ?? "·"}</span>
          <div className="agent-step-body">
            {s.label && s.phase !== "announce" && <span className="agent-step-label">{s.label}</span>}
            <span className="agent-step-message">{s.message}</span>
            {s.image_url ? (
              <img src={s.image_url} alt="" className="agent-step-image" />
            ) : null}
          </div>
        </div>
      ))}
    </div>
  );
}
