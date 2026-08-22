import { createContext, useContext, useState, type ReactNode } from "react";
import type { VoiceStatus } from "../types";

type VoiceStatusState = {
  status: VoiceStatus;
  level: number;
  connected: boolean;
};

type VoiceStatusContextValue = VoiceStatusState & {
  setVoiceStatus: (patch: Partial<VoiceStatusState>) => void;
};

const defaultState: VoiceStatusState = {
  status: "idle",
  level: 0,
  connected: false,
};

const VoiceStatusContext = createContext<VoiceStatusContextValue | null>(null);

export function VoiceStatusProvider({ children }: { children: ReactNode }) {
  const [state, setState] = useState<VoiceStatusState>(defaultState);
  const setVoiceStatus = (patch: Partial<VoiceStatusState>) =>
    setState((s) => ({ ...s, ...patch }));
  return (
    <VoiceStatusContext.Provider value={{ ...state, setVoiceStatus }}>
      {children}
    </VoiceStatusContext.Provider>
  );
}

export function useVoiceStatus() {
  const ctx = useContext(VoiceStatusContext);
  if (!ctx) throw new Error("useVoiceStatus must be used within VoiceStatusProvider");
  return ctx;
}
