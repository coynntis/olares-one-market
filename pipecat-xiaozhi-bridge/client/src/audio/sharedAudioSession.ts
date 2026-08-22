import { BrowserAudioSession } from "./audioSession";

let shared: BrowserAudioSession | null = null;

/** One Web Audio session per page — mic + Opus TTS + REST WAV replies share AEC context. */
export function getSharedAudioSession(): BrowserAudioSession {
  if (!shared) {
    shared = new BrowserAudioSession();
  }
  return shared;
}

/** Call from user gesture (tap/click) before async work — unlocks iOS/Safari AudioContext. */
export async function unlockSharedAudio(): Promise<void> {
  const session = getSharedAudioSession();
  const ctx = await session.ensure();
  if (ctx.state === "suspended") {
    await ctx.resume();
  }
}

export function disposeSharedAudio(): void {
  shared?.dispose();
  shared = null;
}
