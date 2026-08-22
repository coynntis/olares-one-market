import { OpusDecoder } from "opus-decoder";

const DOWNLINK_RATE = 24000;

let decoder: OpusDecoder | null = null;
let decoderReady: Promise<OpusDecoder> | null = null;
let decoderFailed = false;

/** One WASM Opus decoder per page — avoids OOM from repeated instantiate on WS reconnect. */
export async function getSharedOpusDecoder(): Promise<OpusDecoder> {
  if (decoder) return decoder;
  if (decoderFailed) {
    throw new Error("Opus decoder unavailable");
  }
  if (!decoderReady) {
    decoderReady = (async () => {
      const inst = new OpusDecoder({ sampleRate: DOWNLINK_RATE, channels: 1 });
      await inst.ready;
      decoder = inst;
      return inst;
    })().catch((err) => {
      decoderReady = null;
      decoderFailed = true;
      throw err;
    });
  }
  return decoderReady;
}

export const OPUS_DOWNLINK_RATE = DOWNLINK_RATE;
