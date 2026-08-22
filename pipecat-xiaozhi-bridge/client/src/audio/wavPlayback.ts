import { getSharedAudioSession } from "./sharedAudioSession";

function decodeWavPcm(wav: ArrayBuffer): { pcm: Float32Array; sampleRate: number } {
  const view = new DataView(wav);
  const readStr = (off: number, len: number) => {
    let s = "";
    for (let i = 0; i < len; i++) s += String.fromCharCode(view.getUint8(off + i));
    return s;
  };
  if (readStr(0, 4) !== "RIFF" || readStr(8, 4) !== "WAVE") {
    throw new Error("invalid WAV");
  }
  let offset = 12;
  let sampleRate = 24000;
  let channels = 1;
  let bits = 16;
  let dataOffset = 0;
  let dataSize = 0;
  while (offset + 8 <= view.byteLength) {
    const chunkId = readStr(offset, 4);
    const chunkSize = view.getUint32(offset + 4, true);
    const chunkStart = offset + 8;
    if (chunkId === "fmt ") {
      channels = view.getUint16(chunkStart + 2, true);
      sampleRate = view.getUint32(chunkStart + 4, true);
      bits = view.getUint16(chunkStart + 14, true);
    } else if (chunkId === "data") {
      dataOffset = chunkStart;
      dataSize = chunkSize;
      break;
    }
    offset = chunkStart + chunkSize + (chunkSize % 2);
  }
  if (!dataSize || bits !== 16) {
    throw new Error("unsupported WAV format");
  }
  const frameCount = Math.floor(dataSize / (bits / 8) / channels);
  const mono = new Float32Array(frameCount);
  let idx = 0;
  for (let i = 0; i < frameCount; i++) {
    let sum = 0;
    for (let c = 0; c < channels; c++) {
      const sample = view.getInt16(dataOffset + idx * 2, true);
      sum += sample / 32768;
      idx++;
    }
    mono[i] = sum / channels;
  }
  return { pcm: mono, sampleRate };
}

/** Play REST chat TTS through the same Web Audio path as voice Opus (mobile-safe). */
export async function playWavBase64(b64: string): Promise<void> {
  const bin = atob(b64);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  const { pcm, sampleRate } = decodeWavPcm(bytes.buffer);
  const session = getSharedAudioSession();
  const ctx = await session.ensure();
  const buf = ctx.createBuffer(1, pcm.length, sampleRate);
  buf.copyToChannel(pcm, 0);
  await session.playAudioBuffer(buf);
}
