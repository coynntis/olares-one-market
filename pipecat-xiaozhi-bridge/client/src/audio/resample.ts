/** Pull fixed 16 kHz / 60 ms PCM16 mono frames (xiaozhi uplink standard). */

export const UPLINK_RATE = 16000;
const OUTPUT_RATE = UPLINK_RATE;
const FRAME_MS = 60;
const FRAME_SAMPLES = Math.round((OUTPUT_RATE * FRAME_MS) / 1000); // 1440

function resampleLinearToInt16(input: number[], outLen: number): Int16Array {
  const out = new Int16Array(outLen);
  if (input.length === 0) {
    return out;
  }
  if (input.length === 1) {
    const v = Math.max(-1, Math.min(1, input[0]!)) * 32767;
    out.fill(Math.round(v));
    return out;
  }
  const imax = input.length - 1;
  for (let i = 0; i < outLen; i++) {
    const pos = (i / (outLen - 1)) * imax;
    const j = Math.floor(pos);
    const frac = pos - j;
    const a = input[j]!;
    const b = input[Math.min(j + 1, imax)]!;
    const v = a + frac * (b - a);
    out[i] = Math.round(Math.max(-1, Math.min(1, v)) * 32767);
  }
  return out;
}

export function createPcmFramePuller(inputSampleRate: number) {
  const buf: number[] = [];

  return {
    /** Mono float samples [-1, 1] */
    push(f32: Float32Array) {
      for (let i = 0; i < f32.length; i++) {
        buf.push(f32[i]!);
      }
    },

    /** One 60 ms s16le @ 16 kHz frame, or null if not enough input yet */
    tryPullFrame(): Int16Array | null {
      const needIn = Math.ceil((FRAME_SAMPLES * inputSampleRate) / OUTPUT_RATE);
      if (buf.length < needIn) {
        return null;
      }
      const chunk = buf.splice(0, needIn);
      return resampleLinearToInt16(chunk, FRAME_SAMPLES);
    },

    clear() {
      buf.length = 0;
    },
  };
}

export { FRAME_SAMPLES, OUTPUT_RATE };
