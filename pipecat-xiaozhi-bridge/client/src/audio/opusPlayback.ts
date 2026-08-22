import { OpusDecoder } from "opus-decoder";

const OUT_RATE = 24000;

/** Schedule decoded Opus frames on an AudioContext (server uses same rate for TTS PCM → Opus). */
export class OpusPlayback {
  private decoder: OpusDecoder | null = null;
  private ctx: AudioContext | null = null;
  private nextPlay = 0;

  async ensure(): Promise<void> {
    if (this.ctx && this.decoder) {
      return;
    }
    this.ctx = new AudioContext({ sampleRate: OUT_RATE });
    this.decoder = new OpusDecoder({ sampleRate: OUT_RATE, channels: 1 });
    await this.decoder.ready;
  }

  async resume(): Promise<void> {
    await this.ensure();
    if (this.ctx!.state === "suspended") {
      await this.ctx!.resume();
    }
  }

  /** Call when a new assistant utterance starts so scheduling does not lag behind clock. */
  alignSchedule(): void {
    if (!this.ctx) {
      return;
    }
    const now = this.ctx.currentTime;
    this.nextPlay = Math.max(this.nextPlay, now);
  }

  enqueueOpus(packet: ArrayBuffer): void {
    if (!this.decoder || !this.ctx) {
      return;
    }
    const dec = this.decoder.decodeFrame(new Uint8Array(packet));
    if (dec.samplesDecoded === 0 || !dec.channelData[0]) {
      return;
    }
    const ch = dec.channelData[0];
    const buf = this.ctx.createBuffer(1, ch.length, OUT_RATE);
    buf.copyToChannel(ch, 0);
    const src = this.ctx.createBufferSource();
    src.buffer = buf;
    src.connect(this.ctx.destination);
    const now = this.ctx.currentTime;
    if (this.nextPlay < now) {
      this.nextPlay = now;
    }
    src.start(this.nextPlay);
    this.nextPlay += buf.duration;
  }

  dispose(): void {
    try {
      this.decoder?.free();
    } catch {
      /* ignore */
    }
    this.decoder = null;
    void this.ctx?.close();
    this.ctx = null;
    this.nextPlay = 0;
  }
}
