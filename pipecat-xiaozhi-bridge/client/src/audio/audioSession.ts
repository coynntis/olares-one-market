import { getSharedOpusDecoder, OPUS_DOWNLINK_RATE } from "./opusShared";
import { createPcmFramePuller } from "./resample";
import type { OpusDecoder } from "opus-decoder";

/**
 * Single AudioContext for mic + TTS playback — browser AEC works best this way.
 * Uplink is muted while assistant speaks (unless realtime+AEC mode).
 */
export class BrowserAudioSession {
  private ctx: AudioContext | null = null;
  private micStream: MediaStream | null = null;
  private micNode: ScriptProcessorNode | null = null;
  private micSilent: GainNode | null = null;
  private playbackGain: GainNode | null = null;
  private puller: ReturnType<typeof createPcmFramePuller> | null = null;
  private decoder: OpusDecoder | null = null;
  private nextPlay = 0;
  private uplinkOpen = false;
  private allowUplinkDuringPlayback = false;
  private ensurePromise: Promise<AudioContext> | null = null;
  private closed = false;

  async ensure(): Promise<AudioContext> {
    if (this.ctx) {
      if (this.ctx.state === "closed") {
        this.teardownContext();
      } else {
        if (this.ctx.state === "suspended") {
          await this.ctx.resume().catch(() => undefined);
        }
        return this.ctx;
      }
    }
    if (this.ensurePromise) {
      return this.ensurePromise;
    }
    this.ensurePromise = this._initAudio();
    try {
      return await this.ensurePromise;
    } finally {
      this.ensurePromise = null;
    }
  }

  private async _initAudio(): Promise<AudioContext> {
    const ctx = new AudioContext();
    const playbackGain = ctx.createGain();
    playbackGain.connect(ctx.destination);
    let decoder: OpusDecoder;
    try {
      decoder = await getSharedOpusDecoder();
    } catch (err) {
      await ctx.close().catch(() => undefined);
      throw err;
    }
    if (this.closed || this.ctx) {
      await ctx.close().catch(() => undefined);
      if (!this.ctx) {
        throw new Error("Audio init superseded");
      }
      return this.ctx;
    }
    this.ctx = ctx;
    this.playbackGain = playbackGain;
    this.decoder = decoder;
    if (ctx.state === "suspended") {
      await ctx.resume().catch(() => undefined);
    }
    return ctx;
  }

  private teardownContext(): void {
    this.stopMic();
    this.ctx = null;
    this.playbackGain = null;
    this.decoder = null;
    this.nextPlay = 0;
  }

  setRealtimeAec(enabled: boolean): void {
    this.allowUplinkDuringPlayback = enabled;
    this._syncUplinkGate();
  }

  private _syncUplinkGate(): void {
    if (!this.micSilent) {
      return;
    }
    const playing = this.nextPlay > (this.ctx?.currentTime ?? 0) + 0.05;
    const allow = this.uplinkOpen && (!playing || this.allowUplinkDuringPlayback);
    this.micSilent.gain.value = allow ? 1 : 0;
  }

  async startMic(
    onFrame: (pcm: Int16Array) => void,
    onLevel?: (rms: number) => void
  ): Promise<void> {
    if (this.micNode) {
      return;
    }
    let ctx = await this.ensure();
    const micStream = await navigator.mediaDevices.getUserMedia({
      audio: {
        channelCount: 1,
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: true,
      },
    });
    if (this.closed || !this.ctx || this.ctx.state === "closed") {
      micStream.getTracks().forEach((t) => t.stop());
      ctx = await this.ensure();
    } else {
      ctx = this.ctx;
    }
    if (this.micNode) {
      micStream.getTracks().forEach((t) => t.stop());
      return;
    }
    this.micStream = micStream;
    const rate = ctx.sampleRate;
    this.puller = createPcmFramePuller(rate);
    const src = ctx.createMediaStreamSource(this.micStream);
    const proc = ctx.createScriptProcessor(4096, 1, 1);
    this.micSilent = ctx.createGain();
    this.micSilent.gain.value = 0;
    this.uplinkOpen = true;
    this._syncUplinkGate();

    proc.onaudioprocess = (ev) => {
      if (!this.ctx || this.ctx.state === "closed") return;
      const mono = ev.inputBuffer.getChannelData(0);
      if (onLevel) {
        let sum = 0;
        for (let i = 0; i < mono.length; i++) {
          const x = mono[i]!;
          sum += x * x;
        }
        onLevel(Math.min(1, Math.sqrt(sum / mono.length) * 4));
      }
      this.puller!.push(mono);
      let frame: Int16Array | null;
      while ((frame = this.puller!.tryPullFrame())) {
        if (this.micSilent!.gain.value > 0) {
          onFrame(frame);
        }
      }
      this._syncUplinkGate();
    };

    src.connect(proc);
    proc.connect(this.micSilent);
    this.micSilent.connect(ctx.destination);
    this.micNode = proc;
  }

  stopMic(): void {
    this.uplinkOpen = false;
    try {
      this.micNode?.disconnect();
    } catch {
      /* ignore */
    }
    this.micNode = null;
    try {
      this.micSilent?.disconnect();
    } catch {
      /* ignore */
    }
    this.micSilent = null;
    this.puller?.clear();
    this.puller = null;
    this.micStream?.getTracks().forEach((t) => t.stop());
    this.micStream = null;
  }

  alignPlaybackSchedule(): void {
    if (!this.ctx || this.ctx.state === "closed") {
      return;
    }
    const now = this.ctx.currentTime;
    this.nextPlay = Math.max(this.nextPlay, now);
    this._syncUplinkGate();
  }

  enqueueOpus(packet: ArrayBuffer): void {
    if (!this.decoder || !this.ctx || !this.playbackGain || this.ctx.state === "closed") {
      return;
    }
    const dec = this.decoder.decodeFrame(new Uint8Array(packet));
    if (dec.samplesDecoded === 0 || !dec.channelData[0]) {
      return;
    }
    const ch = dec.channelData[0];
    const buf = this.ctx.createBuffer(1, ch.length, OPUS_DOWNLINK_RATE);
    buf.copyToChannel(ch, 0);
    this._scheduleBuffer(buf);
  }

  /** REST chat TTS WAV — same output path as Opus downlink for AEC / mobile coherence. */
  async playAudioBuffer(buf: AudioBuffer): Promise<void> {
    if (!this.ctx || !this.playbackGain || this.ctx.state === "closed") {
      return;
    }
    this.alignPlaybackSchedule();
    await new Promise<void>((resolve, reject) => {
      const src = this.ctx!.createBufferSource();
      src.buffer = buf;
      src.connect(this.playbackGain!);
      const startAt = Math.max(this.nextPlay, this.ctx!.currentTime);
      src.onended = () => {
        this._syncUplinkGate();
        resolve();
      };
      try {
        src.start(startAt);
        this.nextPlay = startAt + buf.duration;
        this._syncUplinkGate();
      } catch (e) {
        reject(e instanceof Error ? e : new Error(String(e)));
      }
    });
  }

  private _scheduleBuffer(buf: AudioBuffer): void {
    if (!this.ctx || !this.playbackGain) return;
    const src = this.ctx.createBufferSource();
    src.buffer = buf;
    src.connect(this.playbackGain);
    const now = this.ctx.currentTime;
    if (this.nextPlay < now) {
      this.nextPlay = now;
    }
    src.start(this.nextPlay);
    this.nextPlay += buf.duration;
    this._syncUplinkGate();
    src.onended = () => this._syncUplinkGate();
  }

  dispose(): void {
    this.closed = true;
    this.stopMic();
    void this.ctx?.close().catch(() => undefined);
    this.teardownContext();
  }
}
