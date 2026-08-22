import { createPcmFramePuller } from "./resample";

export type PcmFrameHandler = (pcmS16le: Int16Array) => void;
export type LevelHandler = (rms01: number) => void;

/**
 * Capture microphone, resample to 24 kHz mono s16le, emit 60 ms frames (xiaozhi uplink_encoding pcm_s16le).
 */
export class PcmMicStream {
  private ctx: AudioContext | null = null;
  private node: ScriptProcessorNode | null = null;
  private silent: GainNode | null = null;
  private stream: MediaStream | null = null;
  private puller: ReturnType<typeof createPcmFramePuller> | null = null;

  constructor(
    private readonly onFrame: PcmFrameHandler,
    private readonly onLevel?: LevelHandler
  ) {}

  async start(): Promise<void> {
    if (this.ctx) {
      return;
    }
    this.stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        channelCount: 1,
        echoCancellation: true,
        noiseSuppression: true,
      },
    });
    this.ctx = new AudioContext();
    const rate = this.ctx.sampleRate;
    this.puller = createPcmFramePuller(rate);

    const src = this.ctx.createMediaStreamSource(this.stream);
    const proc = this.ctx.createScriptProcessor(4096, src.channelCount, 1);

    proc.onaudioprocess = (ev) => {
      const inBuf = ev.inputBuffer;
      const n = inBuf.length;
      const mono = new Float32Array(n);
      if (inBuf.numberOfChannels === 1) {
        mono.set(inBuf.getChannelData(0));
      } else {
        for (let i = 0; i < n; i++) {
          let s = 0;
          for (let c = 0; c < inBuf.numberOfChannels; c++) {
            s += inBuf.getChannelData(c)[i]!;
          }
          mono[i] = s / inBuf.numberOfChannels;
        }
      }

      if (this.onLevel) {
        let sum = 0;
        for (let i = 0; i < mono.length; i++) {
          const x = mono[i]!;
          sum += x * x;
        }
        this.onLevel(Math.min(1, Math.sqrt(sum / mono.length) * 4));
      }

      this.puller!.push(mono);
      let frame: Int16Array | null;
      while ((frame = this.puller!.tryPullFrame())) {
        this.onFrame(frame);
      }
    };

    this.silent = this.ctx.createGain();
    this.silent.gain.value = 0;
    src.connect(proc);
    proc.connect(this.silent);
    this.silent.connect(this.ctx.destination);
    this.node = proc;

    if (this.ctx.state === "suspended") {
      await this.ctx.resume();
    }
  }

  stop(): void {
    try {
      this.node?.disconnect();
    } catch {
      /* ignore */
    }
    this.node = null;
    try {
      this.silent?.disconnect();
    } catch {
      /* ignore */
    }
    this.silent = null;
    this.puller?.clear();
    this.puller = null;
    this.stream?.getTracks().forEach((t) => t.stop());
    this.stream = null;
    void this.ctx?.close();
    this.ctx = null;
  }
}
