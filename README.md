# Orales One Market

A curated app store for the **Olares One** — the first Olares hardware device, packing an RTX 5090M, 96GB DDR5, and a 24-core Intel CPU into a compact form factor.

Every app here is hand-tuned for this exact hardware. No generic configs — just the fastest possible inference on 24GB of GDDR7.

> **One-click install.** Add the market source, and apps appear in your Olares Market alongside the official catalog.

## Apps (16)

### LLM Inference (llama.cpp)

All llama.cpp apps run on **b8667** with **Hadamard rotation** (TurboQuant) and **q4_0 KV cache** for 2x context vs q8_0.

| App | Model | Speed | Context | Details |
|-----|-------|-------|---------|---------|
| llamacppqwen36a3bone | Qwen3.5 35B-A3B | **129 t/s** | 64K | UD-Q4_K_XL, thinking mode |
| gemma426ba4bone | **Gemma 4 26B-A4B** | **119 t/s** | 64K | UD-Q4_K_XL + vision, LMArena 1441 |
| llamacppnemotron30a3bone | Nemotron 3 Nano 30B-A3B | **184 t/s** | 128K | UD-Q4_K_XL, Mamba-2 hybrid |
| cascade230a3bone | Nemotron Cascade 2 30B-A3B | — | 64K | Q4_K_S, math/code specialist |
| qwen36a3bvisionone | Qwen3.5 35B-A3B Vision | **131 t/s** | 32K | UD-Q4_K_XL + mmproj, multimodal |
| qwen35iq4visionone | Qwen3.5 35B-A3B Vision IQ4 | — | 32K | IQ4_XS + mmproj, vision |
| llamacppqwen35iq4one | Qwen3.5 35B-A3B IQ4 | — | — | IQ4_XS, compact |
| llamacppglm47flash | GLM-4.7 Flash | — | 32K | GLM-4 bilingual |

### LLM Inference (vLLM)

| App | Model | Details |
|-----|-------|---------|
| vllmqwen3527bone | Qwen3.5 27B | NVFP4, speculative decoding |

### Voice & Audio (vLLM)

| App | Model | Function | Details |
|-----|-------|----------|---------|
| vllmvoxtral3bone | **Voxtral Mini 3B** | ASR / Audio understanding | 2.7x faster than Whisper, 3.2% WER |
| vllmvoxtralrt4bone | **Voxtral Realtime 4B** | Streaming ASR | Real-time WebSocket, 480ms latency |
| vllmvoxtraltts4bone | **Voxtral 4B TTS** | Text-to-Speech | 20 voices, 9 languages, 70ms latency |

### Creative

| App | Model | Details |
|-----|-------|---------|
| qwen3ttstone | Qwen3-TTS 1.7B | 9 voices, zero-shot voice clone |

### Other

| App | Model | Details |
|-----|-------|---------|
| exl3qwen36a3bone | Qwen3.5 35B-A3B | ExLlamaV3 + TabbyAPI |
| nemotron3nano4bone | Nemotron 3 Nano 4B | Lightweight 4B model |
| devstralsmallone | Devstral Small 24B | Coding agent |

## Quick Start

Add this URL as a market source in **Olares Market > Settings > Add Source**:

```
https://orales-one-market.aamsellem.workers.dev
```

Apps show up within 5 minutes. Install like any other Olares app — models download automatically on first launch.

## Highlights

- **TurboQuant rotation** (Hadamard) on all llama.cpp apps — q4_0 KV cache with same quality as q8_0, enabling 2x context
- **Gemma 4 26B-A4B** — Google's latest MoE with native vision, 119 t/s
- **Voxtral family** — complete voice pipeline: ASR + streaming ASR + TTS
- **128K context** on Nemotron (Mamba-2 hybrid = tiny KV cache)
- **64K context** on Qwen3.5 and Gemma 4

## How It Works

A single Cloudflare Worker serves the full Olares Market Source API. Each app is a Helm chart with GPU-optimized configs, packaged and deployed from this repo.

```bash
npm install
npm run dev              # Local dev (localhost:8787)
npm run deploy           # Deploy to Cloudflare Workers
```

### API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/v1/appstore/hash?version=X` | Catalog hash for cache check |
| GET | `/api/v1/appstore/info?version=X` | Full catalog with app summaries |
| POST | `/api/v1/applications/info` | App details (batched by ID) |
| GET | `/api/v1/applications/{name}/chart?fileName=X` | Chart `.tgz` download |
| GET | `/icons/{name}.png` | App icon |

## Related

- [orales-market](https://github.com/aamsellem/orales-market) — Generic apps for any Olares hardware

## License

MIT
