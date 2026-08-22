# Olares One Market

Custom [Olares Market](https://docs.olares.com) source for **Olares One** (RTX 5090M, 96GB DDR5, Core Ultra 9 275HX). Helm charts live in this repo; a Cloudflare Worker serves the catalog API.

**Live:** https://orales-one-market.coynntis.workers.dev

Add in Olares: **Market → Settings → Add source** → paste the URL above.

## Apps (20)

| App | Title | Backend |
|-----|-------|---------|
| [browserlessone](browserlessone/) | Browserless One | Browserless |
| [dflashqwen3627bone](dflashqwen3627bone/) | Qwen36 27B DFlash | Lucebox DFlash (native CUDA) |
| [dockerbuilderone](dockerbuilderone/) | Docker Builder One | Kaniko + FastAPI + MCP |
| [gemma4e2bone](gemma4e2bone/) | Gemma 4 E2B Voice Brain | llama.cpp |
| [llamacppbonsai8bone](llamacppbonsai8bone/) | Bonsai 8B | llama.cpp |
| [llamacppqwen3627btq34sone](llamacppqwen3627btq34sone/) | Qwen36 27B TQ34S | llama.cpp (turbo-tan tq3) |
| [llamacppqwen3635ba3btq34sone](llamacppqwen3635ba3btq34sone/) | Qwen35 35B Vision TQ34S | llama.cpp (turbo-tan tq3) |
| [locateanything3bone](locateanything3bone/) | LocateAnything 3B | PyTorch + MagiAttention |
| [motifvideo2bone](motifvideo2bone/) | Motif Video 2B | PyTorch Gradio |
| [omnivoiceone](omnivoiceone/) | OmniVoice TTS | OmniVoice |
| [openwebsearchone](openwebsearchone/) | Open WebSearch | Open Web Search (MCP/HTTP) |
| [pipecatxiaozhione](pipecatxiaozhione/) | Pipecat Xiaozhi Bridge | Pipecat |
| [qwen36a3bvisionone](qwen36a3bvisionone/) | Qwen3.6 Vision | llama.cpp |
| [qwen3ttstone](qwen3ttstone/) | Qwen3-TTS 1.7B | Qwen3-TTS |
| [sensenovau1serveone](sensenovau1serveone/) | SenseNova U1 Serve | SenseNova (PyTorch) |
| [sglangernieimageone](sglangernieimageone/) | SGLang ERNIE-Image | SGLang |
| [sglanglfm258ba1bone](sglanglfm258ba1bone/) | LFM2-5 8B | SGLang |
| [sglangminicpm51bone](sglangminicpm51bone/) | MiniCPM5 1B | SGLang |
| [sglangsanasprintone](sglangsanasprintone/) | Sana Sprint 1.6B | SGLang |
| [vllmgemma31bitnvfp4one](vllmgemma31bitnvfp4one/) | Gemma 4 31B NVFP4 | vLLM |

### Backend summary

| Backend | Apps |
|---------|------|
| **llama.cpp** | gemma4e2bone, llamacppbonsai8bone, llamacppqwen3627btq34sone, llamacppqwen3635ba3btq34sone, qwen36a3bvisionone |
| **SGLang** | sglangernieimageone, sglanglfm258ba1bone, sglangminicpm51bone, sglangsanasprintone |
| **vLLM** | vllmgemma31bitnvfp4one |
| **Other** | browserlessone, dflashqwen3627bone, dockerbuilderone, locateanything3bone, motifvideo2bone, omnivoiceone, openwebsearchone, pipecatxiaozhione, qwen3ttstone, sensenovau1serveone |

## Icons

Card PNGs (512×512, title + backend badge) are generated under `icons/` (gitignored). The worker serves them from generated [`src/icons.json`](src/icons.json) after `npm run build:catalog`. App list metadata: [`scripts/apps-icons.json`](scripts/apps-icons.json).

Regenerate after adding an app:

```bash
python3 -m venv .venv-icons   # once
.venv-icons/bin/pip install pillow
.venv-icons/bin/python scripts/generate-app-icons.py
npm run build:catalog           # refresh src/icons.json for the worker
```

Edit app list in [`scripts/generate-app-icons.py`](scripts/generate-app-icons.py) (`APPS` dict), then rerun.

## Build & deploy

```bash
npm install
npm run build:catalog   # src/catalog.json + src/charts.json + src/icons.json
npm run dev             # localhost:8787
npm run deploy          # Cloudflare Workers
```

## Layout

```
olares-one-market/
├── <app-name>/          Chart.yaml + OlaresManifest.yaml + templates/
├── icons/               Market card PNGs (served by worker)
├── scripts/
│   ├── build-catalog.js
│   └── generate-app-icons.py
├── src/
│   ├── index.ts         Worker API
│   ├── catalog.json     Generated
│   ├── charts.json      Generated
│   └── icons.json       Generated (base64)
└── wrangler.toml
```

See [CLAUDE.md](CLAUDE.md) for Olares One tuning notes (llama.cpp flags, VRAM, etc.).
