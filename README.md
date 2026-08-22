# Olares One Market

Custom [Olares Market](https://docs.olares.com) source for **Olares One** — Helm charts in this repo, catalog served by a Cloudflare Worker you deploy yourself.

After `npm run deploy`, add your worker URL in Olares: **Market → Settings → Add source**.

## Target hardware

| Component | Spec |
|-----------|------|
| GPU | NVIDIA RTX 5090M (24 GB GDDR7, sm_120 Blackwell) |
| CPU | Intel Core Ultra 9 275HX (24 cores) |
| RAM | 96 GB DDR5 |

Charts tune quantization, context size, and GPU layers for this stack. Generic Olares hardware apps live in the sibling [`orales-market`](https://github.com/coynntis/orales-market) repo.

## Apps (92)

Grouped by market category. Full metadata: [`src/catalog.json`](src/catalog.json).

### LLM Chat (25)

| App | Description |
|-----|-------------|
| [colibridsv4flash0731one](colibridsv4flash0731one/) | EXPERIMENTAL DeepSeek-V4-Flash-0731 — Colibri SSD expert stream |
| [llamacppbonsai27bone](llamacppbonsai27bone/) | Bonsai-27B Q1_0 VLM — dense 27B + vision at ~4.5GB |
| [llamacppdsv4flash0731one](llamacppdsv4flash0731one/) | EXPERIMENTAL DeepSeek-V4-Flash-0731 IQ2_XXS — cpu-moe (DSpark off) |
| [llamacppgrug27bone](llamacppgrug27bone/) | grug-27b — short-think Qwen3.6-27B, MTP Q4_K_M, turbo3 KV, vision |
| [llamacppkatcoderv25one](llamacppkatcoderv25one/) | KAT-Coder-V2.5-Dev — fastest coding profile: MTP, long ctx |
| [llamacpplagunas21one](llamacpplagunas21one/) | EXPERIMENTAL Laguna S 2.1 Q4_K_M — cpu-moe (DFlash off by default) |
| [llamacppnanbeige423bone](llamacppnanbeige423bone/) | Nanbeige4.2-3B Q8_0 agent LLM — Nanbeige llama.cpp on Olares One |
| [llamacppnemotrondiffusion14b1](llamacppnemotrondiffusion14b1/) | NVIDIA Nemotron-Labs-Diffusion 14B — tri-mode (AR + parallel diffusion + self-spec) |
| [llamacppqwable35bone](llamacppqwable35bone/) | Qwable-v1 IQ4_XS MoE — Opus+Fable distill, buun MTP, GGUF embedded template, text-only 192K |
| [llamacppqwen36a3bdflashone](llamacppqwen36a3bdflashone/) | Qwen3.6 35B-A3B DFlash + Vision — 302 t/s @ 128K, upstream llama.cpp (no fork) |
| [llamacppqwen36a3bone](llamacppqwen36a3bone/) | Qwen3.6 35B-A3B MTP + Vision — GGUF Q3_K_XL, turbo4 KV, MTP n=5, 192K on Olares One |
| [llamacppqwen36beellamaone](llamacppqwen36beellamaone/) | Qwen3.6 27B fastest + longest local — 106 t/s @ FULL 262K via BeeLlama.cpp 0.1.3-rc3 + DFlash + turbo3 KV |
| [llamacppqwen36fable27bone](llamacppqwen36fable27bone/) | Qwen3.6 27B Fable Fusion 711 — MTP Q4_K_M, turbo3 KV, vision, tools |
| [llamacppqwen36mtpone](llamacppqwen36mtpone/) | Qwen3.6 27B longest context — 74 t/s / 86.7% accept @ full 262K (unsloth UD-Q3_K_XL + MTP) |
| [llamacppqwen3827bmtpone](llamacppqwen3827bmtpone/) | Qwen3.8 27B MTP + Vision — Unsloth UD-Q3 default, optional abliterated 12GB / Q4 / Ridge |
| [llamacppqwopus27coder1](llamacppqwopus27coder1/) | Qwopus3.6-27B-Coder Q5_K_M — agentic coding, buun llama.cpp, 128K |
| [llamacppqwopus27mtpone](llamacppqwopus27mtpone/) | Qwopus3.6-27B-v2-MTP Q4_K_M — draft-MTP speculative decode, buun |
| [llamacppqwythos9bone](llamacppqwythos9bone/) | Qwythos 9B Mythos — 64K ctx, buun turbo4, draft-MTP n=2 |
| [llamacpptbonsai27bone](llamacpptbonsai27bone/) | Ternary-Bonsai-27B Q2_0 VLM — PrismML + Web UI |
| [nemotronlabselastic30bnvfp4one](nemotronlabselastic30bnvfp4one/) | NVIDIA Nemotron-Labs Elastic 30B-A3B NVFP4 — ~181 t/s via vLLM v0.22.1 on Olares One |
| [sndrdiffusiongemma26bone](sndrdiffusiongemma26bone/) | SNDR Core Engine — DiffusionGemma 26B FP8 (experimental 1x) |
| [sndrgemma426ba4bone](sndrgemma426ba4bone/) | SNDR Core Engine — Gemma 4 26B-A4B AWQ (1x adapt) |
| [sndrqwen3627bone](sndrqwen3627bone/) | SNDR Core Engine — Qwen3.6 27B INT4 + TQ k8v4 + MTP K=4 (1x) |
| [sndrqwen3635ba3bone](sndrqwen3635ba3bone/) | EXPERIMENTAL — prefer llamacppqwen36a3bone. SNDR FP8 35B cannot do MTP K=5 on 1x24GB |
| [vllmgemma4dflashone](vllmgemma4dflashone/) | Gemma 4 26B-A4B + DFlash + AWQ via vLLM — Olares One (fastest Gemma 4 path) |

### Vision (11)

| App | Description |
|-----|-------------|
| [llamacppagentsa1one](llamacppagentsa1one/) | Agents-A1 MoE agentic VLM — fast vision + long context on Olares One |
| [llamacppnail35bone](llamacppnail35bone/) | Nail Qwen3.6-35B-A3B MTP vision profile — fast long context on Olares One |
| [llamacppqwen36beellamavision1](llamacppqwen36beellamavision1/) | Qwen3.6 27B vision-capable — 105 t/s @ 200K via BeeLlama.cpp + mmproj F16 + DFlash |
| [llamacppthinkingcap27bone](llamacppthinkingcap27bone/) | ThinkingCap Qwen3.6 27B vision — hybrid MoE via llama.cpp b8740 on Olares One |
| [vllmgemma426ba4bvisionone](vllmgemma426ba4bvisionone/) | Gemma 4 26B-A4B vision + MTP via vLLM — 250 t/s @ 128K native + vision + tool calling |
| [vllmgemma431bnvfp4one](vllmgemma431bnvfp4one/) | Gemma 4 31B IT NVFP4 vision — vLLM TurboQuant KV + llama.cpp web UI on Olares One |
| [vllmgemma4e4bone](vllmgemma4e4bone/) | Gemma 4 E4B (Vision + Audio) via vLLM — optimized for Olares One |
| [vllmqwen3627bnvfp4one](vllmqwen3627bnvfp4one/) | Qwen3.6 27B NVFP4 vision — vLLM TurboQuant KV + llama.cpp web UI on Olares One |
| [vllmqwen3635bnvfp4fone](vllmqwen3635bnvfp4fone/) | Qwen3.6 35B-A3B NVFP4-Fast — vLLM TurboQuant KV + llama.cpp web UI on Olares One |
| [vllmqwen3827bnvfp4one](vllmqwen3827bnvfp4one/) | Qwen3.8 27B NVFP4 vision+video — vLLM fp8 KV, medium thinking defaults |
| [vllmtess427bone](vllmtess427bone/) | Tess-4-27B vision + agentic reasoning via vLLM — Qwen3.6 multimodal on Olares One |

### AI (43)

| App | Description |
|-----|-------------|
| [consistcomposeone](consistcomposeone/) | ConsistCompose BAGEL-7B-MoT layout compose — NF4 for 24GB |
| [dflashqwen3627bone](dflashqwen3627bone/) | Qwen3.6-27B DFlash DDTree on Olares One sm_120 |
| [fastwanqad13bone](fastwanqad13bone/) | FastWan QAD 1.3B flagship — ~1.8s lightning T2V on Blackwell NVFP4 |
| [fastwanqad13bsa2one](fastwanqad13bsa2one/) | FastWan QAD 1.3B SA2 — ~2s lightning T2V on Blackwell NVFP4 |
| [fastwanqad13fp8one](fastwanqad13fp8one/) | FastWan QAD FP8 — 4090/Ada fallback when NVFP4 OOM (~3.4s T2V) |
| [gemma4e2bone](gemma4e2bone/) | Gemma 4 E2B (ultra-fast 2.3B) via llama.cpp — voice pipeline brain for Olares One |
| [ideogram4nf4one](ideogram4nf4one/) | Ideogram 4 NF4 + fal Instant — open text-to-image for Olares One |
| [krea2turboone](krea2turboone/) | Krea-2-Turbo — fast T2I Gradio + REST for Olares One |
| [lingbotdepthone](lingbotdepthone/) | Depth completion Gradio + REST (ViT-L v0.5) |
| [lingbotmapone](lingbotmapone/) | Streaming 3D reconstruction (viser + REST) |
| [lingbotvaone](lingbotvaone/) | Video-Action world model (UMT5+VAE CPU offload) |
| [lingbotvideoone](lingbotvideoone/) | T2V / TI2V Gradio (Dense 1.3B single-GPU) |
| [lingbotvisionone](lingbotvisionone/) | Vision backbone PCA Gradio + REST (ViT-L) |
| [lingbotvlaone](lingbotvlaone/) | VLA 6B policy API + Gradio (native depth) |
| [lingbotworldone](lingbotworldone/) | World I2V NF4 + T5 CPU / layer offload (24GB) |
| [llamacppbonsai8bone](llamacppbonsai8bone/) | Bonsai-8B llama-server API — CUDA 13.1, PrismML source build on Olares One |
| [llamacppdiffusiongemma26a4bone](llamacppdiffusiongemma26a4bone/) | DiffusionGemma 26B-A4B Instruct via vLLM — OpenAI API on Olares One |
| [llamacppgemma412agent1](llamacppgemma412agent1/) | Gemma4-12B agentic Fable5 v2 Q4_K_M — tool calling, llama.cpp b8740 |
| [llamacppornith35bone](llamacppornith35bone/) | Ornith 1.0 35B Q4_K_M — dense reasoning via llama.cpp b8740 |
| [llamacppornith9bone](llamacppornith9bone/) | Ornith 1.0 9B Q4_K_M — compact dense model via llama.cpp b8740 |
| [llamacppqwen3627btq34sone](llamacppqwen3627btq34sone/) | Qwen3.6-27B TQ3_4S via llama.cpp-tq3 on Olares One |
| [llamacppqwen3635ba3btq34sone](llamacppqwen3635ba3btq34sone/) | Qwen3.6-35B-A3B Vision TQ3_4S via llama.cpp-tq3 on Olares One |
| [locateanything3bone](locateanything3bone/) | LocateAnything-3B visual grounding (Gradio + REST) |
| [ltx23one](ltx23one/) | LTX-2.3 distilled FP8 T2V/I2V for Olares One 24GB |
| [mageflowone](mageflowone/) | Mage-Flow Turbo + Edit Turbo — T2I and instruction edit on Olares One |
| [minimaxh3nvfp4one](minimaxh3nvfp4one/) | MiniMax H3 Gradio video and audio generation on Olares One |
| [motifvideo2bone](motifvideo2bone/) | Motif Video 2B GGUF Gradio + REST for text-to-video and image-to-video |
| [pipecatxiaozhione](pipecatxiaozhione/) | Pipecat voice bridge compatible with xiaozhi-esp32 WebSocket protocol |
| [qwen36a3bvisionone](qwen36a3bvisionone/) | Qwen3.6 35B-A3B Vision via llama.cpp — image + text, optimized for Olares One |
| [qwen3ttstone](qwen3ttstone/) | Qwen3-TTS 1.7B — text-to-speech with voice cloning, optimized for Olares One |
| [sensenovasi15one](sensenovasi15one/) | SenseNova-SI-1.5 spatial VQA — vision understand only not image gen |
| [sensenovau1infov2one](sensenovau1infov2one/) | SenseNova-U1 Infographic-V2 specialist — charts, posters, structured layouts |
| [sensenovau1lightllmone](sensenovau1lightllmone/) | SenseNova-U1 text-to-image via LightLLM + LightX2V FA3 — Olares One benchmark |
| [sensenovau1serveone](sensenovau1serveone/) | SenseNova-U1 T2I, VQA, editing, and interleave via REST API and Gradio |
| [sensenovavisionone](sensenovavisionone/) | SenseNova-Vision 7B MoT Gradio + REST |
| [sglangernieimageone](sglangernieimageone/) | ERNIE-Image via SGLang — image generation optimized for Olares One |
| [sglangkrea2turboone](sglangkrea2turboone/) | Krea-2-Turbo via SGLang Diffusion — layerwise DiT offload for 24GB |
| [sglanglfm258ba1bone](sglanglfm258ba1bone/) | LFM2.5-8B-A1B MoE via SGLang — agentic tool use and reasoning on Olares One |
| [sglangltx23one](sglangltx23one/) | LTX-2.3 via SGLang Diffusion — layerwise DiT offload for 24GB T2V |
| [sglangminicpm51bone](sglangminicpm51bone/) | MiniCPM5-1B via SGLang — tool use, coding, hybrid reasoning on Olares One |
| [sglangsanasprintone](sglangsanasprintone/) | Sana Sprint 1.6B 1024px via SGLang — image generation for Olares One |
| [splatlabone](splatlabone/) | 3D Gaussian Splatting pipeline — import, reconstruct, view splats on Olares One |
| [vllmgemma31bitnvfp4one](vllmgemma31bitnvfp4one/) | Gemma 4 31B IT NVFP4 Turbo via vLLM modelopt on Olares One |

### TTS (6)

| App | Description |
|-----|-------------|
| [cosyvoice2yueone](cosyvoice2yueone/) | CosyVoice2 Yue TTS — Hong Kong Cantonese zero-shot clone, OpenAI API |
| [cosyvoice3one](cosyvoice3one/) | Fun-CosyVoice3-0.5B TTS — 9 languages, 18 Chinese dialects including Cantonese |
| [mossttslocalone](mossttslocalone/) | MOSS-TTS-Local-Transformer-v1.5 — 31 languages incl. Cantonese, clone, 48kHz stereo |
| [omnivoiceone](omnivoiceone/) | OmniVoice TTS (646 languages, voice cloning, voice design, 0.6B) |
| [vllmgepardone](vllmgepardone/) | Gepard 1.0 streaming TTS via vLLM — Cartesia-compatible API |
| [voxcpmone](voxcpmone/) | VoxCPM2 TTS — multilingual tokenizer-free speech, voice clone |

### STT (1)

| App | Description |
|-----|-------------|
| [sensevoiceone](sensevoiceone/) | SenseVoice STT — Cantonese yue ASR on CPU, OpenAI-compatible API |

### Audio (1)

| App | Description |
|-----|-------------|
| [vllmnemotronaudex30bone](vllmnemotronaudex30bone/) | Nemotron-Labs Audex 30B-A3B — unified audio+speech multimodal MoE via vLLM |

### AI Agents (1)

| App | Description |
|-----|-------------|
| [llamacppagentworld35bone](llamacppagentworld35bone/) | Qwen AgentWorld 35B-A3B — fast long-context text world-model on Olares One |

### Developer (4)

| App | Description |
|-----|-------------|
| [browserlessone](browserlessone/) | Browserless headless Chromium for Puppeteer and Playwright on Olares One |
| [dockerbuilderone](dockerbuilderone/) | Build Docker images from uploaded Dockerfile folders and push to ghcr.io |
| [dshone](dshone/) | DeepSeek Harness Web UI — upgrade dsh from Settings without a chart bump |
| [openwebsearchone](openwebsearchone/) | Multi-engine web search MCP server for Olares One agents — no API keys |

### Backend summary

| Backend | Count | Examples |
|---------|------:|----------|
| **llama.cpp** | 35 | llamacppqwen36a3bone, llamacppqwen3827bmtpone, qwen36a3bvisionone |
| **vLLM** | 14 | vllmgemma4e4bone, vllmqwen3827bnvfp4one, vllmgepardone |
| **SGLang** | 6 | sglanglfm258ba1bone, sglangernieimageone, sglangltx23one |
| **Diffusion / video** | 12 | fastwanqad13bone, motifvideo2bone, ltx23one, lingbotvideoone |
| **TTS / STT / voice** | 9 | omnivoiceone, cosyvoice2yueone, sensevoiceone, pipecatxiaozhione |
| **Other** | 16 | splatlabone, dockerbuilderone, browserlessone, locateanything3bone |

## GitHub Container Registry (`ghcr.io/coynntis`)

Several apps pull pre-built images from `ghcr.io/coynntis`. Packages may be private — use a GitHub PAT with **`read:packages`** scope.

**On Olares:** Settings → Integrations → set **GitHub token** and **GitHub username** (the account that owns the PAT). Charts wire these as `imagePullSecret` automatically via `OLARES_USER_GITHUB_TOKEN` / `OLARES_USER_GITHUB_USERNAME`.

**Manual pull** (replace `TAG` with the chart's pinned tag in `values.yaml`):

```bash
echo "$GITHUB_TOKEN" | docker login ghcr.io -u YOUR_GITHUB_USERNAME --password-stdin
docker pull ghcr.io/coynntis/locate-anything:0.1.2
docker pull ghcr.io/coynntis/pipecat-xiaozhi-bridge:0.5.4
docker pull ghcr.io/coynntis/audio8-tts-onnx:0.6b-int4
docker pull ghcr.io/coynntis/splatlabone:1.1.3
docker pull ghcr.io/coynntis/dsh-runtime:22.19-1
```

Alternatively, make the package **Public** under GitHub → Packages → Package settings.

| Image | Used by |
|-------|---------|
| `ghcr.io/coynntis/locate-anything` | locateanything3bone |
| `ghcr.io/coynntis/pipecat-xiaozhi-bridge` | pipecatxiaozhione |
| `ghcr.io/coynntis/audio8-tts-onnx` | pipecatxiaozhione (CPU TTS sidecar) |
| `ghcr.io/coynntis/splatlabone` | splatlabone |
| `ghcr.io/coynntis/dsh-runtime` | dshone |

Other apps pull from public registries (`ghcr.io/ggml-org/llama.cpp`, `docker.io`, Hugging Face, etc.) and may need `OLARES_USER_HUGGINGFACE_TOKEN` for gated model downloads.

## Chart conventions

GPU and API-serving apps use **server/client split** (Ollama-style shared entrance):

- Root chart: server subchart (`shared: true`) + client proxy subchart
- Shared API: `http://<route-id>.shared.olares.com/v1`
- User-configurable envs: `LLM_CONTEXT_WINDOW`, `LLM_MAX_OUTPUT_TOKENS`, `LLM_API_KEY`, `LLM_REASONING`, `LLM_CHAT_TEMPLATE_SOURCE`

See [CLAUDE.md](CLAUDE.md) for Olares One tuning notes (llama.cpp flags, VRAM budgets, sampling params).

## Icons & featured assets

Card PNGs (512×512) generate under `icons/` (gitignored). The worker serves base64 assets from [`src/icons.json`](src/icons.json) and [`src/featured.json`](src/featured.json) after build.

```bash
python3 -m venv .venv-icons   # once
.venv-icons/bin/pip install -r scripts/requirements-icons.txt
npm run generate:icons          # PNGs + CDN upload + src/icons.json
npm run build:catalog           # refresh catalog + featured
```

App list metadata: [`scripts/apps-icons.json`](scripts/apps-icons.json).

## Build & deploy

```bash
npm install
npm run build          # package charts → src/charts.json, then catalog
npm run dev            # localhost:8787
npm run deploy         # Cloudflare Workers (requires wrangler auth)
```

Optional: if you serve `featured/*.png` from your worker, set `MARKET_BASE_URL` when building the catalog:

```bash
MARKET_BASE_URL=https://your-worker.workers.dev npm run build:catalog
```

Otherwise featured images come from CDN URLs in each app's `OlaresManifest.yaml` (via `npm run generate:icons`).

Requires Olares **≥ 1.12.3** (v3 shared charts need **≥ 1.12.6**).

## Layout

```
olares-one-market/
├── <app-name>/              Chart.yaml + OlaresManifest.yaml + templates/
│   ├── <app-name>/          Client subchart (nginx proxy)
│   └── <app-name>srv/       Server subchart (GPU workload)
├── scripts/
│   ├── build-catalog.js     Parse charts → src/catalog.json
│   ├── package-charts.js    Helm package → src/charts.json
│   └── inject-*.js          ConfigMap injectors (called at build time)
├── shared/                  Shared chat templates, etc.
├── src/
│   ├── index.ts             Worker API (hash / info / applications)
│   ├── catalog.json         Generated
│   ├── charts.json          Generated (base64 tgz)
│   ├── icons.json           Generated
│   └── featured.json        Generated
└── wrangler.toml
```

## Author

**coynntis** — [github.com/coynntis/olares-one-market](https://github.com/coynntis/olares-one-market)

Contributions welcome via pull request. This repo is the public market source; do not commit secrets, tokens, or local venvs.
