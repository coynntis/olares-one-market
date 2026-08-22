# Pipecat Xiaozhi bridge

WebSocket server + **built-in web app** for [xiaozhi-esp32](https://github.com/78/xiaozhi-esp32) protocol at `/xiaozhi/v1/`.

## Web UI (served from chart)

Open the Olares app entrance → three tabs:

| Tab | Purpose |
|-----|---------|
| **Chat** | Text + image attachments → vision LLM; optional TTS playback |
| **Voice** | Push-to-talk xiaozhi session (STT → LLM → TTS → Opus) |
| **Settings** | Paste STT/TTS/LLM shared entrance URLs; persisted in app data |

Settings API: `GET/PUT /api/config` · Chat: `POST /api/chat` · Status: `GET /api/config/status`

Config file: `CONFIG_PATH` (default `/data/config.json` on Olares).

## Pipeline

1. **Opus/PCM** mic from ESP32 or web Voice tab
2. **STT** — OpenAI-compatible `audio/transcriptions` (SenseVoice: set language `yue`)
3. **LLM** — Pipecat + OpenAI fallback
4. **TTS** — `audio/speech` with `wav` (OmniVoice) → decode/resample → Opus downlink

## Run locally

```bash
cd pipecat-xiaozhi-bridge
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export CONFIG_PATH=/tmp/xiaozhi-config.json
python -m xiaozhi_bridge
```

```bash
cd client && npm install && npm run dev
```

Configure pipeline in **Settings** tab at `http://localhost:5173`.

## Docker

Olares is **linux/amd64** only. On Apple Silicon Mac, always build with `--platform`:

```bash
docker buildx build --platform linux/amd64 \
  -t ghcr.io/coynntis/pipecat-xiaozhi-bridge:0.2.2 --push .
```

## Olares chart

`pipecatxiaozhione/` — no hardcoded service URLs; configure via web UI after install.
