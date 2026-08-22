# Xiaozhi voice web client

React + Vite UI for live sessions against `pipecat-xiaozhi-bridge` (xiaozhi WebSocket `/xiaozhi/v1/`).

## Features

- Connect / disconnect, **hold-to-talk** (mouse or **Space**), input level meter
- **PCM uplink** (`pcm_s16le` in hello) resampled to 24 kHz / 60 ms frames (browser-friendly)
- **Opus downlink** via WASM `opus-decoder`, scheduled on `AudioContext`
- **Text** path: `listen` + `state: detect` + `text`
- **Abort** button
- Session log (STT / assistant text / system)

## Dev

Requires the Python bridge on port 8000. Vite proxies WebSocket `ws://localhost:5173/xiaozhi/...` → `ws://127.0.0.1:8000`.

```bash
npm install
npm run dev
```

## Production build

```bash
npm run build
```

Upload `dist/` to static hosting. Users must enter a full `wss://host/xiaozhi/v1` URL (no dev proxy).

## Pipecat

This is **not** the Pipecat JS SDK transport (no Small WebRTC here). UX is aligned with ideas from the [Pipecat voice UI guide](https://docs.pipecat.ai/client/guides/building-a-voice-ui); wire format remains **xiaozhi** for compatibility with your bridge and ESP32 tools.
