# Audio8 TTS ONNX (CPU)

OpenAI-compatible CPU sidecar for [Audio8-TTS-Preview-0.6B-ONNX-INT4](https://huggingface.co/Audio8/Audio8-TTS-Preview-0.6B-ONNX-INT4).

```bash
docker build --platform linux/amd64 -t ghcr.io/coynntis/audio8-tts-onnx:0.6b-int4 .
docker push ghcr.io/coynntis/audio8-tts-onnx:0.6b-int4
```

Runtime:

- `POST /v1/audio/speech` — `{model:"arktts", input, voice, response_format:"wav"}`
- Seed voices: `en_default`, `zh_default`, `yue_default` (from Audio8 demo refs)
- Model download on first start → `ARKTTS_MODEL_DIR` (hostPath in chart)
