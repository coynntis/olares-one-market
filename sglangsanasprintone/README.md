# sglangsanasprintone

This app runs SGLang with `Efficient-Large-Model/Sana_Sprint_1.6B_1024px_diffusers` on port `30000`.

The container startup flow is:

```bash
git clone https://github.com/sgl-project/sglang.git /workspace/sglang
cd /workspace/sglang
pip install -e "python[diffusion]"
sglang serve --model-path Efficient-Large-Model/Sana_Sprint_1.6B_1024px_diffusers --trust-remote-code --host 0.0.0.0 --port 30000
```

Container image: `lmsysorg/sglang:dev-cu13`

Hugging Face credentials are mapped from Olares (`OLARES_USER_HUGGINGFACE_TOKEN` → `HF_TOKEN`, `OLARES_USER_HUGGINGFACE_SERVICE` → `HF_ENDPOINT`) via `OlaresManifest.yaml` `envs`.

Example image generation request:

```bash
curl -X POST http://localhost:30000/v1/images/generations \
  -H "Content-Type: application/json" \
  -d '{"prompt": "A simple geometric logo, flat vector style", "height": 1024, "width": 1024}' \
  --output output.png
```
