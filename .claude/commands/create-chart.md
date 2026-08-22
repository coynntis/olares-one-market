# Create Helm Chart for Olares One

Create a complete Helm chart for a model app optimized for Olares One, ready for Market admin install.

## Argument: $ARGUMENTS

The argument should describe what to build:
- A model + backend (e.g., "Qwen3.5-35B-A3B llama.cpp UD-Q4_K_XL")
- Or reference a previous /research-model output
- Or "from-docker <docker run command>" to convert a working Docker command into a chart

## Olares One Constraints (MUST follow)

- `olaresManifest.version: '0.10.0'`
- `apiVersion: 'v2'` at top level of OlaresManifest.yaml
- CPU values: integer cores (NOT millicores)
- Entrance title: max 30 chars, only `[a-z0-9A-Z-\s]`, NO parentheses
- Proxy image: `beclab/aboveos-bitnami-openresty:1.25.3-2`
- Olares dependency: `>=1.12.3-0`
- **All entrances** (including MCP): `authLevel: internal` — never `public`
- Subchart / entrance names: **max 30 chars**

## New App Defaults (MUST include — see CLAUDE.md)

Every new chart **must** ship with these from day one:

### Shared entrance (server/client split)

```
<app-name>/
├── Chart.yaml                    ← umbrella, lists subCharts
├── OlaresManifest.yaml           ← sharedEntrances + envs + subCharts
├── values.yaml
├── owners
├── .helmignore
├── i18n/en-US/OlaresManifest.yaml
├── templates/
│   ├── keep
│   └── clientproxy.yaml          ← lint safety net (Deployment name == app name)
├── <app-name>/                   ← client subchart (nginx proxy)
│   ├── Chart.yaml
│   ├── values.yaml
│   └── templates/clientproxy.yaml
└── <app-name>srv/                ← server subchart (GPU, shared)
    ├── Chart.yaml
    ├── values.yaml               ← must include olaresEnv: {}
    └── templates/deployment.yaml
```

Manifest requirements:
- `subCharts`: server `shared: true` + client (no `shared` on client)
- `sharedEntrances`: `host: sharedentrances-<app>`, `port: 0` (integer), `invisible: true`, `authLevel: internal`
- `options.apiTimeout: 0`
- Non-admin: mandatory self-dependency on client chart

Reference: `llamacppqwen36beellamaone/`. Scripts: `scripts/split-server-client.js`, `scripts/align-shared-apps.js`.

**Do NOT use Studio devbox** for v2 shared apps — use Market admin install.

### User env vars (Settings → Application)

**LLM serving apps** (llama.cpp / vLLM / SGLang text) — add to `envs:`:

```yaml
  - envName: LLM_CONTEXT_WINDOW
    required: false
    editable: true
    applyOnChange: true
  - envName: LLM_MAX_OUTPUT_TOKENS
    required: false
    editable: true
    applyOnChange: true
  - envName: LLM_API_KEY
    required: false
    editable: true
    applyOnChange: true
```

Wire in server deployment; fall back to ConfigMap defaults when unset. See `scripts/add-llm-serving-envs.js` for backend-specific patterns.

| Env | llama.cpp | vLLM | SGLang |
|-----|-----------|------|--------|
| Context | `--ctx-size` | `--max-model-len` | `--context-length` |
| Max output | `--n-predict` | `--override-generation-config` | per-request (no server default) |
| API key | `--api-key` | `--api-key` | `--api-key` |

### Hugging Face token

If the app downloads from HF, add:

```yaml
  - envName: OLARES_USER_HUGGINGFACE_TOKEN
    required: false
    applyOnChange: true
    valueFrom:
      envName: OLARES_USER_HUGGINGFACE_TOKEN
```

Server deployment:
```yaml
- name: HF_TOKEN
  value: {{ .Values.olaresEnv.OLARES_USER_HUGGINGFACE_TOKEN | default "" | quote }}
```

## Steps

1. **Determine app name**: lowercase alphanumeric, no hyphens/underscores. Convention:
   - llama.cpp: `llamacpp<model><quant>one` (e.g., `llamacppqwen36a3bone`)
   - vLLM: `vllm<model>one`
   - SGLang: `sglang<model>one`

2. **Create server/client split structure** (see above). Use `llamacppqwen36beellamaone/` as template.

3. **Chart.yaml**: Umbrella v2 chart with `dependencies` on both subcharts. Version starts at `1.0.0`.

4. **OlaresManifest.yaml**: Key fields:
   - Resource requirements based on model size + backend needs
   - Icon URL: CDN URL from `npm run generate:icons` (see `scripts/apps-icons.json`), or placeholder until uploaded
   - Categories: `AI` / `LLM Chat` as appropriate
   - Developer: `coynntis`
   - **sharedEntrances + envs + apiTimeout: 0** (defaults section above)
   - License URL: use `ggml-org` (NOT `ggerganov`) for llama.cpp

5. **Server templates/deployment.yaml** (`<app>srv/`):
   - Wrapped in admin guard: `{{- if and .Values.admin .Values.bfl.username (eq .Values.admin .Values.bfl.username) }}`
   - **ConfigMap**: model URL, file name, default context size, tunable parameters
   - **InitContainer for permissions**: `chmod -R 777 <volume-mount>` for hostPath volumes
   - **InitContainer for model download** (if needed): wget/curl to persistent volume
   - **Main container**: inference server with optimized args + **LLM env fallbacks** + **HF_TOKEN**
   - **Probes**: startup (long timeout), liveness
   - **GPU annotation**: `applications.app.bytetrade.io/gpu-inject: "true"`
   - **sharedentrances Service** on server Deployment

6. **Client templates/clientproxy.yaml**: nginx → `http://<backend>.<serverChart>-shared:<port>`. SSE headers (`proxy_buffering off`, etc.).

7. **values.yaml**: `olaresEnv: {}` in server subchart.

8. **i18n/en-US/OlaresManifest.yaml**: Localized metadata.

9. **owners**: `coynntis`

10. **Package**: `npm run build` (or `node scripts/package-charts.js`)

11. **Report**: chart structure, versions, suggest Market admin install.

## Reference

Use `llamacppqwen36beellamaone/` as the gold-standard template.

### llama.cpp optimized args (battle-tested on Olares One)

```
--n-gpu-layers 99 --threads 16
--cache-type-k q8_0 --cache-type-v q8_0
--batch-size 2048 --ubatch-size 1024
--parallel 1 --mlock --swa-full
--flash-attn auto --op-offload
--jinja --no-context-shift
```
Env: `GGML_CUDA_GRAPH_OPT=1`

### Docker images (pinned)
- llama.cpp: `ghcr.io/ggml-org/llama.cpp:server-cuda13-b<N>` — verify tag exists on ghcr.io
- vLLM: `vllm/vllm-openai:v<version>`
- SGLang: `lmsysorg/sglang:dev-cu13`

### Verify Docker image exists before deploying
```bash
TOKEN=$(curl -s "https://ghcr.io/token?scope=repository:ggml-org/llama.cpp:pull" | jq -r '.token')
curl -s -o /dev/null -w "%{http_code}" "https://ghcr.io/v2/ggml-org/llama.cpp/manifests/server-cuda13-b<N>" \
  -H "Authorization: Bearer $TOKEN" -H "Accept: application/vnd.oci.image.index.v1+json"
# 200 = exists, 404 = does not exist
```
