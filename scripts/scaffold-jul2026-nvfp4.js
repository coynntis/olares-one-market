#!/usr/bin/env node
/**
 * Scaffold 4 charts (Jul 2026):
 * - llamacppagentworld35bone  (GGUF + buun turbo4 + native llama.cpp web UI)
 * - vllmqwen3627bnvfp4one      (vLLM + turboquant KV + vision + llama.cpp web UI proxy)
 * - vllmqwen3635bnvfp4fone     (vLLM Fast NVFP4)
 * - vllmgemma431bnvfp4one      (vLLM Unsloth Gemma 4 31B NVFP4)
 */
const fs = require('fs');
const path = require('path');

const ROOT = path.join(__dirname, '..');

function write(rel, content) {
  const p = path.join(ROOT, rel);
  fs.mkdirSync(path.dirname(p), { recursive: true });
  fs.writeFileSync(p, content);
  console.log('wrote', rel);
}

function clientProxy(app) {
  return `---
apiVersion: v1
data:
  nginx.conf: |
    server {

      listen 8080;
      server_name _;
      access_log /opt/bitnami/openresty/nginx/logs/access.log;
      error_log  /opt/bitnami/openresty/nginx/logs/error.log;

      proxy_connect_timeout                          600s;
      proxy_send_timeout                             600s;
      proxy_read_timeout                             1800s;
      proxy_buffering off;
      proxy_cache off;
      chunked_transfer_encoding on;
      proxy_set_header      host                      $host;
      proxy_set_header      x-forwarded-host          $http_host;

      proxy_http_version 1.1;

      proxy_set_header upgrade $http_upgrade;
      proxy_set_header connection "upgrade";

      location / {
        add_header X-Frame-Options "";
        proxy_pass http://${app}:8000;
      }
    }

kind: ConfigMap
metadata:
  name: nginx-config
  namespace: {{ .Release.Namespace }}

---
apiVersion: apps/v1
kind: Deployment
metadata:
  labels:
    io.kompose.service: ${app}cli
  name: ${app}cli
  namespace: '{{ .Release.Namespace }}'
spec:
  replicas: {{ .Values.workloads.${app}cli.replicaCount }}
  selector:
    matchLabels:
      io.kompose.service: ${app}cli
  template:
    metadata:
      labels:
        io.kompose.network/chrome-default: "true"
        io.kompose.service: ${app}cli
    spec:
      volumes:
        - name: nginx-config
          configMap:
            name: nginx-config
            defaultMode: 438
            items:
              - key: nginx.conf
                path: nginx.conf
      containers:
        - name: nginx
          image: "docker.io/beclab/aboveos-bitnami-openresty:1.25.3-2"
          ports:
            - containerPort: 8080
              protocol: TCP
          startupProbe:
            tcpSocket:
              port: 8080
            failureThreshold: 30
            periodSeconds: 10
          resources:
            limits:
              cpu: 500m
              memory: 500Mi
            requests:
              cpu: 10m
              memory: 64Mi
          volumeMounts:
            - name: nginx-config
              mountPath: /opt/bitnami/openresty/nginx/conf/server_blocks/nginx.conf
              subPath: nginx.conf

---
apiVersion: v1
kind: Service
metadata:
  name: ${app}cli
  namespace: {{ .Release.Namespace }}
spec:
  type: ClusterIP
  selector:
    io.kompose.service: ${app}cli
  ports:
    - name: ${app}cli
      protocol: TCP
      port: 8080
      targetPort: 8080
`;
}

function valuesYaml(app) {
  return `workloads:
  ${app}:
    replicaCount: 1
  ${app}cli:
    replicaCount: 1
olaresEnv:
  HF_TOKEN: ""
  HF_ENDPOINT: ""
  GITHUB_TOKEN: ""
  GHCR_USER: ""
  LLM_CONTEXT_WINDOW: ""
  LLM_MAX_OUTPUT_TOKENS: ""
  LLM_API_KEY: ""
`;
}

function helmignore() {
  return `.DS_Store
*.swp
*.bak
README.md
`;
}

function owners() {
  return `owners:
- 'aamsellem'
`;
}

// --- AgentWorld llama.cpp ---
const AW = 'llamacppagentworld35bone';

write(`${AW}/Chart.yaml`, `apiVersion: v2
appVersion: "1.0.0"
description: Qwen AgentWorld 35B-A3B world-model — UD-Q4_K_XL + turbo4 KV on Olares One
name: ${AW}
type: application
version: 1.0.0
`);

write(`${AW}/owners`, owners());
write(`${AW}/.helmignore`, helmignore());
write(`${AW}/values.yaml`, valuesYaml(AW));
write(`${AW}/templates/clientproxy.yaml`, clientProxy(AW));

write(`${AW}/templates/server.yaml`, `---
apiVersion: v1
kind: ConfigMap
metadata:
  name: llamacpp-agentworld-env
  namespace: "{{ .Release.Namespace }}"
data:
  TARGET_MODEL: "unsloth/Qwen-AgentWorld-35B-A3B-GGUF"
  TARGET_FILE: "Qwen-AgentWorld-35B-A3B-UD-Q4_K_XL.gguf"
  MODEL_ALIAS: "qwen-agentworld-35b-a3b"
  THREADS: "16"
  CTX_SIZE: "65536"
---
apiVersion: apps/v1
kind: Deployment
metadata:
  labels:
    io.kompose.service: ${AW}
  name: ${AW}
  namespace: "{{ .Release.Namespace }}"
  annotations:
    applications.app.bytetrade.io/gpu-inject: "true"
spec:
  replicas: {{ .Values.workloads.${AW}.replicaCount }}
  selector:
    matchLabels:
      io.kompose.service: ${AW}
  strategy:
    type: Recreate
  template:
    metadata:
      labels:
        io.kompose.network/chrome-default: "true"
        io.kompose.service: ${AW}
    spec:
      initContainers:
        - name: fix-shared-perms
          image: "docker.io/beclab/aboveos-busybox:1.37.0"
          command: ["sh", "-c", "mkdir -p /shared-models/llms && chmod -R 777 /shared-models && echo perms-fixed"]
          securityContext:
            runAsUser: 0
          volumeMounts:
            - mountPath: "/shared-models"
              name: shared-models
      containers:
        - name: llamacpp-server
          image: "docker.io/aamsellem/buun-llama-cpp:87c351d2"
          command: ["/bin/bash", "-c"]
          args:
            - |
              set -e
              MODELS_DIR=\${MODELS_DIR:-/models}
              SHARED_DIR=\${SHARED_DIR:-/shared-models/llms}
              mkdir -p "$MODELS_DIR" "$SHARED_DIR"
              HF_AUTH=""
              if [ -n "$HF_TOKEN" ]; then HF_AUTH="-H \\"Authorization: Bearer $HF_TOKEN\\""; fi
              if [ ! -f "$SHARED_DIR/$TARGET_FILE.ok" ]; then
                echo "==> Downloading $TARGET_MODEL/$TARGET_FILE (~22.3 GB)"
                eval curl -fL -C - --retry 20 --retry-delay 15 --retry-all-errors --connect-timeout 30 $HF_AUTH \\
                  "https://huggingface.co/\${TARGET_MODEL}/resolve/main/\${TARGET_FILE}" \\
                  -o "$SHARED_DIR/$TARGET_FILE"
                touch "$SHARED_DIR/$TARGET_FILE.ok"
              fi
              EXTRA_LLM_ARGS=()
              if [ -n "\${LLM_MAX_OUTPUT_TOKENS:-}" ]; then EXTRA_LLM_ARGS+=(--n-predict "$LLM_MAX_OUTPUT_TOKENS"); fi
              if [ -n "\${LLM_API_KEY:-}" ]; then EXTRA_LLM_ARGS+=(--api-key "$LLM_API_KEY"); fi
              # Native llama.cpp web UI on :8000 (built into llama-server)
              exec /app/llama-server \\
                --model "$SHARED_DIR/$TARGET_FILE" \\
                --alias "$MODEL_ALIAS" \\
                --host 0.0.0.0 --port 8000 \\
                --ctx-size "\${LLM_CONTEXT_WINDOW:-\${CTX_SIZE:-65536}}" \\
                --n-gpu-layers 99 \\
                --threads "\${THREADS:-16}" \\
                --cache-type-k q8_0 --cache-type-v turbo4 \\
                --batch-size 512 --ubatch-size 512 \\
                --parallel 1 \\
                --flash-attn on \\
                --op-offload --jinja --swa-full \\
                --reasoning on \\
                "\${EXTRA_LLM_ARGS[@]}"
          envFrom:
            - configMapRef:
                name: llamacpp-agentworld-env
          env:
            - name: HF_HOME
              value: "/models/huggingface"
            - name: HF_TOKEN
              value: {{ .Values.olaresEnv.HF_TOKEN | default "" | quote }}
            - name: MODELS_DIR
              value: "/models"
            - name: CUDA_DEVICE_MEMORY_LIMIT_0
              value: "24400m"
            - name: GGML_CUDA_GRAPH_OPT
              value: "1"
            - name: LLM_CONTEXT_WINDOW
              value: {{ .Values.olaresEnv.LLM_CONTEXT_WINDOW | default "" | quote }}
            - name: LLM_MAX_OUTPUT_TOKENS
              value: {{ .Values.olaresEnv.LLM_MAX_OUTPUT_TOKENS | default "" | quote }}
            - name: LLM_API_KEY
              value: {{ .Values.olaresEnv.LLM_API_KEY | default "" | quote }}
          ports:
            - containerPort: 8000
          livenessProbe:
            httpGet:
              path: /health
              port: 8000
              scheme: HTTP
            initialDelaySeconds: 600
            timeoutSeconds: 10
            periodSeconds: 30
            failureThreshold: 5
          startupProbe:
            httpGet:
              path: /health
              port: 8000
              scheme: HTTP
            initialDelaySeconds: 60
            timeoutSeconds: 10
            periodSeconds: 30
            failureThreshold: 240
          resources:
            limits:
              cpu: "12"
              memory: 48Gi
              nvidia.com/gpu: "1"
            requests:
              cpu: "1"
              memory: 4Gi
              nvidia.com/gpu: "1"
          volumeMounts:
            - mountPath: "/models"
              name: models
            - mountPath: "/shared-models"
              name: shared-models
      volumes:
        - name: models
          hostPath:
            path: "{{ .Values.userspace.appData }}/models"
            type: DirectoryOrCreate
        - name: shared-models
          hostPath:
            path: "/olares/share/ai/model"
            type: DirectoryOrCreate
      restartPolicy: Always
---
apiVersion: v1
kind: Service
metadata:
  labels:
    io.kompose.service: ${AW}
  name: ${AW}
  namespace: "{{ .Release.Namespace }}"
spec:
  ports:
    - name: "llamacpp"
      port: 8000
      targetPort: 8000
  selector:
    io.kompose.service: ${AW}
---
apiVersion: v1
kind: Service
metadata:
  labels:
    io.kompose.service: ${AW}
  name: sharedentrances-${AW}
  namespace: "{{ .Release.Namespace }}"
spec:
  ports:
    - name: "${AW}"
      port: 80
      targetPort: 8000
  selector:
    io.kompose.service: ${AW}
`);

write(`${AW}/OlaresManifest.yaml`, `---
olaresManifest.version: 0.12.0
olaresManifest.type: app
apiVersion: v3
workloadReplicas:
  ${AW}: 1
  ${AW}cli: 1
metadata:
  name: ${AW}
  icon: https://cdn.olares.com/images/placeholder-icon.png
  description: Qwen AgentWorld 35B-A3B — UD-Q4_K_XL + turbo4 KV world-model on Olares One
  title: AgentWorld 35B One
  version: 1.0.0
  categories:
    - AI Agents
    - LLM Chat
  bento:
    family: qwen
    size_label: 35B-A3B
    badge: world model
    hero:
      value: 64K ctx
      label: turbo4 KV
    specs:
      - label: context
        value: 64K
      - label: vram
        value: 22 GB
    capabilities:
      tool_calling: true
      vision: false
      audio: false
      mtp:
        enabled: false
    stack: buun-llama.cpp · UD-Q4_K_XL · turbo4 KV
sharedEntrances:
  - name: ${AW}
    host: sharedentrances-${AW}
    port: 0
    title: AgentWorld 35B API
    icon: https://cdn.olares.com/images/placeholder-icon.png
    invisible: true
    authLevel: internal
entrances:
  - name: ${AW}cli
    port: 8080
    host: ${AW}cli
    title: AgentWorld 35B One
    icon: https://cdn.olares.com/images/placeholder-icon.png
    openMethod: window
    authLevel: internal
spec:
  versionName: 1.0.0
  featuredImage: https://cdn.olares.com/images/placeholder-featured.png
  upgradeDescription: |
    v1.0.0: initial release. unsloth/Qwen-AgentWorld-35B-A3B-GGUF UD-Q4_K_XL via buun-llama-cpp with q8_0/turbo4 KV, 64K default context, native llama.cpp web UI. Text-only world model (no mmproj on HF).
  fullDescription: |
    Qwen-AgentWorld-35B-A3B — native language world model for agent environment simulation (MCP, Search, Terminal, SWE, Android, Web, OS).

    Stack (Olares One RTX 5090M 24GB):
    - Image: aamsellem/buun-llama-cpp:87c351d2
    - Model: unsloth/Qwen-AgentWorld-35B-A3B-GGUF → UD-Q4_K_XL (~22.3 GB)
    - KV: q8_0 keys + turbo4 values (TurboQuant) for long-context headroom
    - Default context 64K (override via LLM_CONTEXT_WINDOW; native up to 262K)
    - Entrance opens native llama.cpp web UI on the client proxy

    Text-only — no mmproj in this GGUF repo (world-model simulator, not VLM).
    First launch downloads ~22 GB to /olares/share/ai/model/llms/.
  developer: coynntis
  website: https://huggingface.co/unsloth/Qwen-AgentWorld-35B-A3B-GGUF
  sourceCode: https://github.com/coynntis/olares-one-market
  submitter: coynntis
  locale:
    - en-US
  license:
    - text: Apache 2.0
      url: https://creativecommons.org/licenses/by/4.0/
  supportArch:
    - amd64
  onlyAdmin: true
  accelerator:
    - mode: nvidia
      requiredCpu: "2"
      limitedCpu: "13"
      requiredMemory: 5Gi
      limitedMemory: 49Gi
      requiredDisk: 30Gi
      limitedDisk: 40Gi
      requiredGPUMemory: 1Gi
      limitedGPUMemory: 24Gi
permission:
  appData: true
envs:
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
  - envName: HF_TOKEN
    required: false
    applyOnChange: true
    valueFrom:
      envName: OLARES_USER_HUGGINGFACE_TOKEN
  - envName: HF_ENDPOINT
    required: false
    applyOnChange: true
    valueFrom:
      envName: OLARES_USER_HUGGINGFACE_SERVICE
options:
  shared: true
  apiTimeout: 0
  LLMGatewaySupported: true
  dependencies:
    - name: olares
      version: '>=1.12.6-0'
      type: system
`);

write(`${AW}/i18n/en-US/OlaresManifest.yaml`, `metadata:
  description: "Qwen AgentWorld 35B-A3B — UD-Q4_K_XL + turbo4 KV world-model on Olares One"
  title: AgentWorld 35B One
spec:
  fullDescription: |
    Native language world model via llama.cpp web UI.
    UD-Q4_K_XL (~22.3 GB), q8_0/turbo4 KV, 64K default context.
    Text-only (no mmproj in GGUF repo).
`);

// --- Shared vLLM + llama.cpp webui launcher body ---
function vllmServerYaml(cfg) {
  const {
    app,
    cmName,
    modelName,
    modelAlias,
    maxModelLen,
    gpuUtil,
    vision,
    toolParser,
    reasoningParser,
    speculative,
    title,
  } = cfg;

  const limitMm = vision ? `\n  LIMIT_MM: '{"image": 1}'` : '';
  const limitMmArg = vision
    ? `\n                --limit-mm-per-prompt "\${LIMIT_MM}" \\`
    : '';
  const toolArgs = toolParser
    ? `\n                --enable-auto-tool-choice \\\n                --tool-call-parser ${toolParser} \\`
    : '';
  const reasonArgs = reasoningParser
    ? `\n                --reasoning-parser ${reasoningParser} \\`
    : '';
  const specArgs = speculative
    ? `\n                --speculative-config '${speculative}' \\`
    : '';
  const modalitiesVision = vision ? 'true' : 'false';

  return `{{- $llmCtx := .Values.olaresEnv.LLM_CONTEXT_WINDOW | default "" }}
{{- $llmMaxOut := .Values.olaresEnv.LLM_MAX_OUTPUT_TOKENS | default "" }}
{{- $llmApiKey := .Values.olaresEnv.LLM_API_KEY | default "" }}
---
apiVersion: v1
kind: ConfigMap
metadata:
  name: ${cmName}
  namespace: "{{ .Release.Namespace }}"
data:
  MODEL_NAME: "${modelName}"
  MODEL_ALIAS: "${modelAlias}"
  MAX_MODEL_LEN: "${maxModelLen}"
  GPU_MEMORY_UTILIZATION: "${gpuUtil}"
  MAX_NUM_SEQS: "32"
  KV_CACHE_DTYPE: "turboquant_k8v4"
  WEBUI_REF: "b8740"${limitMm}
---
apiVersion: apps/v1
kind: Deployment
metadata:
  labels:
    io.kompose.service: ${app}
  name: ${app}
  namespace: "{{ .Release.Namespace }}"
  annotations:
    applications.app.bytetrade.io/gpu-inject: "true"
spec:
  replicas: {{ .Values.workloads.${app}.replicaCount }}
  selector:
    matchLabels:
      io.kompose.service: ${app}
  strategy:
    type: Recreate
  template:
    metadata:
      labels:
        io.kompose.network/chrome-default: "true"
        io.kompose.service: ${app}
    spec:
      initContainers:
        - name: fix-shared-llms-perms
          image: "docker.io/beclab/aboveos-busybox:1.37.0"
          command:
            - sh
            - -c
            - mkdir -p /shared-models/llms/huggingface && chmod -R 777 /shared-models/llms && echo shared-llms-dir-ready
          securityContext:
            runAsUser: 0
          volumeMounts:
            - mountPath: "/shared-models"
              name: shared-models
      containers:
        - name: vllm-server
          image: "vllm/vllm-openai:nightly"
          command:
            - "sh"
            - "-c"
            - |
              mkdir -p /app /app/webui
              cat > /app/start-vllm.sh << 'SHEOF'
              #!/bin/bash
              set -euo pipefail
              export HF_HOME="\${HF_HOME:-/shared-models/llms/huggingface}"
              pip install --quiet hf_transfer httpx 2>/dev/null || true
              export HF_HUB_ENABLE_HF_TRANSFER="\${HF_HUB_ENABLE_HF_TRANSFER:-1}"
              echo "$(date -Iseconds) [${app}] starting vLLM on :8001 model=\${MODEL_NAME} kv=\${KV_CACHE_DTYPE}"
              exec vllm serve "\${MODEL_NAME}" \\
                --served-model-name "\${MODEL_ALIAS}" \\
                --host 127.0.0.1 \\
                --port 8001 \\
                --max-model-len \\
{{- if $llmCtx }}
                {{ $llmCtx | quote }} \\
{{- else }}
                "\${MAX_MODEL_LEN}" \\
{{- end }}
                --gpu-memory-utilization "\${GPU_MEMORY_UTILIZATION}" \\
                --max-num-seqs "\${MAX_NUM_SEQS}" \\
                --kv-cache-dtype "\${KV_CACHE_DTYPE}" \\
                --dtype auto \\
                --compilation-config '{"cudagraph_mode": 1, "max_cudagraph_capture_size": 4, "cudagraph_capture_sizes": [1, 2, 4]}' \\${limitMmArg}${toolArgs}${reasonArgs}${specArgs}
                --trust-remote-code \\
                --download-dir /shared-models/llms/huggingface \\
{{- if $llmApiKey }}
                --api-key {{ $llmApiKey | quote }} \\
{{- end }}
{{- if $llmMaxOut }}
                --override-generation-config {{ printf "{\\"max_tokens\\":%s}" $llmMaxOut | quote }} \\
{{- end }}
                --enable-prefix-caching
              SHEOF
              chmod +x /app/start-vllm.sh

              # Fetch llama.cpp web UI (same UI as llama-server) and serve it in front of vLLM
              WEBUI_REF="\${WEBUI_REF:-b8740}"
              BASE="https://cdn.jsdelivr.net/gh/ggml-org/llama.cpp@\${WEBUI_REF}/tools/server/public"
              for f in index.html bundle.js bundle.css loading.html; do
                echo "[${app}] fetching webui \$f"
                curl -fsSL "\$BASE/\$f" -o "/app/webui/\$f" || true
              done

              cat > /app/launcher.py << 'PYEOF'
              import asyncio, glob, os, subprocess, threading, traceback, urllib.request
              from contextlib import asynccontextmanager
              from typing import Optional
              from fastapi import FastAPI, HTTPException, Request, Response
              from fastapi.responses import JSONResponse, FileResponse
              from fastapi.staticfiles import StaticFiles

              _phase = "starting"
              _bootstrap_error: Optional[str] = None
              _vllm_proc: Optional[subprocess.Popen] = None
              _backend_port = 8001
              APP = "${app}"
              ALIAS = os.environ.get("MODEL_ALIAS", "${modelAlias}")
              CTX = int(os.environ.get("LLM_CONTEXT_WINDOW") or os.environ.get("MAX_MODEL_LEN") or "${maxModelLen}")
              VISION = ${modalitiesVision == 'true' ? 'True' : 'False'}

              def log(msg: str) -> None:
                  print(f"[{APP}] {msg}", flush=True)

              def backend_healthy_sync() -> bool:
                  try:
                      with urllib.request.urlopen(f"http://127.0.0.1:{_backend_port}/v1/models", timeout=10) as resp:
                          return resp.status == 200
                  except Exception:
                      return False

              def relay_vllm_output(proc: subprocess.Popen) -> None:
                  if proc.stdout is None:
                      return
                  def pump() -> None:
                      for raw in iter(proc.stdout.readline, b""):
                          line = raw.decode(errors="replace").rstrip()
                          if line:
                              print(f"[vllm] {line}", flush=True)
                      proc.stdout.close()
                  threading.Thread(target=pump, daemon=True).start()

              def wait_for_backend_sync() -> None:
                  global _phase
                  import time
                  max_wait = int(os.environ.get("VLLM_BOOTSTRAP_TIMEOUT_SEC", "21600"))
                  interval = 5
                  for i in range(max(1, max_wait // interval)):
                      if _vllm_proc is not None and _vllm_proc.poll() is not None:
                          raise RuntimeError(f"vLLM exited {_vllm_proc.returncode}")
                      if backend_healthy_sync():
                          _phase = "ready"
                          log(f"vLLM ready after ~{i * interval}s — llama.cpp web UI proxying")
                          return
                      if i > 0 and i % 60 == 0:
                          log(f"still loading ({i * interval}s)")
                      time.sleep(interval)
                  raise RuntimeError(f"vLLM health timeout after {max_wait}s")

              def bootstrap_sync() -> None:
                  global _phase, _bootstrap_error, _vllm_proc
                  try:
                      _vllm_proc = subprocess.Popen(["/app/start-vllm.sh"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1)
                      relay_vllm_output(_vllm_proc)
                      _phase = "starting_vllm"
                      wait_for_backend_sync()
                  except Exception as exc:
                      _bootstrap_error = str(exc)
                      _phase = "error"
                      log(f"bootstrap failed: {exc}")
                      traceback.print_exc()

              @asynccontextmanager
              async def lifespan(app: FastAPI):
                  log("launcher :8000 — llama.cpp web UI + vLLM :8001")
                  asyncio.create_task(asyncio.to_thread(bootstrap_sync))
                  yield

              app = FastAPI(title="${title} launcher", lifespan=lifespan)

              @app.get("/health")
              async def health():
                  ready = _phase == "ready"
                  backend_ok = ready and await asyncio.to_thread(backend_healthy_sync)
                  body = {"status": _phase, "ready": ready and backend_ok, "model": os.environ.get("MODEL_NAME"), "bootstrap_error": _bootstrap_error, "ui": "llama.cpp"}
                  if ready and not backend_ok:
                      return JSONResponse(status_code=503, content=body)
                  return JSONResponse(status_code=200, content=body)

              @app.get("/props")
              async def props():
                  return {
                      "default_generation_settings": {
                          "id": 0, "id_task": -1, "n_ctx": CTX, "speculative": False, "is_processing": False,
                          "params": {"n_predict": -1, "temperature": 0.8, "top_k": 40, "top_p": 0.95, "min_p": 0.05, "stream": True, "max_tokens": -1},
                          "prompt": "", "next_token": {"has_next_token": True, "has_new_line": False, "n_remain": -1, "n_decoded": 0, "stopping_word": ""}
                      },
                      "total_slots": 1,
                      "model_path": os.environ.get("MODEL_NAME", ALIAS),
                      "model_alias": ALIAS,
                      "chat_template": "",
                      "chat_template_caps": {},
                      "modalities": {"vision": VISION},
                      "build_info": "vllm-proxy+llama.cpp-webui",
                      "is_sleeping": _phase != "ready",
                  }

              @app.get("/slots")
              async def slots():
                  return [{"id": 0, "is_processing": False, "n_ctx": CTX}]

              async def proxy_request(request: Request, full_path: str):
                  if _phase != "ready":
                      raise HTTPException(503, detail=f"not ready (phase={_phase})")
                  try:
                      import httpx
                  except ImportError:
                      raise HTTPException(503, detail="httpx missing")
                  url = f"http://127.0.0.1:{_backend_port}/{full_path}"
                  if request.url.query:
                      url = f"{url}?{request.url.query}"
                  headers = {k: v for k, v in request.headers.items() if k.lower() not in ("host", "content-length")}
                  body = await request.body()
                  async with httpx.AsyncClient(timeout=None) as client:
                      # stream SSE
                      if "text/event-stream" in (request.headers.get("accept") or "") or b'"stream": true' in body.lower() or b'"stream":true' in body.lower():
                          upstream = await client.send(client.build_request(request.method, url, headers=headers, content=body if body else None), stream=True)
                          async def gen():
                              async for chunk in upstream.aiter_bytes():
                                  yield chunk
                              await upstream.aclose()
                          return Response(content=None, status_code=upstream.status_code, headers=dict(upstream.headers), media_type=upstream.headers.get("content-type"))
                      upstream = await client.request(request.method, url, headers=headers, content=body if body else None)
                  return Response(content=upstream.content, status_code=upstream.status_code, headers=dict(upstream.headers))

              @app.api_route("/v1/{full_path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS", "HEAD"])
              async def proxy_v1(full_path: str, request: Request):
                  return await proxy_request(request, f"v1/{full_path}")

              @app.api_route("/openai/{full_path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS", "HEAD"])
              async def proxy_openai(full_path: str, request: Request):
                  return await proxy_request(request, f"v1/{full_path}")

              webui = "/app/webui"
              if os.path.isfile(os.path.join(webui, "index.html")):
                  @app.get("/")
                  async def root_ui():
                      return FileResponse(os.path.join(webui, "index.html"))
                  app.mount("/", StaticFiles(directory=webui, html=True), name="webui")
              else:
                  @app.get("/")
                  async def root_fallback():
                      return JSONResponse({"error": "webui missing", "hint": "API at /v1", "phase": _phase})

              if __name__ == "__main__":
                  import uvicorn
                  uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
              PYEOF

              exec python3 /app/launcher.py
          envFrom:
            - configMapRef:
                name: ${cmName}
          env:
            - name: HF_HOME
              value: "/shared-models/llms/huggingface"
            - name: LLM_CONTEXT_WINDOW
              value: {{ .Values.olaresEnv.LLM_CONTEXT_WINDOW | default "" | quote }}
            - name: LLM_MAX_OUTPUT_TOKENS
              value: {{ .Values.olaresEnv.LLM_MAX_OUTPUT_TOKENS | default "" | quote }}
            - name: LLM_API_KEY
              value: {{ .Values.olaresEnv.LLM_API_KEY | default "" | quote }}
            - name: VLLM_BOOTSTRAP_TIMEOUT_SEC
              value: "21600"
            - name: HF_HUB_ENABLE_HF_TRANSFER
              value: "1"
            - name: HF_TOKEN
              value: {{ .Values.olaresEnv.HF_TOKEN | default "" | quote }}
            - name: HUGGING_FACE_HUB_TOKEN
              value: {{ .Values.olaresEnv.HF_TOKEN | default "" | quote }}
            {{- if .Values.olaresEnv.HF_ENDPOINT }}
            - name: HF_ENDPOINT
              value: "{{ .Values.olaresEnv.HF_ENDPOINT }}"
            {{- end }}
            - name: CUDA_DEVICE_MEMORY_LIMIT_0
              value: "24400m"
          ports:
            - containerPort: 8000
          livenessProbe:
            httpGet:
              path: /health
              port: 8000
              scheme: HTTP
            initialDelaySeconds: 120
            timeoutSeconds: 10
            periodSeconds: 30
            failureThreshold: 5
          startupProbe:
            httpGet:
              path: /health
              port: 8000
              scheme: HTTP
            initialDelaySeconds: 60
            timeoutSeconds: 10
            periodSeconds: 30
            failureThreshold: 240
          resources:
            limits:
              cpu: "12"
              memory: 48Gi
              nvidia.com/gpu: "1"
            requests:
              cpu: "1"
              memory: 4Gi
              nvidia.com/gpu: "1"
          volumeMounts:
            - mountPath: "/models"
              name: models
            - mountPath: "/shared-models"
              name: shared-models
      volumes:
        - name: models
          hostPath:
            path: "{{ .Values.userspace.appData }}/models"
            type: DirectoryOrCreate
        - name: shared-models
          hostPath:
            path: "/olares/share/ai/model"
            type: DirectoryOrCreate
      restartPolicy: Always
---
apiVersion: v1
kind: Service
metadata:
  labels:
    io.kompose.service: ${app}
  name: ${app}
  namespace: "{{ .Release.Namespace }}"
spec:
  ports:
    - name: "vllm"
      port: 8000
      targetPort: 8000
  selector:
    io.kompose.service: ${app}
---
apiVersion: v1
kind: Service
metadata:
  labels:
    io.kompose.service: ${app}
  name: sharedentrances-${app}
  namespace: "{{ .Release.Namespace }}"
spec:
  ports:
    - name: "${app}"
      port: 80
      targetPort: 8000
  selector:
    io.kompose.service: ${app}
`;
}

function vllmManifest(cfg) {
  const {
    app,
    title,
    description,
    shortTitle,
    modelHf,
    fullDesc,
    categories,
    badge,
    ctxLabel,
    vision,
    stack,
    diskGi,
  } = cfg;
  return `---
olaresManifest.version: 0.12.0
olaresManifest.type: app
apiVersion: v3
workloadReplicas:
  ${app}: 1
  ${app}cli: 1
metadata:
  name: ${app}
  icon: https://cdn.olares.com/images/placeholder-icon.png
  description: ${description}
  title: ${title}
  version: 1.0.0
  categories:
${categories.map((c) => `    - ${c}`).join('\n')}
  bento:
    family: ${app.includes('gemma') ? 'gemma' : 'qwen'}
    size_label: ${badge}
    badge: NVFP4
    hero:
      value: ${ctxLabel}
      label: turboquant KV
    specs:
      - label: context
        value: ${ctxLabel}
      - label: vram
        value: 24 GB
    capabilities:
      tool_calling: true
      vision: ${vision}
      audio: false
      mtp:
        enabled: true
    stack: ${stack}
sharedEntrances:
  - name: ${app}
    host: sharedentrances-${app}
    port: 0
    title: ${shortTitle} API
    icon: https://cdn.olares.com/images/placeholder-icon.png
    invisible: true
    authLevel: internal
entrances:
  - name: ${app}cli
    port: 8080
    host: ${app}cli
    title: ${shortTitle}
    icon: https://cdn.olares.com/images/placeholder-icon.png
    openMethod: window
    authLevel: internal
spec:
  versionName: 1.0.0
  featuredImage: https://cdn.olares.com/images/placeholder-featured.png
  upgradeDescription: |
    v1.0.0: initial release. ${modelHf} via vLLM nightly with turboquant_k8v4 KV for long context, native multimodal vision, MTP where available. Entrance serves llama.cpp web UI (b8740 public/) proxying /v1 to vLLM.
  fullDescription: |
${fullDesc.split('\n').map((l) => `    ${l}`).join('\n')}
  developer: coynntis
  website: https://huggingface.co/${modelHf}
  sourceCode: https://github.com/coynntis/olares-one-market
  submitter: coynntis
  locale:
    - en-US
  license:
    - text: Apache 2.0
      url: https://www.apache.org/licenses/LICENSE-2.0
  supportArch:
    - amd64
  onlyAdmin: true
  accelerator:
    - mode: nvidia
      requiredCpu: "2"
      limitedCpu: "13"
      requiredMemory: 5Gi
      limitedMemory: 49Gi
      requiredDisk: ${diskGi}Gi
      limitedDisk: ${diskGi + 10}Gi
      requiredGPUMemory: 1Gi
      limitedGPUMemory: 24Gi
permission:
  appData: true
envs:
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
  - envName: HF_TOKEN
    required: false
    applyOnChange: true
    valueFrom:
      envName: OLARES_USER_HUGGINGFACE_TOKEN
  - envName: HF_ENDPOINT
    required: false
    applyOnChange: true
    valueFrom:
      envName: OLARES_USER_HUGGINGFACE_SERVICE
options:
  shared: true
  apiTimeout: 0
  LLMGatewaySupported: true
  dependencies:
    - name: olares
      version: '>=1.12.6-0'
      type: system
`;
}

function scaffoldVllm(cfg) {
  const { app, chartDesc } = cfg;
  write(`${app}/Chart.yaml`, `apiVersion: v2
appVersion: "1.0.0"
description: ${chartDesc}
name: ${app}
type: application
version: 1.0.0
`);
  write(`${app}/owners`, owners());
  write(`${app}/.helmignore`, helmignore());
  write(`${app}/values.yaml`, valuesYaml(app));
  write(`${app}/templates/clientproxy.yaml`, clientProxy(app));
  write(`${app}/templates/server.yaml`, vllmServerYaml(cfg));
  write(`${app}/OlaresManifest.yaml`, vllmManifest(cfg));
  write(
    `${app}/i18n/en-US/OlaresManifest.yaml`,
    `metadata:
  description: "${cfg.description}"
  title: ${cfg.title}
spec:
  fullDescription: |
    vLLM + turboquant_k8v4 KV + llama.cpp web UI entrance.
    Model: ${cfg.modelHf}
`
  );
}

scaffoldVllm({
  app: 'vllmqwen3627bnvfp4one',
  cmName: 'vllm-qwen3627b-nvfp4-env',
  modelName: 'unsloth/Qwen3.6-27B-NVFP4',
  modelAlias: 'qwen3.6-27b-nvfp4',
  maxModelLen: '65536',
  gpuUtil: '0.92',
  vision: true,
  toolParser: 'hermes',
  reasoningParser: 'qwen3',
  speculative: '{"method": "mtp", "num_speculative_tokens": 2}',
  title: 'Qwen3.6 27B NVFP4 One',
  shortTitle: 'Qwen36 27B NVFP4',
  description: 'Qwen3.6 27B NVFP4 vision — vLLM turboquant KV + llama.cpp web UI on Olares One',
  chartDesc: 'Qwen3.6 27B Unsloth NVFP4 via vLLM + llama.cpp web UI',
  modelHf: 'unsloth/Qwen3.6-27B-NVFP4',
  categories: ['Vision', 'LLM Chat', 'AI Agents'],
  badge: '27B',
  ctxLabel: '64K',
  stack: 'vLLM nightly · NVFP4 · turboquant_k8v4 · llama.cpp UI',
  diskGi: 40,
  fullDesc: `Unsloth Qwen3.6-27B-NVFP4 (multimodal VLM, ~23.5 GB) on Olares One RTX 5090M.

Stack:
- Image: vllm/vllm-openai:nightly (needs >=0.25 for Unsloth NVFP4 cute-DSL kernels)
- --kv-cache-dtype turboquant_k8v4 for long-context KV compression
- --limit-mm-per-prompt image=1 (native vision — no GGUF mmproj; projector baked into checkpoint)
- MTP speculative decoding (num_speculative_tokens=2)
- hermes tool parser + qwen3 reasoning parser
- Default max_model_len 65536 (override via LLM_CONTEXT_WINDOW)

Entrance serves the real llama.cpp web UI (tools/server/public @ b8740) with /props stub + /v1 proxy to vLLM — same chat UI you know from llama-server.`,
});

scaffoldVllm({
  app: 'vllmqwen3635bnvfp4fone',
  cmName: 'vllm-qwen3635b-nvfp4f-env',
  modelName: 'unsloth/Qwen3.6-35B-A3B-NVFP4-Fast',
  modelAlias: 'qwen3.6-35b-a3b-nvfp4-fast',
  maxModelLen: '32768',
  gpuUtil: '0.90',
  vision: true,
  toolParser: 'hermes',
  reasoningParser: 'qwen3',
  speculative: '{"method": "mtp", "num_speculative_tokens": 2}',
  title: 'Qwen3.6 35B NVFP4 Fast One',
  shortTitle: 'Qwen36 35B NVFP4 Fast',
  description: 'Qwen3.6 35B-A3B NVFP4-Fast — vLLM turboquant KV + llama.cpp web UI on Olares One',
  chartDesc: 'Qwen3.6 35B-A3B Unsloth NVFP4-Fast via vLLM + llama.cpp web UI',
  modelHf: 'unsloth/Qwen3.6-35B-A3B-NVFP4-Fast',
  categories: ['Vision', 'LLM Chat', 'AI Agents'],
  badge: '35B-A3B',
  ctxLabel: '32K',
  stack: 'vLLM nightly · NVFP4-Fast · turboquant_k8v4 · llama.cpp UI',
  diskGi: 40,
  fullDesc: `Unsloth Qwen3.6-35B-A3B-NVFP4-Fast (MoE VLM, ~23.6 GB weights). Upstream notes 32GB VRAM preferred — tight on 24GB 5090M; turboquant KV + PIECEWISE CUDA graphs + 32K default ctx to leave headroom.

Stack:
- Image: vllm/vllm-openai:nightly
- --kv-cache-dtype turboquant_k8v4
- Native multimodal vision (limit-mm image=1) — no separate mmproj
- MTP speculative decoding
- Default max_model_len 32768 (raise via LLM_CONTEXT_WINDOW if VRAM allows)

Entrance: llama.cpp web UI proxying OpenAI API to vLLM.`,
});

scaffoldVllm({
  app: 'vllmgemma431bnvfp4one',
  cmName: 'vllm-gemma431b-nvfp4-env',
  modelName: 'unsloth/gemma-4-31B-it-NVFP4',
  modelAlias: 'gemma-4-31b-it-nvfp4',
  maxModelLen: '16384',
  gpuUtil: '0.95',
  vision: true,
  toolParser: 'gemma4',
  reasoningParser: null,
  speculative: null,
  title: 'Gemma 4 31B NVFP4 One',
  shortTitle: 'Gemma4 31B NVFP4',
  description: 'Gemma 4 31B IT NVFP4 vision — vLLM turboquant KV + llama.cpp web UI on Olares One',
  chartDesc: 'Unsloth Gemma 4 31B IT NVFP4 via vLLM + llama.cpp web UI',
  modelHf: 'unsloth/gemma-4-31B-it-NVFP4',
  categories: ['Vision', 'LLM Chat'],
  badge: '31B',
  ctxLabel: '16K',
  stack: 'vLLM nightly · NVFP4 · turboquant_k8v4 · llama.cpp UI',
  diskGi: 40,
  fullDesc: `Unsloth gemma-4-31B-it-NVFP4 (~24.8 GB on disk) — multimodal Gemma 4 dense IT. Extremely tight on 24GB; default 16K ctx + turboquant_k8v4 + PIECEWISE graphs.

Stack:
- Image: vllm/vllm-openai:nightly
- --kv-cache-dtype turboquant_k8v4 for long-context KV budget
- Native vision (limit-mm image=1) — vision encoder in checkpoint (not GGUF mmproj)
- gemma4 tool-call parser
- Default max_model_len 16384

Sibling chart vllmgemma31bitnvfp4one uses LilaRest turbo text-only checkpoint; this chart is the Unsloth multimodal NVFP4.

Entrance: llama.cpp web UI in front of vLLM.`,
});

console.log('done');
