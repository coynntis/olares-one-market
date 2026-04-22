#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# Olares Market - Model App Generator
#
# Generates a complete Olares app package for deploying an LLM model
# with either llama.cpp or vLLM backend, including auto-registration
# with LiteLLM gateway.
#
# Usage:
#   ./generate-model-app.sh \
#     --backend llamacpp|vllm \
#     --app-id <unique-app-id> \
#     --model-name <display-name> \
#     --model-alias <short-alias> \
#     --model-source <hf-repo-or-url> \
#     [--quant <quantization>] \
#     [--context-size <ctx>] \
#     [--gpu-memory <mem>] \
#     [--ram <mem>] \
#     [--gpu-vram <mem>] \
#     [--arch amd64,arm64]
#
# Examples:
#   # llama.cpp with GGUF model
#   ./generate-model-app.sh \
#     --backend llamacpp \
#     --app-id llamacppqwen36a3b \
#     --model-name "Qwen3.5 35B-A3B Q4_K_M" \
#     --model-alias qwen3.5-35b-a3b \
#     --model-source "https://huggingface.co/Qwen/Qwen3.5-35B-A3B-GGUF/resolve/main/qwen3.5-35b-a3b-q4_k_m.gguf" \
#     --quant Q4_K_M \
#     --context-size 8192 \
#     --gpu-memory 8Gi \
#     --ram 12Gi
#
#   # vLLM with HuggingFace model
#   ./generate-model-app.sh \
#     --backend vllm \
#     --app-id vllmmistral7b \
#     --model-name "Mistral 7B Instruct v0.3" \
#     --model-alias mistral-7b \
#     --model-source "mistralai/Mistral-7B-Instruct-v0.3" \
#     --gpu-vram 16Gi \
#     --ram 20Gi
# =============================================================================

# Defaults
BACKEND=""
APP_ID=""
MODEL_NAME=""
MODEL_ALIAS=""
MODEL_SOURCE=""
QUANT=""
CONTEXT_SIZE="8192"
GPU_MEMORY="12Gi"
RAM="12Gi"
GPU_VRAM="16Gi"
DISK="15Gi"
ARCH="amd64,arm64"
GPU_LAYERS="99"
THREADS="4"
GPU_UTIL="0.90"
DTYPE="auto"

usage() {
  head -35 "$0" | tail -30
  exit 1
}

while [[ $# -gt 0 ]]; do
  case $1 in
    --backend)       BACKEND="$2"; shift 2;;
    --app-id)        APP_ID="$2"; shift 2;;
    --model-name)    MODEL_NAME="$2"; shift 2;;
    --model-alias)   MODEL_ALIAS="$2"; shift 2;;
    --model-source)  MODEL_SOURCE="$2"; shift 2;;
    --quant)         QUANT="$2"; shift 2;;
    --context-size)  CONTEXT_SIZE="$2"; shift 2;;
    --gpu-memory)    GPU_MEMORY="$2"; shift 2;;
    --ram)           RAM="$2"; shift 2;;
    --gpu-vram)      GPU_VRAM="$2"; shift 2;;
    --disk)          DISK="$2"; shift 2;;
    --arch)          ARCH="$2"; shift 2;;
    --gpu-layers)    GPU_LAYERS="$2"; shift 2;;
    --threads)       THREADS="$2"; shift 2;;
    --gpu-util)      GPU_UTIL="$2"; shift 2;;
    --dtype)         DTYPE="$2"; shift 2;;
    -h|--help)       usage;;
    *)               echo "Unknown option: $1"; usage;;
  esac
done

[[ -z "$BACKEND" ]]      && echo "Error: --backend required" && usage
[[ -z "$APP_ID" ]]       && echo "Error: --app-id required" && usage
[[ -z "$MODEL_NAME" ]]   && echo "Error: --model-name required" && usage
[[ -z "$MODEL_ALIAS" ]]  && echo "Error: --model-alias required" && usage
[[ -z "$MODEL_SOURCE" ]] && echo "Error: --model-source required" && usage

SERVER_ID="${APP_ID}server"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
APP_DIR="${SCRIPT_DIR}/${APP_ID}"

if [[ -d "$APP_DIR" ]]; then
  echo "Error: Directory ${APP_DIR} already exists"
  exit 1
fi

echo "Generating Olares app: ${APP_ID}"
echo "  Backend: ${BACKEND}"
echo "  Model: ${MODEL_NAME}"
echo "  Alias: ${MODEL_ALIAS}"

# Create directory structure
mkdir -p "${APP_DIR}/${APP_ID}/templates"
mkdir -p "${APP_DIR}/${SERVER_ID}/templates"
mkdir -p "${APP_DIR}/i18n/en-US"
mkdir -p "${APP_DIR}/templates"

# Determine backend-specific values
case "$BACKEND" in
  llamacpp)
    BACKEND_LABEL="llama.cpp"
    BACKEND_IMAGE="ghcr.io/ggerganov/llama.cpp:server"
    BACKEND_PORT="8080"
    SERVER_SERVICE="llamacpp"
    CLIENT_SERVICE="${APP_ID}client"
    HEALTH_PATH="/health"
    ICON="https://raw.githubusercontent.com/ggerganov/llama.cpp/master/docs/images/logo2.png"
    DEVELOPER="ggerganov"
    WEBSITE="https://github.com/ggerganov/llama.cpp"
    LICENSE_TEXT="MIT"
    LICENSE_URL="https://github.com/ggerganov/llama.cpp/blob/master/LICENSE"
    STORAGE_DIR="LlamaCpp"
    # Extract filename from URL
    MODEL_FILE=$(basename "$MODEL_SOURCE")
    QUANT_LABEL="${QUANT:+ ${QUANT}}"
    ;;
  vllm)
    BACKEND_LABEL="vLLM"
    BACKEND_IMAGE="vllm/vllm-openai:v0.6.6.post1"
    BACKEND_PORT="8000"
    SERVER_SERVICE="vllm"
    CLIENT_SERVICE="${APP_ID}client"
    HEALTH_PATH="/health"
    ICON="https://docs.vllm.ai/en/latest/_images/vllm-logo-text-light.png"
    DEVELOPER="vLLM Team"
    WEBSITE="https://vllm.ai/"
    LICENSE_TEXT="Apache-2.0"
    LICENSE_URL="https://github.com/vllm-project/vllm/blob/main/LICENSE"
    STORAGE_DIR="VLLM"
    MODEL_FILE=""
    QUANT_LABEL=""
    # vLLM only supports amd64 (CUDA)
    ARCH="amd64"
    ;;
  *)
    echo "Error: backend must be 'llamacpp' or 'vllm'"
    exit 1
    ;;
esac

# Build arch list for YAML
IFS=',' read -ra ARCH_ARRAY <<< "$ARCH"
ARCH_YAML=""
for a in "${ARCH_ARRAY[@]}"; do
  ARCH_YAML="${ARCH_YAML}    - ${a}
"
done

TITLE="${MODEL_NAME}${QUANT_LABEL} (${BACKEND_LABEL})"

# --- Root Chart.yaml ---
cat > "${APP_DIR}/Chart.yaml" << EOF
apiVersion: v2
appVersion: '1.0.0'
description: ${MODEL_NAME} served via ${BACKEND_LABEL}
name: ${APP_ID}
type: application
version: '1.0.0'
EOF

# --- Root values.yaml ---
cat > "${APP_DIR}/values.yaml" << EOF
admin: ""
bfl:
  username: ""
domain:
  ${CLIENT_SERVICE}: ""
userspace:
  userData: ""
olaresEnv: {}
EOF

# --- owners ---
cat > "${APP_DIR}/owners" << EOF
owners:
- 'orales-market'
EOF

# --- templates/keep ---
touch "${APP_DIR}/templates/keep"

# --- OlaresManifest.yaml ---
cat > "${APP_DIR}/OlaresManifest.yaml" << MANIFEST
---
olaresManifest.version: '0.10.0'
olaresManifest.type: app
apiVersion: 'v2'
metadata:
  name: ${APP_ID}
  icon: ${ICON}
  description: "${MODEL_NAME} served via ${BACKEND_LABEL}"
  appid: ${APP_ID}
  title: ${TITLE}
  version: '1.0.0'
  categories:
    - AI
sharedEntrances:
  - name: ${APP_ID}
    host: sharedentrances-${SERVER_SERVICE}
    port: 0
    title: ${MODEL_NAME} (${BACKEND_LABEL})
    invisible: true
    authLevel: internal
    icon: ${ICON}
entrances:
  - name: ${CLIENT_SERVICE}
    port: 8080
    host: ${CLIENT_SERVICE}
    title: ${TITLE}
    icon: ${ICON}
    openMethod: window
    authLevel: internal
spec:
  versionName: '1.0.0'
  fullDescription: |
    ## IMPORTANT NOTE ##
    This is a shared app. Once installed by the Olares Admin, all users in the cluster can use it through reference app.

    ## MODEL ##
    ${MODEL_NAME} served via ${BACKEND_LABEL}.
    Auto-registers with LiteLLM Gateway if installed.

  developer: ${DEVELOPER}
  website: ${WEBSITE}
  submitter: orales-market
  locale:
    - en-US
  license:
    - text: ${LICENSE_TEXT}
      url: ${LICENSE_URL}

  {{- if and .Values.admin .Values.bfl.username (eq .Values.admin .Values.bfl.username) }}
  limitedCpu: 8000m
  requiredCpu: 1000m
  requiredDisk: ${DISK}
  limitedDisk: 30Gi
  limitedMemory: ${RAM}
  requiredMemory: 4Gi
  requiredGpu: 1Gi
  limitedGpu: ${GPU_VRAM}
  {{- else }}
  requiredMemory: 64Mi
  limitedMemory: 800Mi
  requiredDisk: 50Mi
  limitedDisk: 200Mi
  requiredCpu: 10m
  limitedCpu: 800m
  {{- end }}

  supportArch:
${ARCH_YAML}
  subCharts:
  - name: ${SERVER_ID}
    shared: true
  - name: ${APP_ID}
permission:
  appData: true
  appCache: true
  userData:
    - Home/${STORAGE_DIR}
options:
  apiTimeout: 0
  appScope:
  {{- if and .Values.admin .Values.bfl.username (eq .Values.admin .Values.bfl.username) }}
    clusterScoped: true
    appRef:
      - ${APP_ID}
  {{- else }}
    clusterScoped: false
  {{- end }}
  dependencies:
    - name: olares
      version: '>=1.12.3-0'
      type: system
  {{- if and .Values.admin .Values.bfl.username (eq .Values.admin .Values.bfl.username) }}
    - name: litellmgateway
      type: application
      version: '>=1.0.0'
      mandatory: false
  {{- else }}
    - name: ${APP_ID}
      type: application
      version: '>=1.0.0'
      mandatory: true
  {{- end }}
MANIFEST

# --- i18n ---
cat > "${APP_DIR}/i18n/en-US/OlaresManifest.yaml" << EOF
metadata:
  title: ${TITLE}
  description: "${MODEL_NAME} served via ${BACKEND_LABEL}"

spec:
  fullDescription: |
    ## IMPORTANT NOTE ##
    This is a shared app. Once installed by the Olares Admin, all users in the cluster can use it through reference app.

    ## MODEL ##
    ${MODEL_NAME} served via ${BACKEND_LABEL}.
    Auto-registers with LiteLLM Gateway if installed.
EOF

# --- Per-user proxy subchart ---
cat > "${APP_DIR}/${APP_ID}/Chart.yaml" << EOF
apiVersion: v2
appVersion: '1.25.3-2'
description: ${BACKEND_LABEL} client proxy
name: ${APP_ID}
type: application
version: '1.0.0'
EOF

touch "${APP_DIR}/${APP_ID}/values.yaml"

cat > "${APP_DIR}/${APP_ID}/templates/clientproxy.yaml" << PROXY
---
apiVersion: v1
data:
  nginx.conf: |
    server {

      listen 8080;
      access_log /opt/bitnami/openresty/nginx/logs/access.log;
      error_log  /opt/bitnami/openresty/nginx/logs/error.log;

      proxy_connect_timeout                          600s;
      proxy_send_timeout                             600s;
      proxy_read_timeout                             1800s;
      proxy_set_header      host                      \$host;
      proxy_set_header      x-forwarded-host          \$http_host;

      proxy_http_version 1.1;

      proxy_set_header upgrade \$http_upgrade;
      proxy_set_header connection "upgrade";

      location / {
        add_header X-Frame-Options "";
        proxy_pass http://${SERVER_SERVICE}.${SERVER_ID}-shared:${BACKEND_PORT};
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
    io.kompose.service: ${CLIENT_SERVICE}
  name: {{ .Release.Name }}
  namespace: '{{ .Release.Namespace }}'
spec:
  replicas: 1
  selector:
    matchLabels:
      io.kompose.service: ${CLIENT_SERVICE}
  template:
    metadata:
      labels:
        io.kompose.network/chrome-default: "true"
        io.kompose.service: ${CLIENT_SERVICE}
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
          env:
            - name: OPENRESTY_CONF_FILE
              value: /etc/nginx/nginx.conf
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
              mountPath: /etc/nginx/nginx.conf
              subPath: nginx.conf
            - name: nginx-config
              mountPath: /opt/bitnami/openresty/nginx/conf/server_blocks/nginx.conf
              subPath: nginx.conf

---
apiVersion: v1
kind: Service
metadata:
  name: ${CLIENT_SERVICE}
  namespace: {{ .Release.Namespace }}
spec:
  type: ClusterIP
  selector:
    io.kompose.service: ${CLIENT_SERVICE}
  ports:
    - name: ${CLIENT_SERVICE}
      protocol: TCP
      port: 8080
      targetPort: 8080
PROXY

# --- Server subchart ---
cat > "${APP_DIR}/${SERVER_ID}/Chart.yaml" << EOF
apiVersion: v2
appVersion: '1.0.0'
description: ${BACKEND_LABEL} server for ${MODEL_NAME}
name: ${SERVER_ID}
type: application
version: '1.0.0'
EOF

touch "${APP_DIR}/${SERVER_ID}/values.yaml"

# --- Server configmap ---
if [[ "$BACKEND" == "llamacpp" ]]; then
cat > "${APP_DIR}/${SERVER_ID}/templates/configmap.yaml" << CFGEOF
{{- if and .Values.admin .Values.bfl.username (eq .Values.admin .Values.bfl.username) }}
---
apiVersion: v1
kind: ConfigMap
metadata:
  name: ${SERVER_SERVICE}-env
  namespace: "{{ .Release.Namespace }}"
  labels:
    app: ${APP_ID}
    backend: llamacpp
data:
  MODEL_URL: "${MODEL_SOURCE}"
  MODEL_FILE: "${MODEL_FILE}"
  MODEL_ALIAS: "${MODEL_ALIAS}"
  CONTEXT_SIZE: "${CONTEXT_SIZE}"
  N_GPU_LAYERS: "${GPU_LAYERS}"
  THREADS: "${THREADS}"
  LITELLM_MODEL_NAME: "${MODEL_ALIAS}"
  LITELLM_SERVICE_URL: "http://litellm.litellmgatewayserver-shared:4000"
{{- end }}
CFGEOF
else
cat > "${APP_DIR}/${SERVER_ID}/templates/configmap.yaml" << CFGEOF
{{- if and .Values.admin .Values.bfl.username (eq .Values.admin .Values.bfl.username) }}
---
apiVersion: v1
kind: ConfigMap
metadata:
  name: ${SERVER_SERVICE}-env
  namespace: "{{ .Release.Namespace }}"
  labels:
    app: ${APP_ID}
    backend: vllm
data:
  MODEL_NAME: "${MODEL_SOURCE}"
  MODEL_ALIAS: "${MODEL_ALIAS}"
  MAX_MODEL_LEN: "${CONTEXT_SIZE}"
  GPU_MEMORY_UTILIZATION: "${GPU_UTIL}"
  DTYPE: "${DTYPE}"
  LITELLM_MODEL_NAME: "${MODEL_ALIAS}"
  LITELLM_SERVICE_URL: "http://litellm.litellmgatewayserver-shared:4000"
{{- end }}
CFGEOF
fi

# --- Server deployment ---
if [[ "$BACKEND" == "llamacpp" ]]; then
cat > "${APP_DIR}/${SERVER_ID}/templates/deployment.yaml" << 'DEPEOF'
{{- if and .Values.admin .Values.bfl.username (eq .Values.admin .Values.bfl.username) }}
---
apiVersion: apps/v1
kind: Deployment
metadata:
  labels:
    io.kompose.service: SVCNAME
    llm-backend: llamacpp
  name: SVCNAME
  namespace: "{{ .Release.Namespace }}"
  annotations:
    applications.app.bytetrade.io/gpu-inject: "true"
spec:
  replicas: 1
  selector:
    matchLabels:
      io.kompose.service: SVCNAME
  strategy:
    type: Recreate
  template:
    metadata:
      labels:
        io.kompose.network/chrome-default: "true"
        io.kompose.service: SVCNAME
    spec:
      initContainers:
        - name: model-downloader
          image: "docker.io/curlimages/curl:8.11.0"
          command:
            - sh
            - '-c'
            - |
              MODEL_PATH="/models/${MODEL_FILE}"
              if [ -f "$MODEL_PATH" ]; then
                echo "Model already downloaded: $MODEL_PATH"
                ls -lh "$MODEL_PATH"
              else
                echo "Downloading model from $MODEL_URL ..."
                curl -L --progress-bar -o "$MODEL_PATH" "$MODEL_URL"
                echo "Download complete."
                ls -lh "$MODEL_PATH"
              fi
          envFrom:
            - configMapRef:
                name: SVCNAME-env
          resources:
            limits:
              cpu: "1"
              memory: 512Mi
            requests:
              cpu: 100m
              memory: 128Mi
          volumeMounts:
            - mountPath: "/models"
              name: models
      containers:
        - name: llamacpp-server
          image: "BACKEND_IMAGE_PLACEHOLDER"
          args:
            - "--host"
            - "0.0.0.0"
            - "--port"
            - "PORTNUM"
            - "--model"
            - "/models/$(MODEL_FILE)"
            - "--alias"
            - "$(MODEL_ALIAS)"
            - "--ctx-size"
            - "$(CONTEXT_SIZE)"
            - "--n-gpu-layers"
            - "$(N_GPU_LAYERS)"
            - "--threads"
            - "$(THREADS)"
          envFrom:
            - configMapRef:
                name: SVCNAME-env
          ports:
            - containerPort: PORTNUM
          livenessProbe:
            httpGet:
              path: /health
              port: PORTNUM
              scheme: HTTP
            initialDelaySeconds: 60
            timeoutSeconds: 10
            periodSeconds: 30
            failureThreshold: 5
          startupProbe:
            httpGet:
              path: /health
              port: PORTNUM
              scheme: HTTP
            initialDelaySeconds: 30
            timeoutSeconds: 10
            periodSeconds: 10
            failureThreshold: 60
          resources:
            limits:
              cpu: "6"
              memory: RAMLIMIT
            requests:
              cpu: 500m
              memory: 2Gi
          volumeMounts:
            - mountPath: "/models"
              name: models
        - name: litellm-register
          image: "docker.io/curlimages/curl:8.11.0"
          command:
            - sh
            - '-c'
            - |
              until curl -sf http://localhost:PORTNUM/health > /dev/null 2>&1; do
                sleep 5
              done
              LITELLM_KEY="sk-olares-litellm-{{ .Release.Namespace }}"
              LITELLM_URL="${LITELLM_SERVICE_URL}"
              BACKEND_URL="http://SVCNAME.SERVERID-shared:PORTNUM/v1"
              register_model() {
                curl -sf -o /dev/null -w "%{http_code}" \
                  -X POST "${LITELLM_URL}/model/new" \
                  -H "Authorization: Bearer ${LITELLM_KEY}" \
                  -H "Content-Type: application/json" \
                  -d "{\"model_name\": \"${LITELLM_MODEL_NAME}\", \"litellm_params\": {\"model\": \"openai/${MODEL_ALIAS}\", \"api_base\": \"${BACKEND_URL}\"}}"
              }
              for i in $(seq 1 5); do register_model && break; sleep 30; done
              while true; do sleep 3600; register_model; done
          envFrom:
            - configMapRef:
                name: SVCNAME-env
          resources:
            limits:
              cpu: 50m
              memory: 32Mi
            requests:
              cpu: 5m
              memory: 16Mi
      volumes:
        - name: models
          hostPath:
            path: "{{ .Values.userspace.userData }}/STORAGEDIR/{{ .Release.Name }}/models"
            type: DirectoryOrCreate
      restartPolicy: Always
---
apiVersion: v1
kind: Service
metadata:
  labels:
    io.kompose.service: SVCNAME
  name: SVCNAME
  namespace: "{{ .Release.Namespace }}"
spec:
  ports:
    - name: "SVCNAME"
      port: PORTNUM
      targetPort: PORTNUM
  selector:
    io.kompose.service: SVCNAME
---
apiVersion: v1
kind: Service
metadata:
  labels:
    io.kompose.service: SVCNAME
  name: sharedentrances-SVCNAME
  namespace: "{{ .Release.Namespace }}"
spec:
  ports:
    - name: "SVCNAME"
      port: 80
      targetPort: PORTNUM
  selector:
    io.kompose.service: SVCNAME
{{- end }}
DEPEOF

  # Replace placeholders
  sed -i '' \
    -e "s|SVCNAME|${SERVER_SERVICE}|g" \
    -e "s|SERVERID|${SERVER_ID}|g" \
    -e "s|BACKEND_IMAGE_PLACEHOLDER|${BACKEND_IMAGE}|g" \
    -e "s|PORTNUM|${BACKEND_PORT}|g" \
    -e "s|RAMLIMIT|${RAM}|g" \
    -e "s|STORAGEDIR|${STORAGE_DIR}|g" \
    "${APP_DIR}/${SERVER_ID}/templates/deployment.yaml"

else
  # vLLM deployment
cat > "${APP_DIR}/${SERVER_ID}/templates/deployment.yaml" << 'DEPEOF'
{{- if and .Values.admin .Values.bfl.username (eq .Values.admin .Values.bfl.username) }}
---
apiVersion: apps/v1
kind: Deployment
metadata:
  labels:
    io.kompose.service: SVCNAME
    llm-backend: vllm
  name: SVCNAME
  namespace: "{{ .Release.Namespace }}"
  annotations:
    applications.app.bytetrade.io/gpu-inject: "true"
spec:
  replicas: 1
  selector:
    matchLabels:
      io.kompose.service: SVCNAME
  strategy:
    type: Recreate
  template:
    metadata:
      labels:
        io.kompose.network/chrome-default: "true"
        io.kompose.service: SVCNAME
    spec:
      containers:
        - name: vllm-server
          image: "BACKEND_IMAGE_PLACEHOLDER"
          args:
            - "--model"
            - "$(MODEL_NAME)"
            - "--served-model-name"
            - "$(MODEL_ALIAS)"
            - "--host"
            - "0.0.0.0"
            - "--port"
            - "PORTNUM"
            - "--max-model-len"
            - "$(MAX_MODEL_LEN)"
            - "--gpu-memory-utilization"
            - "$(GPU_MEMORY_UTILIZATION)"
            - "--dtype"
            - "$(DTYPE)"
            - "--trust-remote-code"
            - "--download-dir"
            - "/models"
          envFrom:
            - configMapRef:
                name: SVCNAME-env
          env:
            - name: HF_HOME
              value: "/models/huggingface"
          ports:
            - containerPort: PORTNUM
          livenessProbe:
            httpGet:
              path: /health
              port: PORTNUM
              scheme: HTTP
            initialDelaySeconds: 120
            timeoutSeconds: 10
            periodSeconds: 30
            failureThreshold: 5
          startupProbe:
            httpGet:
              path: /health
              port: PORTNUM
              scheme: HTTP
            initialDelaySeconds: 60
            timeoutSeconds: 10
            periodSeconds: 15
            failureThreshold: 40
          resources:
            limits:
              cpu: "6"
              memory: RAMLIMIT
            requests:
              cpu: 1000m
              memory: 6Gi
          volumeMounts:
            - mountPath: "/models"
              name: models
        - name: litellm-register
          image: "docker.io/curlimages/curl:8.11.0"
          command:
            - sh
            - '-c'
            - |
              until curl -sf http://localhost:PORTNUM/health > /dev/null 2>&1; do
                sleep 10
              done
              LITELLM_KEY="sk-olares-litellm-{{ .Release.Namespace }}"
              LITELLM_URL="${LITELLM_SERVICE_URL}"
              BACKEND_URL="http://SVCNAME.SERVERID-shared:PORTNUM/v1"
              register_model() {
                curl -sf -o /dev/null -w "%{http_code}" \
                  -X POST "${LITELLM_URL}/model/new" \
                  -H "Authorization: Bearer ${LITELLM_KEY}" \
                  -H "Content-Type: application/json" \
                  -d "{\"model_name\": \"${LITELLM_MODEL_NAME}\", \"litellm_params\": {\"model\": \"openai/${MODEL_ALIAS}\", \"api_base\": \"${BACKEND_URL}\"}}"
              }
              for i in $(seq 1 5); do register_model && break; sleep 30; done
              while true; do sleep 3600; register_model; done
          envFrom:
            - configMapRef:
                name: SVCNAME-env
          resources:
            limits:
              cpu: 50m
              memory: 32Mi
            requests:
              cpu: 5m
              memory: 16Mi
      volumes:
        - name: models
          hostPath:
            path: "{{ .Values.userspace.userData }}/STORAGEDIR/{{ .Release.Name }}/models"
            type: DirectoryOrCreate
      restartPolicy: Always
---
apiVersion: v1
kind: Service
metadata:
  labels:
    io.kompose.service: SVCNAME
  name: SVCNAME
  namespace: "{{ .Release.Namespace }}"
spec:
  ports:
    - name: "SVCNAME"
      port: PORTNUM
      targetPort: PORTNUM
  selector:
    io.kompose.service: SVCNAME
---
apiVersion: v1
kind: Service
metadata:
  labels:
    io.kompose.service: SVCNAME
  name: sharedentrances-SVCNAME
  namespace: "{{ .Release.Namespace }}"
spec:
  ports:
    - name: "SVCNAME"
      port: 80
      targetPort: PORTNUM
  selector:
    io.kompose.service: SVCNAME
{{- end }}
DEPEOF

  sed -i '' \
    -e "s|SVCNAME|${SERVER_SERVICE}|g" \
    -e "s|SERVERID|${SERVER_ID}|g" \
    -e "s|BACKEND_IMAGE_PLACEHOLDER|${BACKEND_IMAGE}|g" \
    -e "s|PORTNUM|${BACKEND_PORT}|g" \
    -e "s|RAMLIMIT|${RAM}|g" \
    -e "s|STORAGEDIR|${STORAGE_DIR}|g" \
    "${APP_DIR}/${SERVER_ID}/templates/deployment.yaml"
fi

echo ""
echo "App generated at: ${APP_DIR}/"
echo ""
find "${APP_DIR}" -type f | sort | sed "s|${SCRIPT_DIR}/||"
echo ""
echo "Next steps:"
echo "  1. Review the generated files"
echo "  2. Submit to the Olares market repository"
