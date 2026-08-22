#!/usr/bin/env python3
"""Scaffold all LingBot Olares One charts (v3 shared + soft-ready boot)."""

from __future__ import annotations

import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SHARED = Path(__file__).resolve().parent / "lingbot" / "shared"

APPS = [
    {
        "name": "lingbotdepthone",
        "title": "LingBot Depth One",
        "short": "Depth completion Gradio + REST (ViT-L v0.5)",
        "desc": "LingBot-Depth metric depth refinement / completion for Olares One.",
        "repo": "https://github.com/Robbyant/lingbot-depth",
        "hf": "robbyant/lingbot-depth-pretrain-vitl-14-v0.5",
        "website": "https://technology.robbyant.com/lingbot-depth",
        "disk_req": "20Gi",
        "disk_lim": "60Gi",
        "mem_lim": "32Gi",
        "cpu_lim": "10",
        "gpu_lim": "24Gi",
        "accent": (14, 165, 233),
        "pill": "Depth",
        "pipeline": "depth-estimation",
        "extra_env": {},
        "git_url": "https://github.com/Robbyant/lingbot-depth.git",
        "git_dir": "lingbot-depth",
        "pip_extra": "opencv-python-headless pillow numpy",
        "kind": "depth",
    },
    {
        "name": "lingbotvisionone",
        "title": "LingBot Vision One",
        "short": "Vision backbone PCA Gradio + REST (ViT-L)",
        "desc": "LingBot-Vision dense spatial features (masked boundary modeling).",
        "repo": "https://github.com/Robbyant/lingbot-vision",
        "hf": "robbyant/lingbot-vision-vit-large",
        "website": "https://technology.robbyant.com/lingbot-vision",
        "disk_req": "15Gi",
        "disk_lim": "40Gi",
        "mem_lim": "24Gi",
        "cpu_lim": "8",
        "gpu_lim": "24Gi",
        "accent": (59, 130, 246),
        "pill": "Vision",
        "pipeline": "image-feature-extraction",
        "extra_env": {"LINGBOT_VISION_VARIANT": "large"},
        "git_url": "https://github.com/Robbyant/lingbot-vision.git",
        "git_dir": "lingbot-vision",
        "pip_extra": "omegaconf scikit-learn opencv-python-headless pillow numpy",
        "kind": "vision",
    },
    {
        "name": "lingbotmapone",
        "title": "LingBot Map One",
        "short": "Streaming 3D reconstruction (viser + REST)",
        "desc": "LingBot-Map Geometric Context Transformer for streaming 3D reconstruction.",
        "repo": "https://github.com/Robbyant/lingbot-map",
        "hf": "robbyant/lingbot-map",
        "website": "https://technology.robbyant.com/lingbot-map",
        "disk_req": "30Gi",
        "disk_lim": "80Gi",
        "mem_lim": "48Gi",
        "cpu_lim": "12",
        "gpu_lim": "24Gi",
        "accent": (16, 185, 129),
        "pill": "3D Map",
        "pipeline": "image-to-3d",
        "extra_env": {"LINGBOT_MAP_CKPT": "lingbot-map-long.pt"},
        "git_url": "https://github.com/Robbyant/lingbot-map.git",
        "git_dir": "lingbot-map",
        "pip_extra": "opencv-python-headless pillow numpy viser imageio",
        "kind": "map",
    },
    {
        "name": "lingbotvideoone",
        "title": "LingBot Video Dense One",
        "short": "T2V / TI2V Gradio (Dense 1.3B single-GPU)",
        "desc": "LingBot-Video Dense 1.3B — embodied video generation on 5090M.",
        "repo": "https://github.com/Robbyant/lingbot-video",
        "hf": "robbyant/lingbot-video-dense-1.3b",
        "website": "https://technology.robbyant.com/lingbot-video",
        "disk_req": "40Gi",
        "disk_lim": "100Gi",
        "mem_lim": "48Gi",
        "cpu_lim": "12",
        "gpu_lim": "24Gi",
        "accent": (236, 72, 153),
        "pill": "T2V Dense",
        "pipeline": "text-to-video",
        "extra_env": {},
        "git_url": "https://github.com/Robbyant/lingbot-video.git",
        "git_dir": "lingbot-video",
        "pip_extra": "opencv-python-headless pillow numpy imageio imageio-ffmpeg einops",
        "kind": "video",
    },
    {
        "name": "lingbotvlaone",
        "title": "LingBot VLA 2.0 One",
        "short": "VLA 6B policy API + Gradio (native depth)",
        "desc": "LingBot-VLA 2.0 — vision-language-action foundation for robot control.",
        "repo": "https://github.com/Robbyant/lingbot-vla-v2",
        "hf": "robbyant/lingbot-vla-v2-6b",
        "website": "https://technology.robbyant.com/lingbot-vla-v2",
        "disk_req": "50Gi",
        "disk_lim": "120Gi",
        "mem_lim": "48Gi",
        "cpu_lim": "12",
        "gpu_lim": "24Gi",
        "accent": (249, 115, 22),
        "pill": "VLA 6B",
        "pipeline": "robotics",
        "extra_env": {},
        "git_url": "https://github.com/Robbyant/lingbot-vla-v2.git",
        "git_dir": "lingbot-vla-v2",
        "pip_extra": "opencv-python-headless pillow numpy",
        "kind": "vla",
    },
    {
        "name": "lingbotvaone",
        "title": "LingBot VA One",
        "short": "Video-Action world model (UMT5+VAE CPU offload)",
        "desc": "LingBot-VA causal video-action model with layer/CPU offload for 24GB.",
        "repo": "https://github.com/Robbyant/lingbot-va",
        "hf": "robbyant/lingbot-va-base",
        "website": "https://technology.robbyant.com/lingbot-va",
        "disk_req": "50Gi",
        "disk_lim": "120Gi",
        "mem_lim": "64Gi",
        "cpu_lim": "16",
        "gpu_lim": "24Gi",
        "accent": (168, 85, 247),
        "pill": "VA Offload",
        "pipeline": "robotics",
        "extra_env": {"LINGBOT_VA_OFFLOAD": "1"},
        "git_url": "https://github.com/Robbyant/lingbot-va.git",
        "git_dir": "lingbot-va",
        "pip_extra": "opencv-python-headless pillow numpy einops imageio",
        "kind": "va",
    },
    {
        "name": "lingbotworldone",
        "title": "LingBot World NF4 One",
        "short": "World I2V NF4 + T5 CPU / layer offload (24GB)",
        "desc": "LingBot-World Base Cam NF4 (community) with aggressive offload for 5090M.",
        "repo": "https://github.com/Robbyant/lingbot-world",
        "hf": "cahlen/lingbot-world-base-cam-nf4",
        "website": "https://technology.robbyant.com/lingbot-world",
        "disk_req": "80Gi",
        "disk_lim": "200Gi",
        "mem_lim": "64Gi",
        "cpu_lim": "16",
        "gpu_lim": "24Gi",
        "accent": (234, 88, 12),
        "pill": "World NF4",
        "pipeline": "image-to-video",
        "extra_env": {
            "LINGBOT_WORLD_T5_CPU": "1",
            "LINGBOT_WORLD_LAYER_OFFLOAD": "1",
            "LINGBOT_WORLD_MAX_FRAMES": "49",
        },
        "git_url": "https://github.com/Robbyant/lingbot-world.git",
        "git_dir": "lingbot-world",
        "pip_extra": "bitsandbytes safetensors einops imageio imageio-ffmpeg pillow numpy tqdm easydict",
        "kind": "world",
    },
]


def indent_block(text: str, n: int = 4) -> str:
    pad = " " * n
    return "\n".join(pad + line if line else line for line in text.replace("\r\n", "\n").split("\n"))


def write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content if content.endswith("\n") else content + "\n", encoding="utf-8")
    print("wrote", path.relative_to(ROOT))


def clientproxy(app: dict) -> str:
    name = app["name"]
    cli = f"{name}cli"
    return f"""---
apiVersion: v1
data:
  nginx.conf: |
    server {{

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

      location / {{
        add_header X-Frame-Options "";
        proxy_pass http://{name}:7860;
      }}
    }}

kind: ConfigMap
metadata:
  name: nginx-config
  namespace: {{{{ .Release.Namespace }}}}

---
apiVersion: apps/v1
kind: Deployment
metadata:
  labels:
    io.kompose.service: {cli}
  name: {cli}
  namespace: '{{{{ .Release.Namespace }}}}'
spec:
  replicas: {{{{ .Values.workloads.{cli}.replicaCount }}}}
  selector:
    matchLabels:
      io.kompose.service: {cli}
  template:
    metadata:
      labels:
        io.kompose.network/chrome-default: "true"
        io.kompose.service: {cli}
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
  name: {cli}
  namespace: {{{{ .Release.Namespace }}}}
spec:
  type: ClusterIP
  selector:
    io.kompose.service: {cli}
  ports:
    - name: {cli}
      protocol: TCP
      port: 8080
      targetPort: 8080
"""


def server_yaml(app: dict) -> str:
    name = app["name"]
    git_url = app["git_url"]
    git_dir = app["git_dir"]
    hf = app["hf"]
    kind = app["kind"]
    pip_extra = app["pip_extra"]

    extra_env_yaml = ""
    for k, v in app.get("extra_env", {}).items():
        extra_env_yaml += f"""
            - name: {k}
              value: {{{{ .Values.olaresEnv.{k} | default "{v}" | quote }}}}"""

    return f"""---
apiVersion: apps/v1
kind: Deployment
metadata:
  labels:
    io.kompose.service: {name}
  name: {name}
  namespace: "{{{{ .Release.Namespace }}}}"
  annotations:
    applications.app.bytetrade.io/gpu-inject: "true"
spec:
  replicas: {{{{ .Values.workloads.{name}.replicaCount }}}}
  selector:
    matchLabels:
      io.kompose.service: {name}
  strategy:
    type: Recreate
  template:
    metadata:
      labels:
        io.kompose.network/chrome-default: "true"
        io.kompose.service: {name}
    spec:
      initContainers:
        - name: volume-perms
          image: "docker.io/beclab/aboveos-busybox:1.37.0"
          command:
            - sh
            - "-c"
            - |
              set -eux
              chmod -R 777 /models /workspace /output || true
          volumeMounts:
            - mountPath: "/models"
              name: models
            - mountPath: "/workspace"
              name: workspace
            - mountPath: "/output"
              name: output
          securityContext:
            runAsUser: 0
      containers:
        - name: lingbot
          image: {{{{ printf "%s:%s" (.Values.image.repository | default "pytorch/pytorch") (.Values.image.tag | default "2.12.0-cuda13.0-cudnn9-devel") | quote }}}}
          command:
            - "bash"
            - "-lc"
          args:
            - |
              set -euo pipefail
              APP_DIR="/workspace/{name}"
              SITE_PKGS="$APP_DIR/site-packages"
              PHASE_FILE="$APP_DIR/.boot-phase"
              ATTEMPTS_FILE="$APP_DIR/bootstrap.log"
              SRC_DIR="$APP_DIR/src/{git_dir}"
              MODEL_DIR="/models/{name}/weights"
              mkdir -p "$APP_DIR" "$SITE_PKGS" /output/gradio "$MODEL_DIR" "$APP_DIR/src" "$APP_DIR/bin" "$APP_DIR/uv-cache"
              cp /app-src/app.py "$APP_DIR/app.py"
              cp /app-src/requirements.txt "$APP_DIR/requirements.txt"
              cp /app-src/soft_ready.py "$APP_DIR/soft_ready.py"
              cp /app-src/download_models.py "$APP_DIR/download_models.py"
              cp /app-src/_common.py "$APP_DIR/_common.py"
              cd "$APP_DIR"

              export LINGBOT_APP="{name}"
              export LINGBOT_KIND="{kind}"
              export LINGBOT_BOOT_PHASE_FILE="$PHASE_FILE"
              export LINGBOT_BOOT_ATTEMPTS_FILE="$ATTEMPTS_FILE"
              export SERVER_PORT="${{SERVER_PORT:-7860}}"
              export HF_HOME="/models/huggingface"
              export HF_HUB_CACHE="$HF_HOME/hub"
              export TRANSFORMERS_CACHE="$HF_HOME/transformers"
              export GRADIO_TEMP_DIR="/output/gradio"
              export PYTHONNOUSERSITE=1
              export UV_CACHE_DIR="$APP_DIR/uv-cache"
              export UV_CONCURRENT_DOWNLOADS="${{UV_CONCURRENT_DOWNLOADS:-32}}"

              set_phase() {{
                printf '%s\\n' "$1" > "$PHASE_FILE"
                echo "[{name}] phase=$1" | tee -a "$ATTEMPTS_FILE"
              }}
              log_attempt() {{
                printf '%s %s\\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$1" | tee -a "$ATTEMPTS_FILE"
              }}

              # Pick image python with torch (do not put SITE_PKGS on PYTHONPATH yet).
              rm -rf "$APP_DIR/.venv" "$APP_DIR/venv" 2>/dev/null || true
              BASE_PY=""
              for c in /opt/conda/bin/python /opt/conda/bin/python3 /usr/local/bin/python /usr/local/bin/python3 "$(command -v python || true)" "$(command -v python3 || true)"; do
                [ -n "$c" ] && [ -x "$c" ] || continue
                if PYTHONPATH= PYTHONNOUSERSITE=1 "$c" -c "import torch" 2>/dev/null; then
                  BASE_PY="$c"
                  break
                fi
              done
              if [ -z "$BASE_PY" ]; then
                log_attempt "FATAL: no python with torch"
                exit 1
              fi
              export UV_PYTHON="$BASE_PY"
              log_attempt "base_python=$BASE_PY"
              PYTHONPATH= PYTHONNOUSERSITE=1 "$BASE_PY" -c "import torch; print('[{name}] image torch', torch.__version__, 'cuda', torch.version.cuda)"

              purge_torch_shadow() {{
                log_attempt "purging torch/nvidia stacks from site-packages (image owns torch)"
                rm -rf \\
                  "$SITE_PKGS"/torch "$SITE_PKGS"/torch-* "$SITE_PKGS"/torchgen "$SITE_PKGS"/functorch \\
                  "$SITE_PKGS"/torchvision "$SITE_PKGS"/torchvision-* \\
                  "$SITE_PKGS"/torchaudio "$SITE_PKGS"/torchaudio-* \\
                  "$SITE_PKGS"/triton "$SITE_PKGS"/triton-* \\
                  "$SITE_PKGS"/nvidia "$SITE_PKGS"/nvidia_* 2>/dev/null || true
              }}

              # Soft-ready FIRST
              set_phase "installing:soft_ready_starting"
              PYTHONPATH= "$BASE_PY" "$APP_DIR/soft_ready.py" &
              SOFT_PID=$!
              log_attempt "soft-ready pid=$SOFT_PID"
              sleep 1

              stop_soft_ready() {{
                if [ -n "${{SOFT_PID:-}}" ] && kill -0 "$SOFT_PID" 2>/dev/null; then
                  log_attempt "stopping soft-ready pid=$SOFT_PID"
                  kill "$SOFT_PID" 2>/dev/null || true
                  wait "$SOFT_PID" 2>/dev/null || true
                fi
                if command -v fuser >/dev/null 2>&1; then
                  fuser -k "${{SERVER_PORT}}/tcp" 2>/dev/null || true
                fi
                sleep 1
              }}
              trap 'stop_soft_ready' EXIT

              purge_torch_shadow
              export PYTHONPATH="$SITE_PKGS${{PYTHONPATH:+:$PYTHONPATH}}"

              # Optional PyPI mirror
              if [ -z "${{UV_INDEX_URL:-}}" ] && [ -n "${{PIP_INDEX_URL:-}}" ]; then
                export UV_INDEX_URL="$PIP_INDEX_URL"
              fi
              if [ -z "${{UV_INDEX_URL:-}}" ]; then
                case "${{HF_ENDPOINT:-}}" in
                  *hf-mirror*|*huggingface.co.cn*|*hf-mirror.com*)
                    export UV_INDEX_URL="https://pypi.tuna.tsinghua.edu.cn/simple"
                    log_attempt "HF China mirror → UV_INDEX_URL=$UV_INDEX_URL"
                    ;;
                esac
              fi

              UV="$APP_DIR/bin/uv"
              if [ ! -x "$UV" ]; then
                set_phase "installing:uv"
                log_attempt "bootstrapping uv into $APP_DIR/bin"
                if ! command -v curl >/dev/null 2>&1; then
                  apt-get update
                  apt-get install -y --no-install-recommends curl ca-certificates git
                  rm -rf /var/lib/apt/lists/*
                fi
                if ! command -v git >/dev/null 2>&1; then
                  apt-get update
                  apt-get install -y --no-install-recommends git
                  rm -rf /var/lib/apt/lists/*
                fi
                if ! curl -fsSL https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="$APP_DIR/bin" UV_NO_MODIFY_PATH=1 sh; then
                  tmpd="$(mktemp -d)"
                  curl -fL "https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-unknown-linux-gnu.tar.gz" -o "$tmpd/uv.tgz"
                  tar -xzf "$tmpd/uv.tgz" -C "$tmpd"
                  find "$tmpd" -type f -name uv -exec mv {{}} "$UV" \\;
                  rm -rf "$tmpd"
                  chmod +x "$UV"
                fi
              fi
              if [ ! -x "$UV" ]; then
                log_attempt "FATAL: uv missing"
                exit 1
              fi
              log_attempt "uv=$("$UV" --version 2>&1 | head -1)"

              uvpip() {{
                local extra=()
                if [ -n "${{UV_INDEX_URL:-}}" ]; then
                  extra+=(--index-url "$UV_INDEX_URL")
                fi
                "$UV" pip install --python "$BASE_PY" --target "$SITE_PKGS" "${{extra[@]}}" "$@"
              }}

              REQ_HASH="$("$BASE_PY" - <<'PY'
              import hashlib, pathlib
              print(hashlib.sha256(pathlib.Path("requirements.txt").read_bytes()).hexdigest()[:16])
              PY
              )"
              DEPS_MARK="$APP_DIR/.deps-ok-v2-$REQ_HASH"
              deps_ok=0
              if [ -f "$DEPS_MARK" ]; then
                if "$BASE_PY" -c "import gradio, fastapi, uvicorn, huggingface_hub" 2>/dev/null \\
                  && ! [ -d "$SITE_PKGS/torch" ]; then
                  deps_ok=1
                  log_attempt "deps marker OK — skip uv pip -r"
                else
                  rm -f "$DEPS_MARK"
                fi
              fi
              if [ "$deps_ok" != "1" ]; then
                set_phase "installing:uv_requirements"
                uvpip --upgrade "pydantic>=2.10,<2.12" "fastapi>=0.115.2,<1.0" "uvicorn[standard]>=0.30.0,<1.0"
                uvpip --upgrade -r requirements.txt
                # extras for this chart
                uvpip --upgrade {pip_extra} || true
                purge_torch_shadow
                touch "$DEPS_MARK"
                log_attempt "wrote $DEPS_MARK"
              fi

              # Clone upstream source (editable path for imports)
              if [ ! -d "$SRC_DIR/.git" ]; then
                set_phase "installing:git_clone"
                log_attempt "git clone {git_url}"
                rm -rf "$SRC_DIR"
                git clone --depth 1 "{git_url}" "$SRC_DIR" || {{
                  log_attempt "git clone FAIL — retry once"
                  sleep 5
                  rm -rf "$SRC_DIR"
                  git clone --depth 1 "{git_url}" "$SRC_DIR"
                }}
              else
                log_attempt "git src present — skip clone"
              fi
              export LINGBOT_SRC="$SRC_DIR"
              export PYTHONPATH="$SRC_DIR:$SITE_PKGS${{PYTHONPATH:+:$PYTHONPATH}}"

              # Soft-install upstream package if setup exists
              if [ -f "$SRC_DIR/pyproject.toml" ] || [ -f "$SRC_DIR/setup.py" ] || [ -f "$SRC_DIR/setup.cfg" ]; then
                set_phase "installing:upstream_editable"
                log_attempt "uv pip install upstream (no-deps when possible)"
                uvpip --upgrade --no-deps "$SRC_DIR" || uvpip --upgrade "$SRC_DIR" || log_attempt "upstream install soft-fail"
                purge_torch_shadow
              fi

              # Model download with progress / attempts logged
              export HF_REPO_ID="${{HF_REPO_ID:-{hf}}}"
              export HF_LOCAL_DIR="$MODEL_DIR"
              set_phase "installing:hf_download"
              "$BASE_PY" "$APP_DIR/download_models.py" || {{
                log_attempt "download_models failed — continuing if partial weights exist"
              }}
              export LINGBOT_MODEL_DIR="$MODEL_DIR"

              set_phase "starting:app"
              stop_soft_ready
              trap - EXIT
              log_attempt "launching app.py PYTHONPATH=$PYTHONPATH"
              exec "$BASE_PY" app.py
          env:
            - name: HF_HOME
              value: "/models/huggingface"
            - name: HF_TOKEN
              value: "{{{{ .Values.olaresEnv.HF_TOKEN }}}}"
            - name: HF_ENDPOINT
              value: "{{{{ .Values.olaresEnv.HF_ENDPOINT }}}}"
            - name: HF_REPO_ID
              value: {{{{ .Values.olaresEnv.HF_REPO_ID | default "{hf}" | quote }}}}
            - name: UV_INDEX_URL
              value: {{{{ .Values.olaresEnv.UV_INDEX_URL | default "" | quote }}}}
            - name: PIP_INDEX_URL
              value: {{{{ .Values.olaresEnv.PIP_INDEX_URL | default "" | quote }}}}
            - name: UV_CONCURRENT_DOWNLOADS
              value: {{{{ .Values.olaresEnv.UV_CONCURRENT_DOWNLOADS | default "32" | quote }}}}
            - name: PYTORCH_CUDA_ALLOC_CONF
              value: "expandable_segments:True"
            - name: TORCHDYNAMO_DISABLE
              value: "1"
            - name: GRADIO_TEMP_DIR
              value: "/output/gradio"
            - name: SERVER_PORT
              value: "7860"
            - name: LINGBOT_APP
              value: "{name}"
            - name: LINGBOT_KIND
              value: "{kind}"{extra_env_yaml}
          ports:
            - containerPort: 7860
          startupProbe:
            tcpSocket:
              port: 7860
            initialDelaySeconds: 5
            timeoutSeconds: 5
            periodSeconds: 10
            failureThreshold: 2160
          livenessProbe:
            tcpSocket:
              port: 7860
            initialDelaySeconds: 60
            timeoutSeconds: 10
            periodSeconds: 30
            failureThreshold: 8
          resources:
            limits:
              cpu: "{app['cpu_lim']}"
              memory: {app['mem_lim']}
            requests:
              cpu: "500m"
              memory: 2Gi
          volumeMounts:
            - mountPath: "/models"
              name: models
            - mountPath: "/workspace"
              name: workspace
            - mountPath: "/output"
              name: output
            - mountPath: "/dev/shm"
              name: dshm
            - mountPath: "/app-src"
              name: app-source
      volumes:
        - name: models
          hostPath:
            path: "{{{{ .Values.userspace.appData }}}}/models"
            type: DirectoryOrCreate
        - name: workspace
          hostPath:
            path: "{{{{ .Values.userspace.appData }}}}/workspace"
            type: DirectoryOrCreate
        - name: output
          hostPath:
            path: "{{{{ .Values.userspace.appData }}}}/output"
            type: DirectoryOrCreate
        - name: dshm
          emptyDir:
            medium: Memory
            sizeLimit: 32Gi
        - name: app-source
          configMap:
            name: {name}-source
      restartPolicy: Always
---
apiVersion: v1
kind: Service
metadata:
  name: {name}
  namespace: "{{{{ .Release.Namespace }}}}"
spec:
  type: ClusterIP
  selector:
    io.kompose.service: {name}
  ports:
    - name: {name}
      protocol: TCP
      port: 7860
      targetPort: 7860
"""


def manifest(app: dict) -> str:
    name = app["name"]
    cli = f"{name}cli"
    title = app["title"]
    # placeholder icons — generate:icons patches later
    icon = "https://cdn.olares.com/images/2026/07/7b49b96c0f6e947b29dea43b25e1f235.png"
    featured = "https://cdn.olares.com/images/2026/07/42094b0b721bafe30bc25692a88bdc8f.png"
    extra_env_block = ""
    for k, v in app.get("extra_env", {}).items():
        extra_env_block += f"""
  - envName: {k}
    required: false
    editable: true
    applyOnChange: true
    value: '{v}'"""
    # bump limitedCpu for nginx +0.5 → use cpu_lim+1 roughly; skill says +500m
    cpu_acc = str(int(float(app["cpu_lim"])) + 1)
    mem_acc = app["mem_lim"].replace("Gi", "")
    mem_acc_i = str(int(mem_acc) + 1) + "Gi"
    return f"""---
olaresManifest.version: 0.12.0
olaresManifest.type: app
apiVersion: v3
workloadReplicas:
  {name}: 1
  {cli}: 1
metadata:
  name: {name}
  icon: {icon}
  description: {app['short']}
  title: {title}
  version: 1.0.0
  categories:
    - AI
sharedEntrances:
  - name: {name}
    host: sharedentrances-{name}
    port: 0
    title: {title} API
    icon: {icon}
    invisible: true
    authLevel: internal
entrances:
  - name: {cli}
    port: 8080
    host: {cli}
    title: {title}
    icon: {icon}
    openMethod: window
    authLevel: internal
spec:
  versionName: 1.0.0
  featuredImage: {featured}
  fullDescription: |
    {app['desc']}

    Hardware: Olares One — RTX 5090M 24GB + 96GB DDR5.

    Weights: {app['hf']}
    Source: {app['repo']}
    Project: {app['website']}

    Boot lessons (Krea soft-ready):
    - soft-ready HTTP on :7860 before uv/pip/HF download (pass ~30m install gate)
    - uv pip --target /workspace/{name}/site-packages (persist uv + cache)
    - purge torch shadow from site-packages (image owns torch)
    - bootstrap attempts + download progress → /workspace/{name}/bootstrap.log
    - GET /health shows phase / attempts_tail while installing

    UI: / and /ui · REST under /api/v1 · health: GET /health
  developer: Robbyant
  website: {app['website']}
  sourceCode: {app['repo']}
  submitter: coynntis
  locale:
    - en-US
  license:
    - text: Apache-2.0
      url: {app['repo']}
  supportArch:
    - amd64
  onlyAdmin: true
  accelerator:
    - mode: nvidia
      requiredCpu: '1'
      limitedCpu: '{cpu_acc}'
      requiredMemory: 4Gi
      limitedMemory: {mem_acc_i}
      requiredDisk: {app['disk_req']}
      limitedDisk: {app['disk_lim']}
      requiredGPUMemory: 1Gi
      limitedGPUMemory: {app['gpu_lim']}
  upgradeDescription: |
    v1.0.0: initial chart — soft-ready + uv target site-packages + HF download attempts log + Gradio/REST.
permission:
  appData: true
envs:
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
  - envName: HF_REPO_ID
    required: false
    editable: true
    applyOnChange: true
    value: '{app['hf']}'
  - envName: UV_INDEX_URL
    required: false
    editable: true
    applyOnChange: true
    value: ''{extra_env_block}
options:
  shared: true
  apiTimeout: 0
  dependencies:
    - name: olares
      version: '>=1.12.6-0'
      type: system
"""


def values_yaml(app: dict) -> str:
    name = app["name"]
    lines = [
        "admin: ''",
        "bfl:",
        "  username: ''",
        "userspace:",
        "  appData: ''",
        "  appCache: ''",
        "  userData: ''",
        "olaresEnv:",
        "  HF_TOKEN: ''",
        "  HF_ENDPOINT: ''",
        f"  HF_REPO_ID: '{app['hf']}'",
        "  UV_INDEX_URL: ''",
        "  PIP_INDEX_URL: ''",
        "  UV_CONCURRENT_DOWNLOADS: '32'",
    ]
    for k, v in app.get("extra_env", {}).items():
        lines.append(f"  {k}: '{v}'")
    lines += [
        "image:",
        "  repository: pytorch/pytorch",
        "  tag: 2.12.0-cuda13.0-cudnn9-devel",
        "workloads:",
        f"  {name}:",
        "    replicaCount: 1",
        f"  {name}cli:",
        "    replicaCount: 1",
        "",
    ]
    return "\n".join(lines)


def requirements(app: dict) -> str:
    return textwrap.dedent(
        """\
        # App deps only — NEVER pin torch/torchvision (image owns them).
        gradio>=4.44.0,<6
        fastapi>=0.115.2,<1.0
        uvicorn[standard]>=0.30.0,<1.0
        pydantic>=2.10,<2.12
        huggingface_hub>=0.26.0
        hf_transfer>=0.1.8
        accelerate>=0.33.0
        transformers>=4.44.0
        safetensors>=0.4.5
        pillow>=10.0.0
        numpy>=1.24.0,<2.0.0
        tqdm>=4.66.0
        """
    ).replace("fastapi", "fastapi")  # keep


def scaffold_one(app: dict) -> None:
    name = app["name"]
    chart = ROOT / name
    soft = (SHARED / "soft_ready.py").read_text(encoding="utf-8")
    soft = soft.replace('os.environ.get("LINGBOT_APP", "lingbot")', f'os.environ.get("LINGBOT_APP", "{name}")')
    dl = (SHARED / "download_models.py").read_text(encoding="utf-8")

    write(chart / "Chart.yaml", f"""apiVersion: v2
appVersion: "1.0.0"
description: {app['short']}
name: {name}
type: application
version: 1.0.0
""")
    write(chart / "owners", "owners:\n- 'coynntis'\n")
    write(chart / ".helmignore", "docker/\n*.tgz\n.DS_Store\napp/\n")
    write(chart / "OlaresManifest.yaml", manifest(app))
    write(chart / "i18n/en-US/OlaresManifest.yaml", manifest(app))
    write(chart / "values.yaml", values_yaml(app))
    write(chart / "templates/clientproxy.yaml", clientproxy(app))
    write(chart / "templates/server.yaml", server_yaml(app))
    write(chart / "app/soft_ready.py", soft)
    write(chart / "app/download_models.py", dl)
    write(chart / "app/_common.py", (SHARED.parent / "apps" / "_common.py").read_text(encoding="utf-8"))
    write(chart / "app/requirements.txt", requirements(app))
    # app.py written separately by kind modules
    app_py = ROOT / "scripts" / "lingbot" / "apps" / f"{app['kind']}_app.py"
    if app_py.is_file():
        write(chart / "app/app.py", app_py.read_text(encoding="utf-8"))
    else:
        write(chart / "app/app.py", f'# placeholder — see scripts/lingbot/apps/{app["kind"]}_app.py\nraise SystemExit("app not generated")\n')


def main() -> None:
    for app in APPS:
        scaffold_one(app)
    # dump APPS for icons script
    print(f"scaffolded {len(APPS)} charts")


if __name__ == "__main__":
    main()
