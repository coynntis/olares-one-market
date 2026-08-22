#!/usr/bin/env node
/**
 * Scaffold / refresh SNDR Core Engine charts from upstream recommendations.
 * Image: sndr/pins.yaml current_image_digest (dev748)
 * Boot: soft-ready → uv → site-packages cache → sndr.apply → vllm serve
 * Pattern mirrors lingbotworldone (attempt log + UV_CACHE_DIR + --target).
 */
const fs = require('fs');
const path = require('path');
const { execSync } = require('child_process');

const ROOT = path.join(__dirname, '..');
const IMAGE =
  'vllm/vllm-openai@sha256:6a93ae4316826f3dd8a92bee5442cbed50184a9cbd688d310f9e56ecad1eabeb';
const SNDR_GIT = 'https://github.com/Sandermage/sndr_core_engine.git';
const SNDR_REF = 'main';
const VERSION = '1.0.6';
const SOFT_READY = fs.readFileSync(path.join(__dirname, 'sndr/soft_ready.py'), 'utf8');

function yamlBlock(text, indent) {
  const pad = ' '.repeat(indent);
  return text
    .replace(/\r\n/g, '\n')
    .split('\n')
    .map((l) => (l.length ? pad + l : pad.trimEnd() ? pad : ''))
    .join('\n');
}

const GENESIS_QWEN_CORE = {
  PYTORCH_CUDA_ALLOC_CONF: 'expandable_segments:True,max_split_size_mb:512',
  CUDA_MODULE_LOADING: 'LAZY',
  NCCL_P2P_DISABLE: '1',
  NCCL_CUMEM_ENABLE: '0',
  OMP_NUM_THREADS: '1',
  CUDA_DEVICE_MAX_CONNECTIONS: '8',
  VLLM_ALLOW_LONG_MAX_MODEL_LEN: '1',
  VLLM_FLOAT32_MATMUL_PRECISION: 'high',
  VLLM_NO_USAGE_STATS: '1',
  VLLM_USE_AOT_COMPILE: '1',
  VLLM_USE_STANDALONE_COMPILE: '1',
  VLLM_ENABLE_PREGRAD_PASSES: '1',
  VLLM_TQ_DECODE_BLOCK_KV: '32',
  VLLM_TQ_DECODE_NUM_WARPS: '8',
  VLLM_TQ_DECODE_NUM_STAGES: '3',
  VLLM_MARLIN_USE_ATOMIC_ADD: '1',
  VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS: '1',
  VLLM_MOE_USE_DEEP_GEMM: '0',
  VLLM_USE_DEEP_GEMM: '0',
  VLLM_USE_FLASHINFER_MOE_FP8: '0',
  VLLM_USE_FLASHINFER_SAMPLER: '1',
  VLLM_USE_FUSED_MOE_GROUPED_TOPK: '1',
  VLLM_WORKER_MULTIPROC_METHOD: 'spawn',
  GENESIS_ENFORCE_VERSION_RANGE: '1',
  GENESIS_P67_NUM_WARPS: '4',
  GENESIS_ENABLE_P58_ASYNC_PLACEHOLDER_FIX: '1',
  GENESIS_ENABLE_P60_GDN_NGRAM_FIX: '1',
  GENESIS_ENABLE_P60B_TRITON_KERNEL: '1',
  GENESIS_ENABLE_P61B_STREAMING_OVERLAP: '1',
  GENESIS_ENABLE_P61C_QWEN3CODER_DEFERRED_COMMIT: '1',
  GENESIS_ENABLE_P62_STRUCT_OUT_SPEC_TIMING: '1',
  GENESIS_ENABLE_P64_QWEN3CODER_MTP_STREAMING: '1',
  // P65 conflicts with P67/P67b — never ENABLE P65; hard DISABLE (club-3090#25)
  GENESIS_DISABLE_P65_TURBOQUANT_SPEC_CG_DOWNGRADE: '1',
  GENESIS_ENABLE_P66_CUDAGRAPH_SIZE_FILTER: '1',
  GENESIS_ENABLE_P67_TQ_MULTI_QUERY_KERNEL: '1',
  GENESIS_P67_USE_UPSTREAM: '1',
  GENESIS_P67_NUM_KV_SPLITS: '32',
  GENESIS_ENABLE_P95: '1',
  GENESIS_ENABLE_P98: '1',
  GENESIS_ENABLE_G4_61_TQ_SHARED_WORKSPACE: '1',
  GENESIS_ENABLE_G4_62_TQ_KERNEL_WARMUP: '1',
  GENESIS_ENABLE_PN14_TQ_DECODE_OOB_CLAMP: '1',
  GENESIS_ENABLE_PN11_GDN_AB_CONTIGUOUS: '1',
  GENESIS_ENABLE_PN59_STREAMING_GDN: '1',
  GENESIS_ENABLE_P68_AUTO_FORCE_TOOL: '1',
  GENESIS_ENABLE_P69_LONG_CTX_TOOL_REMINDER: '1',
  GENESIS_ENABLE_P72_PROFILE_RUN_CAP: '1',
  GENESIS_PROFILE_RUN_CAP_M: '4096',
  GENESIS_ENABLE_P74_CHUNK_CLAMP: '1',
  GENESIS_ENABLE_PN17_FA2_LSE_CLAMP: '1',
  GENESIS_ENABLE_P99: '1',
  GENESIS_ENABLE_P101: '1',
  GENESIS_ENABLE_P103: '1',
  GENESIS_ENABLE_PN66: '1',
  GENESIS_ENABLE_PN30_DS_LAYOUT_SPEC_DECODE: '1',
  GENESIS_ENABLE_PN33_SPEC_DECODE_WARMUP_K: '1',
  GENESIS_ENABLE_PN35_INPUTS_EMBEDS_OPTIONAL: '1',
  GENESIS_ENABLE_PN116: '1',
  GENESIS_ENABLE_PN118: '1',
  GENESIS_ENABLE_PN119: '1',
  GENESIS_ENABLE_P109: '1',
  GENESIS_ENABLE_PN110: '1',
  GENESIS_ENABLE_P107_MTP_TRUNCATION_DETECTOR: '1',
  GENESIS_ENABLE_PN399_TQ_DECODE_SCRATCH_IMA: '1',
  GENESIS_ENABLE_PN401_TQ_PREFILL_CONTINUATION_GUARD: '1',
  GENESIS_ENABLE_PN521_TQ_RAW_TAIL_VERIFY: '1',
  GENESIS_ENABLE_PN522_TQ_RAW_TAIL_WARMUP: '1',
  GENESIS_ENABLE_PN25_SILU_INDUCTOR_SAFE: '1',
  GENESIS_ENABLE_P82: '1',
  GENESIS_P82_THRESHOLD_SINGLE: '0.1',
  GENESIS_ENABLE_PN31_FA_VARLEN_PERSISTENT_OUT: '1',
  GENESIS_ENABLE_PN125_HYBRID_FULL_AND_PIECEWISE: '1',
  GENESIS_ENABLE_PN126_V1_DECODE_WARMUP: '1',
  GENESIS_ENABLE_PN128_SPEC_DECODE_WARMUP: '1',
  GENESIS_ENABLE_PN130_TQ_DECODE_WARMUP: '1',
  GENESIS_ENABLE_PN133_MTP_EMPTY_OUTPUT_FIX: '1',
  GENESIS_ENABLE_PN402_SANITIZE_INVALID_DRAFT_TOKENS: '1',
  GENESIS_ENABLE_P108: '1',
  GENESIS_BUFFER_MODE: 'shared',
  GENESIS_PREALLOC_TOKEN_BUDGET: '4096',
  CUDA_DEVICE_MEMORY_LIMIT_0: '24300m',
  HF_HUB_ENABLE_HF_TRANSFER: '1',
};

/** 1×24GB adapt: smaller prealloc, no CUDA mem-limit fight with HAMI */
const GENESIS_QWEN_35B_1X = {
  ...GENESIS_QWEN_CORE,
  GENESIS_PREALLOC_TOKEN_BUDGET: '512',
  GENESIS_PROFILE_RUN_CAP_M: '1024',
  CUDA_DEVICE_MEMORY_LIMIT_0: '',
};

function omitKeys(obj, keys) {
  const out = { ...obj };
  for (const k of keys) delete out[k];
  return out;
}

const GENESIS_GEMMA = {
  ...omitKeys(GENESIS_QWEN_CORE, [
    'GENESIS_ENABLE_PN521_TQ_RAW_TAIL_VERIFY',
    'GENESIS_ENABLE_PN522_TQ_RAW_TAIL_WARMUP',
    'VLLM_TQ_DECODE_BLOCK_KV',
    'VLLM_TQ_DECODE_NUM_WARPS',
    'VLLM_TQ_DECODE_NUM_STAGES',
    'GENESIS_ENABLE_P67_TQ_MULTI_QUERY_KERNEL',
    'GENESIS_ENABLE_G4_61_TQ_SHARED_WORKSPACE',
    'GENESIS_ENABLE_G4_62_TQ_KERNEL_WARMUP',
    'GENESIS_ENABLE_PN14_TQ_DECODE_OOB_CLAMP',
    'GENESIS_ENABLE_PN130_TQ_DECODE_WARMUP',
    'GENESIS_ENABLE_PN399_TQ_DECODE_SCRATCH_IMA',
    'GENESIS_ENABLE_PN401_TQ_PREFILL_CONTINUATION_GUARD',
  ]),
  GENESIS_ENABLE_G4_01_GEMMA4_FP8_BLOCK_GUARD: '1',
  GENESIS_ENABLE_G4_03_GEMMA4_NON_CAUSAL_DRAFTER_GUARD: '1',
  GENESIS_ENABLE_G4_12_GEMMA4_FP8_E4NV_GUARD: '1',
  GENESIS_ENABLE_G4_04_GEMMA4_AWQ_MOE_KEYS_REMAP: '1',
  GENESIS_ENABLE_G4_09_GEMMA4_SWA_PREFILL_CHUNKER: '1',
  GENESIS_ENABLE_G4_14_GEMMA4_TOOL_CALL_PARSER_PAD: '1',
  GENESIS_ENABLE_G4_16_GEMMA4_FULL_AND_PIECEWISE: '1',
  GENESIS_ENABLE_G4_23_GEMMA4_VISION_FP16_OVERFLOW: '1',
  GENESIS_ENABLE_G4_25_GEMMA4_RoPE_DUAL_BASE_GUARD: '1',
  GENESIS_ENABLE_G4_08_MARLIN_KDIM_PAD: '1',
  GENESIS_ENABLE_G4_18_GEMMA4_PER_LAYER_KV_PAGE_SIZE: '1',
  GENESIS_ENABLE_PN286_FA_LAYOUT_REVERT_SM86: '1',
};

const GENESIS_DIFF = {
  ...omitKeys(GENESIS_QWEN_CORE, [
    'GENESIS_ENABLE_PN521_TQ_RAW_TAIL_VERIFY',
    'GENESIS_ENABLE_PN522_TQ_RAW_TAIL_WARMUP',
    'VLLM_TQ_DECODE_BLOCK_KV',
    'VLLM_TQ_DECODE_NUM_WARPS',
    'VLLM_TQ_DECODE_NUM_STAGES',
    'GENESIS_ENABLE_P67_TQ_MULTI_QUERY_KERNEL',
    'GENESIS_ENABLE_G4_61_TQ_SHARED_WORKSPACE',
    'GENESIS_ENABLE_G4_62_TQ_KERNEL_WARMUP',
    'GENESIS_ENABLE_PN14_TQ_DECODE_OOB_CLAMP',
    'GENESIS_ENABLE_PN130_TQ_DECODE_WARMUP',
    'GENESIS_ENABLE_PN399_TQ_DECODE_SCRATCH_IMA',
    'GENESIS_ENABLE_PN401_TQ_PREFILL_CONTINUATION_GUARD',
    'GENESIS_ENABLE_P64_QWEN3CODER_MTP_STREAMING',
    'GENESIS_ENABLE_PN133_MTP_EMPTY_OUTPUT_FIX',
  ]),
  GENESIS_ENABLE_PN_FP8MOE_KPAD: '1',
  GENESIS_ENABLE_PN524_DIFFUSION_SPEC_PADDING_SKIP: '1',
  VLLM_USE_V2_MODEL_RUNNER: '1',
};

const APPS = [
  {
    id: 'sndrqwen3627bone',
    title: 'SNDR Qwen36 27B One',
    entrance_title: 'SNDR Qwen36 27B One',
    api_title: 'SNDR Qwen36 27B API',
    desc: 'SNDR Core Engine — Qwen3.6 27B INT4 + TQ k8v4 + MTP K=4 (1x)',
    preset: 'qa-qwen3.6-27b-tq-1x',
    model: 'Lorbus/Qwen3.6-27B-int4-AutoRound',
    alias: 'qwen3.6-27b',
    max_len: '49152',
    gmu: '0.95',
    // Hybrid Mamba align: block_size ~2128 → batched must be >= that
    batched: '4096',
    seqs: '1',
    kv: 'turboquant_k8v4',
    dtype: 'float16',
    quant: 'auto_round',
    spec: '{"method":"mtp","num_speculative_tokens":4}',
    tool_parser: 'qwen3_xml',
    reasoning: 'qwen3',
    language_only: true,
    // 24GB: 48K ctx (78K KV OOM). torch.compile OK if SAM/other GPU apps stopped at boot.
    enforce_eager: false,
    extra_flags: [['--override-generation-config', '{"temperature":0.6,"top_k":20,"top_p":0.95}']],
    genesis: {
      ...GENESIS_QWEN_CORE,
      // 27B+TQ+MTP: FULL_AND_PIECEWISE + P67b can .tolist()-crash under capture
      GENESIS_ENABLE_PN125_HYBRID_FULL_AND_PIECEWISE: '0',
    },
    bento: {
      family: 'qwen',
      size_label: '27B',
      badge: 'sndr mtp4',
      hero: { value: 'MTP K=4', label: 'TQ k8v4' },
      specs: [
        { label: 'context', value: '48K' },
        { label: 'pin', value: 'dev748' },
      ],
      capabilities: { tool_calling: true, vision: false, audio: false, mtp: { enabled: true, accept: 70 } },
      stack: 'SNDR · nightly digest · MTP K=4',
    },
    categories: ['LLM Chat', 'AI Agents'],
    full: `SNDR Core Engine (Sandermage) on Olares One — official pin.

Image: vllm/vllm-openai@sha256:6a93ae43… (pins.yaml current = 0.23.1rc1.dev748+g2dfaae752)
Boot: soft-ready → uv cache → editable -e sndr (not --target) → sndr.apply → vllm serve
Attempts: /shared-models/sndr/<app>/bootstrap.log
Site-packages target: only optional deps (pyyaml); SNDR itself is editable/--system so vLLM plugin entry points work.
Preset base: qa-qwen3.6-27b-tq-1x (1×24GB)
Model: Lorbus/Qwen3.6-27B-int4-AutoRound
KV turboquant_k8v4 · MTP K=4 (2026-07-03 coherence retune; K=5 broke 27B tool-calls)
PN521/PN522 required for TQ×MTP

Upstream: https://github.com/Sandermage/sndr_core_engine`,
  },
  {
    id: 'sndrqwen3635ba3bone',
    title: 'SNDR Qwen36 35B One',
    entrance_title: 'SNDR Qwen36 35B One',
    api_title: 'SNDR Qwen36 35B API',
    desc: 'SNDR Core Engine — Qwen3.6 35B-A3B FP8 + TQ (1x24GB, no MTP; official MTP K=5 needs TP=2)',
    preset: 'prod-qwen3.6-35b-balanced (TP=1 no-MTP adapt)',
    model: 'Qwen/Qwen3.6-35B-A3B-FP8',
    alias: 'qwen3.6-35b-a3b',
    max_len: '8192',
    gmu: '0.88',
    batched: '1024',
    seqs: '1',
    kv: 'turboquant_k8v4',
    dtype: 'float16',
    quant: null,
    // MTP K=5 is the SNDR family default but needs TP=2 / >24GB; weights alone ~18GiB → OOM with MTP on 5090M
    spec: null,
    tool_parser: 'qwen3_xml',
    reasoning: 'qwen3',
    language_only: true,
    enforce_eager: false,
    extra_flags: [['--override-generation-config', '{"temperature":0.6,"top_k":20,"top_p":0.95}']],
    genesis: GENESIS_QWEN_35B_1X,
    bento: {
      family: 'qwen',
      size_label: '35B-A3B',
      badge: '1x tight',
      hero: { value: 'FP8', label: 'no MTP 1x' },
      specs: [
        { label: 'context', value: '8K' },
        { label: 'need', value: 'TP=2 ideal' },
      ],
      capabilities: { tool_calling: true, vision: false, audio: false, mtp: { enabled: false } },
      stack: 'SNDR · FP8 · TQ · no MTP (24GB)',
    },
    categories: ['LLM Chat', 'AI Agents'],
    full: `SNDR Core Engine — Qwen3.6-35B-A3B-FP8 on Olares One (1×24GB adapt).

Official prod preset (prod-qwen3.6-35b-balanced) is TP=2 + MTP K=5.
On RTX 5090M 24GB, FP8 weights alone peak ~18GiB — MTP K=5 OOMs during load.
This chart: TP=1, no MTP, 8K ctx, lowered Genesis prealloc. Prefer sndrqwen3627bone (INT4+MTP K=4) for agentic use.
For 249 t/s-class 35B MTP on 24GB use llama.cpp GGUF (SNDR escape hatch), not vLLM FP8.

Upstream: https://github.com/Sandermage/sndr_core_engine`,
  },
  {
    id: 'sndrgemma426ba4bone',
    title: 'SNDR Gemma4 26B One',
    entrance_title: 'SNDR Gemma4 26B One',
    api_title: 'SNDR Gemma4 26B API',
    desc: 'SNDR Core Engine — Gemma 4 26B-A4B AWQ (1x adapt)',
    preset: 'prod-gemma4-26b-default (TP=1 adapt)',
    model: 'cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit',
    alias: 'gemma-4-26b-a4b',
    max_len: '32768',
    gmu: '0.90',
    batched: '4096',
    seqs: '1',
    kv: 'auto',
    dtype: 'bfloat16',
    quant: null,
    spec: null,
    tool_parser: 'gemma4',
    reasoning: null,
    language_only: false,
    enforce_eager: false,
    extra_flags: [
      [
        '--override-generation-config',
        '{"temperature":0.7,"top_p":0.95,"top_k":64,"frequency_penalty":0.6,"presence_penalty":0.4}',
      ],
    ],
    genesis: GENESIS_GEMMA,
    bento: {
      family: 'gemma',
      size_label: '26B-A4B',
      badge: 'sndr',
      hero: { value: 'MoE A4B', label: 'AWQ' },
      specs: [
        { label: 'context', value: '32K' },
        { label: 'MTP', value: 'off' },
      ],
      capabilities: { tool_calling: true, vision: true, audio: false, mtp: { enabled: false } },
      stack: 'SNDR · G4 patches · kv-auto',
    },
    categories: ['LLM Chat', 'Vision'],
    full: `SNDR Core Engine — Gemma 4 26B-A4B AWQ (family default = MTP off).

Upstream prod uses TP=2; this chart TP=1 / 32K for Olares One.
Upstream: https://github.com/Sandermage/sndr_core_engine`,
  },
  {
    id: 'sndrdiffusiongemma26bone',
    title: 'SNDR DiffusionGemma One',
    entrance_title: 'SNDR DiffusionGemma One',
    api_title: 'SNDR DiffusionGemma API',
    desc: 'SNDR Core Engine — DiffusionGemma 26B FP8 (experimental 1x)',
    preset: 'prod-diffusiongemma-tp2 (TP=1 adapt)',
    model: 'RedHatAI/diffusiongemma-26B-A4B-it-FP8-dynamic',
    alias: 'diffusiongemma',
    max_len: '8192',
    gmu: '0.90',
    batched: '1024',
    seqs: '1',
    kv: 'auto',
    dtype: 'bfloat16',
    quant: null,
    spec: null,
    tool_parser: 'gemma4',
    reasoning: null,
    language_only: false,
    enforce_eager: true,
    extra_flags: [
      ['--num-gpu-blocks-override', '512'],
      ['--attention-backend', 'TRITON_ATTN'],
      ['--generation-config', 'vllm'],
    ],
    genesis: GENESIS_DIFF,
    bento: {
      family: 'gemma',
      size_label: 'Diff26B',
      badge: 'experimental',
      hero: { value: 'block', label: 'diffusion' },
      specs: [
        { label: 'context', value: '8K' },
        { label: 'need', value: 'TP=2 ideal' },
      ],
      capabilities: { tool_calling: true, vision: false, audio: false, mtp: { enabled: false } },
      stack: 'SNDR · FP8 MoE · eager',
    },
    categories: ['LLM Chat', 'AI'],
    full: `SNDR Core Engine — DiffusionGemma 26B-A4B FP8-dynamic.

Upstream validates TP=2 only. Olares One is 1× — experimental 8K.
Requires PN-FP8MOE-KPAD + sndr.apply before serve.
Upstream: https://github.com/Sandermage/sndr_core_engine`,
  },
];

function q(s) {
  return JSON.stringify(String(s));
}

function envYaml(envs) {
  return Object.entries(envs)
    .map(([k, v]) => `            - name: ${k}\n              value: ${q(v)}`)
    .join('\n');
}

function keepCdn(appId) {
  const p = path.join(ROOT, appId, 'OlaresManifest.yaml');
  let icon = 'https://cdn.olares.com/placeholder-icon.png';
  let featured = 'https://cdn.olares.com/placeholder-featured.png';
  if (fs.existsSync(p)) {
    const t = fs.readFileSync(p, 'utf8');
    const mi = t.match(/^  icon: (https:\/\/cdn\.olares\.com\S+)/m);
    const mf = t.match(/^  featuredImage: (https:\/\/cdn\.olares\.com\S+)/m);
    if (mi) icon = mi[1];
    if (mf) featured = mf[1];
  }
  return { icon, featured };
}

function clientproxy(app) {
  return `---
apiVersion: v1
data:
  nginx.conf: |
    server {
      listen 8080;
      server_name _;
      access_log /opt/bitnami/openresty/nginx/logs/access.log;
      error_log  /opt/bitnami/openresty/nginx/logs/error.log;
      proxy_connect_timeout 600s;
      proxy_send_timeout 600s;
      proxy_read_timeout 1800s;
      proxy_buffering off;
      proxy_cache off;
      chunked_transfer_encoding on;
      proxy_set_header host $host;
      proxy_set_header x-forwarded-host $http_host;
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
              mountPath: /opt/bitnami/openresty/nginx/conf/server_blocks
---
apiVersion: v1
kind: Service
metadata:
  name: ${app}cli
  namespace: '{{ .Release.Namespace }}'
  labels:
    io.kompose.service: ${app}cli
spec:
  ports:
    - name: "8080"
      port: 8080
      targetPort: 8080
  selector:
    io.kompose.service: ${app}cli
`;
}

function bootScript(app) {
  // Shell vars: escape $ for JS template literals as \${...}
  return `set -euo pipefail
              APP="${app}"
              CHART_BOOT="${VERSION}"
              APP_DIR="/shared-models/sndr/\${APP}"
              SITE_PKGS="\$APP_DIR/site-packages"
              PHASE_FILE="\$APP_DIR/.boot-phase"
              ATTEMPTS_FILE="\$APP_DIR/bootstrap.log"
              SNDR_HOME="/shared-models/sndr/sndr_core_engine"
              UV_BIN_DIR="\$APP_DIR/bin"
              mkdir -p "\$APP_DIR" "\$SITE_PKGS" "\$UV_BIN_DIR" "\$APP_DIR/uv-cache" /shared-models/llms/huggingface
              cp /app-src/soft_ready.py "\$APP_DIR/soft_ready.py"

              export HF_HOME="\${HF_HOME:-/shared-models/llms/huggingface}"
              export HF_HUB_CACHE="\$HF_HOME/hub"
              export PYTHONNOUSERSITE=1
              export UV_CACHE_DIR="\$APP_DIR/uv-cache"
              export UV_CONCURRENT_DOWNLOADS="\${UV_CONCURRENT_DOWNLOADS:-16}"
              export SERVER_PORT=8000
              export SNDR_APP="\$APP"
              export SNDR_BOOT_PHASE_FILE="\$PHASE_FILE"
              export SNDR_BOOT_ATTEMPTS_FILE="\$ATTEMPTS_FILE"

              set_phase() {
                printf '%s\\n' "\$1" > "\$PHASE_FILE"
                echo "[\${APP}] phase=\$1" | tee -a "\$ATTEMPTS_FILE"
              }
              log_attempt() {
                printf '%s %s\\n' "\$(date -u +%Y-%m-%dT%H:%M:%SZ)" "\$1" | tee -a "\$ATTEMPTS_FILE"
              }
              retry() {
                local n="\$1"; shift
                local i=1
                while true; do
                  log_attempt "attempt \$i/\$n: \$*"
                  if "\$@"; then
                    log_attempt "OK: \$*"
                    return 0
                  fi
                  log_attempt "FAIL (\$i/\$n): \$*"
                  if [ "\$i" -ge "\$n" ]; then
                    return 1
                  fi
                  i=\$((i + 1))
                  sleep \$((i * 3))
                done
              }

              log_attempt "SNDR_CHART_BOOT=\$CHART_BOOT mode=pythonpath+editable (never --target sndr)"

              # ALWAYS scrub broken --target sndr copies — namespace merge with repo causes EngineAdapter unknown location
              rm -rf "\$SITE_PKGS/sndr" \\
                "\$SITE_PKGS"/sndr[-_]platform* \\
                "\$SITE_PKGS"/sndr_platform* \\
                "\$SITE_PKGS"/sndr_platform-*.dist-info \\
                "\$SITE_PKGS"/sndr_platform-*.egg-info \\
                "\$SITE_PKGS"/__editable__.sndr* \\
                "\$SITE_PKGS"/__editable___sndr* 2>/dev/null || true
              rm -f "\$APP_DIR"/.sndr-ok-* 2>/dev/null || true
              log_attempt "purged \$SITE_PKGS/sndr (+ markers) if any"

              BASE_PY=""
              for c in /usr/bin/python3 /usr/local/bin/python3 "\$(command -v python3 || true)" "\$(command -v python || true)"; do
                [ -n "\$c" ] && [ -x "\$c" ] || continue
                if PYTHONPATH= PYTHONNOUSERSITE=1 "\$c" -c "import vllm" 2>/dev/null \\
                  || PYTHONPATH= PYTHONNOUSERSITE=1 "\$c" -c "import torch" 2>/dev/null; then
                  BASE_PY="\$c"
                  break
                fi
              done
              if [ -z "\$BASE_PY" ]; then
                BASE_PY="\$(command -v python3 || command -v python)"
              fi
              if [ -z "\$BASE_PY" ] || [ ! -x "\$BASE_PY" ]; then
                log_attempt "FATAL: no python"
                exit 1
              fi
              export UV_PYTHON="\$BASE_PY"
              log_attempt "base_python=\$BASE_PY"
              PYTHONPATH= PYTHONNOUSERSITE=1 "\$BASE_PY" -c "import sys; print('[${app}] python', sys.version.split()[0], sys.executable)" || true

              set_phase "installing:soft_ready_starting"
              PYTHONPATH= "\$BASE_PY" "\$APP_DIR/soft_ready.py" &
              SOFT_PID=\$!
              log_attempt "soft-ready pid=\$SOFT_PID port=\$SERVER_PORT"
              sleep 1

              stop_soft_ready() {
                if [ -n "\${SOFT_PID:-}" ] && kill -0 "\$SOFT_PID" 2>/dev/null; then
                  log_attempt "stopping soft-ready pid=\$SOFT_PID"
                  kill "\$SOFT_PID" 2>/dev/null || true
                  wait "\$SOFT_PID" 2>/dev/null || true
                fi
                if command -v fuser >/dev/null 2>&1; then
                  fuser -k "\${SERVER_PORT}/tcp" 2>/dev/null || true
                fi
                sleep 1
              }
              trap 'stop_soft_ready' EXIT

              if [ -z "\${UV_INDEX_URL:-}" ] && [ -n "\${PIP_INDEX_URL:-}" ]; then
                export UV_INDEX_URL="\$PIP_INDEX_URL"
              fi
              if [ -z "\${UV_INDEX_URL:-}" ]; then
                case "\${HF_ENDPOINT:-}" in
                  *hf-mirror*|*huggingface.co.cn*|*hf-mirror.com*)
                    export UV_INDEX_URL="https://pypi.tuna.tsinghua.edu.cn/simple"
                    log_attempt "HF China mirror → UV_INDEX_URL=\$UV_INDEX_URL"
                    ;;
                esac
              fi

              # ensure curl/git for uv bootstrap + clone
              if ! command -v curl >/dev/null 2>&1 || ! command -v git >/dev/null 2>&1; then
                set_phase "installing:apt_tools"
                log_attempt "apt install curl ca-certificates git"
                apt-get update -qq
                apt-get install -y -qq --no-install-recommends curl ca-certificates git
                rm -rf /var/lib/apt/lists/*
              fi

              UV="\$UV_BIN_DIR/uv"
              if [ ! -x "\$UV" ]; then
                set_phase "installing:uv"
                log_attempt "bootstrapping uv into \$UV_BIN_DIR"
                if ! curl -fsSL https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="\$UV_BIN_DIR" UV_NO_MODIFY_PATH=1 sh; then
                  log_attempt "uv install.sh failed — tarball fallback"
                  tmpd="\$(mktemp -d)"
                  curl -fL "https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-unknown-linux-gnu.tar.gz" -o "\$tmpd/uv.tgz"
                  tar -xzf "\$tmpd/uv.tgz" -C "\$tmpd"
                  find "\$tmpd" -type f -name uv -exec mv {} "\$UV" \\;
                  rm -rf "\$tmpd"
                  chmod +x "\$UV"
                fi
              fi
              if [ ! -x "\$UV" ]; then
                log_attempt "FATAL: uv missing"
                exit 1
              fi
              log_attempt "uv=\$("\$UV" --version 2>&1 | head -1)"

              uvpip_target() {
                local extra=()
                if [ -n "\${UV_INDEX_URL:-}" ]; then
                  extra+=(--index-url "\$UV_INDEX_URL")
                fi
                # ONLY for non-sndr deps. Never install sndr via --target.
                "\$UV" pip install --python "\$BASE_PY" --target "\$SITE_PKGS" "\${extra[@]}" "\$@"
              }
              uvpip_editable() {
                local extra=()
                if [ -n "\${UV_INDEX_URL:-}" ]; then
                  extra+=(--index-url "\$UV_INDEX_URL")
                fi
                "\$UV" pip install --python "\$BASE_PY" --system "\${extra[@]}" "\$@"
              }

              # clone / update SNDR (shared tree, retries logged)
              set_phase "installing:git_clone"
              if [ ! -d "\$SNDR_HOME/.git" ]; then
                rm -rf "\$SNDR_HOME"
                retry 3 git clone --depth 1 --branch "\${SNDR_REF}" "\${SNDR_GIT}" "\$SNDR_HOME" \\
                  || retry 2 git clone --depth 1 "\${SNDR_GIT}" "\$SNDR_HOME"
              else
                log_attempt "sndr repo present — fetch \${SNDR_REF}"
                git -C "\$SNDR_HOME" fetch --depth 1 origin "\${SNDR_REF}" && \\
                  git -C "\$SNDR_HOME" checkout -q FETCH_HEAD \\
                  || log_attempt "git fetch soft-fail — using existing tree"
              fi
              log_attempt "sndr HEAD=\$(git -C "\$SNDR_HOME" rev-parse --short HEAD 2>/dev/null || echo unknown)"

              # Repo root FIRST (install.sh --no-plugin). Never put target/sndr on path.
              export PYTHONPATH="\$SNDR_HOME:\$SITE_PKGS\${PYTHONPATH:+:\$PYTHONPATH}"

              # Optional runtime deps into persistent SITE_PKGS (not sndr)
              if ! "\$BASE_PY" -c "import yaml, packaging" 2>/dev/null; then
                set_phase "installing:uv_deps"
                log_attempt "uv pip --target pyyaml packaging (not sndr)"
                uvpip_target --upgrade "pyyaml>=6" "packaging>=23" || log_attempt "target deps soft-fail"
                # scrub again in case a bad dep tree dropped sndr
                rm -rf "\$SITE_PKGS/sndr" "\$SITE_PKGS"/sndr[-_]platform* 2>/dev/null || true
              fi

              sndr_import_ok() {
                PYTHONPATH="\$SNDR_HOME:\$SITE_PKGS" "\$BASE_PY" -c \\
                  "from sndr.engines import EngineAdapter; import sndr; print('sndr', sndr.__file__, sndr.__version__)" 
              }

              set_phase "installing:uv_sndr"
              # Primary: PYTHONPATH to clone (no --target). Secondary: editable for vllm plugin entry points.
              if sndr_import_ok 2>&1 | tee -a "\$ATTEMPTS_FILE" | grep -q 'sndr '; then
                log_attempt "PYTHONPATH import OK — installing editable for vllm.general_plugins entry point"
              else
                log_attempt "PYTHONPATH import failed — will try editable then re-check"
              fi
              log_attempt "uv pip install --system --no-deps -e \$SNDR_HOME"
              retry 3 uvpip_editable --upgrade --no-deps -e "\$SNDR_HOME" \\
                || log_attempt "editable --system soft-fail — relying on PYTHONPATH=\$SNDR_HOME"
              # purge target again after any uv ops
              rm -rf "\$SITE_PKGS/sndr" "\$SITE_PKGS"/sndr[-_]platform* 2>/dev/null || true
              export PYTHONPATH="\$SNDR_HOME:\$SITE_PKGS\${PYTHONPATH:+:\$PYTHONPATH}"
              if ! sndr_import_ok 2>&1 | tee -a "\$ATTEMPTS_FILE"; then
                log_attempt "FATAL: import sndr.engines.EngineAdapter failed (chart=\$CHART_BOOT)"
                "\$BASE_PY" -c "import sys; print('sys.path'); [print(p) for p in sys.path]" 2>&1 | tee -a "\$ATTEMPTS_FILE" || true
                ls -la "\$SITE_PKGS" 2>&1 | tee -a "\$ATTEMPTS_FILE" || true
                ls -la "\$SNDR_HOME/sndr/engines" 2>&1 | tee -a "\$ATTEMPTS_FILE" || true
                exit 1
              fi
              touch "\$APP_DIR/.sndr-ok-\$CHART_BOOT-\$(git -C "\$SNDR_HOME" rev-parse --short HEAD 2>/dev/null || echo na)"
              log_attempt "sndr ready chart=\$CHART_BOOT"

              set_phase "installing:sndr_apply"
              log_attempt "python -m sndr.apply"
              "\$BASE_PY" -m sndr.apply 2>&1 | tee -a "\$ATTEMPTS_FILE" | tail -80 \\
                || "\$BASE_PY" -c "from sndr.plugin import register; register()" \\
                || log_attempt "sndr.apply soft-fail — continuing"

              CTX="\${LLM_CONTEXT_WINDOW:-\${MAX_MODEL_LEN}}"
              EXTRA=()
              if [ -n "\${LLM_API_KEY:-}" ]; then EXTRA+=(--api-key "\$LLM_API_KEY"); fi
              if [ -n "\${LLM_MAX_OUTPUT_TOKENS:-}" ]; then
                EXTRA+=(--override-generation-config "{\\"max_tokens\\":\$LLM_MAX_OUTPUT_TOKENS}")
              fi

              set_phase "starting:vllm"
              stop_soft_ready
              trap - EXIT
              log_attempt "exec vllm serve model=\$MODEL_NAME ctx=\$CTX PYTHONPATH=\$PYTHONPATH chart=\$CHART_BOOT"
              echo "$(date -Iseconds) [${app}] vllm serve model=\$MODEL_NAME"`;
}

function buildFlags(cfg) {
  const f = [];
  const push = (...xs) => f.push(...xs);
  push('serve', '"$MODEL_NAME"');
  push('--served-model-name', '"$MODEL_ALIAS"');
  push('--host', '0.0.0.0', '--port', '8000');
  push('--max-model-len', '"$CTX"');
  push('--gpu-memory-utilization', '"$GPU_MEMORY_UTILIZATION"');
  push('--max-num-seqs', '"$MAX_NUM_SEQS"');
  push('--max-num-batched-tokens', '"$MAX_NUM_BATCHED_TOKENS"');
  push('--dtype', `"${cfg.dtype}"`);
  push('--kv-cache-dtype', `"${cfg.kv}"`);
  push('--download-dir', '/shared-models/llms/huggingface');
  push('--trust-remote-code');
  push('--enable-prefix-caching', '--enable-chunked-prefill');
  push('--enable-auto-tool-choice', '--tool-call-parser', `"${cfg.tool_parser}"`);
  push('--tensor-parallel-size', '1');
  push('--disable-custom-all-reduce');
  if (cfg.quant) push('--quantization', `"${cfg.quant}"`);
  if (cfg.language_only) push('--language-model-only');
  if (cfg.reasoning) push('--reasoning-parser', `"${cfg.reasoning}"`);
  if (cfg.spec) push('--speculative-config', `'${cfg.spec}'`);
  if (cfg.enforce_eager) push('--enforce-eager');
  for (const [fl, val] of cfg.extra_flags) {
    push(fl, val.startsWith('{') ? `'${val}'` : `"${val}"`);
  }
  return f.join(' \\\n                ');
}

function serverYaml(cfg) {
  const app = cfg.id;
  const cm = `sndr-${app}-env`;
  const softCm = `sndr-${app}-soft-ready`;
  const joined = buildFlags(cfg);
  const boot = bootScript(app);
  return `{{- $llmCtx := .Values.olaresEnv.LLM_CONTEXT_WINDOW | default "" }}
{{- $llmMaxOut := .Values.olaresEnv.LLM_MAX_OUTPUT_TOKENS | default "" }}
{{- $llmApiKey := .Values.olaresEnv.LLM_API_KEY | default "" }}
---
apiVersion: v1
kind: ConfigMap
metadata:
  name: ${cm}
  namespace: "{{ .Release.Namespace }}"
data:
  MODEL_NAME: "${cfg.model}"
  MODEL_ALIAS: "${cfg.alias}"
  MAX_MODEL_LEN: "${cfg.max_len}"
  GPU_MEMORY_UTILIZATION: "${cfg.gmu}"
  MAX_NUM_SEQS: "${cfg.seqs}"
  MAX_NUM_BATCHED_TOKENS: "${cfg.batched}"
  SNDR_GIT: "${SNDR_GIT}"
  SNDR_REF: "${SNDR_REF}"
---
apiVersion: v1
kind: ConfigMap
metadata:
  name: ${softCm}
  namespace: "{{ .Release.Namespace }}"
data:
  soft_ready.py: |
${yamlBlock(SOFT_READY, 4)}
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
            - mkdir -p /shared-models/llms/huggingface /shared-models/sndr && chmod -R 777 /shared-models && echo ok
          securityContext:
            runAsUser: 0
          volumeMounts:
            - mountPath: "/shared-models"
              name: shared-models
      containers:
        - name: vllm-server
          image: "${IMAGE}"
          command: ["/bin/bash", "-lc"]
          args:
            - |
              ${boot}
              exec vllm ${joined} \\
                "\${EXTRA[@]}"
          envFrom:
            - configMapRef:
                name: ${cm}
          env:
            - name: HF_HOME
              value: "/shared-models/llms/huggingface"
            - name: HF_TOKEN
              value: {{ .Values.olaresEnv.HF_TOKEN | default "" | quote }}
            - name: HUGGING_FACE_HUB_TOKEN
              value: {{ .Values.olaresEnv.HF_TOKEN | default "" | quote }}
            {{- if .Values.olaresEnv.HF_ENDPOINT }}
            - name: HF_ENDPOINT
              value: "{{ .Values.olaresEnv.HF_ENDPOINT }}"
            {{- end }}
            - name: UV_INDEX_URL
              value: {{ .Values.olaresEnv.UV_INDEX_URL | default "" | quote }}
            - name: PIP_INDEX_URL
              value: {{ .Values.olaresEnv.PIP_INDEX_URL | default "" | quote }}
            - name: UV_CONCURRENT_DOWNLOADS
              value: {{ .Values.olaresEnv.UV_CONCURRENT_DOWNLOADS | default "16" | quote }}
            - name: LLM_CONTEXT_WINDOW
              value: {{ .Values.olaresEnv.LLM_CONTEXT_WINDOW | default "" | quote }}
            - name: LLM_MAX_OUTPUT_TOKENS
              value: {{ .Values.olaresEnv.LLM_MAX_OUTPUT_TOKENS | default "" | quote }}
            - name: LLM_API_KEY
              value: {{ .Values.olaresEnv.LLM_API_KEY | default "" | quote }}
${envYaml(cfg.genesis)}
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
            initialDelaySeconds: 10
            timeoutSeconds: 10
            periodSeconds: 15
            failureThreshold: 480
          resources:
            limits:
              cpu: "16"
              memory: 48Gi
            requests:
              cpu: "4"
              memory: 24Gi
          volumeMounts:
            - mountPath: "/shared-models"
              name: shared-models
            - mountPath: "/app-src"
              name: soft-ready-src
      volumes:
        - name: shared-models
          hostPath:
            path: "{{ .Values.userspace.appData }}/shared-llms"
            type: DirectoryOrCreate
        - name: soft-ready-src
          configMap:
            name: ${softCm}
            defaultMode: 0555
      restartPolicy: Always
---
apiVersion: v1
kind: Service
metadata:
  name: ${app}
  namespace: "{{ .Release.Namespace }}"
  labels:
    io.kompose.service: ${app}
spec:
  ports:
    - name: "vllm"
      port: 8000
      targetPort: 8000
  selector:
    io.kompose.service: ${app}
`;
}

function manifest(cfg) {
  const app = cfg.id;
  const b = cfg.bento;
  const { icon, featured } = keepCdn(app);
  const cats = cfg.categories.map((c) => `    - ${c}`).join('\n');
  const specs = b.specs.map((s) => `      - label: ${s.label}\n        value: ${s.value}`).join('\n');
  let mtp = `enabled: ${b.capabilities.mtp.enabled}`;
  if (b.capabilities.mtp.enabled && b.capabilities.mtp.accept != null) {
    mtp += `\n        accept: ${b.capabilities.mtp.accept}`;
  }
  const full = cfg.full
    .split('\n')
    .map((l) => `    ${l}`)
    .join('\n');
  return `---
olaresManifest.version: 0.12.0
olaresManifest.type: app
apiVersion: v3
workloadReplicas:
  ${app}: 1
  ${app}cli: 1
metadata:
  name: ${app}
  icon: ${icon}
  description: ${cfg.desc}
  title: ${cfg.title}
  version: ${VERSION}
  categories:
${cats}
  bento:
    family: ${b.family}
    size_label: "${b.size_label}"
    badge: ${b.badge}
    hero:
      value: "${b.hero.value}"
      label: "${b.hero.label}"
    specs:
${specs}
    capabilities:
      tool_calling: ${b.capabilities.tool_calling}
      vision: ${b.capabilities.vision}
      audio: ${b.capabilities.audio}
      mtp:
        ${mtp}
    stack: ${b.stack}
sharedEntrances:
  - name: ${app}
    host: sharedentrances-${app}
    port: 0
    title: ${cfg.api_title}
    icon: ${icon}
    invisible: true
    authLevel: internal
entrances:
  - name: ${app}cli
    port: 8080
    host: ${app}cli
    title: ${cfg.entrance_title}
    icon: ${icon}
    openMethod: window
    authLevel: internal
spec:
  versionName: ${VERSION}
  featuredImage: ${featured}
  upgradeDescription: |
    v1.0.6: fix P65↔P67 conflict (P65 off, keep P67); 35B 1×24GB: drop MTP K=5 + 8K ctx (FP8 weights ~18GiB OOM with MTP); prefer sndrqwen3627bone for agents.
    v1.0.5: MUST upgrade — your logs show old --target install. Boot stamps SNDR_CHART_BOOT=1.0.5; always purge site-packages/sndr; PYTHONPATH=repo first; editable -e only (never --target sndr).
    v1.0.4: fix EngineAdapter ImportError — drop --target for sndr; uv --system --no-deps -e (upstream install.sh) + PYTHONPATH=repo; purge broken site-packages/sndr.
    v1.0.3: bump accelerator requiredCpu/Memory to cover server+client request sums (4+10m CPU, 24Gi+64Mi).
    v1.0.2: soft-ready /health+/v1/models during boot; uv → persistent site-packages + UV_CACHE; attempt log at bootstrap.log; retries on git/uv.
    v1.0.1: drop aamsellem image — official SNDR pin vllm/vllm-openai@sha256:6a93ae43 (dev748) + runtime sndr_core_engine install + sndr.apply. Qwen27 MTP K=4 + PN521/522; add 35B-A3B MTP K=5; Gemma/Diffusion from upstream ModelDefs.
    v1.0.0: initial.
  fullDescription: |
${full}
  developer: coynntis
  website: https://github.com/Sandermage/sndr_core_engine
  sourceCode: https://github.com/Sandermage/sndr_core_engine
  submitter: coynntis
  locale:
    - en-US
  license:
    - text: Apache-2.0
      url: https://www.apache.org/licenses/LICENSE-2.0
  supportArch:
    - amd64
  onlyAdmin: true
  accelerator:
    - mode: nvidia
      requiredCpu: '5'
      limitedCpu: '17'
      requiredMemory: 25Gi
      limitedMemory: 49Gi
      requiredDisk: 40Gi
      limitedDisk: 80Gi
      requiredGPUMemory: 1Gi
      limitedGPUMemory: 24Gi
permission:
  appData: true
  appCache: true
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
  - envName: UV_INDEX_URL
    required: false
    editable: true
    applyOnChange: true
  - envName: PIP_INDEX_URL
    required: false
    editable: true
    applyOnChange: true
options:
  shared: true
  apiTimeout: 0
  dependencies:
    - name: olares
      version: '>=1.12.6-0'
      type: system
  LLMGatewaySupported: true
`;
}

function writeApp(cfg) {
  const app = cfg.id;
  const d = path.join(ROOT, app);
  fs.mkdirSync(path.join(d, 'templates'), { recursive: true });
  fs.mkdirSync(path.join(d, 'i18n', 'en-US'), { recursive: true });
  fs.writeFileSync(
    path.join(d, 'Chart.yaml'),
    `apiVersion: v2\nappVersion: "${VERSION}"\ndescription: ${JSON.stringify(cfg.desc)}\nname: ${app}\ntype: application\nversion: ${VERSION}\n`,
  );
  fs.writeFileSync(
    path.join(d, 'values.yaml'),
    `workloads:\n  ${app}:\n    replicaCount: 1\n  ${app}cli:\n    replicaCount: 1\nolaresEnv:\n  HF_TOKEN: ''\n  HF_ENDPOINT: ''\n  UV_INDEX_URL: ''\n  PIP_INDEX_URL: ''\n  LLM_CONTEXT_WINDOW: ''\n  LLM_MAX_OUTPUT_TOKENS: ''\n  LLM_API_KEY: ''\n`,
  );
  fs.writeFileSync(path.join(d, 'owners'), 'coynntis\n');
  fs.writeFileSync(path.join(d, '.helmignore'), '*.md\nREADME*\n');
  fs.writeFileSync(path.join(d, 'OlaresManifest.yaml'), manifest(cfg));
  fs.writeFileSync(
    path.join(d, 'i18n', 'en-US', 'OlaresManifest.yaml'),
    `metadata:\n  description: "${cfg.desc}"\n  title: ${cfg.title}\nspec:\n  fullDescription: |\n${cfg.full
      .split('\n')
      .map((l) => `    ${l}`)
      .join('\n')}\n`,
  );
  fs.writeFileSync(path.join(d, 'templates', 'server.yaml'), serverYaml(cfg));
  fs.writeFileSync(path.join(d, 'templates', 'clientproxy.yaml'), clientproxy(app));
  console.log('wrote', app);
}

for (const cfg of APPS) writeApp(cfg);

for (const cfg of APPS) {
  execSync(
    `helm template t ./${cfg.id} --set bfl.username=u --set bfl.namespace=ns --set userspace.appData=/d --set userspace.appCache=/c >/dev/null`,
    { cwd: ROOT, stdio: 'inherit' },
  );
  console.log('helm ok', cfg.id);
}
console.log('done', APPS.length);
