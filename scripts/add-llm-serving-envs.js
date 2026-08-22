#!/usr/bin/env node
/**
 * Add user-configurable LLM env vars (Settings → Application) for llama.cpp / vLLM / SGLang servers:
 *   LLM_CONTEXT_WINDOW, LLM_MAX_OUTPUT_TOKENS, LLM_API_KEY
 * Falls back to chart defaults when unset.
 */
const fs = require('fs');
const path = require('path');
const REPO = path.resolve(__dirname, '..');

const LLM_ENV_NAMES = ['LLM_CONTEXT_WINDOW', 'LLM_MAX_OUTPUT_TOKENS', 'LLM_API_KEY'];

const LLM_MANIFEST_ENVS = `  - envName: LLM_CONTEXT_WINDOW
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
    applyOnChange: true`;

const LLM_DEPLOY_ENV = `            - name: LLM_CONTEXT_WINDOW
              value: {{ .Values.olaresEnv.LLM_CONTEXT_WINDOW | default "" | quote }}
            - name: LLM_MAX_OUTPUT_TOKENS
              value: {{ .Values.olaresEnv.LLM_MAX_OUTPUT_TOKENS | default "" | quote }}
            - name: LLM_API_KEY
              value: {{ .Values.olaresEnv.LLM_API_KEY | default "" | quote }}`;

const HELM_LLM_VARS = `{{- $llmCtx := .Values.olaresEnv.LLM_CONTEXT_WINDOW | default "" }}
{{- $llmMaxOut := .Values.olaresEnv.LLM_MAX_OUTPUT_TOKENS | default "" }}
{{- $llmApiKey := .Values.olaresEnv.LLM_API_KEY | default "" }}`;

const EXTRA_LLM_BASH = `              EXTRA_LLM_ARGS=()
              if [ -n "\${LLM_MAX_OUTPUT_TOKENS:-}" ]; then EXTRA_LLM_ARGS+=(--n-predict "\$LLM_MAX_OUTPUT_TOKENS"); fi
              if [ -n "\${LLM_API_KEY:-}" ]; then EXTRA_LLM_ARGS+=(--api-key "\$LLM_API_KEY"); fi`;

const EXTRA_SGLANG_BASH = `              EXTRA_LLM_ARGS=()
              if [ -n "\${LLM_API_KEY:-}" ]; then EXTRA_LLM_ARGS+=(--api-key "\$LLM_API_KEY"); fi`;

const VLLM_HELM_EXTRAS = `
{{- if $llmApiKey }}
            - "--api-key"
            - {{ $llmApiKey | quote }}
{{- end }}
{{- if $llmMaxOut }}
            - "--override-generation-config"
            - {{ printf "{\\"max_tokens\\":%s}" $llmMaxOut | quote }}
{{- end }}`;

function bumpPatch(version) {
  const parts = String(version).replace(/['"]/g, '').split('.');
  const last = parseInt(parts[parts.length - 1], 10);
  parts[parts.length - 1] = String(Number.isNaN(last) ? 1 : last + 1);
  return parts.join('.');
}

function bumpAppVersion(appDir) {
  const manifestPath = path.join(appDir, 'OlaresManifest.yaml');
  const chartPath = path.join(appDir, 'Chart.yaml');
  if (!fs.existsSync(manifestPath)) return null;
  let raw = fs.readFileSync(manifestPath, 'utf8');
  const m = raw.match(/^  version:\s*['"]?([^'"\n]+)['"]?/m);
  const oldV = (m ? m[1] : '1.0.0').replace(/['"]/g, '');
  const newV = bumpPatch(oldV);
  raw = raw.replace(
    new RegExp(`(^  version:\\s*)(['"]?)${oldV.replace(/\./g, '\\.')}\\2`, 'm'),
    `$1'${newV}'`,
  );
  raw = raw.replace(
    new RegExp(`(^  versionName:\\s*)(['"]?)${oldV.replace(/\./g, '\\.')}\\2`, 'm'),
    `$1'${newV}'`,
  );
  fs.writeFileSync(manifestPath, raw);
  if (fs.existsSync(chartPath)) {
    let chartRaw = fs.readFileSync(chartPath, 'utf8').replace(/^apiVersion: v2\n(?=apiVersion: v2\n)/, '');
    if (/^version:/m.test(chartRaw)) {
      chartRaw = chartRaw.replace(/^version:.*$/m, `version: ${newV}`);
    }
    if (/^appVersion:/m.test(chartRaw)) {
      chartRaw = chartRaw.replace(/^appVersion:.*$/m, `appVersion: ${newV}`);
    }
    fs.writeFileSync(chartPath, chartRaw);
  }
  return newV;
}

function ensureManifestEnvs(appDir) {
  const manifestPath = path.join(appDir, 'OlaresManifest.yaml');
  if (!fs.existsSync(manifestPath)) return false;
  let raw = fs.readFileSync(manifestPath, 'utf8');
  if (raw.includes('envName: LLM_CONTEXT_WINDOW')) return false;
  if (/^envs:/m.test(raw)) {
    raw = raw.replace(/^envs:\n/m, `envs:\n${LLM_MANIFEST_ENVS}\n`);
  } else {
    raw = raw.replace(/^permission:/m, `envs:\n${LLM_MANIFEST_ENVS}\npermission:`);
  }
  fs.writeFileSync(manifestPath, raw);
  return true;
}

function insertDeployEnv(content) {
  if (content.includes('name: LLM_CONTEXT_WINDOW')) return content;
  const markers = [
    /(\n          env:\n(?:            - name:[^\n]+\n(?:              value:[^\n]+\n)*)+)/,
    /(\n          envFrom:\n)/,
  ];
  for (const re of markers) {
    if (re.test(content)) {
      return content.replace(re, (m) => {
        if (m.includes('envFrom:')) return `\n          env:\n${LLM_DEPLOY_ENV}${m}`;
        return `${m}${LLM_DEPLOY_ENV}\n`;
      });
    }
  }
  return content;
}

function ensureHelmLlmVars(content) {
  if (content.includes('$llmCtx :=')) return content;
  return content.replace(
    /^(\{\{- if and \.Values\.admin[^\n]+\}\}\n)/,
    `$1${HELM_LLM_VARS}\n`,
  );
}

function patchVllmDeployment(content) {
  let out = ensureHelmLlmVars(content);
  if (!out.includes('$llmCtx :=')) return out;

  out = out.replace(
    /            - "--max-model-len"\n            - "\$\((MAX_MODEL_LEN|CONTEXT_SIZE)\)"/g,
    (_, varName) => `            - "--max-model-len"
{{- if $llmCtx }}
            - {{ $llmCtx | quote }}
{{- else }}
            - "$(${varName})"
{{- end }}`,
  );

  if (!out.includes('$llmMaxOut }}') && !out.includes('override-generation-config')) {
    out = out.replace(/(\n          envFrom:\n)/, `${VLLM_HELM_EXTRAS}$1`);
  }
  return insertDeployEnv(out);
}

function patchLlamacppArgsDeployment(content) {
  let out = ensureHelmLlmVars(content);
  if (!out.includes('$llmCtx :=')) return out;

  out = out.replace(
    /            - "--ctx-size"\n            - "\$\(CONTEXT_SIZE\)"/,
    `            - "--ctx-size"
{{- if $llmCtx }}
            - {{ $llmCtx | quote }}
{{- else }}
            - "$(CONTEXT_SIZE)"
{{- end }}`,
  );

  if (!out.includes('$llmApiKey }}') && !out.includes('--n-predict')) {
    out = out.replace(
      /(\n            - "--no-context-shift"\n)(          env:)/,
      `$1{{- if $llmMaxOut }}
            - "--n-predict"
            - {{ $llmMaxOut | quote }}
{{- end }}
{{- if $llmApiKey }}
            - "--api-key"
            - {{ $llmApiKey | quote }}
{{- end }}
$2`,
    );
  }
  return insertDeployEnv(out);
}

function patchLlamacppBashDeployment(content) {
  let out = content;
  if (!out.includes('EXTRA_LLM_ARGS=()')) {
    out = out.replace(/(\n              exec .*llama-server)/, `\n${EXTRA_LLM_BASH}$1`);
  }

  const ctxPatterns = [
    [/(--ctx-size "\$\{CTX_SIZE:-(\d+)\}")/, '--ctx-size "${LLM_CONTEXT_WINDOW:-${CTX_SIZE:-$1}}"', false],
    [/(--ctx-size "\$\{CTX_SIZE:-(\d+)\}")/, null, true],
    [/(--ctx-size "\$\{CTX_SIZE:-(\d+)\}")/g, '--ctx-size "${LLM_CONTEXT_WINDOW:-${CTX_SIZE:-$1}}"', false],
    [/--ctx-size "\$\{CTX_SIZE:-(\d+)\}"/g, '--ctx-size "${LLM_CONTEXT_WINDOW:-${CTX_SIZE:-$1}}"', false],
    [/--ctx-size "\$\{CTX_SIZE:-(\d+)\}"/, '--ctx-size "${LLM_CONTEXT_WINDOW:-${CTX_SIZE:-$1}}"', false],
    [/--ctx-size "\$\{CONTEXT_SIZE\}"/, '--ctx-size "${LLM_CONTEXT_WINDOW:-${CONTEXT_SIZE}}"', false],
  ];

  for (const [re, repl] of [
    [/--ctx-size "\$\{CTX_SIZE:-(\d+)\}"/g, (m, n) => `--ctx-size "\${LLM_CONTEXT_WINDOW:-\${CTX_SIZE:-${n}}}"`],
    [/--ctx-size "\$\{CONTEXT_SIZE\}"/, '--ctx-size "${LLM_CONTEXT_WINDOW:-${CONTEXT_SIZE}}"'],
  ]) {
    out = out.replace(re, repl);
  }

  if (!out.includes('${EXTRA_LLM_ARGS[@]}')) {
    out = out.replace(
      /(exec .*llama-server \\\n(?:                [^\n]+\\\n)*)(                [^\n\\]+)\n(          envFrom:)/,
      (m, head, lastLine, tail) => {
        const line = lastLine.trimEnd();
        if (line.endsWith('\\')) return m;
        return `${head}${lastLine} \\\n                "\${EXTRA_LLM_ARGS[@]}"\n${tail}`;
      },
    );
    out = out.replace(
      /(exec \/app\/llama-server \\\n(?:                [^\n]+\\\n)*)(                [^\n\\]+)\n(          envFrom:)/,
      (m, head, lastLine, tail) => {
        const line = lastLine.trimEnd();
        if (line.endsWith('\\')) return m;
        return `${head}${lastLine} \\\n                "\${EXTRA_LLM_ARGS[@]}"\n${tail}`;
      },
    );
    out = out.replace(
      /(exec \.\/build\/bin\/llama-server \\\n(?:                [^\n]+\\\n)*)(                [^\n\\]+)\n(          env:)/,
      (m, head, lastLine, tail) => {
        if (lastLine.includes('EXTRA_LLM_ARGS')) return m;
        return `${head}${lastLine} \\\n                "\${EXTRA_LLM_ARGS[@]}"\n${tail}`;
      },
    );
  }

  // Apps without --ctx-size: add once after --port line
  if (!out.includes('--ctx-size') && /exec .*llama-server/.test(out)) {
    out = out.replace(
      /(--host 0\.0\.0\.0 --port \d+ \\\n)/,
      `$1                --ctx-size "\${LLM_CONTEXT_WINDOW:-16384}" \\\n`,
    );
  }

  if (out.includes('${EXTRA_LLM_ARGS[@]}') && (out.match(/\$\{EXTRA_LLM_ARGS\[@\]\}/g) || []).length > 1) {
    out = out.replace(/\n\s+"\$\{EXTRA_LLM_ARGS\[@\]\}"\s*\\\n\s+"\$\{EXTRA_LLM_ARGS\[@\]\}"/, '\n                "${EXTRA_LLM_ARGS[@]}"');
  }

  return insertDeployEnv(out);
}

function patchSglangDeployment(content) {
  let out = content;
  if (!out.includes('EXTRA_LLM_ARGS=()')) {
    out = out.replace(/(\n              exec python3 -m sglang\.launch_server)/, `\n${EXTRA_SGLANG_BASH}$1`);
  }
  out = out.replace(
    /--context-length "\$\(CONTEXT_LENGTH\)"/,
    '--context-length "${LLM_CONTEXT_WINDOW:-$(CONTEXT_LENGTH)}"',
  );
  if (!out.includes('${EXTRA_LLM_ARGS[@]}')) {
    out = out.replace(
      /(--trust-remote-code \\\n)(\s+)"\$\{EXTRA_ARGS\[@\]\}"/,
      `$1$2"\${EXTRA_ARGS[@]}" "\${EXTRA_LLM_ARGS[@]}"`,
    );
    out = out.replace(
      /(--trust-remote-code\n)(\s+)"\$\{EXTRA_ARGS\[@\]\}"/,
      `$1$2"\${EXTRA_ARGS[@]}" "\${EXTRA_LLM_ARGS[@]}"`,
    );
  }
  return insertDeployEnv(out);
}

function patchDflashDeployment(content) {
  let out = content.replace(
    /--max-ctx "\$\{MAX_CTX\}"/,
    '--max-ctx "${LLM_CONTEXT_WINDOW:-$MAX_CTX}"',
  );
  return insertDeployEnv(out);
}

function classifyDeployment(content) {
  if (/sglang\.launch_server/.test(content)) return 'sglang';
  if (/--max-ctx "\$\{MAX_CTX\}"/.test(content)) return 'dflash';
  if (/vllm serve|command:\s*\n\s+- "vllm"/.test(content)) return 'vllm';
  if (/image:.*vllm/i.test(content) && /--max-model-len/.test(content)) return 'vllm';
  if (/command:\s*\[.*llama-server/.test(content)) return 'llamacpp-args';
  if (/llama-server/.test(content)) return 'llamacpp-bash';
  return null;
}

function patchDeployment(deployPath) {
  let content = fs.readFileSync(deployPath, 'utf8');
  const kind = classifyDeployment(content);
  if (!kind) return null;

  const before = content;
  switch (kind) {
    case 'vllm':
      content = patchVllmDeployment(content);
      break;
    case 'llamacpp-args':
      content = patchLlamacppArgsDeployment(content);
      break;
    case 'llamacpp-bash':
      content = patchLlamacppBashDeployment(content);
      break;
    case 'sglang':
      content = patchSglangDeployment(content);
      break;
    case 'dflash':
      content = patchDflashDeployment(content);
      break;
    default:
      return null;
  }

  if (content !== before) {
    fs.writeFileSync(deployPath, content);
    return kind;
  }
  return null;
}

function findApps() {
  const apps = [];
  for (const name of fs.readdirSync(REPO)) {
    const appDir = path.join(REPO, name);
    if (!fs.statSync(appDir).isDirectory()) continue;
    if (!fs.existsSync(path.join(appDir, 'OlaresManifest.yaml'))) continue;
    const srvDirs = fs.readdirSync(appDir).filter((d) => {
      const p = path.join(appDir, d);
      return fs.statSync(p).isDirectory() && d.endsWith('srv');
    });
    for (const srv of srvDirs) {
      const deploy = path.join(appDir, srv, 'templates', 'deployment.yaml');
      if (fs.existsSync(deploy)) apps.push({ appDir, deploy, appName: name });
    }
  }
  return apps;
}

function main() {
  const results = [];
  for (const { appDir, deploy, appName } of findApps()) {
    const kind = patchDeployment(deploy);
    if (!kind) continue;
    const manifestChanged = ensureManifestEnvs(appDir);
    const newV = bumpAppVersion(appDir);
    results.push({ appName, kind, newV, manifestChanged });
  }
  console.log(`Patched ${results.length} LLM serving apps:`);
  for (const r of results) {
    console.log(`  ${r.appName} (${r.kind}) → v${r.newV}`);
  }
}

main();
