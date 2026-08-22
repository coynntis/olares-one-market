#!/usr/bin/env node
/**
 * Patch llama.cpp server charts: LLM_REASONING toggle, chat template source env,
 * optional bundled chat_template.jinja via ConfigMap mount.
 */
const fs = require('fs');
const path = require('path');

const ROOT = path.join(__dirname, '..');

const CHARTS = [
  {
    dir: 'llamacppqwen3827bmtpone',
    cmName: 'llamacpp-qwen3827bmtp-chat-template',
    bundled: true,
    reasoningDefault: 'on',
    templateDefault: 'bundled',
    bump: { from: '1.0.5', to: '1.0.6' },
    upgradeLine:
      'v1.0.6: remove hardcoded --reasoning off — LLM_REASONING on|off|auto (default on). Bundled Qwen3.8 chat_template.jinja default via LLM_CHAT_TEMPLATE_SOURCE=bundled|froggeric|url; LLM_ENABLE_THINKING / LLM_PRESERVE_THINKING for --chat-template-kwargs.',
  },
  {
    dir: 'llamacppqwen36a3bone',
    cmName: 'llamacpp-qwen36a3b-chat-template',
    bundled: true,
    reasoningDefault: 'on',
    templateDefault: 'bundled',
    bump: { from: '2.0.38', to: '2.0.39' },
    upgradeLine:
      'v2.0.39: LLM_REASONING + LLM_CHAT_TEMPLATE_SOURCE (bundled default) + LLM_ENABLE_THINKING / LLM_PRESERVE_THINKING — no more hardcoded --reasoning off.',
  },
  {
    dir: 'llamacppqwen36a3bdflashone',
    cmName: 'llamacpp-qwen36a3bdflash-chat-template',
    bundled: true,
    reasoningDefault: 'on',
    templateDefault: 'bundled',
    bump: { from: '1.0.6', to: '1.0.7' },
    upgradeLine:
      'v1.0.7: LLM_REASONING + bundled chat template env toggles (default thinking on).',
  },
  {
    dir: 'llamacppqwen36mtpone',
    cmName: null,
    bundled: false,
    reasoningDefault: 'on',
    templateDefault: 'froggeric',
    bump: { from: '1.0.40', to: '1.0.41' },
    upgradeLine:
      'v1.0.41: LLM_REASONING on|off|auto + LLM_CHAT_TEMPLATE_SOURCE froggeric|url + thinking kwargs envs.',
  },
  {
    dir: 'llamacppkatcoderv25one',
    cmName: null,
    bundled: false,
    reasoningDefault: 'off',
    templateDefault: 'froggeric',
    bump: { from: '1.0.4', to: '1.0.5' },
    upgradeLine:
      'v1.0.5: LLM_REASONING env (default off for coding agents) + chat template / thinking toggles.',
  },
  {
    dir: 'llamacppqwen36fable27bone',
    cmName: null,
    bundled: false,
    reasoningDefault: 'on',
    templateDefault: 'froggeric',
    bump: { from: '1.0.3', to: '1.0.4' },
    upgradeLine:
      'v1.0.4: LLM_REASONING + chat template source / thinking env toggles.',
  },
  {
    dir: 'llamacppqwable35bone',
    cmName: null,
    bundled: false,
    reasoningOnly: true,
    reasoningDefault: 'on',
    bump: { from: '1.0.12', to: '1.0.13' },
    upgradeLine:
      'v1.0.13: LLM_REASONING on|off|auto env — no more hardcoded --reasoning off.',
  },
];

function reasoningBlock(defaultMode) {
  return `              REASONING_ARGS=()
              REASONING_MODE="$(echo "\${LLM_REASONING:-${defaultMode}}" | tr '[:upper:]' '[:lower:]')"
              case "$REASONING_MODE" in
                off|0|false|no) REASONING_ARGS=(--reasoning off) ;;
                auto) REASONING_ARGS=(--reasoning auto) ;;
                *) REASONING_ARGS=(--reasoning on) ;;
              esac`;
}

function ctkBlock() {
  return `              CTK_ARGS=()
              if [ -n "\${LLM_ENABLE_THINKING:-}" ] || [ -n "\${LLM_PRESERVE_THINKING:-}" ]; then
                json_bool() { case "$(echo "$1" | tr '[:upper:]' '[:lower:]')" in 1|true|yes|on) echo true ;; *) echo false ;; esac; }
                kw="{"
                [ -n "\${LLM_ENABLE_THINKING:-}" ] && kw="\${kw}\\"enable_thinking\\":$(json_bool "$LLM_ENABLE_THINKING"),"
                [ -n "\${LLM_PRESERVE_THINKING:-}" ] && kw="\${kw}\\"preserve_thinking\\":$(json_bool "$LLM_PRESERVE_THINKING"),"
                kw="\${kw%,}}"
                CTK_ARGS=(--chat-template-kwargs "$kw")
              fi`;
}

function templateBlock(defaultSrc) {
  return `              CHAT_TEMPLATE="\${MODELS_DIR}/\${CHAT_TEMPLATE_FILE:-chat_template.jinja}"
              TEMPLATE_SRC="$(echo "\${LLM_CHAT_TEMPLATE_SOURCE:-${defaultSrc}}" | tr '[:upper:]' '[:lower:]')"
              case "$TEMPLATE_SRC" in
                froggeric|remote|hf)
                  log "Fetching chat template (froggeric)"
                  HF_HDR=()
                  if [ -n "\${HF_TOKEN:-}" ]; then HF_HDR=(-H "Authorization: Bearer $HF_TOKEN"); fi
                  curl -fL --retry 10 --retry-delay 5 "\${HF_HDR[@]}" \\
                    "\${CHAT_TEMPLATE_URL:-https://huggingface.co/froggeric/Qwen-Fixed-Chat-Templates/resolve/main/chat_template.jinja}" \\
                    -o "$CHAT_TEMPLATE"
                  ;;
                url|custom)
                  if [ -z "\${LLM_CHAT_TEMPLATE_URL:-}" ]; then log "LLM_CHAT_TEMPLATE_URL required when source=url"; exit 1; fi
                  log "Fetching chat template from $LLM_CHAT_TEMPLATE_URL"
                  curl -fL --retry 10 --retry-delay 5 -o "$CHAT_TEMPLATE" "$LLM_CHAT_TEMPLATE_URL"
                  ;;
                bundled|local|chart|*)
                  if [ -f "/etc/llamacpp/chat-template/chat_template.jinja" ]; then
                    cp "/etc/llamacpp/chat-template/chat_template.jinja" "$CHAT_TEMPLATE"
                    log "Using bundled chart chat template"
                  else
                    log "Bundled template missing; falling back to froggeric"
                    HF_HDR=()
                    if [ -n "\${HF_TOKEN:-}" ]; then HF_HDR=(-H "Authorization: Bearer $HF_TOKEN"); fi
                    curl -fL --retry 10 --retry-delay 5 "\${HF_HDR[@]}" \\
                      "\${CHAT_TEMPLATE_URL:-https://huggingface.co/froggeric/Qwen-Fixed-Chat-Templates/resolve/main/chat_template.jinja}" \\
                      -o "$CHAT_TEMPLATE"
                  fi
                  ;;
              esac`;
}

function configMapBlock(cmName) {
  return `---
apiVersion: v1
kind: ConfigMap
metadata:
  name: ${cmName}
  namespace: "{{ .Release.Namespace }}"
data:
  chat_template.jinja: |
{{ .Files.Get "chat_template.jinja" | nindent 4 }}
`;
}

function envYamlExtra() {
  return `  - envName: LLM_REASONING
    required: false
    editable: true
    applyOnChange: true
  - envName: LLM_CHAT_TEMPLATE_SOURCE
    required: false
    editable: true
    applyOnChange: true
  - envName: LLM_CHAT_TEMPLATE_URL
    required: false
    editable: true
    applyOnChange: true
  - envName: LLM_ENABLE_THINKING
    required: false
    editable: true
    applyOnChange: true
  - envName: LLM_PRESERVE_THINKING
    required: false
    editable: true
    applyOnChange: true`;
}

function valuesEnvExtra(chart) {
  const lines = [
    '  LLM_REASONING: \'\'',
    '  LLM_CHAT_TEMPLATE_SOURCE: \'\'',
    '  LLM_CHAT_TEMPLATE_URL: \'\'',
    '  LLM_ENABLE_THINKING: \'\'',
    '  LLM_PRESERVE_THINKING: \'\'',
  ];
  if (chart.reasoningOnly) return `  LLM_REASONING: ''\n`;
  return lines.join('\n') + '\n';
}

function deploymentEnvExtra(chart) {
  if (chart.reasoningOnly) {
    return `            - name: LLM_REASONING
              value: {{ .Values.olaresEnv.LLM_REASONING | default "" | quote }}`;
  }
  return `            - name: LLM_REASONING
              value: {{ .Values.olaresEnv.LLM_REASONING | default "" | quote }}
            - name: LLM_CHAT_TEMPLATE_SOURCE
              value: {{ .Values.olaresEnv.LLM_CHAT_TEMPLATE_SOURCE | default "" | quote }}
            - name: LLM_CHAT_TEMPLATE_URL
              value: {{ .Values.olaresEnv.LLM_CHAT_TEMPLATE_URL | default "" | quote }}
            - name: LLM_ENABLE_THINKING
              value: {{ .Values.olaresEnv.LLM_ENABLE_THINKING | default "" | quote }}
            - name: LLM_PRESERVE_THINKING
              value: {{ .Values.olaresEnv.LLM_PRESERVE_THINKING | default "" | quote }}`;
}

function patchServerYaml(chart) {
  const file = path.join(ROOT, chart.dir, 'templates/server.yaml');
  let s = fs.readFileSync(file, 'utf8');

  if (chart.bundled && chart.cmName && !s.includes(chart.cmName)) {
    s = s.replace(/^---\napiVersion: v1\nkind: ConfigMap\n/m, configMapBlock(chart.cmName) + '---\napiVersion: v1\nkind: ConfigMap\n');
  }

  // Replace chat template fetch block
  const fetchRe =
    /              CHAT_TEMPLATE="\$\{MODELS_DIR\}\/\$\{CHAT_TEMPLATE_FILE:-chat_template\.jinja\}"\n(?:              log "Fetching chat template"\n)?              HF_HDR=\(\)\n              if \[ -n "\$\{HF_TOKEN:-\}" \]; then HF_HDR=\(-H "Authorization: Bearer \$HF_TOKEN"\); fi\n              curl -fL[^\n]+\n                "\$\{CHAT_TEMPLATE_URL[^\n]+\n                -o "\$CHAT_TEMPLATE"/;

  const altFetchRe =
    /              CHAT_TEMPLATE="\$\{MODELS_DIR:-\/models\}\/\$\{CHAT_TEMPLATE_FILE:-chat_template\.jinja\}"\n              curl[^\n]+\n                "\$\{CHAT_TEMPLATE_URL[^\n]+\n                -o "\$CHAT_TEMPLATE"/;

  const replacement =
    reasoningBlock(chart.reasoningDefault) +
    '\n' +
    (chart.reasoningOnly ? '' : ctkBlock() + '\n') +
    (chart.reasoningOnly ? '' : templateBlock(chart.templateDefault));

  if (fetchRe.test(s)) {
    s = s.replace(fetchRe, replacement);
  } else if (altFetchRe.test(s)) {
    s = s.replace(altFetchRe, replacement);
  } else if (chart.reasoningOnly) {
    // qwable: insert before exec
    s = s.replace(
      /              EXTRA_LLM_ARGS=\(\)\n              if \[ -n "\$\{LLM_MAX_OUTPUT_TOKENS:-\}" \]; then EXTRA_LLM_ARGS\+\=\(--n-predict "\$LLM_MAX_OUTPUT_TOKENS"\); fi\n              if \[ -n "\$\{LLM_API_KEY:-\}" \]; then EXTRA_LLM_ARGS\+\=\(--api-key "\$LLM_API_KEY"\); fi\n\n              exec/,
      `              EXTRA_LLM_ARGS=()\n              if [ -n "\${LLM_MAX_OUTPUT_TOKENS:-}" ]; then EXTRA_LLM_ARGS+=(--n-predict "$LLM_MAX_OUTPUT_TOKENS"); fi\n              if [ -n "\${LLM_API_KEY:-}" ]; then EXTRA_LLM_ARGS+=(--api-key "$LLM_API_KEY"); fi\n${reasoningBlock(chart.reasoningDefault)}\n\n              exec`,
    );
  } else {
    console.warn(`WARN: no chat template block matched in ${chart.dir}`);
  }

  s = s.replace(/\n                --reasoning off \\?\n/g, '\n                "${REASONING_ARGS[@]}" \\\n');
  s = s.replace(/\n                --reasoning off\n/g, '\n                "${REASONING_ARGS[@]}"\n');

  if (!chart.reasoningOnly) {
    s = s.replace(
      /(\$\{EXTRA_LLM_ARGS\[@\]\})/,
      '"${CTK_ARGS[@]}" \\\n                "${REASONING_ARGS[@]}" \\\n                $1',
    );
    // If reasoning already inserted before EXTRA, dedupe
    s = s.replace(
      /"\$\{REASONING_ARGS\[@\]\}" \\\n                "\$\{CTK_ARGS\[@\]\}" \\\n                "\$\{REASONING_ARGS\[@\]\}"/g,
      '"${CTK_ARGS[@]}" \\\n                "${REASONING_ARGS[@]}"',
    );
  }

  if (chart.bundled && chart.cmName) {
    if (!s.includes('name: chat-template')) {
      s = s.replace(
        /(          volumeMounts:\n(?:            - mountPath:[^\n]+\n              name: [^\n]+\n)+)/,
        `$1            - mountPath: /etc/llamacpp/chat-template\n              name: chat-template\n              readOnly: true\n`,
      );
      s = s.replace(
        /(      volumes:\n(?:        - name:[^\n]+\n          hostPath:[^\n]+\n            path:[^\n]+\n            type:[^\n]+\n)+)/,
        `$1        - name: chat-template\n          configMap:\n            name: ${chart.cmName}\n`,
      );
    }
  }

  if (!s.includes('LLM_REASONING')) {
    s = s.replace(
      /(            - name: LLM_SPEC_DRAFT_N_MAX\n              value:[^\n]+\n)/,
      `$1${deploymentEnvExtra(chart)}\n`,
    );
    if (!s.includes('LLM_REASONING') && chart.reasoningOnly) {
      s = s.replace(
        /(            - name: LLM_API_KEY\n              value:[^\n]+\n)/,
        `$1${deploymentEnvExtra(chart)}\n`,
      );
    }
  }

  fs.writeFileSync(file, s);
  console.log(`patched ${chart.dir}/templates/server.yaml`);
}

function patchManifest(chart) {
  const file = path.join(ROOT, chart.dir, 'OlaresManifest.yaml');
  let s = fs.readFileSync(file, 'utf8');
  const { from, to } = chart.bump;
  s = s.replaceAll(`version: ${from}`, `version: ${to}`);
  s = s.replaceAll(`versionName: ${from}`, `versionName: ${to}`);
  s = s.replaceAll(`appVersion: "${from}"`, `appVersion: "${to}"`);

  if (!s.includes('LLM_REASONING')) {
    if (chart.reasoningOnly) {
      s = s.replace(
        /(  - envName: LLM_API_KEY\n    required: false\n    editable: true\n    applyOnChange: true\n)/,
        `$1  - envName: LLM_REASONING\n    required: false\n    editable: true\n    applyOnChange: true\n`,
      );
    } else {
      s = s.replace(/(options:\n)/, `${envYamlExtra()}\n$1`);
      // envs inserted before options — fix order: should be inside envs
    }
  }

  if (!s.includes('LLM_REASONING') && !chart.reasoningOnly) {
    s = fs.readFileSync(file, 'utf8');
    s = s.replace(
      /(  - envName: LLM_SPEC_DRAFT_N_MAX\n    required: false\n    editable: true\n    applyOnChange: true\n)/,
      `$1${envYamlExtra().split('\n').slice(1).join('\n')}\n`,
    );
    if (chart.dir === 'llamacppqwen3827bmtpone') {
      s = s.replace(
        /(  - envName: QWEN38_VARIANT\n    required: false\n    editable: true\n    applyOnChange: true\n)/,
        `$1${envYamlExtra().split('\n').slice(1).join('\n')}\n`,
      );
    }
  }

  if (!s.includes(chart.upgradeLine.slice(0, 20))) {
    s = s.replace(/(  upgradeDescription: \|\n)/, `$1    ${chart.upgradeLine}\n`);
  }

  // Fix doc lines mentioning hardcoded reasoning off
  s = s.replace(/    - --reasoning off, --jinja, froggeric chat template\n/g, '    - LLM_REASONING (default on) + bundled Qwen3.8 chat template; froggeric via LLM_CHAT_TEMPLATE_SOURCE\n');

  fs.writeFileSync(file, s);
  console.log(`patched ${chart.dir}/OlaresManifest.yaml`);
}

function patchChartYaml(chart) {
  const file = path.join(ROOT, chart.dir, 'Chart.yaml');
  let s = fs.readFileSync(file, 'utf8');
  s = s.replace(`version: ${chart.bump.from}`, `version: ${chart.bump.to}`);
  s = s.replace(`appVersion: "${chart.bump.from}"`, `appVersion: "${chart.bump.to}"`);
  fs.writeFileSync(file, s);
}

function patchValues(chart) {
  const file = path.join(ROOT, chart.dir, 'values.yaml');
  if (!fs.existsSync(file)) return;
  let s = fs.readFileSync(file, 'utf8');
  if (s.includes('LLM_REASONING')) return;
  s = s.replace(/(olaresEnv:\n(?:  [^\n]+\n)*)/, `$1${valuesEnvExtra(chart)}`);
  fs.writeFileSync(file, s);
  console.log(`patched ${chart.dir}/values.yaml`);
}

for (const chart of CHARTS) {
  patchServerYaml(chart);
  patchManifest(chart);
  patchChartYaml(chart);
  patchValues(chart);
}

console.log('done');
