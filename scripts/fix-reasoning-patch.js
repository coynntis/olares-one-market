#!/usr/bin/env node
/** Fix mangled llama-server arg lines + chat-template mounts from add-llm-reasoning-template-envs.js */
const fs = require('fs');
const path = require('path');
const ROOT = path.join(__dirname, '..');

const BUNDLED = [
  { dir: 'llamacppqwen3827bmtpone', cm: 'llamacpp-qwen3827bmtp-chat-template' },
  { dir: 'llamacppqwen36a3bone', cm: 'llamacpp-qwen36a3b-chat-template' },
  { dir: 'llamacppqwen36a3bdflashone', cm: 'llamacpp-qwen36a3bdflash-chat-template' },
];

const ALL = [
  ...BUNDLED.map((b) => b.dir),
  'llamacppqwen36mtpone',
  'llamacppkatcoderv25one',
  'llamacppqwen36fable27bone',
  'llamacppqwable35bone',
];

const REASONING_BLOCK = `              REASONING_ARGS=()
              REASONING_MODE="$(echo "\${LLM_REASONING:-on}" | tr '[:upper:]' '[:lower:]')"
              case "$REASONING_MODE" in
                off|0|false|no) REASONING_ARGS=(--reasoning off) ;;
                auto) REASONING_ARGS=(--reasoning auto) ;;
                *) REASONING_ARGS=(--reasoning on) ;;
              esac
              CTK_ARGS=()
              if [ -n "\${LLM_ENABLE_THINKING:-}" ] || [ -n "\${LLM_PRESERVE_THINKING:-}" ]; then
                json_bool() { case "$(echo "$1" | tr '[:upper:]' '[:lower:]')" in 1|true|yes|on) echo true ;; *) echo false ;; esac; }
                kw="{"
                [ -n "\${LLM_ENABLE_THINKING:-}" ] && kw="\${kw}\\"enable_thinking\\":$(json_bool "$LLM_ENABLE_THINKING"),"
                [ -n "\${LLM_PRESERVE_THINKING:-}" ] && kw="\${kw}\\"preserve_thinking\\":$(json_bool "$LLM_PRESERVE_THINKING"),"
                kw="\${kw%,}}"
                CTK_ARGS=(--chat-template-kwargs "$kw")
              fi`;

const KAT_REASONING = REASONING_BLOCK.replace('LLM_REASONING:-on', 'LLM_REASONING:-off');

const TEMPLATE_BLOCK_FROG = `              CHAT_TEMPLATE="\${MODELS_DIR:-/models}/\${CHAT_TEMPLATE_FILE:-chat_template.jinja}"
              TEMPLATE_SRC="$(echo "\${LLM_CHAT_TEMPLATE_SOURCE:-froggeric}" | tr '[:upper:]' '[:lower:]')"
              case "$TEMPLATE_SRC" in
                froggeric|remote|hf)
                  echo "==> Fetching chat template (froggeric)"
                  eval curl -fL --retry 10 --retry-delay 5 $HF_AUTH \\
                    "\${CHAT_TEMPLATE_URL:-https://huggingface.co/froggeric/Qwen-Fixed-Chat-Templates/resolve/main/chat_template.jinja}" -o "$CHAT_TEMPLATE"
                  ;;
                url|custom)
                  if [ -z "\${LLM_CHAT_TEMPLATE_URL:-}" ]; then echo "LLM_CHAT_TEMPLATE_URL required when source=url"; exit 1; fi
                  echo "==> Fetching chat template from $LLM_CHAT_TEMPLATE_URL"
                  curl -fL --retry 10 --retry-delay 5 -o "$CHAT_TEMPLATE" "$LLM_CHAT_TEMPLATE_URL"
                  ;;
                bundled|local|chart|*)
                  if [ -f "/etc/llamacpp/chat-template/chat_template.jinja" ]; then
                    cp "/etc/llamacpp/chat-template/chat_template.jinja" "$CHAT_TEMPLATE"
                    echo "==> Using bundled chart chat template"
                  else
                    echo "==> Bundled template missing; falling back to froggeric"
                    eval curl -fL --retry 10 --retry-delay 5 $HF_AUTH \\
                      "\${CHAT_TEMPLATE_URL:-https://huggingface.co/froggeric/Qwen-Fixed-Chat-Templates/resolve/main/chat_template.jinja}" -o "$CHAT_TEMPLATE"
                  fi
                  ;;
              esac`;

function fixArgs(s) {
  // Remove duplicate reasoning before spec-type; fix broken quoting
  s = s.replace(
    /(\n                --op-offload --jinja --chat-template-file "\$CHAT_TEMPLATE" \\)\n                "\$\{REASONING_ARGS\[@\]\}" \\/g,
    '$1',
  );
  s = s.replace(
    /(\n                --flash-attn on --jinja --chat-template-file "\$CHAT_TEMPLATE" \\)\n                "\$\{REASONING_ARGS\[@\]\}" \\/g,
    '$1',
  );
  s = s.replace(
    /(\n                --flash-attn on \\)\n                "\$\{REASONING_ARGS\[@\]\}" \\/g,
    '$1',
  );
  s = s.replace(
    /(\n                --flash-attn on \\)\n                "\$\{REASONING_ARGS\[@\]\}" \\/g,
    '$1',
  );
  s = s.replace(
    /(\n                --op-offload --jinja --chat-template-file "\$CHAT_TEMPLATE" \\)\n                "\$\{REASONING_ARGS\[@\]\}" \\/g,
    '$1',
  );
  s = s.replace(
    /\n                ""\$\{CTK_ARGS\[@\]\}" \\\n                "\$\{REASONING_ARGS\[@\]\}" \\\n                \$\{EXTRA_LLM_ARGS\[@\]\}(")/g,
    '\n                "${CTK_ARGS[@]}" \\\n                "${REASONING_ARGS[@]}" \\\n                "${EXTRA_LLM_ARGS[@]}$1',
  );
  s = s.replace(
    /\n                ""\$\{CTK_ARGS\[@\]\}" \\\n                "\$\{REASONING_ARGS\[@\]\}" \\\n                \$\{EXTRA_LLM_ARGS\[@\]\}" 2>&1/g,
    '\n                "${CTK_ARGS[@]}" \\\n                "${REASONING_ARGS[@]}" \\\n                "${EXTRA_LLM_ARGS[@]}" 2>&1',
  );
  return s;
}

function fixMounts(s, cm) {
  // Remove wrong initContainer mount
  s = s.replace(
    /\n            - mountPath: \/etc\/llamacpp\/chat-template\n              name: chat-template\n              readOnly: true(?=\n      containers:)/,
    '',
  );
  // Add main container mount if missing
  if (!s.includes('containers:\n        - name:') || s.includes('name: chat-template\n              readOnly: true\n      volumes:')) {
    // already on main or fixed
  }
  if (cm && !s.match(/containers:[\s\S]*volumeMounts:[\s\S]*name: chat-template/)) {
    s = s.replace(
      /(          volumeMounts:\n            - mountPath: "\/comfyui-llms"\n              name: comfyui-llms\n)(      volumes:)/,
      `$1            - mountPath: /etc/llamacpp/chat-template\n              name: chat-template\n              readOnly: true\n$2`,
    );
  }
  if (cm && !s.includes(`name: ${cm}`)) {
    // configmap exists at top
  }
  if (cm && !s.match(/      volumes:[\s\S]*- name: chat-template/)) {
    s = s.replace(
      /(        - name: comfyui-llms\n          hostPath:\n            path: "[^"]+"\n            type: DirectoryOrCreate\n)(      restartPolicy:)/,
      `$1        - name: chat-template\n          configMap:\n            name: ${cm}\n$2`,
    );
  }
  return s;
}

function fixMtpone(s) {
  if (!s.includes('REASONING_ARGS=()')) {
    s = s.replace(
      /              CHAT_TEMPLATE="\$\{MODELS_DIR:-\/models\}\/\$\{CHAT_TEMPLATE_FILE:-chat_template\.jinja\}"\n              echo "==> Fetching chat template"\n              eval curl[^\n]+\n                "\$\{CHAT_TEMPLATE_URL[^\n]+\n                -o "\$CHAT_TEMPLATE"/,
      REASONING_BLOCK + '\n' + TEMPLATE_BLOCK_FROG,
    );
  }
  return s;
}

function fixKat(s) {
  if (s.includes('REASONING_ARGS=()') && s.includes('LLM_REASONING:-on')) {
    s = s.replace('LLM_REASONING:-on', 'LLM_REASONING:-off');
  }
  return s;
}

function addDeploymentEnvs(s, reasoningOnly = false) {
  const block = reasoningOnly
    ? `            - name: LLM_REASONING\n              value: {{ .Values.olaresEnv.LLM_REASONING | default "" | quote }}\n`
    : `            - name: LLM_REASONING\n              value: {{ .Values.olaresEnv.LLM_REASONING | default "" | quote }}\n            - name: LLM_CHAT_TEMPLATE_SOURCE\n              value: {{ .Values.olaresEnv.LLM_CHAT_TEMPLATE_SOURCE | default "" | quote }}\n            - name: LLM_CHAT_TEMPLATE_URL\n              value: {{ .Values.olaresEnv.LLM_CHAT_TEMPLATE_URL | default "" | quote }}\n            - name: LLM_ENABLE_THINKING\n              value: {{ .Values.olaresEnv.LLM_ENABLE_THINKING | default "" | quote }}\n            - name: LLM_PRESERVE_THINKING\n              value: {{ .Values.olaresEnv.LLM_PRESERVE_THINKING | default "" | quote }}\n`;
  if (s.includes('name: LLM_REASONING')) return s;
  return s.replace(
    /(            - name: LLM_API_KEY\n              value:[^\n]+\n)/,
    `$1${block}`,
  );
}

for (const dir of ALL) {
  const file = path.join(ROOT, dir, 'templates/server.yaml');
  let s = fs.readFileSync(file, 'utf8');
  s = fixArgs(s);
  const bundled = BUNDLED.find((b) => b.dir === dir);
  if (bundled) s = fixMounts(s, bundled.cm);
  if (dir === 'llamacppqwen36mtpone') s = fixMtpone(s);
  if (dir === 'llamacppkatcoderv25one') s = fixKat(s);
  s = addDeploymentEnvs(s, dir === 'llamacppqwable35bone');
  if (dir === 'llamacppqwen3827bmtpone') {
    s = addDeploymentEnvs(s, false);
    // qwen3827 has QWEN38_VARIANT after LLM_SPEC - insert after that if still missing
    if (!s.includes('name: LLM_REASONING')) {
      s = s.replace(
        /(            - name: QWEN38_VARIANT\n              value:[^\n]+\n)/,
        `$1            - name: LLM_REASONING\n              value: {{ .Values.olaresEnv.LLM_REASONING | default "" | quote }}\n            - name: LLM_CHAT_TEMPLATE_SOURCE\n              value: {{ .Values.olaresEnv.LLM_CHAT_TEMPLATE_SOURCE | default "" | quote }}\n            - name: LLM_CHAT_TEMPLATE_URL\n              value: {{ .Values.olaresEnv.LLM_CHAT_TEMPLATE_URL | default "" | quote }}\n            - name: LLM_ENABLE_THINKING\n              value: {{ .Values.olaresEnv.LLM_ENABLE_THINKING | default "" | quote }}\n            - name: LLM_PRESERVE_THINKING\n              value: {{ .Values.olaresEnv.LLM_PRESERVE_THINKING | default "" | quote }}\n`,
      );
    }
  }
  fs.writeFileSync(file, s);
  console.log('fixed', dir);
}

console.log('done');
