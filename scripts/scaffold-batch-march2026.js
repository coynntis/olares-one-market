#!/usr/bin/env node
/**
 * Scaffold new Olares One apps + patch froggeric Qwen chat template into existing charts.
 */
const fs = require('fs');
const path = require('path');
const { execSync } = require('child_process');

const REPO = path.resolve(__dirname, '..');
const TEMPLATE_SRC = path.join(REPO, 'shared/chat_templates/qwen-fixed-chat-template.jinja');
const TEMPLATE_HF_URL =
  'https://huggingface.co/froggeric/Qwen-Fixed-Chat-Templates/resolve/main/chat_template.jinja';

function walk(dir, fn) {
  for (const ent of fs.readdirSync(dir, { withFileTypes: true })) {
    const p = path.join(dir, ent.name);
    if (ent.isDirectory()) walk(p, fn);
    else fn(p);
  }
}

function replaceAll(dir, pairs) {
  walk(dir, (file) => {
    if (!/\.(yaml|yml|jinja|md|py|txt|json)$/.test(file)) return;
    let s = fs.readFileSync(file, 'utf8');
    let changed = false;
    for (const [from, to] of pairs) {
      if (s.includes(from)) {
        s = s.split(from).join(to);
        changed = true;
      }
    }
    if (changed) fs.writeFileSync(file, s);
  });
}

function copyApp(srcName, dstName, pairs) {
  const src = path.join(REPO, srcName);
  const dst = path.join(REPO, dstName);
  if (fs.existsSync(dst)) {
    console.log(`skip copy (exists): ${dstName}`);
  } else {
    execSync(`cp -R "${src}" "${dst}"`);
    console.log(`copied ${srcName} -> ${dstName}`);
  }
  replaceAll(dst, [[srcName, dstName]]);
  for (const pair of pairs) {
    if (pair[0] instanceof RegExp) continue;
    replaceAll(dst, [[pair[0], pair[1]]]);
  }
  walk(dst, (file) => {
    if (!file.endsWith('.yaml')) return;
    let s = fs.readFileSync(file, 'utf8');
    let changed = false;
    for (const pair of pairs) {
      if (pair[0] instanceof RegExp) {
        const next = s.replace(pair[0], pair[1]);
        if (next !== s) {
          s = next;
          changed = true;
        }
      }
    }
    if (changed) fs.writeFileSync(file, s);
  });
}

function bumpManifestVersion(manifestPath, version = '1.0.0') {
  if (!fs.existsSync(manifestPath)) return;
  let raw = fs.readFileSync(manifestPath, 'utf8');
  raw = raw.replace(/^  version:.*$/m, `  version: '${version}'`);
  raw = raw.replace(/^  versionName:.*$/m, `  versionName: '${version}'`);
  fs.writeFileSync(manifestPath, raw);
  const chartPath = path.join(path.dirname(manifestPath), 'Chart.yaml');
  if (fs.existsSync(chartPath)) {
    let c = fs.readFileSync(chartPath, 'utf8').replace(/^apiVersion: v2\n(?=apiVersion: v2\n)/, '');
    c = c.replace(/^version:.*$/m, `version: ${version}`);
    c = c.replace(/^appVersion:.*$/m, `appVersion: ${version}`);
    fs.writeFileSync(chartPath, c);
  }
}

function patchQwenChatTemplate(appName, srvDeployRel) {
  const deployPath = path.join(REPO, appName, srvDeployRel, 'templates/deployment.yaml');
  if (!fs.existsSync(deployPath)) return;
  let d = fs.readFileSync(deployPath, 'utf8');
  if (d.includes('froggeric/Qwen-Fixed-Chat-Templates')) return;

  if (!d.includes('CHAT_TEMPLATE_URL')) {
    d = d.replace(
      /(data:\n(?:  [^\n]+\n)*?)(---\napiVersion: apps)/,
      `$1  CHAT_TEMPLATE_URL: "${TEMPLATE_HF_URL}"\n  CHAT_TEMPLATE_FILE: "chat_template.jinja"\n$2`,
    );
  }

  const dlBlock = `              CHAT_TEMPLATE="\${MODELS_DIR:-/models}/\${CHAT_TEMPLATE_FILE:-chat_template.jinja}"
              if [ ! -f "$CHAT_TEMPLATE" ]; then
                echo "==> Downloading froggeric Qwen fixed chat template"
                eval curl -fL --retry 10 --retry-delay 5 $HF_AUTH \\
                  "\${CHAT_TEMPLATE_URL:-${TEMPLATE_HF_URL}}" -o "$CHAT_TEMPLATE"
              fi
`;

  if (!d.includes('froggeric Qwen fixed chat template')) {
    d = d.replace(/(\n              exec .*llama-server)/, `\n${dlBlock}$1`);
  }

  if (d.includes('--jinja') && !d.includes('--chat-template-file')) {
    d = d.replace(
      /--jinja( --no-mmap --mlock)?/,
      '--jinja --chat-template-file "$CHAT_TEMPLATE"$1',
    );
    d = d.replace(/--jinja \\\n/, '--jinja \\\n                --chat-template-file "$CHAT_TEMPLATE" \\\n');
  }

  // fix duplicate EXTRA_LLM_ARGS
  d = d.replace(
    /"\$\{EXTRA_LLM_ARGS\[@\]\}" \\\n\s+"\$\{EXTRA_LLM_ARGS\[@\]\}"/g,
    '"${EXTRA_LLM_ARGS[@]}"',
  );

  // ensure envFrom has CHAT_TEMPLATE from configmap - add envFrom keys via configmap already

  fs.writeFileSync(deployPath, d);

  // copy local jinja for apps that keep one in repo root
  const localJinja = path.join(REPO, appName, 'chat_template.jinja');
  if (fs.existsSync(localJinja) && fs.existsSync(TEMPLATE_SRC)) {
    fs.copyFileSync(TEMPLATE_SRC, localJinja);
  }
  console.log(`patched Qwen template: ${appName}`);
}

function patchQwenArgsStyle(deployPath) {
  if (!fs.existsSync(deployPath)) return;
  let d = fs.readFileSync(deployPath, 'utf8');
  if (d.includes('froggeric') && d.includes('--chat-template-file')) return;
  const jinjaPath = '/models/chat_template.jinja';
  if (!d.includes('--chat-template-file')) {
    d = d.replace(
      /(\n            - "--jinja"\n)/,
      `$1            - "--chat-template-file"\n            - "${jinjaPath}"\n`,
    );
  }
  fs.writeFileSync(deployPath, d);
}

// --- scaffold apps ---
const apps = [
  {
    src: 'llamacppqwen36mtpone',
    dst: 'llamacppqwythos9bone',
    srv: 'llamacppqwythos9srv',
    extra: [
      ['llamacppqwen36mtponesrv', 'llamacppqwythos9srv'],
      ['llamacpp-mtp-env', 'llamacpp-qwythos-env'],
      ['Qwen36 27B MTP', 'Qwythos 9B'],
      ['qwen3.6-27b-mtp', 'qwythos-9b'],
      ['unsloth/Qwen3.6-27B-MTP-GGUF', 'empero-ai/Qwythos-9B-Claude-Mythos-5-1M-GGUF'],
      ['Qwen3.6-27B-UD-Q3_K_XL.gguf', 'Qwythos-9B-Claude-Mythos-5-1M-MTP-Q4_K_M.gguf'],
      ['CTX_SIZE: "262144"', 'CTX_SIZE: "262144"'],
      ['SPEC_DRAFT_N_MAX: "3"', 'SPEC_DRAFT_N_MAX: "6"'],
    ],
    manifest: {
      title: 'Qwythos 9B One',
      description: 'Qwythos 9B Mythos reasoning — 1M ctx YaRN, buun MTP, froggeric chat template',
      version: '1.0.0',
    },
  },
  {
    src: 'llamacppqwen36mtpone',
    dst: 'llamacppqwopus27coder1',
    srv: 'llamacppqwopus27cdsrv',
    extra: [
      ['llamacppqwen36mtponesrv', 'llamacppqwopus27cdsrv'],
      ['llamacpp-mtp-env', 'llamacpp-qwopus-env'],
      ['Qwen36 27B MTP', 'Qwopus 27B Coder'],
      ['qwen3.6-27b-mtp', 'qwopus-27b-coder'],
      ['unsloth/Qwen3.6-27B-MTP-GGUF', 'Jackrong/Qwopus3.6-27B-Coder-GGUF'],
      ['Qwen3.6-27B-UD-Q3_K_XL.gguf', 'Qwopus3.6-27B-Coder-Q5_K_M.gguf'],
      ['--spec-type draft-mtp \\\n                --spec-draft-n-max "${SPEC_DRAFT_N_MAX:-3}" \\\n                --reasoning off', '--reasoning off'],
      ['SPEC_DRAFT_N_MAX: "3"', 'SPEC_DRAFT_N_MAX: "3"'],
      ['CTX_SIZE: "262144"', 'CTX_SIZE: "131072"'],
    ],
    manifest: {
      title: 'Qwopus 27B Coder One',
      description: 'Qwopus3.6-27B-Coder Q5_K_M — agentic coding, buun llama.cpp, froggeric template',
      version: '1.0.0',
    },
  },
  {
    src: 'llamacppqwen36a3bone',
    dst: 'llamacppqwable35bone',
    srv: 'llamacppqwable35bsrv',
    extra: [
      ['llamacppqwen36a3bonesrv', 'llamacppqwable35bsrv'],
      ['llamacpp-qwen36a3b-env', 'llamacpp-qwable-env'],
      ['Qwen36 35B MTP Vision', 'Qwable 35B MoE'],
      ['qwen3.6-35b-a3b-mtp-vision', 'qwable-v1'],
      ['unsloth/Qwen3.6-35B-A3B-MTP-GGUF', 'lordx64/Qwable-v1-GGUF'],
      ['Qwen3.6-35B-A3B-UD-Q3_K_XL.gguf', 'Qwable-v1.IQ4_XS.gguf'],
      ['--mmproj "$SHARED_DIR/$MMPROJ_FILE" \\\n                --mmproj-gpu-swap \\\n', ''],
      ['MMPROJ_FILE: "mmproj-BF16.gguf"\n', ''],
      [/Migrate \$MMPROJ_FILE[\s\S]*?touch "\$SHARED_DIR\/\$MMPROJ_FILE\.ok"\n              fi\n/g, ''],
    ],
    manifest: {
      title: 'Qwable 35B One',
      description: 'Qwable-v1 IQ4_XS MoE — Opus+Fable distill, buun MTP, froggeric template',
      version: '1.0.0',
    },
  },
  {
    src: 'gemma4e2bone',
    dst: 'llamacppgemma412agent1',
    srv: 'llamacppgemma412agsrv',
    extra: [
      ['gemma4e2bonesrv', 'llamacppgemma412agsrv'],
      ['gemma4e2bone', 'llamacppgemma412agent1'],
      ['llamacpp-gemma4e2b-env', 'llamacpp-gemma412ag-env'],
      ['gemma-4-e2b', 'gemma-4-12b-agentic-v2'],
      ['gemma-4-E2B-it-Q8_0.gguf', 'gemma-4-12B-agentic-fable5-composer2.5-v2-3.5x-tau2-Q4_K_M.gguf'],
      [
        'unsloth/gemma-4-E2B-it-GGUF/resolve/main/gemma-4-E2B-it-Q8_0.gguf',
        'yuxinlu1/gemma-4-12B-agentic-fable5-composer2.5-v2-3.5x-tau2-GGUF/resolve/main/gemma-4-12B-agentic-fable5-composer2.5-v2-3.5x-tau2-Q4_K_M.gguf',
      ],
      ['CONTEXT_SIZE: "8192"', 'CONTEXT_SIZE: "131072"'],
      ['Gemma 4 E2B', 'Gemma 4 12B Agentic v2'],
    ],
    manifest: {
      title: 'Gemma 4 12B Agentic One',
      description: 'Gemma4-12B agentic Fable5 v2 Q4_K_M — tool calling, llama.cpp b8740',
      version: '1.0.0',
    },
  },
];

for (const app of apps) {
  copyApp(app.src, app.dst, app.extra);
  const manifestPath = path.join(REPO, app.dst, 'OlaresManifest.yaml');
  if (fs.existsSync(manifestPath) && app.manifest) {
    let m = fs.readFileSync(manifestPath, 'utf8');
    m = m.replace(/title:.*$/m, `title: ${app.manifest.title}`);
    m = m.replace(/^  description:.*$/m, `  description: ${app.manifest.description}`);
    m = m.replace(/fullDescription: \|[\s\S]*?(?=  developer:)/m, `fullDescription: |\n    ${app.manifest.description}\n\n    Olares One optimized. OpenAI API on port 8000 (LLM) or 8080 (Gradio proxy).\n`);
    fs.writeFileSync(manifestPath, m);
    bumpManifestVersion(manifestPath, app.manifest.version);
  }
  if (app.dst.startsWith('llamacpp') && app.dst.includes('gemma')) {
    // gemma - no froggeric
  } else if (app.dst.startsWith('llamacpp')) {
    patchQwenChatTemplate(app.dst, app.srv);
  }
}

// Qwable: strip mmproj download block manually if regex failed
const qwableDeploy = path.join(REPO, 'llamacppqwable35bone/llamacppqwable35bsrv/templates/deployment.yaml');
if (fs.existsSync(qwableDeploy)) {
  let d = fs.readFileSync(qwableDeploy, 'utf8');
  d = d.replace(/# Migrate \$MMPROJ_FILE[\s\S]*?touch "\$SHARED_DIR\/\$MMPROJ_FILE\.ok"\n              fi\n/g, '');
  d = d.replace(/if \[ ! -f "\$SHARED_DIR\/\$MMPROJ_FILE\.ok" \]; then[\s\S]*?touch "\$SHARED_DIR\/\$MMPROJ_FILE\.ok"\n              fi\n/g, '');
  fs.writeFileSync(qwableDeploy, d);
  patchQwenChatTemplate('llamacppqwable35bone', 'llamacppqwable35bsrv');
}

// Patch existing Qwen charts
const existingQwen = [
  ['llamacppqwen36a3bone', 'llamacppqwen36a3bonesrv'],
  ['llamacppqwen36mtpone', 'llamacppqwen36mtponesrv'],
  ['llamacppqwen36beellamaone', 'llamacppqwen36beellamaonesrv'],
  ['llamacppqwen36beellamavision1', 'llamacppqwen36beellamavisiosrv'],
  ['llamacppqwen3627btq34sone', 'llamacppqwen3627btq34sonesrv'],
  ['llamacppqwen3635ba3btq34sone', 'llamacppqwen3635ba3btq34sonsrv'],
];
for (const [app, srv] of existingQwen) {
  patchQwenChatTemplate(app, srv);
}

patchQwenArgsStyle(path.join(REPO, 'qwen36a3bvisionone/qwen36a3bvisiononesrv/templates/deployment.yaml'));
if (fs.existsSync(path.join(REPO, 'qwen36a3bvisionone/chat_template.jinja'))) {
  fs.copyFileSync(TEMPLATE_SRC, path.join(REPO, 'qwen36a3bvisionone/chat_template.jinja'));
}

console.log('scaffold-batch-march2026 done');
