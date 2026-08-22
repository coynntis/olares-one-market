#!/usr/bin/env node
/**
 * Wire HF-based LLM servers (vLLM, SGLang) to shared cache:
 *   /olares/share/ai/model/llms/huggingface  (container: /shared-models/llms/huggingface)
 *
 * llama.cpp GGUF apps use /shared-models/llms directly (separate migration).
 */
const fs = require('fs');
const path = require('path');

const ROOT = path.join(__dirname, '..');
const SHARED_HF = '/shared-models/llms/huggingface';

const INIT_BLOCK = `      initContainers:
        - name: fix-shared-llms-perms
          image: "docker.io/alpine:3.20"
          command:
            - sh
            - -c
            - mkdir -p /shared-models/llms/huggingface && chmod -R 777 /shared-models && echo shared-llms-dir-ready
          securityContext:
            runAsUser: 0
          volumeMounts:
            - mountPath: "/shared-models"
              name: shared-models
`;

const SHARED_MOUNT = `            - mountPath: "/shared-models"
              name: shared-models`;

const SHARED_VOLUME = `        - name: shared-models
          hostPath:
            path: "/olares/share/ai/model"
            type: DirectoryOrCreate`;

function isLlmServerDeployment(content) {
  return (
    /vllm\/vllm-openai|vllm serve|sglang\.launch_server|sglang serve/.test(content) &&
    !content.includes('sensevoiceone') &&
    !content.includes('omnivoiceone') &&
    !content.includes('cosyvoice2yueone') &&
    !content.includes('voxcpmone')
  );
}

function patch(content) {
  let out = content;
  let changed = false;

  if (!out.includes('name: shared-models')) {
    // HF cache env vars
    const hfReplacements = [
      ['value: "/models/huggingface"', `value: "${SHARED_HF}"`],
      ['value: "/models"', `value: "${SHARED_HF}"`], // only safe after download-dir fix below
    ];
    for (const [from, to] of hfReplacements) {
      if (out.includes(from) && (from !== 'value: "/models"' || out.includes('HF_HOME'))) {
        const next = out.split(from).join(to);
        if (next !== out) {
          out = next;
          changed = true;
        }
      }
    }

    // vLLM --download-dir
    for (const dir of ['/models/huggingface', '/models']) {
      const needle = `            - "${dir}"`;
      const idx = out.indexOf('            - "--download-dir"');
      if (idx !== -1) {
        const slice = out.slice(idx, idx + 200);
        if (slice.includes(needle)) {
          out = out.replace(needle, `            - "${SHARED_HF}"`);
          changed = true;
        }
      }
    }

    // initContainer
    if (!out.includes('fix-shared-llms-perms')) {
      if (out.includes('      initContainers:')) {
        if (!out.includes('fix-shared-llms-perms')) {
          out = out.replace(
            '      initContainers:\n',
            `${INIT_BLOCK}`
          );
          changed = true;
        }
      } else {
        out = out.replace('    spec:\n      containers:', `    spec:\n${INIT_BLOCK}      containers:`);
        changed = true;
      }
    }

    // volumeMount on main server container — insert after models mount
    if (!out.includes('mountPath: "/shared-models"')) {
      out = out.replace(
        /(mountPath: "\/models"\n\s+name: models\n)/g,
        `$1${SHARED_MOUNT}\n`
      );
      changed = true;
    }

    // shared volume
    if (!out.includes('name: shared-models')) {
      out = out.replace(
        /(        - name: models\n          hostPath:\n            path: "\{\{ \.Values\.userspace\.appData \}\}\/models"\n            type: DirectoryOrCreate\n)/,
        `$1${SHARED_VOLUME}\n`
      );
      changed = true;
    }
  }

  // Fix HF_HOME if we accidentally replaced wrong /models env — restore MODELS_DIR scratch
  out = out.replace(
    /- name: MODELS_DIR\n\s+value: "\/shared-models\/llms\/huggingface"/g,
    '- name: MODELS_DIR\n              value: "/models"'
  );

  return { out, changed };
}

function findDeploymentYaml(appDir) {
  const appPath = path.join(ROOT, appDir);
  if (!fs.existsSync(appPath)) return null;
  for (const name of fs.readdirSync(appPath)) {
    if (!name.endsWith('srv')) continue;
    const p = path.join(appPath, name, 'templates', 'deployment.yaml');
    if (fs.existsSync(p)) return p;
  }
  return null;
}

function main() {
  const dirs = fs.readdirSync(ROOT, { withFileTypes: true }).filter((d) => d.isDirectory());
  let patched = 0;

  for (const d of dirs) {
    const p = findDeploymentYaml(d.name);
    if (!p) continue;
    const content = fs.readFileSync(p, 'utf8');
    if (!isLlmServerDeployment(content)) continue;
    if (content.includes('name: shared-models') && content.includes(SHARED_HF)) {
      console.log(`  skip (already patched): ${path.relative(ROOT, p)}`);
      continue;
    }
    const { out, changed } = patch(content);
    if (changed) {
      fs.writeFileSync(p, out);
      console.log(`  patched: ${path.relative(ROOT, p)}`);
      patched++;
    }
  }

  console.log(`\nDone. Patched ${patched} deployment(s).`);
}

main();
