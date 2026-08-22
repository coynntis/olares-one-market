#!/usr/bin/env node
/**
 * Point llama.cpp GGUF init-downloaders at /shared-models/llms (host: /olares/share/ai/model/llms).
 */
const fs = require('fs');
const path = require('path');

const ROOT = path.join(__dirname, '..');
const INIT_PERMS = `        - name: fix-shared-llms-perms
          image: "docker.io/alpine:3.20"
          command:
            - sh
            - -c
            - mkdir -p /shared-models/llms && chmod -R 777 /shared-models && echo shared-llms-dir-ready
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

const GGUF_DOWNLOAD_SNIPPET = `              SHARED_DIR="/shared-models/llms"
              MODELS_DIR="/models"
              mkdir -p "$SHARED_DIR" "$MODELS_DIR"`;

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

function patch(content) {
  if (!content.includes('llama-server') || content.includes('path: "/olares/share/ai/model"')) {
    return { out: content, changed: false };
  }

  let out = content;
  let changed = false;

  if (!out.includes('fix-shared-llms-perms')) {
    out = out.replace('      initContainers:\n', `      initContainers:\n${INIT_PERMS}`);
    changed = true;
  }

  if (out.includes('MODEL_PATH="/models/${MODEL_FILE}"')) {
    out = out.replace(
      /MODEL_PATH="\/models\/\$\{MODEL_FILE\}"/g,
      'MODEL_PATH="$SHARED_DIR/${MODEL_FILE}"'
    );
    out = out.replace(
      /MMPROJ_PATH="\/models\/\$\{MMPROJ_FILE\}"/g,
      'MMPROJ_PATH="$SHARED_DIR/${MMPROJ_FILE}"'
    );
    // inject SHARED_DIR setup after first line of init script block if missing
    if (!out.includes('SHARED_DIR="/shared-models/llms"')) {
      out = out.replace(
        /(command:\n            - sh\n            - '-c'\n            - \|\n)(              )/,
        `$1${GGUF_DOWNLOAD_SNIPPET}\n$2`
      );
      // migration before download check
      out = out.replace(
        /(MODEL_PATH="\$SHARED_DIR\/\$\{MODEL_FILE\}")\n(              if \[ -f "\$MODEL_PATH" \])/,
        '$1\n              if [ -f "$MODELS_DIR/${MODEL_FILE}" ] && [ ! -f "$MODEL_PATH" ]; then mv "$MODELS_DIR/${MODEL_FILE}" "$MODEL_PATH"; fi\n$2'
      );
      out = out.replace(
        /(MMPROJ_PATH="\$SHARED_DIR\/\$\{MMPROJ_FILE\}")\n(              if \[ -f "\$MMPROJ_PATH" \])/,
        '$1\n              if [ -f "$MODELS_DIR/${MMPROJ_FILE}" ] && [ ! -f "$MMPROJ_PATH" ]; then mv "$MODELS_DIR/${MMPROJ_FILE}" "$MMPROJ_PATH"; fi\n$2'
      );
    }
    changed = true;
  }

  if (out.includes('- "/models/$(MODEL_FILE)"')) {
    out = out.replace(/- "\/models\/\$\(MODEL_FILE\)"/g, '- "/shared-models/llms/$(MODEL_FILE)"');
    changed = true;
  }
  if (out.includes('- "/models/$(MMPROJ_FILE)"')) {
    out = out.replace(/- "\/models\/\$\(MMPROJ_FILE\)"/g, '- "/shared-models/llms/$(MMPROJ_FILE)"');
    changed = true;
  }

  if (!out.includes('mountPath: "/shared-models"')) {
    out = out.replace(
      /(mountPath: "\/models"\n\s+name: models\n)/g,
      `$1${SHARED_MOUNT}\n`
    );
    changed = true;
  }

  if (!out.includes('path: "/olares/share/ai/model"')) {
    out = out.replace(
      /(        - name: models\n          hostPath:\n            path: "\{\{ \.Values\.userspace\.appData \}\}\/models"\n            type: DirectoryOrCreate\n)/,
      `$1${SHARED_VOLUME}\n`
    );
    changed = true;
  }

  return { out, changed };
}

let n = 0;
for (const d of fs.readdirSync(ROOT, { withFileTypes: true }).filter((x) => x.isDirectory())) {
  const p = findDeploymentYaml(d.name);
  if (!p) continue;
  const content = fs.readFileSync(p, 'utf8');
  const { out, changed } = patch(content);
  if (changed) {
    fs.writeFileSync(p, out);
    console.log(`  patched: ${path.relative(ROOT, p)}`);
    n++;
  }
}
console.log(`Done. Patched ${n} llama.cpp deployment(s).`);
