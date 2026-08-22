#!/usr/bin/env node
/** Add missing shared-models volumeMount + volume on LLM server pods. */
const fs = require('fs');
const path = require('path');

const ROOT = path.join(__dirname, '..');
const SHARED_MOUNT = `            - mountPath: "/shared-models"
              name: shared-models`;
const SHARED_VOLUME = `        - name: shared-models
          hostPath:
            path: "/olares/share/ai/model"
            type: DirectoryOrCreate`;

function needsVolumeFix(content) {
  if (!content.includes('path: "/olares/share/ai/model"')) {
    return content.includes('fix-shared-llms-perms') || content.includes('fix-shared-llms-perms');
  }
  const afterContainers = content.split('      containers:')[1] || '';
  return !afterContainers.includes('mountPath: "/shared-models"');
}

function fix(content) {
  let out = content;
  let changed = false;

  const afterContainers = out.split('      containers:')[1] || '';
  if (afterContainers && !afterContainers.includes('mountPath: "/shared-models"')) {
    const newTail = afterContainers.replace(
      /(mountPath: "\/models"\n\s+name: models\n)/g,
      `$1${SHARED_MOUNT}\n`
    );
    if (newTail !== afterContainers) {
      out = out.split('      containers:')[0] + '      containers:' + newTail;
      changed = true;
    }
  }

  if (!out.includes('path: "/olares/share/ai/model"')) {
    const withVol = out.replace(
      /(        - name: models\n          hostPath:\n            path: "\{\{ \.Values\.userspace\.appData \}\}\/models"\n            type: DirectoryOrCreate\n)/,
      `$1${SHARED_VOLUME}\n`
    );
    if (withVol !== out) {
      out = withVol;
      changed = true;
    }
  }

  // init downloaders that write to /shared-models need the mount too
  if (out.includes('SHARED_DIR="/shared-models/llms"')) {
    const fixed = out.replace(
      /(        - name: model-downloader[\s\S]*?          volumeMounts:\n            - mountPath: "\/models"\n              name: models\n)(?!            - mountPath: "\/shared-models")/,
      `$1${SHARED_MOUNT}\n`
    );
    if (fixed !== out) {
      out = fixed;
      changed = true;
    }
  }

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

let n = 0;
for (const d of fs.readdirSync(ROOT, { withFileTypes: true }).filter((x) => x.isDirectory())) {
  const p = findDeploymentYaml(d.name);
  if (!p) continue;
  const content = fs.readFileSync(p, 'utf8');
  if (!needsVolumeFix(content)) continue;
  const { out, changed } = fix(content);
  if (changed) {
    fs.writeFileSync(p, out);
    console.log(`  fixed volumes: ${path.relative(ROOT, p)}`);
    n++;
  }
}
console.log(`Done. Fixed ${n} file(s).`);
