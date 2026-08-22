#!/usr/bin/env node
/** Repair broken dependencies blocks from bad regex insert. */
const fs = require('fs');
const path = require('path');
const yaml = require('js-yaml');

const REPO = path.resolve(__dirname, '..');

function stripHelm(content) {
  const lines = content.split('\n');
  const result = [];
  let inElse = false;
  for (const line of lines) {
    const t = line.trim();
    if (/^\{\{-?\s*if\b/.test(t)) continue;
    if (/^\{\{-?\s*else\b/.test(t)) { inElse = true; continue; }
    if (/^\{\{-?\s*end\b/.test(t)) { inElse = false; continue; }
    if (inElse) continue;
    result.push(line.replace(/\{\{.*?\}\}/g, ''));
  }
  return result.join('\n');
}

function depBlock(appName, adminExtra = '') {
  return `  dependencies:
    - name: olares
      version: '>=1.12.3-0'
      type: system
  {{- if and .Values.admin .Values.bfl.username (eq .Values.admin .Values.bfl.username) }}${adminExtra}
  {{- else }}
    - name: ${appName}
      type: application
      version: '>=1.0.0'
      mandatory: true
  {{- end }}
`;
}

let fixed = 0;
for (const entry of fs.readdirSync(REPO, { withFileTypes: true }).filter((e) => e.isDirectory())) {
  const manifestPath = path.join(REPO, entry.name, 'OlaresManifest.yaml');
  if (!fs.existsSync(manifestPath)) continue;
  const raw = fs.readFileSync(manifestPath, 'utf8');
  if (!/    - name: olares\n  \{\{- if/m.test(raw)) continue;

  let manifest;
  try {
    manifest = yaml.load(stripHelm(raw));
  } catch {
    continue;
  }
  const appName = manifest.metadata?.name || entry.name;

  // Preserve admin-only extra deps (e.g. litellmgateway on diffusion)
  let adminExtra = '';
  const adminMatch = raw.match(
    /\{\{- if and \.Values\.admin[\s\S]*?\{\{- else \}\}[\s\S]*?\{\{- end \}\}\n      version: '>=1\.12\.3-0'/,
  );
  if (adminMatch) {
    const inner = raw.match(
      /\{\{- if and \.Values\.admin \.Values\.bfl\.username \(eq \.Values\.admin \.Values\.bfl\.username\) \}\}\n([\s\S]*?)\n  \{\{- else \}\}/,
    );
    if (inner && inner[1].trim().startsWith('- name:')) {
      adminExtra = `\n${inner[1].trimEnd()}`;
    }
  }

  const replacement = depBlock(appName, adminExtra);
  const out = raw.replace(
    /  dependencies:\n[\s\S]*$/m,
    replacement.trimEnd(),
  );
  if (out === raw) continue;
  fs.writeFileSync(manifestPath, `${out}\n`);
  console.log(`repaired ${entry.name}`);
  fixed++;
}

console.log(`Done: ${fixed}`);
