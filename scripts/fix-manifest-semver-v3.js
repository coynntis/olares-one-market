#!/usr/bin/env node
/** Strip invalid quoted semver in v3 OlaresManifest (metadata.version, spec.versionName). */
const fs = require('fs');
const path = require('path');

const REPO = path.resolve(__dirname, '..');

function normalizeSemverFields(content) {
  function unquoteSemver(match, indent, key, v) {
    if (!/^\d+\.\d+\.\d+(-[0-9A-Za-z.]+)?$/.test(v)) return match;
    return `${indent}${key} ${v}`;
  }

  let out = content;
  out = out.replace(/^(\s+)(version:)\s*'''([^']+)'''$/gm, unquoteSemver);
  out = out.replace(/^(\s+)(versionName:)\s*'''([^']+)'''$/gm, unquoteSemver);
  out = out.replace(/^(\s+)(version:)\s*'([^']+)'$/gm, unquoteSemver);
  out = out.replace(/^(\s+)(versionName:)\s*'([^']+)'$/gm, unquoteSemver);
  out = out.replace(/^(\s+version:)\s*(>=.+)$/gm, "$1 '$2'");
  return out;
}

function findManifests(dir, out = []) {
  for (const e of fs.readdirSync(dir, { withFileTypes: true })) {
    const p = path.join(dir, e.name);
    if (e.isDirectory()) {
      if (e.name === 'node_modules' || e.name === 'charts') continue;
      findManifests(p, out);
    } else if (e.name === 'OlaresManifest.yaml') out.push(p);
  }
  return out;
}

let fixed = 0;
if (require.main === module) {
for (const file of findManifests(REPO)) {
  const raw = fs.readFileSync(file, 'utf8');
  if (!/apiVersion:\s*'?v3'?/.test(raw)) continue;
  const next = normalizeSemverFields(raw);
  if (next !== raw) {
    fs.writeFileSync(file, next);
    console.log(`fixed ${path.relative(REPO, file)}`);
    fixed++;
  }
}
console.log(`\nDone: ${fixed} manifests normalized`);
}

module.exports = { normalizeSemverFields };
