#!/usr/bin/env node
'use strict';

const fs = require('fs');
const path = require('path');

const REPO = path.resolve(__dirname, '..');

function bumpPatch(v) {
  const m = String(v).trim().replace(/^['"]|['"]$/g, '').match(/^(\d+)\.(\d+)\.(\d+)$/);
  if (!m) return null;
  return `${m[1]}.${m[2]}.${parseInt(m[3], 10) + 1}`;
}

function bumpChartYaml(file) {
  let text = fs.readFileSync(file, 'utf8');
  const orig = text;
  const vm = text.match(/^version:\s*(\S+)/m);
  if (!vm) return null;
  const next = bumpPatch(vm[1]);
  if (!next) return null;
  text = text.replace(/^version:\s*\S+/m, `version: ${next}`);
  if (/^appVersion:\s*\S+/m.test(text)) {
    text = text.replace(/^appVersion:\s*\S+/m, `appVersion: ${next}`);
  }
  if (text !== orig) {
    fs.writeFileSync(file, text);
    return next;
  }
  return null;
}

function bumpManifest(file) {
  let text = fs.readFileSync(file, 'utf8');
  const orig = text;
  const vm = text.match(/^  version:\s*['"]?([^'"\n]+)['"]?/m);
  if (!vm) return null;
  const next = bumpPatch(vm[1]);
  if (!next) return null;
  text = text.replace(/^  version:\s*['"]?[^'"\n]+['"]?/m, `  version: '${next}'`);
  if (/^  versionName:/m.test(text)) {
    text = text.replace(/^  versionName:\s*['"]?[^'"\n]+['"]?/m, `  versionName: '${next}'`);
  }
  if (text !== orig) {
    fs.writeFileSync(file, text);
    return next;
  }
  return null;
}

const bumped = [];

for (const ent of fs.readdirSync(REPO, { withFileTypes: true })) {
  if (!ent.isDirectory() || ent.name.startsWith('.')) continue;
  const appDir = path.join(REPO, ent.name);
  const chartPath = path.join(appDir, 'Chart.yaml');
  const manifestPath = path.join(appDir, 'OlaresManifest.yaml');
  if (!fs.existsSync(chartPath) || !fs.existsSync(manifestPath)) continue;
  const next = bumpChartYaml(chartPath);
  if (!next) continue;
  bumpManifest(manifestPath);
  bumped.push({ app: ent.name, version: next });
}

bumped.sort((a, b) => a.app.localeCompare(b.app));
for (const { app, version } of bumped) {
  console.log(`${app} -> v${version}`);
}
console.log(`\nBumped ${bumped.length} apps`);
