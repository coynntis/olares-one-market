#!/usr/bin/env node
/** v3 lint: initContainers with runAsUser 0 must use beclab/aboveos-busybox, not alpine. */
const fs = require('fs');
const path = require('path');
const yaml = require('js-yaml');
const { normalizeSemverFields } = require('./fix-manifest-semver-v3.js');

const REPO = path.resolve(__dirname, '..');
const FROM = /docker\.io\/alpine:3\.20/g;
const TO = 'docker.io/beclab/aboveos-busybox:1.37.0';

function bumpPatch(version) {
  const parts = String(version).replace(/['"]/g, '').split('.');
  const last = parseInt(parts[parts.length - 1], 10);
  parts[parts.length - 1] = String(Number.isNaN(last) ? 1 : last + 1);
  return parts.join('.');
}

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
    result.push(line);
  }
  return result.join('\n');
}

function updateChartVersion(chartPath, newVersion) {
  const raw = fs.readFileSync(chartPath, 'utf8').replace(/^apiVersion: v2\n(?=apiVersion: v2\n)/, '');
  const chart = yaml.load(raw);
  chart.version = newVersion;
  if (chart.appVersion !== undefined) chart.appVersion = newVersion;
  fs.writeFileSync(chartPath, yaml.dump(chart, { lineWidth: -1, noRefs: true }));
}

function bumpManifest(appDir, newVersion) {
  const manifestPath = path.join(appDir, 'OlaresManifest.yaml');
  let raw = fs.readFileSync(manifestPath, 'utf8');
  raw = raw.replace(/^(\s+version:)\s*.+$/m, `$1 ${newVersion}`);
  raw = raw.replace(/^(\s+versionName:)\s*.+$/m, `$1 ${newVersion}`);
  if (!raw.includes('aboveos-busybox')) {
    const note = `    v${newVersion}: initContainers use beclab/aboveos-busybox (v3 rejects alpine+runAsUser 0).\n`;
    raw = raw.replace(/(  upgradeDescription: \|\n)/, `$1${note}`);
  }
  fs.writeFileSync(manifestPath, normalizeSemverFields(raw));
}

const dryRun = process.argv.includes('--dry-run');
const only = process.argv.find((a) => a.startsWith('--app='))?.split('=')[1];

let files = 0;
let apps = new Set();

for (const entry of fs.readdirSync(REPO, { withFileTypes: true })) {
  if (!entry.isDirectory()) continue;
  const appName = entry.name;
  if (only && appName !== only) continue;
  const tpl = path.join(REPO, appName, 'templates');
  if (!fs.existsSync(tpl)) continue;
  let appChanged = false;
  for (const f of fs.readdirSync(tpl)) {
    if (!f.endsWith('.yaml') && !f.endsWith('.yml')) continue;
    const p = path.join(tpl, f);
    const raw = fs.readFileSync(p, 'utf8');
    if (!FROM.test(raw)) { FROM.lastIndex = 0; continue; }
    FROM.lastIndex = 0;
    const next = raw.replace(FROM, TO);
    if (dryRun) {
      console.log(`would fix ${path.relative(REPO, p)}`);
    } else {
      fs.writeFileSync(p, next);
      console.log(`fixed ${path.relative(REPO, p)}`);
    }
    files++;
    appChanged = true;
  }
  if (appChanged) apps.add(appName);
}

if (!dryRun) {
  for (const appName of [...apps].sort()) {
    const appDir = path.join(REPO, appName);
    const chartPath = path.join(appDir, 'Chart.yaml');
    if (!fs.existsSync(chartPath)) continue;
    const raw = fs.readFileSync(chartPath, 'utf8').replace(/^apiVersion: v2\n(?=apiVersion: v2\n)/, '');
    const oldV = String(yaml.load(raw).version);
    const newV = bumpPatch(oldV);
    updateChartVersion(chartPath, newV);
    bumpManifest(appDir, newV);
    console.log(`bumped ${appName} → v${newV}`);
  }
}

console.log(`\nDone: ${files} template files, ${apps.size} apps`);
