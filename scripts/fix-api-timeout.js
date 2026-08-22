#!/usr/bin/env node
/** Add options.apiTimeout: 0 for long-lived LLM/SSE streams (Envoy default is 15s). */
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

function bumpPatch(version) {
  const parts = String(version).replace(/['"]/g, '').split('.');
  const last = parseInt(parts[parts.length - 1], 10);
  parts[parts.length - 1] = String(Number.isNaN(last) ? 1 : last + 1);
  return parts.join('.');
}

function bumpVersionsInRaw(raw, oldVersion, newVersion) {
  const old = String(oldVersion).replace(/['"]/g, '');
  let out = raw;
  out = out.replace(
    new RegExp(`(^  version:\\s*)(['"]?)${old.replace(/\./g, '\\.')}\\2`, 'm'),
    `$1'${newVersion}'`,
  );
  out = out.replace(
    new RegExp(`(^  versionName:\\s*)(['"]?)${old.replace(/\./g, '\\.')}\\2`, 'm'),
    `$1'${newVersion}'`,
  );
  return out;
}

function updateChartVersion(chartPath, newVersion) {
  const raw = fs.readFileSync(chartPath, 'utf8').replace(/^apiVersion: v2\n(?=apiVersion: v2\n)/, '');
  const chart = yaml.load(raw);
  chart.version = newVersion;
  if (chart.appVersion !== undefined) chart.appVersion = newVersion;
  fs.writeFileSync(chartPath, yaml.dump(chart, { lineWidth: -1, noRefs: true }));
}

const SSE_NGINX_BLOCK = `      proxy_buffering off;
      proxy_cache off;
      chunked_transfer_encoding on;
`;

function patchClientProxy(content) {
  let out = content;
  if (!out.includes('proxy_buffering off')) {
    out = out.replace(
      /(proxy_read_timeout\s+\d+s;\n)/,
      `$1${SSE_NGINX_BLOCK}`,
    );
  }
  return out;
}

let updated = 0;
for (const entry of fs.readdirSync(REPO, { withFileTypes: true }).filter((e) => e.isDirectory())) {
  const appDir = path.join(REPO, entry.name);
  const manifestPath = path.join(appDir, 'OlaresManifest.yaml');
  const chartPath = path.join(appDir, 'Chart.yaml');
  if (!fs.existsSync(manifestPath) || !fs.existsSync(chartPath)) continue;

  let raw = fs.readFileSync(manifestPath, 'utf8');
  if (/^\s+apiTimeout:/m.test(raw)) {
    // still patch client proxies if needed
  } else if (/^options:/m.test(raw)) {
    raw = raw.replace(/^options:\n/m, 'options:\n  apiTimeout: 0\n');
  } else {
    continue;
  }

  let manifest;
  try {
    manifest = yaml.load(stripHelm(raw));
  } catch {
    continue;
  }

  const hadTimeout = /^\s+apiTimeout:/m.test(fs.readFileSync(manifestPath, 'utf8'));
  const oldVersion = String(manifest.metadata?.version || '1.0.0').replace(/['"]/g, '');
  const newVersion = hadTimeout ? oldVersion : bumpPatch(oldVersion);
  let out = hadTimeout ? fs.readFileSync(manifestPath, 'utf8') : bumpVersionsInRaw(raw, oldVersion, newVersion);

  if (!hadTimeout) {
    fs.writeFileSync(manifestPath, out);
    updateChartVersion(chartPath, newVersion);
  }

  for (const rel of [
    'templates/clientproxy.yaml',
    `${entry.name}/templates/clientproxy.yaml`,
  ]) {
    const p = path.join(appDir, rel);
    if (!fs.existsSync(p)) continue;
    const patched = patchClientProxy(fs.readFileSync(p, 'utf8'));
    if (patched !== fs.readFileSync(p, 'utf8')) {
      fs.writeFileSync(p, patched);
    }
  }

  if (!hadTimeout) {
    console.log(`apiTimeout:0 ${entry.name} v${newVersion}`);
    updated++;
  }
}

console.log(`\nDone: ${updated} manifests updated`);
