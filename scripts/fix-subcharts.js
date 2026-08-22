#!/usr/bin/env node
/**
 * Fix broken subCharts blocks (duplicate client + shared:true on client).
 * Copy client proxy to root templates/ so OAC root-only render finds app-named Deployment.
 */
const fs = require('fs');
const path = require('path');
const yaml = require('js-yaml');

const REPO = path.resolve(__dirname, '..');
const MAX_LEN = 30;
const SERVER_SUFFIX = 'srv';

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

function fitSuffix(base, suffix) {
  return `${String(base).slice(0, MAX_LEN - suffix.length)}${suffix}`;
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

function fixSubChartsBlock(raw, serverChart, clientChart) {
  const block = `  subCharts:\n  - name: ${serverChart}\n    shared: true\n  - name: ${clientChart}\n`;
  if (!/^  subCharts:/m.test(raw)) return raw;
  return raw.replace(
    /^  subCharts:\n(?:  - name: [^\n]+\n(?:    shared: true\n)?)+/m,
    block,
  );
}

function fixClientProxyDeployment(content) {
  const lines = content.split('\n');
  let afterDeploy = false;
  let afterMeta = false;
  for (let i = 0; i < lines.length; i++) {
    if (lines[i].trim() === 'kind: Deployment') {
      afterDeploy = true;
      afterMeta = false;
      continue;
    }
    if (afterDeploy && lines[i].startsWith('metadata:')) {
      afterMeta = true;
      continue;
    }
    if (afterDeploy && afterMeta && /^  name:\s/.test(lines[i])) {
      if (!lines[i].includes('{{ .Release.Name }}')) {
        lines[i] = '  name: {{ .Release.Name }}';
      }
      break;
    }
  }
  return lines.join('\n');
}

let fixed = 0;
for (const entry of fs.readdirSync(REPO, { withFileTypes: true }).filter((e) => e.isDirectory())) {
  const appDir = path.join(REPO, entry.name);
  const manifestPath = path.join(appDir, 'OlaresManifest.yaml');
  const chartPath = path.join(appDir, 'Chart.yaml');
  if (!fs.existsSync(manifestPath) || !fs.existsSync(chartPath)) continue;

  const raw = fs.readFileSync(manifestPath, 'utf8');
  let manifest;
  try {
    manifest = yaml.load(stripHelm(raw));
  } catch {
    continue;
  }
  if (!manifest?.spec?.subCharts?.length) continue;

  const appName = manifest.metadata?.name || entry.name;
  const serverChart = fitSuffix(appName, SERVER_SUFFIX);
  const clientChart = appName.length <= MAX_LEN ? appName : appName.slice(0, MAX_LEN);
  const clientDir = path.join(appDir, clientChart);
  const clientProxy = path.join(clientDir, 'templates/clientproxy.yaml');

  const newRaw = fixSubChartsBlock(raw, serverChart, clientChart);
  if (newRaw === raw && !fs.existsSync(clientProxy)) continue;

  const oldVersion = String(manifest.metadata?.version || '1.0.0').replace(/['"]/g, '');
  const newVersion = bumpPatch(oldVersion);
  let out = bumpVersionsInRaw(newRaw !== raw ? newRaw : raw, oldVersion, newVersion);

  fs.writeFileSync(manifestPath, out);
  updateChartVersion(chartPath, newVersion);

  if (fs.existsSync(clientProxy)) {
    let proxy = fs.readFileSync(clientProxy, 'utf8');
    const updatedProxy = fixClientProxyDeployment(proxy);
    if (updatedProxy !== proxy) {
      fs.writeFileSync(clientProxy, updatedProxy);
      proxy = updatedProxy;
    }
    const rootTemplates = path.join(appDir, 'templates');
    fs.mkdirSync(rootTemplates, { recursive: true });
    fs.writeFileSync(path.join(rootTemplates, 'clientproxy.yaml'), proxy);
    const keep = path.join(rootTemplates, 'keep');
    if (fs.existsSync(keep)) fs.unlinkSync(keep);
  }

  console.log(`fixed ${entry.name}: srv=${serverChart} client=${clientChart} v${newVersion}`);
  fixed++;
}

console.log(`\nDone: ${fixed} apps`);
