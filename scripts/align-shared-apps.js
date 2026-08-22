#!/usr/bin/env node
/**
 * Align split shared apps with beclab/apps pattern (ollamav2, ollamaqwen*, vllmqwen*):
 * - subCharts: server shared + client (no shared on client)
 * - client Deployment metadata.name == app name (literal, like diffusion reference)
 * - server HF_TOKEN safe when olaresEnv missing
 * - olaresEnv: {} in values.yaml
 * - mandatory self application dep for non-admin installs
 * - root templates/clientproxy.yaml for lint (parent-chart render path)
 */
const fs = require('fs');
const path = require('path');
const yaml = require('js-yaml');

const REPO = path.resolve(__dirname, '..');
const MAX_LEN = 30;
const SERVER_SUFFIX = 'srv';
const HF_SAFE = '{{ .Values.olaresEnv.OLARES_USER_HUGGINGFACE_TOKEN | default "" }}';

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

function ensureOlaresEnvInValues(valuesPath) {
  if (!fs.existsSync(valuesPath)) return;
  let raw = fs.readFileSync(valuesPath, 'utf8');
  if (/^olaresEnv:/m.test(raw)) return;
  fs.writeFileSync(valuesPath, `${raw.replace(/\s*$/, '\n')}olaresEnv: {}\n`);
}

function fixSubChartsBlock(raw, serverChart, clientChart) {
  if (!/^  subCharts:/m.test(raw)) return raw;
  const block = `  subCharts:\n  - name: ${serverChart}\n    shared: true\n  - name: ${clientChart}\n`;
  return raw.replace(
    /^  subCharts:\n(?:  - name: [^\n]+\n(?:    shared: true\n)?)+/m,
    block,
  );
}

function addMandatoryDep(raw, appName) {
  if (new RegExp(`- name: ${appName.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}\\s*\\n\\s+type: application`).test(raw)) {
    return raw;
  }
  const tail = `  {{- if and .Values.admin .Values.bfl.username (eq .Values.admin .Values.bfl.username) }}
  {{- else }}
    - name: ${appName}
      type: application
      version: '>=1.0.0'
      mandatory: true
  {{- end }}
`;
  return raw.replace(
    /(    - name: olares\n      version: '>=1\.12\.3-0'\n      type: system\n)/m,
    `$1${tail}`,
  );
}

function fixClientDeploymentName(content, appName) {
  const lines = content.split('\n');
  let inDeploy = false;
  let inMeta = false;
  for (let i = 0; i < lines.length; i++) {
    if (lines[i].trim() === 'kind: Deployment') {
      inDeploy = true;
      inMeta = false;
      continue;
    }
    if (inDeploy && lines[i].startsWith('metadata:')) {
      inMeta = true;
      continue;
    }
    if (inDeploy && inMeta && /^  name:\s/.test(lines[i])) {
      lines[i] = `  name: ${appName}`;
      break;
    }
  }
  return lines.join('\n');
}

function fixHfToken(content) {
  return content.replace(
    /value:\s*"\{\{\s*\.Values\.olaresEnv\.OLARES_USER_HUGGINGFACE_TOKEN\s*\}\}"/g,
    `value: "${HF_SAFE}"`,
  );
}

let updated = 0;
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
  const serverDir = path.join(appDir, serverChart);
  const clientProxy = path.join(clientDir, 'templates/clientproxy.yaml');
  if (!fs.existsSync(clientProxy)) continue;

  const oldVersion = String(manifest.metadata?.version || '1.0.0').replace(/['"]/g, '');
  const newVersion = bumpPatch(oldVersion);
  let out = fixSubChartsBlock(raw, serverChart, clientChart);
  out = addMandatoryDep(out, appName);
  out = bumpVersionsInRaw(out, oldVersion, newVersion);
  fs.writeFileSync(manifestPath, out);
  updateChartVersion(chartPath, newVersion);

  let proxy = fixClientDeploymentName(fs.readFileSync(clientProxy, 'utf8'), appName);
  fs.writeFileSync(clientProxy, proxy);

  const deployPath = path.join(serverDir, 'templates/deployment.yaml');
  if (fs.existsSync(deployPath)) {
    fs.writeFileSync(deployPath, fixHfToken(fs.readFileSync(deployPath, 'utf8')));
  }

  ensureOlaresEnvInValues(path.join(appDir, 'values.yaml'));
  ensureOlaresEnvInValues(path.join(serverDir, 'values.yaml'));
  ensureOlaresEnvInValues(path.join(clientDir, 'values.yaml'));

  const rootTemplates = path.join(appDir, 'templates');
  fs.mkdirSync(rootTemplates, { recursive: true });
  fs.writeFileSync(path.join(rootTemplates, 'clientproxy.yaml'), proxy);
  const keep = path.join(rootTemplates, 'keep');
  if (fs.existsSync(keep)) fs.unlinkSync(keep);

  console.log(`aligned ${entry.name} v${newVersion}`);
  updated++;
}

console.log(`\nDone: ${updated} apps`);
