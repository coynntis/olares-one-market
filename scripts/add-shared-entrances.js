#!/usr/bin/env node
/**
 * Add sharedEntrances + provider to all app OlaresManifest.yaml files that expose HTTP entrances.
 * Bumps metadata.version, spec.versionName, and Chart.yaml version.
 */
const fs = require('fs');
const path = require('path');
const yaml = require('js-yaml');

const REPO = path.resolve(__dirname, '..');

// Strip Helm template directives from YAML (keeps if-branch, drops else-branch).
function stripHelmTemplates(content) {
  const lines = content.split('\n');
  const result = [];
  let inElse = false;
  for (const line of lines) {
    const trimmed = line.trim();
    if (/^\{\{-?\s*if\b/.test(trimmed)) continue;
    if (/^\{\{-?\s*else\b/.test(trimmed)) { inElse = true; continue; }
    if (/^\{\{-?\s*end\b/.test(trimmed)) { inElse = false; continue; }
    if (inElse) continue;
    result.push(line.replace(/\{\{.*?\}\}/g, ''));
  }
  return result.join('\n');
}

function hasHelmTemplates(content) {
  return /\{\{/.test(content);
}

function bumpPatch(version) {
  const parts = String(version).replace(/['"]/g, '').split('.');
  const last = parseInt(parts[parts.length - 1], 10);
  parts[parts.length - 1] = String(Number.isNaN(last) ? 1 : last + 1);
  return parts.join('.');
}

const MAX_NAME_LEN = 30;

function apiEntranceName(appName, sharedEntrances) {
  if (sharedEntrances?.length) return sharedEntrances[0].name;
  const candidates = [
    `${appName}api`,
    `${appName}mcp`,
    appName.endsWith('one') ? `${appName.slice(0, -3)}api` : null,
    appName.startsWith('llamacpp') ? `${appName.slice(8)}api` : null,
    appName.startsWith('llamacpp') ? `${appName.slice(8)}mcp` : null,
    appName.replace('labs', '') + 'api',
  ].filter(Boolean);
  for (const candidate of candidates) {
    if (candidate.length > 0 && candidate.length <= MAX_NAME_LEN) return candidate;
  }
  return `${appName.slice(0, MAX_NAME_LEN - 3)}api`.slice(0, MAX_NAME_LEN);
}

function apiTitle(entranceTitle) {
  let title = entranceTitle.trim();
  if (/\bAPI$/i.test(title) || /\bMCP$/i.test(title)) {
    if (!/\bAPI$/i.test(title)) title = `${title} API`;
  } else if (/ One$/i.test(title)) {
    title = title.replace(/ One$/i, ' API');
  } else {
    title = `${title} API`;
  }
  title = title.replace(/\s+API\s+API$/i, ' API');
  if (title.length > 30) title = title.slice(0, 30).trim();
  return title;
}

function mapEntrance(e) {
  return {
    name: e.name || '',
    host: e.host || '',
    port: e.port ?? 0,
    title: e.title || '',
    icon: e.icon || '',
    authLevel: e.authLevel || 'internal',
    invisible: e.invisible ?? true,
    ...(e.openMethod ? { openMethod: e.openMethod } : {}),
    ...(e.disablePreload ? { disablePreload: e.disablePreload } : {}),
  };
}

function orderedManifest(doc) {
  const order = [
    'olaresManifest.version',
    'olaresManifest.type',
    'apiVersion',
    'metadata',
    'sharedEntrances',
    'entrances',
    'provider',
    'spec',
    'envs',
    'permission',
    'options',
    'middleware',
    'ports',
    'tailscale',
  ];
  const out = {};
  for (const key of order) {
    if (doc[key] !== undefined) out[key] = doc[key];
  }
  for (const key of Object.keys(doc)) {
    if (!(key in out)) out[key] = doc[key];
  }
  return out;
}

function dumpManifest(doc) {
  return `---\n${yaml.dump(orderedManifest(doc), {
    lineWidth: -1,
    quotingType: "'",
    forceQuotes: false,
    noRefs: true,
  })}`;
}

function updateChartVersion(chartPath, newVersion) {
  const raw = fs.readFileSync(chartPath, 'utf8').replace(/^apiVersion: v2\n(?=apiVersion: v2\n)/, '');
  const chart = yaml.load(raw);
  chart.version = newVersion;
  if (chart.appVersion !== undefined) chart.appVersion = newVersion;
  fs.writeFileSync(chartPath, yaml.dump(chart, { lineWidth: -1, noRefs: true }));
}

function yamlSection(key, value) {
  return yaml.dump({ [key]: value }, { lineWidth: -1, quotingType: "'", forceQuotes: false, noRefs: true });
}

function insertBeforeAnchor(raw, anchor, block) {
  const idx = raw.search(anchor);
  if (idx < 0) return raw;
  return raw.slice(0, idx) + block + raw.slice(idx);
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

function patchHelmManifest(raw, { sharedEntrances, provider, oldVersion, newVersion }) {
  let out = raw;
  if (sharedEntrances && !/^sharedEntrances:/m.test(out)) {
    out = insertBeforeAnchor(out, /^entrances:/m, yamlSection('sharedEntrances', sharedEntrances));
  }
  if (provider && !/^provider:/m.test(out)) {
    out = insertBeforeAnchor(out, /^spec:/m, yamlSection('provider', provider));
  }
  return bumpVersionsInRaw(out, oldVersion, newVersion);
}

const entries = fs.readdirSync(REPO, { withFileTypes: true }).filter((e) => e.isDirectory());
let updated = 0;
let skipped = 0;

for (const entry of entries) {
  const appDir = path.join(REPO, entry.name);
  const manifestPath = path.join(appDir, 'OlaresManifest.yaml');
  const chartPath = path.join(appDir, 'Chart.yaml');
  if (!fs.existsSync(manifestPath) || !fs.existsSync(chartPath)) continue;

  const rawManifest = fs.readFileSync(manifestPath, 'utf8');
  let manifest;
  try {
    manifest = yaml.load(stripHelmTemplates(rawManifest));
  } catch (e) {
    console.log(`skip ${entry.name}: parse error — ${e.message}`);
    skipped++;
    continue;
  }

  const entrances = manifest.entrances || [];
  if (entrances.length === 0) {
    console.log(`skip ${entry.name}: no entrances`);
    skipped++;
    continue;
  }

  const needsShared = !manifest.sharedEntrances?.length;
  const needsProvider = !manifest.provider?.length;
  if (!needsShared && !needsProvider) {
    console.log(`skip ${entry.name}: already configured`);
    skipped++;
    continue;
  }

  const primary = entrances[0];
  const appName = manifest.metadata?.name || entry.name;
  const icon = primary.icon || manifest.metadata?.icon || '';
  const apiName = apiEntranceName(appName, manifest.sharedEntrances);
  const title = apiTitle(primary.title || manifest.metadata?.title || appName);

  const serviceHost = primary.host || appName;
  const sharedEntrances = needsShared ? [{
    name: apiName,
    host: `sharedentrances-${serviceHost}`,
    port: 0,
    title,
    icon,
    invisible: true,
    authLevel: 'internal',
  }] : null;

  const sharedName = needsShared ? apiName : manifest.sharedEntrances[0].name;
  const provider = needsProvider ? [{
    name: sharedName,
    entrance: primary.name || appName,
    paths: ['/*'],
    verbs: ['*'],
  }] : null;

  const oldVersion = String(manifest.metadata?.version || '1.0.0').replace(/['"]/g, '');
  const newVersion = bumpPatch(oldVersion);

  if (hasHelmTemplates(rawManifest)) {
    const patched = patchHelmManifest(rawManifest, {
      sharedEntrances,
      provider,
      oldVersion,
      newVersion,
    });
    fs.writeFileSync(manifestPath, patched);
  } else {
    if (sharedEntrances) manifest.sharedEntrances = sharedEntrances;
    if (provider) manifest.provider = provider;
    manifest.metadata.version = newVersion;
    if (manifest.spec) manifest.spec.versionName = newVersion;
    fs.writeFileSync(manifestPath, dumpManifest(manifest));
  }

  updateChartVersion(chartPath, newVersion);
  console.log(`updated ${entry.name} -> ${sharedName} v${newVersion}`);
  updated++;
}

console.log(`\nDone: ${updated} updated, ${skipped} skipped`);
