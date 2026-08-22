#!/usr/bin/env node
/**
 * Ensure OlaresManifest spec.requiredMemory is greater than the sum of all
 * container memory requests across deployment templates (Studio lint rule).
 */
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

function hasHelmTemplates(content) {
  return /\{\{/.test(content);
}

function parseMem(s) {
  if (!s) return 0;
  s = String(s).replace(/['"]/g, '').trim();
  const m = s.match(/^(\d+(?:\.\d+)?)(Mi|Gi|Ti|Ki|M|G|T|K)?$/i);
  if (!m) return NaN;
  const n = parseFloat(m[1]);
  const u = (m[2] || 'M').toLowerCase();
  if (u === 'gi' || u === 'g') return n * 1024;
  if (u === 'mi' || u === 'm') return n;
  if (u === 'ki' || u === 'k') return n / 1024;
  return n;
}

function toManifestMem(mi) {
  if (mi % 1024 === 0) return `${mi / 1024}Gi`;
  return `${mi}Mi`;
}

function suggestedRequiredMi(totalMi) {
  if (totalMi < 1024) return 1024;
  const gi = Math.ceil((totalMi + 1) / 1024);
  return gi * 1024;
}

function bumpPatch(version) {
  const parts = String(version).replace(/['"]/g, '').split('.');
  const last = parseInt(parts[parts.length - 1], 10);
  parts[parts.length - 1] = String(Number.isNaN(last) ? 1 : last + 1);
  return parts.join('.');
}

function findDeployments(appDir) {
  const out = [];
  function walk(dir) {
    for (const e of fs.readdirSync(dir, { withFileTypes: true })) {
      const p = path.join(dir, e.name);
      if (e.isDirectory()) walk(p);
      else if (e.name === 'deployment.yaml' && dir.endsWith('templates')) out.push(p);
    }
  }
  walk(appDir);
  return out;
}

function sumRequests(appDir) {
  let totalMi = 0;
  for (const deployPath of findDeployments(appDir)) {
    const docs = yaml.loadAll(stripHelm(fs.readFileSync(deployPath, 'utf8')));
    for (const doc of docs) {
      if (!doc?.spec?.template?.spec) continue;
      const spec = doc.spec.template.spec;
      for (const c of [...(spec.initContainers || []), ...(spec.containers || [])]) {
        const mem = c.resources?.requests?.memory;
        if (mem) totalMi += parseMem(mem);
      }
    }
  }
  return totalMi;
}

function updateChartVersion(chartPath, newVersion) {
  const raw = fs.readFileSync(chartPath, 'utf8').replace(/^apiVersion: v2\n(?=apiVersion: v2\n)/, '');
  const chart = yaml.load(raw);
  chart.version = newVersion;
  if (chart.appVersion !== undefined) chart.appVersion = newVersion;
  fs.writeFileSync(chartPath, yaml.dump(chart, { lineWidth: -1, noRefs: true }));
}

function patchRequiredMemory(raw, currentMi, newMem) {
  let replaced = false;
  return raw.replace(/^(\s+requiredMemory:\s*)(['"]?)([^\n'"]+)\2\s*$/gm, (match, prefix, quote, value) => {
    const mi = parseMem(value);
    if (replaced || mi !== currentMi) return match;
    replaced = true;
    return `${prefix}'${newMem}'`;
  });
}

function patchLimitedMemory(raw, minMi) {
  return raw.replace(/^(\s+limitedMemory:\s*)(['"]?)([^\n'"]+)\2\s*$/m, (match, prefix, quote, value) => {
    const mi = parseMem(value);
    if (mi >= minMi) return match;
    return `${prefix}'${toManifestMem(minMi)}'`;
  });
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

let updated = 0;
let skipped = 0;

for (const entry of fs.readdirSync(REPO, { withFileTypes: true }).filter((e) => e.isDirectory())) {
  const appDir = path.join(REPO, entry.name);
  const manifestPath = path.join(appDir, 'OlaresManifest.yaml');
  const chartPath = path.join(appDir, 'Chart.yaml');
  if (!fs.existsSync(manifestPath) || !fs.existsSync(chartPath)) continue;

  const totalMi = sumRequests(appDir);
  if (!totalMi) {
    console.log(`skip ${entry.name}: no memory requests`);
    skipped++;
    continue;
  }

  const rawManifest = fs.readFileSync(manifestPath, 'utf8');
  let manifest;
  try {
    manifest = yaml.load(stripHelm(rawManifest));
  } catch (e) {
    console.log(`skip ${entry.name}: parse error`);
    skipped++;
    continue;
  }

  const currentMi = parseMem(manifest.spec?.requiredMemory);
  if (!currentMi || Number.isNaN(currentMi)) {
    console.log(`skip ${entry.name}: no requiredMemory`);
    skipped++;
    continue;
  }

  if (totalMi < currentMi) {
    console.log(`ok ${entry.name}: requests=${toManifestMem(totalMi)} required=${manifest.spec.requiredMemory}`);
    skipped++;
    continue;
  }

  const newMi = suggestedRequiredMi(totalMi);
  const newMem = toManifestMem(newMi);
  const oldVersion = String(manifest.metadata?.version || '1.0.0').replace(/['"]/g, '');
  const newVersion = bumpPatch(oldVersion);

  let out = patchRequiredMemory(rawManifest, currentMi, newMem);
  out = patchLimitedMemory(out, newMi);
  out = bumpVersionsInRaw(out, oldVersion, newVersion);
  fs.writeFileSync(manifestPath, out);
  updateChartVersion(chartPath, newVersion);
  console.log(`updated ${entry.name}: requests=${toManifestMem(totalMi)} required ${manifest.spec.requiredMemory} -> ${newMem} v${newVersion}`);
  updated++;
}

console.log(`\nDone: ${updated} updated, ${skipped} skipped`);
