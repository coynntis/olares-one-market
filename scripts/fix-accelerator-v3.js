#!/usr/bin/env node
/**
 * v3 chart lint: sum of ALL workload container resources (initContainers excluded)
 * must fit within spec.accelerator per mode.
 * Post-migration charts include nginx clientproxy — bump accelerator to match.
 */
const fs = require('fs');
const path = require('path');
const yaml = require('js-yaml');
const { normalizeSemverFields } = require('./fix-manifest-semver-v3.js');

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

function parseCpu(s) {
  if (!s) return 0;
  const v = String(s).replace(/['"]/g, '').trim();
  if (v.endsWith('m')) return parseInt(v, 10);
  return Math.round(parseFloat(v) * 1000);
}

function parseMemMi(s) {
  if (!s) return 0;
  const v = String(s).replace(/['"]/g, '').trim();
  const m = v.match(/^(\d+(?:\.\d+)?)(Ki|Mi|Gi|Ti)?$/i);
  if (!m) return 0;
  const n = parseFloat(m[1]);
  const u = (m[2] || 'Mi').toLowerCase();
  if (u === 'gi') return n * 1024;
  if (u === 'mi') return n;
  if (u === 'ki') return n / 1024;
  return n;
}

function formatCpuMillicores(m) {
  if (m % 1000 === 0) return String(m / 1000);
  return `${m}m`;
}

function formatMemMi(mi) {
  if (mi >= 1024) return `${Math.ceil(mi / 1024)}Gi`;
  return `${Math.ceil(mi)}Mi`;
}

function ceilCpuCores(millicores) {
  return Math.max(1, Math.ceil(millicores / 1000));
}

function ceilMemGi(mi) {
  return Math.max(1, Math.ceil(mi / 1024));
}

function bumpPatch(version) {
  const parts = String(version).replace(/['"]/g, '').split('.');
  const last = parseInt(parts[parts.length - 1], 10);
  parts[parts.length - 1] = String(Number.isNaN(last) ? 1 : last + 1);
  return parts.join('.');
}

function readChartVersion(chartPath) {
  const raw = fs.readFileSync(chartPath, 'utf8').replace(/^apiVersion: v2\n(?=apiVersion: v2\n)/, '');
  const chart = yaml.load(raw);
  return String(chart.version);
}

function updateChartVersion(chartPath, newVersion) {
  if (!fs.existsSync(chartPath)) return;
  const raw = fs.readFileSync(chartPath, 'utf8').replace(/^apiVersion: v2\n(?=apiVersion: v2\n)/, '');
  const chart = yaml.load(raw);
  chart.version = newVersion;
  if (chart.appVersion !== undefined) chart.appVersion = newVersion;
  fs.writeFileSync(chartPath, yaml.dump(chart, { lineWidth: -1, noRefs: true }));
}

function workloadTemplateFiles(appDir) {
  const tpl = path.join(appDir, 'templates');
  if (!fs.existsSync(tpl)) return [];
  return fs.readdirSync(tpl)
    .filter((f) => f.endsWith('.yaml') || f.endsWith('.yml'))
    .map((f) => path.join(tpl, f));
}

function sumContainerResources(appDir) {
  const totals = { reqCpu: 0, limCpu: 0, reqMem: 0, limMem: 0 };
  for (const file of workloadTemplateFiles(appDir)) {
    let docs;
    try {
      docs = yaml.loadAll(stripHelm(fs.readFileSync(file, 'utf8')));
    } catch {
      continue;
    }
    for (const doc of docs) {
      if (!doc || !['Deployment', 'StatefulSet'].includes(doc.kind)) continue;
      const containers = doc.spec?.template?.spec?.containers || [];
      for (const c of containers) {
        totals.reqCpu += parseCpu(c.resources?.requests?.cpu);
        totals.limCpu += parseCpu(c.resources?.limits?.cpu);
        totals.reqMem += parseMemMi(c.resources?.requests?.memory);
        totals.limMem += parseMemMi(c.resources?.limits?.memory);
      }
    }
  }
  return totals;
}

function manifestCpuToMillicores(v) {
  if (v === undefined || v === null) return 0;
  const s = String(v).replace(/['"]/g, '').trim();
  if (s.endsWith('m')) return parseInt(s, 10);
  return Math.round(parseFloat(s) * 1000);
}

function manifestMemToMi(v) {
  return parseMemMi(String(v).replace(/['"]/g, ''));
}

function acceleratorNeedsBump(accel, totals) {
  const reqCpu = manifestCpuToMillicores(accel.requiredCpu);
  const limCpu = manifestCpuToMillicores(accel.limitedCpu);
  const reqMem = manifestMemToMi(accel.requiredMemory);
  const limMem = manifestMemToMi(accel.limitedMemory);
  return totals.reqCpu > reqCpu
    || totals.limCpu > limCpu
    || totals.reqMem > reqMem
    || totals.limMem > limMem;
}

function buildAcceleratorValues(accel, totals) {
  const reqCpuMi = Math.max(totals.reqCpu, manifestCpuToMillicores(accel.requiredCpu));
  const limCpuMi = Math.max(totals.limCpu, manifestCpuToMillicores(accel.limitedCpu));
  const reqMemMi = Math.max(totals.reqMem, manifestMemToMi(accel.requiredMemory));
  const limMemMi = Math.max(totals.limMem, manifestMemToMi(accel.limitedMemory));

  return {
    ...accel,
    requiredCpu: String(ceilCpuCores(reqCpuMi)),
    limitedCpu: String(ceilCpuCores(limCpuMi)),
    requiredMemory: formatMemMi(ceilMemGi(reqMemMi) * 1024),
    limitedMemory: formatMemMi(ceilMemGi(limMemMi) * 1024),
  };
}

function fixManifestAccelerator(raw, totals, newVersion) {
  const doc = yaml.load(stripHelm(raw)) || {};
  if (!/apiVersion:\s*'?v3'?/.test(raw) && doc.apiVersion !== 'v3') {
    return null;
  }
  const spec = doc.spec || {};
  const accelerators = spec.accelerator || [];
  if (!accelerators.length) return null;

  let changed = false;
  const nextAccel = accelerators.map((accel) => {
    if (!acceleratorNeedsBump(accel, totals)) return accel;
    changed = true;
    return buildAcceleratorValues(accel, totals);
  });

  if (!changed) return null;

  spec.accelerator = nextAccel;
  spec.versionName = newVersion;
  if (doc.metadata) doc.metadata.version = newVersion;

  const note = `    v${newVersion}: bump spec.accelerator to cover server + client container resource sums (v3 lint).\n`;
  const ud = spec.upgradeDescription || '';
  if (!ud.includes('accelerator')) {
    spec.upgradeDescription = note + ud;
  }

  doc.spec = spec;
  let out = yaml.dump(doc, { lineWidth: -1, noRefs: true, quotingType: "'", forceQuotes: false });
  out = normalizeSemverFields(out);
  if (!out.startsWith('---')) out = `---\n${out}`;
  return out;
}

const dryRun = process.argv.includes('--dry-run');
const only = process.argv.find((a) => a.startsWith('--app='))?.split('=')[1];

const apps = fs.readdirSync(REPO, { withFileTypes: true })
  .filter((e) => e.isDirectory())
  .map((e) => e.name)
  .filter((name) => fs.existsSync(path.join(REPO, name, 'OlaresManifest.yaml')))
  .filter((name) => !only || name === only)
  .sort();

let fixed = 0;
let skipped = 0;

for (const appName of apps) {
  const appDir = path.join(REPO, appName);
  const manifestPath = path.join(appDir, 'OlaresManifest.yaml');
  const raw = fs.readFileSync(manifestPath, 'utf8');
  if (!/apiVersion:\s*'?v3'?/.test(raw)) {
    skipped++;
    continue;
  }

  const totals = sumContainerResources(appDir);
  if (!totals.reqCpu && !totals.limCpu && !totals.reqMem && !totals.limMem) {
    skipped++;
    continue;
  }

  const doc = yaml.load(stripHelm(raw)) || {};
  const accel = doc.spec?.accelerator?.[0];
  if (!accel || !acceleratorNeedsBump(accel, totals)) {
    skipped++;
    continue;
  }

  const oldVersion = readChartVersion(path.join(appDir, 'Chart.yaml'));
  const newVersion = bumpPatch(oldVersion);
  const next = fixManifestAccelerator(raw, totals, newVersion);
  if (!next) {
    skipped++;
    continue;
  }

  const nextAccel = yaml.load(stripHelm(next)).spec.accelerator[0];
  if (dryRun) {
    console.log(`${appName}: v${oldVersion}→v${newVersion} accel req ${accel.requiredCpu}/${accel.requiredMemory} lim ${accel.limitedCpu}/${accel.limitedMemory} → req ${nextAccel.requiredCpu}/${nextAccel.requiredMemory} lim ${nextAccel.limitedCpu}/${nextAccel.limitedMemory} (Σ lim ${totals.limCpu}m ${totals.limMem}Mi)`);
    fixed++;
    continue;
  }

  fs.writeFileSync(manifestPath, next);
  updateChartVersion(path.join(appDir, 'Chart.yaml'), newVersion);
  console.log(`fixed ${appName} v${newVersion} (lim cpu ${totals.limCpu}m mem ${totals.limMem}Mi)`);
  fixed++;
}

console.log(`\nDone: ${fixed} ${dryRun ? 'would fix' : 'fixed'}, ${skipped} ok/skipped`);
