#!/usr/bin/env node
/**
 * Trim CPU/memory requests (and bloated limits) so more Olares apps can schedule.
 * Packing uses requests + spec.accelerator.required*; limits stay for spikes.
 *
 * Usage: node scripts/trim-chart-resources.js [--dry-run]
 */
const fs = require('fs');
const path = require('path');
const yaml = require('js-yaml');

const ROOT = path.resolve(__dirname, '..');
const DRY = process.argv.includes('--dry-run');

const CLIENT = { reqCpu: 0.01, reqMemGi: 64 / 1024, limCpu: 0.5, limMemGi: 0.5 };

/** Cap targets — never raise existing limits; only lower when above these. */
const TIERS = {
  llm: { reqCpu: 0.5, reqMemGi: 2, limCpu: 8, limMemGi: 24 },
  media: { reqCpu: 1, reqMemGi: 4, limCpu: 12, limMemGi: 48 },
  videoheavy: { reqCpu: 1, reqMemGi: 8, limCpu: 12, limMemGi: 48 },
  light: { reqCpu: 0.25, reqMemGi: 1, limCpu: 4, limMemGi: 8 },
  tiny: { reqCpu: 0.1, reqMemGi: 0.5, limCpu: 2, limMemGi: 4 },
  /** Non-GPU sidecars / download helpers — cut hard */
  sidecar: { reqCpu: 0.2, reqMemGi: 0.5, limCpu: 2, limMemGi: 8 },
};

function classify(name, manifest) {
  const n = name.toLowerCase();
  const blob = `${n} ${manifest?.metadata?.title || ''} ${manifest?.metadata?.description || ''}`.toLowerCase();
  if (/browserless|dockerbuilder/.test(n)) return 'light';
  if (/openwebsearch/.test(n)) return 'tiny';
  if (/fastwan/.test(n)) return 'videoheavy';
  if (/splatlab|locateanything/.test(n)) return 'media';
  if (/motif|ltx|krea|ideogram|consistcompose|qwen3tts|cosyvoice|sensenova|i2v|t2v|wan/.test(blob)) {
    return 'media';
  }
  // Large vLLM audio / long-context that previously reserved 48Gi host RAM
  if (/audex|tess427/.test(n)) return 'media';
  return 'llm';
}

function bumpPatch(ver) {
  const parts = String(ver).split('.').map((x) => parseInt(x, 10) || 0);
  while (parts.length < 3) parts.push(0);
  parts[2] += 1;
  return parts.join('.');
}

function parseCpu(v) {
  if (v == null) return 0;
  const s = String(v).trim().replace(/^"|"$/g, '');
  if (s.endsWith('m')) return parseInt(s, 10) / 1000;
  return parseFloat(s) || 0;
}

function parseMemGi(v) {
  if (v == null) return 0;
  const s = String(v).trim().replace(/^"|"$/g, '');
  if (s.endsWith('Gi')) return parseFloat(s);
  if (s.endsWith('Mi')) return parseFloat(s) / 1024;
  if (s.endsWith('G')) return parseFloat(s);
  if (s.endsWith('M')) return parseFloat(s) / 1024;
  return parseFloat(s) || 0;
}

function fmtCpu(cores) {
  if (cores < 1) return `${Math.round(cores * 1000)}m`;
  // Quote integers for helm safety on string-ish fields
  return `"${Math.round(cores * 1000) / 1000}"`;
}

function fmtMem(gi) {
  if (gi < 1) return `${Math.round(gi * 1024)}Mi`;
  return `${Math.round(gi * 10) / 10}Gi`.replace(/\.0Gi$/, 'Gi');
}

function ceilCpuCores(cores) {
  return Math.max(1, Math.ceil(cores - 1e-9));
}

function ceilMemGi(gi) {
  return Math.max(1, Math.ceil(gi - 1e-9));
}

const RES_RE =
  /(\n\s+resources:\n\s+limits:\n\s+cpu:\s*)("[^"]+"|[^\n]+)(\n\s+memory:\s*)([^\n]+)(\n(?:\s+nvidia\.com\/gpu:\s*[^\n]+\n)?)(\s+requests:\n\s+cpu:\s*)([^\n]+)(\n\s+memory:\s*)([^\n]+)/g;

function applyCap(oldReqCpu, oldReqMem, oldLimCpu, oldLimMem, tier) {
  const newReqCpu = Math.min(oldReqCpu, tier.reqCpu);
  const newReqMem = Math.min(oldReqMem, tier.reqMemGi);
  const limCpu = Math.max(Math.min(oldLimCpu, tier.limCpu), newReqCpu);
  const limMem = Math.max(Math.min(oldLimMem, tier.limMemGi), newReqMem);
  const eps = 0.05;
  const changed =
    Math.abs(newReqCpu - oldReqCpu) >= eps ||
    Math.abs(newReqMem - oldReqMem) >= eps ||
    Math.abs(limCpu - oldLimCpu) >= eps ||
    Math.abs(limMem - oldLimMem) >= eps;
  return { newReqCpu, newReqMem, limCpu, limMem, changed };
}

/** Trim every bloated container: GPU/main → app tier; other fat containers → sidecar. */
function replaceMainResources(serverYaml, tierName) {
  const mainTier = TIERS[tierName];
  const sidecar = TIERS.sidecar;
  const matches = [...serverYaml.matchAll(RES_RE)];
  if (!matches.length) return { text: serverYaml, changed: false, reason: 'no resource blocks' };

  // Replace from end so indices stay valid
  let text = serverYaml;
  let any = false;
  const details = [];
  for (let i = matches.length - 1; i >= 0; i--) {
    const best = matches[i];
    const hasGpu = Boolean(best[5] && best[5].includes('gpu'));
    const oldReqCpu = parseCpu(best[7]);
    const oldReqMem = parseMemGi(best[9]);
    const oldLimCpu = parseCpu(best[2]);
    const oldLimMem = parseMemGi(best[4]);
    const bloated = oldReqCpu >= 1 || oldReqMem >= 2 || oldLimCpu >= 8 || oldLimMem >= 16;
    if (!bloated && !hasGpu) continue;

    const maxMem = Math.max(...matches.map((m) => parseMemGi(m[9])));
    // Prefer main tier for GPU container or largest memory request; else sidecar
    const useTier = hasGpu || oldReqMem === maxMem ? mainTier : sidecar;

    const { newReqCpu, newReqMem, limCpu, limMem, changed } = applyCap(
      oldReqCpu,
      oldReqMem,
      oldLimCpu,
      oldLimMem,
      useTier
    );
    if (!changed) continue;
    any = true;
    const replacement =
      best[1] +
      fmtCpu(limCpu) +
      best[3] +
      fmtMem(limMem) +
      (best[5] || '\n') +
      best[6] +
      fmtCpu(newReqCpu) +
      best[8] +
      fmtMem(newReqMem);
    text = text.slice(0, best.index) + replacement + text.slice(best.index + best[0].length);
    details.unshift({
      tier: useTier === mainTier ? tierName : 'sidecar',
      before: `${best[7].trim()}/${best[9].trim()}`,
      after: `${fmtCpu(newReqCpu)}/${fmtMem(newReqMem)}`,
      lim: `${fmtCpu(limCpu)}/${fmtMem(limMem)}`,
    });
  }

  if (!any) return { text: serverYaml, changed: false, reason: 'already lean' };

  const primary = details.find((d) => d.tier === tierName) || details[0];
  return {
    text,
    changed: true,
    before: { cpu: primary.before.split('/')[0], mem: primary.before.split('/')[1], limCpu: '', limMem: '' },
    after: {
      cpu: primary.after.split('/')[0],
      mem: primary.after.split('/')[1],
      limCpu: primary.lim.split('/')[0],
      limMem: primary.lim.split('/')[1],
    },
    details,
  };
}

function sumServerResources(serverYaml) {
  let reqCpu = 0;
  let reqMem = 0;
  let limCpu = 0;
  let limMem = 0;
  for (const m of serverYaml.matchAll(RES_RE)) {
    limCpu += parseCpu(m[2]);
    limMem += parseMemGi(m[4]);
    reqCpu += parseCpu(m[7]);
    reqMem += parseMemGi(m[9]);
  }
  return { reqCpu, reqMem, limCpu, limMem };
}

/** Surgically patch first accelerator entry cpu/mem required+limited. */
function patchAcceleratorText(manText, sums) {
  const rCpu = ceilCpuCores(sums.reqCpu + CLIENT.reqCpu);
  const rMem = ceilMemGi(sums.reqMem + CLIENT.reqMemGi);
  const lCpu = Math.max(rCpu, ceilCpuCores(sums.limCpu + CLIENT.limCpu));
  const lMem = Math.max(rMem, ceilMemGi(sums.limMem + CLIENT.limMemGi));

  const blockRe =
    /(accelerator:\n(?:\s*-\s*mode:\s*(?:nvidia|cpu)\n)(?:\s+[^\n]+\n)*?)(\s+requiredCpu:\s*)([^\n]+)(\n\s+limitedCpu:\s*)([^\n]+)(\n\s+requiredMemory:\s*)([^\n]+)(\n\s+limitedMemory:\s*)([^\n]+)/;

  if (!blockRe.test(manText)) {
    return { text: manText, ok: false, reason: 'accelerator block not found' };
  }
  const before = {};
  const text = manText.replace(blockRe, (_, head, a, rc, b, lc, c, rm, d, lm) => {
    before.requiredCpu = rc.trim();
    before.limitedCpu = lc.trim();
    before.requiredMemory = rm.trim();
    before.limitedMemory = lm.trim();
    return (
      head +
      a +
      `'${rCpu}'` +
      b +
      `'${lCpu}'` +
      c +
      `${rMem}Gi` +
      d +
      `${lMem}Gi`
    );
  });
  return {
    text,
    ok: true,
    before,
    after: {
      requiredCpu: `'${rCpu}'`,
      limitedCpu: `'${lCpu}'`,
      requiredMemory: `${rMem}Gi`,
      limitedMemory: `${lMem}Gi`,
    },
  };
}

function patchVersions(manText, chartText, i18nText, next, tier) {
  const line = `    v${next}: trim CPU/memory requests for denser multi-app packing (${tier}).\n`;

  let man = manText
    .replace(/^(metadata:\n(?:  [^\n]+\n)*?  version:\s*)([^\n]+)/m, `$1${next}`)
    .replace(/^(spec:\n(?:  [^\n]+\n)*?  versionName:\s*)([^\n]+)/m, `$1${next}`);

  // Prefer version under metadata (standard): also handle versionName near top of spec
  man = man.replace(/^(  version:\s*)(\d+\.\d+\.\d+)\s*$/m, `$1${next}`);
  man = man.replace(/^(  versionName:\s*)(\d+\.\d+\.\d+)\s*$/m, `$1${next}`);

  if (!man.includes('denser multi-app packing')) {
    man = man.replace(
      /(upgradeDescription:\s*\|\s*\n)/,
      `$1${line}`
    );
  }

  let chart = chartText
    .replace(/^version:\s*.*$/m, `version: ${next}`)
    .replace(/^appVersion:\s*.*$/m, `appVersion: "${next}"`);

  let i18n = i18nText;
  if (i18n) {
    i18n = i18n
      .replace(/^(  version:\s*)(\d+\.\d+\.\d+)\s*$/m, `$1${next}`)
      .replace(/^(  versionName:\s*)(\d+\.\d+\.\d+)\s*$/m, `$1${next}`);
    if (!i18n.includes('denser multi-app packing')) {
      i18n = i18n.replace(/(upgradeDescription:\s*\|\s*\n)/, `$1${line}`);
    }
  }
  return { man, chart, i18n };
}

function readVersion(manifest) {
  return String(manifest?.metadata?.version || manifest?.spec?.versionName || '0.0.0');
}

function listCharts() {
  return fs
    .readdirSync(ROOT, { withFileTypes: true })
    .filter((d) => d.isDirectory())
    .map((d) => d.name)
    .filter((n) => {
      const base = path.join(ROOT, n);
      return (
        fs.existsSync(path.join(base, 'Chart.yaml')) &&
        fs.existsSync(path.join(base, 'OlaresManifest.yaml'))
      );
    })
    .sort();
}

function shouldProcess(name, manifest) {
  if (manifest?.options?.shared) return true;
  if (['browserlessone', 'dockerbuilderone', 'openwebsearchone'].includes(name)) return true;
  const acc = manifest?.spec?.accelerator;
  if (Array.isArray(acc) && acc.some((a) => a.mode === 'nvidia')) return true;
  return false;
}

function main() {
  const report = [];
  for (const name of listCharts()) {
    const base = path.join(ROOT, name);
    const manPath = path.join(base, 'OlaresManifest.yaml');
    const chartPath = path.join(base, 'Chart.yaml');
    const serverPath = path.join(base, 'templates', 'server.yaml');
    if (!fs.existsSync(serverPath)) continue;

    const manText0 = fs.readFileSync(manPath, 'utf8');
    const manifest = yaml.load(manText0);
    if (!shouldProcess(name, manifest)) continue;

    const tier = classify(name, manifest);
    let server = fs.readFileSync(serverPath, 'utf8');
    const rep = replaceMainResources(server, tier);

    // Even if containers already lean, fix bloated accelerator required*
    const sumsBefore = sumServerResources(server);
    const serverAfter = rep.changed ? rep.text : server;
    const sums = sumServerResources(serverAfter);
    const accPatch = patchAcceleratorText(manText0, sums);

    const accBloated =
      accPatch.ok &&
      (parseCpu(accPatch.before.requiredCpu) > ceilCpuCores(sums.reqCpu + CLIENT.reqCpu) + 0.01 ||
        parseMemGi(accPatch.before.requiredMemory) > ceilMemGi(sums.reqMem + CLIENT.reqMemGi) + 0.01 ||
        parseCpu(accPatch.before.limitedCpu) > ceilCpuCores(sums.limCpu + CLIENT.limCpu) + 0.5 ||
        parseMemGi(accPatch.before.limitedMemory) > ceilMemGi(sums.limMem + CLIENT.limMemGi) + 0.5);

    if (!rep.changed && !accBloated) {
      report.push({ name, tier, skipped: rep.reason || 'already lean' });
      continue;
    }

    const oldVer = readVersion(manifest);
    const next = bumpPatch(oldVer);
    const i18nPath = path.join(base, 'i18n', 'en-US', 'OlaresManifest.yaml');
    const i18nText0 = fs.existsSync(i18nPath) ? fs.readFileSync(i18nPath, 'utf8') : null;
    const chartText0 = fs.readFileSync(chartPath, 'utf8');

    let manText = accPatch.ok ? accPatch.text : manText0;
    const verPatch = patchVersions(manText, chartText0, i18nText0, next, tier);

    if (!DRY) {
      if (rep.changed) fs.writeFileSync(serverPath, serverAfter);
      fs.writeFileSync(manPath, verPatch.man);
      fs.writeFileSync(chartPath, verPatch.chart);
      if (i18nText0 != null) fs.writeFileSync(i18nPath, verPatch.i18n);
    }

    report.push({
      name,
      tier,
      version: `${oldVer} → ${next}`,
      resources: rep.changed
        ? `${rep.before.cpu}/${rep.before.mem} → ${rep.after.cpu}/${rep.after.mem} (lim ${rep.after.limCpu}/${rep.after.limMem})`
        : 'containers unchanged',
      accelerator: accPatch.ok
        ? `cpu ${accPatch.before.requiredCpu}→${accPatch.after.requiredCpu} mem ${accPatch.before.requiredMemory}→${accPatch.after.requiredMemory} (lim ${accPatch.after.limitedCpu}/${accPatch.after.limitedMemory})`
        : accPatch.reason,
      sums: {
        req: `${sums.reqCpu}c/${sums.reqMem}Gi`,
        lim: `${sums.limCpu}c/${sums.limMem}Gi`,
        beforeReqMem: `${sumsBefore.reqMem}Gi`,
      },
    });
  }

  const changed = report.filter((r) => r.version);
  const skipped = report.filter((r) => r.skipped);
  console.log(JSON.stringify({ dry: DRY, changed: changed.length, skipped: skipped.length, items: report }, null, 2));
}

main();
