#!/usr/bin/env node
/**
 * Align llama.cpp charts:
 *  - remove --mlock (safe; no whitespace collapse)
 *  - ensure --op-offload
 *  - ensure GGML_CUDA_GRAPH_OPT=1
 * Optional: --bump to bump patch version + upgradeDescription
 *
 * Usage: node scripts/align-llamacpp-flags.js [--dry-run] [--bump]
 */
const fs = require('fs');
const path = require('path');
const yaml = require('js-yaml');

const ROOT = path.resolve(__dirname, '..');
const DRY = process.argv.includes('--dry-run');
const BUMP = process.argv.includes('--bump');

function isLlamaChart(serverText) {
  return /llama-server|ggml-org\/llama\.cpp|buun-llama-cpp|beellama-cpp/.test(serverText);
}

function bumpPatch(ver) {
  const parts = String(ver).split('.').map((x) => parseInt(x, 10) || 0);
  while (parts.length < 3) parts.push(0);
  parts[2] += 1;
  return parts.join('.');
}

function stripMlock(text) {
  let t = text;
  // YAML list entries only
  t = t.replace(/^\s*-\s*"--mlock"\s*\n/gm, '');
  t = t.replace(/^\s*-\s*'--mlock'\s*\n/gm, '');
  t = t.replace(/^\s*-\s*--mlock\s*\n/gm, '');
  // Bash: standalone line with only --mlock \
  t = t.replace(/^[ \t]*--mlock[ \t]*\\?[ \t]*\n/gm, '');
  // Bash: inline --mlock between other flags (same line)
  t = t.replace(/[ \t]+--mlock(?=[ \t])/g, '');
  t = t.replace(/[ \t]+--mlock\\/g, ' \\');
  t = t.replace(/[ \t]+--mlock$/gm, '');
  return t;
}

function ensureOpOffload(text) {
  if (/--op-offload/.test(text)) return text;

  // YAML args: after flash-attn value
  if (/^\s*-\s*["']?--flash-attn["']?\s*$/m.test(text)) {
    return text.replace(
      /(^\s*-\s*["']?--flash-attn["']?\s*\n\s*-\s*["']?(?:on|auto|off)["']?\s*\n)/m,
      `$1            - "--op-offload"\n`
    );
  }

  // Bash: --flash-attn on|auto|off  → add next continued line
  if (/--flash-attn\s+(on|auto|off)/.test(text)) {
    return text.replace(
      /(--flash-attn\s+(?:on|auto|off))([ \t]*\\)?/,
      (m, fa, cont) => {
        if (cont) return `${fa} \\\n                --op-offload`;
        return `${fa} --op-offload`;
      }
    );
  }

  // Bash exec block: before EXTRA_LLM_ARGS
  if (/\$\{EXTRA_LLM_ARGS\[@\]\}/.test(text)) {
    return text.replace(
      /^([ \t]*)("\$\{EXTRA_LLM_ARGS\[@\]\}")/m,
      `$1--op-offload \\\n$1$2`
    );
  }

  return text;
}

function ensureGraphOpt(text) {
  if (/GGML_CUDA_GRAPH_OPT/.test(text)) {
    return text.replace(
      /(name:\s*GGML_CUDA_GRAPH_OPT\s*\n\s*value:\s*)(["']?)[^"'\n]+\2/,
      `$1"1"`
    );
  }

  const envEntry = `            - name: GGML_CUDA_GRAPH_OPT\n              value: "1"\n`;

  if (/^\s*-\s*name:\s*LLM_CONTEXT_WINDOW\s*$/m.test(text)) {
    return text.replace(/^(\s*-\s*name:\s*LLM_CONTEXT_WINDOW\s*$)/m, `${envEntry}$1`);
  }

  const re = /(llama-server|buun-llama-cpp|beellama-cpp|ggml-org\/llama\.cpp)([\s\S]{0,3500}?)(^\s+env:\s*\n)/m;
  if (re.test(text)) {
    return text.replace(re, `$1$2$3${envEntry}`);
  }

  if (/envFrom:\s*\n\s*-\s*configMapRef:/.test(text)) {
    return text.replace(
      /(^\s+)(envFrom:\s*\n\s+-\s+configMapRef:)/m,
      `$1env:\n${envEntry}$1$2`
    );
  }

  return text;
}

function patchVersions(manText, chartText, i18nText, next) {
  const line = `    v${next}: llama.cpp defaults — drop --mlock; ensure GGML_CUDA_GRAPH_OPT=1 + --op-offload.\n`;
  let man = manText
    .replace(/^(  version:\s*)(\d+\.\d+\.\d+)\s*$/m, `$1${next}`)
    .replace(/^(  versionName:\s*)(\d+\.\d+\.\d+)\s*$/m, `$1${next}`);
  if (!man.includes('drop --mlock; ensure GGML_CUDA_GRAPH_OPT')) {
    man = man.replace(/(upgradeDescription:\s*\|2?\s*\n)/, `$1${line}`);
  }
  let chart = chartText
    .replace(/^version:\s*.*$/m, `version: ${next}`)
    .replace(/^appVersion:\s*.*$/m, `appVersion: "${next}"`);
  let i18n = i18nText;
  if (i18n) {
    i18n = i18n
      .replace(/^(  version:\s*)(\d+\.\d+\.\d+)\s*$/m, `$1${next}`)
      .replace(/^(  versionName:\s*)(\d+\.\d+\.\d+)\s*$/m, `$1${next}`);
  }
  return { man, chart, i18n };
}

function listCharts() {
  return fs
    .readdirSync(ROOT, { withFileTypes: true })
    .filter((d) => d.isDirectory())
    .map((d) => d.name)
    .filter((n) => fs.existsSync(path.join(ROOT, n, 'templates', 'server.yaml')))
    .sort();
}

function main() {
  const report = [];
  for (const name of listCharts()) {
    const base = path.join(ROOT, name);
    const serverPath = path.join(base, 'templates', 'server.yaml');
    let server = fs.readFileSync(serverPath, 'utf8');
    if (!isLlamaChart(server)) continue;

    const before = server;
    const actions = [];
    if (/--mlock/.test(server)) {
      server = stripMlock(server);
      actions.push('remove-mlock');
    }
    const beforeOp = server;
    server = ensureOpOffload(server);
    if (server !== beforeOp) actions.push('add-op-offload');
    else if (!/--op-offload/.test(server)) actions.push('WARN-op-offload-failed');

    const beforeGraph = server;
    server = ensureGraphOpt(server);
    if (server !== beforeGraph) actions.push('add-GRAPH_OPT');
    else if (!/GGML_CUDA_GRAPH_OPT/.test(server)) actions.push('WARN-GRAPH_OPT-failed');

    // Detect indent corruption
    if (/^ (limits|resources|env|value):/m.test(server)) {
      actions.push('WARN-indent-corrupt');
    }

    if (server === before && !actions.some((a) => a.startsWith('WARN'))) {
      report.push({ name, skipped: 'already aligned' });
      continue;
    }

    if (!DRY) fs.writeFileSync(serverPath, server);

    let verInfo = null;
    if (BUMP) {
      const manPath = path.join(base, 'OlaresManifest.yaml');
      const chartPath = path.join(base, 'Chart.yaml');
      const manText = fs.readFileSync(manPath, 'utf8');
      const chartText = fs.readFileSync(chartPath, 'utf8');
      const manifest = yaml.load(manText);
      const oldVer = String(manifest?.metadata?.version || '0.0.0');
      // If upgrade note already present for this theme, don't double-bump
      const alreadyNote = manText.includes('drop --mlock; ensure GGML_CUDA_GRAPH_OPT');
      const next = alreadyNote ? oldVer : bumpPatch(oldVer);
      const i18nPath = path.join(base, 'i18n', 'en-US', 'OlaresManifest.yaml');
      const i18nText = fs.existsSync(i18nPath) ? fs.readFileSync(i18nPath, 'utf8') : null;
      if (!alreadyNote) {
        const patched = patchVersions(manText, chartText, i18nText, next);
        if (!DRY) {
          fs.writeFileSync(manPath, patched.man);
          fs.writeFileSync(chartPath, patched.chart);
          if (i18nText != null) fs.writeFileSync(i18nPath, patched.i18n);
        }
        verInfo = `${oldVer} → ${next}`;
      } else {
        verInfo = `${oldVer} (note exists)`;
      }
    }

    report.push({
      name,
      version: verInfo,
      actions,
      mlockGone: !/--mlock/.test(server),
      hasOp: /--op-offload/.test(server),
      hasGraph: /GGML_CUDA_GRAPH_OPT/.test(server),
    });
  }
  console.log(JSON.stringify({ dry: DRY, bump: BUMP, items: report }, null, 2));
}

main();
