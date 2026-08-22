#!/usr/bin/env node
/**
 * Package every Olares app chart (Chart.yaml + OlaresManifest.yaml) into charts/*.tgz.
 * Removes stale tgz for the same app when version changes.
 */
const fs = require('fs');
const path = require('path');
const { execSync } = require('child_process');
const yaml = require('js-yaml');

const REPO = path.resolve(__dirname, '..');
const CHARTS_DIR = path.join(REPO, 'charts');

// Inline app sources into Helm configmaps before packaging (must run before helm package).
for (const fn of [
  './inject-sensenovau1-configmap.js',
  './inject-sensenovau1infov2-configmap.js',
  './inject-sensenovau1lightllm-configmap.js',
  './inject-sensenovasi15-configmap.js',
  './inject-locateanything-configmap.js',
  './inject-dockerbuilder-configmap.js',
  './inject-ideogram4-configmap.js',
  './inject-fastwan-configmap.js',
  './inject-fastwanqad13bone-configmap.js',
  './inject-fastwanqad13fp8-configmap.js',
  './inject-motifvideo-configmap.js',
  './inject-splatlab-configmap.js',
  './inject-ltx23one-configmap.js',
  './inject-consistcompose-configmap.js',
  './inject-krea2turbo-configmap.js',
  './inject-mageflow-configmap.js',
  './inject-minimaxh3-configmap.js',
  './inject-lingbot-configmaps.js',
  './inject-sensenovavision-configmap.js',
]) {
  const mod = require(fn);
  const key = Object.keys(mod).find((k) => k.startsWith('inject'));
  if (key) mod[key]();
}

function readChartVersion(chartPath) {
  const raw = fs.readFileSync(chartPath, 'utf8').replace(/^apiVersion: v2\n(?=apiVersion: v2\n)/, '');
  const chart = yaml.load(raw);
  const name = chart.name || path.basename(path.dirname(chartPath));
  return [name, String(chart.version)];
}

function removeStaleTgz(appName, currentVersion) {
  if (!fs.existsSync(CHARTS_DIR)) return;
  const keep = `${appName}-${currentVersion}.tgz`;
  for (const file of fs.readdirSync(CHARTS_DIR)) {
    if (!file.endsWith('.tgz')) continue;
    if (file.startsWith(`${appName}-`) && file !== keep) {
      fs.unlinkSync(path.join(CHARTS_DIR, file));
      console.log(`  removed stale ${file}`);
    }
  }
}

fs.mkdirSync(CHARTS_DIR, { recursive: true });

const apps = fs.readdirSync(REPO, { withFileTypes: true })
  .filter((e) => e.isDirectory())
  .map((e) => e.name)
  .filter((name) => {
    const dir = path.join(REPO, name);
    return fs.existsSync(path.join(dir, 'Chart.yaml'))
      && fs.existsSync(path.join(dir, 'OlaresManifest.yaml'));
  })
  .sort();

let ok = 0;
let failed = 0;

for (const appName of apps) {
  const appDir = path.join(REPO, appName);
  const chartPath = path.join(appDir, 'Chart.yaml');
  try {
    const [name, version] = readChartVersion(chartPath);
    removeStaleTgz(name, version);
    execSync(`helm package "${appDir}" -d "${CHARTS_DIR}"`, { stdio: 'pipe' });
    const tgz = path.join(CHARTS_DIR, `${name}-${version}.tgz`);
    const kb = Math.round(fs.statSync(tgz).size / 1024);
    console.log(`packaged ${name} v${version} (${kb}KB)`);
    ok++;
  } catch (err) {
    console.error(`FAILED ${appName}: ${err.stderr?.toString() || err.message}`);
    failed++;
  }
}

console.log(`\nDone: ${ok} packaged, ${failed} failed, ${apps.length} apps`);
if (failed > 0) process.exit(1);
