#!/usr/bin/env node
/**
 * Fix sharedEntrances to Olares convention:
 *   host: sharedentrances-<service>
 *   port: 0
 * Add sharedentrances-<service> K8s Service (port 80 -> app port) when missing.
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

function primaryServiceFromDeployment(deployPath) {
  const docs = yaml.loadAll(stripHelm(fs.readFileSync(deployPath, 'utf8')));
  for (const doc of docs) {
    if (doc?.kind !== 'Service') continue;
    const name = doc.metadata?.name;
    if (!name || name.startsWith('sharedentrances-')) continue;
    const port = doc.spec?.ports?.[0]?.port;
    const targetPort = doc.spec?.ports?.[0]?.targetPort ?? port;
    const selector = doc.spec?.selector?.['io.kompose.service'] || name;
    if (name && port) return { name, port, targetPort, selector, namespace: doc.metadata?.namespace };
  }
  return null;
}

function sharedServiceBlock(svc) {
  const sharedName = `sharedentrances-${svc.name}`;
  return `---
apiVersion: v1
kind: Service
metadata:
  labels:
    io.kompose.service: ${svc.selector}
  name: ${sharedName}
  namespace: ${svc.namespace || '"{{ .Release.Namespace }}"'}
spec:
  ports:
    - name: "${svc.name}"
      port: 80
      targetPort: ${svc.targetPort}
  selector:
    io.kompose.service: ${svc.selector}
`;
}

function ensureSharedService(deployPath, svc) {
  const raw = fs.readFileSync(deployPath, 'utf8');
  const sharedName = `sharedentrances-${svc.name}`;
  if (raw.includes(`name: ${sharedName}`)) return false;
  fs.writeFileSync(deployPath, `${raw.replace(/\s*$/, '\n')}${sharedServiceBlock(svc)}`);
  return true;
}

function patchSharedEntranceInManifest(raw, serviceName, sharedHost) {
  let out = raw;
  out = out.replace(
    /^(\s+host:\s*)(['"]?)([^'"\n]+)\3\s*$/m,
    (match, prefix, quote, host) => {
      if (!match.includes('sharedEntrances') && out.indexOf(match) > out.indexOf('sharedEntrances')) {
        return `${prefix}${sharedHost}`;
      }
      return match;
    },
  );
  // Fix host under sharedEntrances block only
  out = out.replace(
    /(sharedEntrances:\n(?:  - .*\n)*?    host:\s*)(['"]?)([^'"\n]+)\2/g,
    `$1'${sharedHost}'`,
  );
  out = out.replace(
    /(sharedEntrances:\n(?:  - [^\n]+\n)*?    host:[^\n]+\n    port:\s*)(['"]?)[^'"\n]+\2/g,
    `$10`,
  );
  return out;
}

let updated = 0;

for (const entry of fs.readdirSync(REPO, { withFileTypes: true }).filter((e) => e.isDirectory())) {
  const appDir = path.join(REPO, entry.name);
  const manifestPath = path.join(appDir, 'OlaresManifest.yaml');
  const chartPath = path.join(appDir, 'Chart.yaml');
  if (!fs.existsSync(manifestPath) || !fs.existsSync(chartPath)) continue;

  const rawManifest = fs.readFileSync(manifestPath, 'utf8');
  let manifest;
  try {
    manifest = yaml.load(stripHelm(rawManifest));
  } catch {
    continue;
  }

  const shared = manifest.sharedEntrances?.[0];
  const primary = manifest.entrances?.[0];
  if (!shared || !primary) continue;

  const serviceName = primary.host || manifest.metadata?.name || entry.name;
  const sharedHost = shared.host?.startsWith('sharedentrances-')
    ? shared.host
    : `sharedentrances-${serviceName}`;

  const alreadyOk = shared.host === sharedHost && String(shared.port) === '0';
  const deployPaths = findDeployments(appDir);
  let addedService = false;
  for (const deployPath of deployPaths) {
    const svc = primaryServiceFromDeployment(deployPath);
    if (!svc) continue;
    if (svc.name === serviceName || svc.name === manifest.metadata?.name) {
      addedService = ensureSharedService(deployPath, svc) || addedService;
    }
  }

  if (alreadyOk && !addedService) {
    console.log(`ok ${entry.name}`);
    continue;
  }

  const oldVersion = String(manifest.metadata?.version || '1.0.0').replace(/['"]/g, '');
  const newVersion = bumpPatch(oldVersion);
  let out = patchSharedEntranceInManifest(rawManifest, serviceName, sharedHost);
  out = bumpVersionsInRaw(out, oldVersion, newVersion);
  fs.writeFileSync(manifestPath, out);
  updateChartVersion(chartPath, newVersion);
  console.log(`updated ${entry.name}: host=${sharedHost} port=0 v${newVersion}${addedService ? ' +service' : ''}`);
  updated++;
}

console.log(`\nDone: ${updated} updated`);
