#!/usr/bin/env node
/**
 * Migrate split shared apps (v2 subCharts) → apiVersion v3 flat chart.
 * - Flatten *srv templates into root templates/
 * - Remove OLARES_USER_* from chart files (manifest valueFrom OK)
 * - workloadReplicas + options.shared
 */
const fs = require('fs');
const path = require('path');
const yaml = require('js-yaml');
const { normalizeSemverFields } = require('./fix-manifest-semver-v3.js');

const REPO = path.resolve(__dirname, '..');
const ENV_MAP = [
  ['OLARES_USER_HUGGINGFACE_TOKEN', 'HF_TOKEN'],
  ['OLARES_USER_HUGGINGFACE_SERVICE', 'HF_ENDPOINT'],
  ['OLARES_USER_GITHUB_TOKEN', 'GITHUB_TOKEN'],
  ['OLARES_USER_GITHUB_USERNAME', 'GHCR_USER'],
];

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
    result.push(line);
  }
  return result.join('\n');
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
  return [chart.name || path.basename(path.dirname(chartPath)), String(chart.version)];
}

function updateChartVersion(chartPath, newVersion) {
  if (!fs.existsSync(chartPath)) return;
  const raw = fs.readFileSync(chartPath, 'utf8').replace(/^apiVersion: v2\n(?=apiVersion: v2\n)/, '');
  const chart = yaml.load(raw);
  chart.version = newVersion;
  if (chart.appVersion !== undefined) chart.appVersion = newVersion;
  fs.writeFileSync(chartPath, yaml.dump(chart, { lineWidth: -1, noRefs: true }));
}

function fixOlaresUserRefs(content) {
  let out = content;
  for (const [oldName, newName] of ENV_MAP) {
    out = out.replaceAll(`.Values.olaresEnv.${oldName}`, `.Values.olaresEnv.${newName}`);
    out = out.replaceAll(`olaresEnv.${oldName}`, `olaresEnv.${newName}`);
    out = out.replaceAll(oldName, newName);
    out = out.replace(new RegExp(`^\\s+${newName}:`, 'gm'), `  ${newName}:`);
  }
  return out;
}

function isChartYamlFile(filePath) {
  if (filePath.endsWith('OlaresManifest.yaml')) return false;
  if (filePath.includes('/i18n/')) return false;
  return filePath.includes('/templates/') || filePath.endsWith('values.yaml');
}

function chartHasOlaresUser(appDir) {
  return scanYamlFiles(appDir).some((f) => isChartYamlFile(f) && fs.readFileSync(f, 'utf8').includes('OLARES_USER'));
}

function scrubChartOlaresUser(appDir) {
  for (const f of scanYamlFiles(appDir)) {
    if (!isChartYamlFile(f)) continue;
    const content = fs.readFileSync(f, 'utf8');
    if (!content.includes('OLARES_USER')) continue;
    fs.writeFileSync(f, fixOlaresUserRefs(content));
  }
}

function stripAdminGuard(content) {
  if (!/^\{\{-?\s*if\s+and\s+\.Values\.admin/.test(content)) return content;
  const lines = content.split('\n');
  let depth = 0;
  let started = false;
  const out = [];
  for (const line of lines) {
    const t = line.trim();
    if (!started && /^\{\{-?\s*if\s+and\s+\.Values\.admin/.test(t)) {
      started = true;
      depth = 1;
      continue;
    }
    if (!started) { out.push(line); continue; }
    if (/^\{\{-?\s*if\b/.test(t)) depth++;
    else if (/^\{\{-?\s*end\b/.test(t)) {
      depth--;
      if (depth === 0) continue;
    }
    if (depth > 0) out.push(line);
  }
  return out.join('\n').trimEnd() + '\n';
}

function findServerChart(appDir, manifest) {
  const subs = manifest?.spec?.subCharts || [];
  const shared = subs.find((s) => s.shared);
  const srvDirs = fs.readdirSync(appDir, { withFileTypes: true })
    .filter((e) => e.isDirectory() && e.name.endsWith('srv'))
    .map((e) => e.name);
  if (shared) {
    if (fs.existsSync(path.join(appDir, shared.name, 'Chart.yaml'))) return shared.name;
    if (srvDirs.length === 1) return srvDirs[0];
    const match = srvDirs.find((d) => shared.name.startsWith(d) || d.startsWith(shared.name.slice(0, 20)));
    if (match) return match;
  }
  return srvDirs[0] || null;
}

function deploymentNameFromYaml(content) {
  const m = content.match(/kind:\s*Deployment[\s\S]*?^\s*name:\s*(\S+)/m);
  if (m) return m[1].replace(/['"]/g, '');
  try {
    const docs = yaml.loadAll(stripHelm(content));
    for (const doc of docs) {
      if (doc?.kind === 'Deployment' && doc.metadata?.name) return doc.metadata.name;
    }
  } catch {
    // helm-heavy deployment templates may not parse as YAML
  }
  return null;
}

function proxyBackendFromClient(content) {
  const m = content.match(/proxy_pass\s+http:\/\/([^;]+);/);
  if (!m) return null;
  const host = m[1].trim();
  if (host.includes('-shared')) {
    const parts = host.split('.');
    const svc = parts[0];
    const port = parts[parts.length - 1].includes(':') ? parts[parts.length - 1].split(':').pop() : '8080';
    return { serverSvc: svc, port };
  }
  const [svc, port] = host.includes(':') ? host.split(':') : [host, '8080'];
  return { serverSvc: svc, port };
}

function clientCliName(manifest, appName) {
  const ent = (manifest.entrances || []).find((e) => String(e.host || '').endsWith('cli') || String(e.name || '').endsWith('cli'));
  if (ent?.host) return ent.host;
  if (ent?.name) return ent.name;
  return `${appName}cli`;
}

function fixClientProxy(content, appName, serverDeploy, port, cliName) {
  let out = fixOlaresUserRefs(content);
  out = out.replace(
    /proxy_pass\s+http:\/\/[^;]+;/,
    `proxy_pass http://${serverDeploy}:${port};`,
  );
  // deployment metadata.name → cliName
  const lines = out.split('\n');
  let inDeploy = false;
  let inMeta = false;
  for (let i = 0; i < lines.length; i++) {
    if (lines[i].trim() === 'kind: Deployment') { inDeploy = true; inMeta = false; continue; }
    if (inDeploy && lines[i].startsWith('metadata:')) { inMeta = true; continue; }
    if (inDeploy && inMeta && /^  name:\s/.test(lines[i])) {
      lines[i] = `  name: ${cliName}`;
      break;
    }
  }
  out = lines.join('\n');
  out = out.replace(
    /spec:\n  replicas: \d+/,
    `spec:\n  replicas: {{ .Values.workloads.${cliName}.replicaCount }}`,
  );
  if (!out.includes(`workloads.${cliName}`)) {
    out = out.replace(/spec:\n  replicas: 1/, `spec:\n  replicas: {{ .Values.workloads.${cliName}.replicaCount }}`);
  }
  return out;
}

function fixServerReplicas(content, serverDeploy) {
  let out = content;
  if (!out.includes('workloads.')) {
    out = out.replace(
      /(kind: Deployment[\s\S]*?name: ${serverDeploy}[\s\S]*?spec:\n)  replicas: \d+/m,
      `$1  replicas: {{ .Values.workloads.${serverDeploy}.replicaCount }}`,
    );
    out = out.replace(
      /(kind: Deployment[\s\S]*?spec:\n)  replicas: 1/m,
      `$1  replicas: {{ .Values.workloads.${serverDeploy}.replicaCount }}`,
    );
  }
  return out;
}

function buildAccelerator(spec) {
  const hasGpu = spec.requiredGpu || spec.limitedGpu;
  if (hasGpu) {
    return [{
      mode: 'nvidia',
      requiredCpu: String(spec.requiredCpu || '1'),
      limitedCpu: String(spec.limitedCpu || '8'),
      requiredMemory: spec.requiredMemory || '4Gi',
      limitedMemory: spec.limitedMemory || '16Gi',
      requiredDisk: spec.requiredDisk || '10Gi',
      limitedDisk: spec.limitedDisk || '50Gi',
      requiredGPUMemory: spec.requiredGpu || '1Gi',
      limitedGPUMemory: spec.limitedGpu || '24Gi',
    }];
  }
  return [{
    mode: 'cpu',
    requiredCpu: String(spec.requiredCpu || '100m'),
    limitedCpu: String(spec.limitedCpu || '2'),
    requiredMemory: spec.requiredMemory || '256Mi',
    limitedMemory: spec.limitedMemory || '2Gi',
    requiredDisk: spec.requiredDisk || '1Gi',
    limitedDisk: spec.limitedDisk || '20Gi',
  }];
}

function fixManifestEnvs(envs) {
  const out = [];
  const seen = new Set();
  for (const e of envs || []) {
    let env = { ...e };
    if (env.envName === 'OLARES_USER_HUGGINGFACE_TOKEN') {
      env = { ...env, envName: 'HF_TOKEN' };
    }
    if (env.envName === 'OLARES_USER_HUGGINGFACE_SERVICE') {
      env = { ...env, envName: 'HF_ENDPOINT' };
    }
    if (seen.has(env.envName)) continue;
    seen.add(env.envName);
    out.push(env);
  }
  // ensure HF_TOKEN if HF download app
  if (!seen.has('HF_TOKEN') && envs?.some((e) => e.envName?.includes('HUGGINGFACE'))) {
    out.push({
      envName: 'HF_TOKEN',
      required: false,
      applyOnChange: true,
      valueFrom: { envName: 'OLARES_USER_HUGGINGFACE_TOKEN' },
    });
  }
  return out;
}

function bumpManifestVersion(raw, newVersion) {
  let out = raw;
  out = out.replace(/^metadata:\n([\s\S]*?)^  version:.*$/m, (block) => block.replace(/^  version:.*$/m, `  version: ${newVersion}`));
  out = out.replace(/^  versionName:.*$/m, `  versionName: ${newVersion}`);
  if (!out.includes('v3 shared') && !out.includes('OLARES_USER')) {
    const note = `    v${newVersion}: scrub OLARES_USER_* from chart templates (v3 upload rule).\n`;
    out = out.replace(/(  upgradeDescription: \|\n)/, `$1${note}`);
  }
  return out;
}

function rewriteManifest(raw, appName, serverDeploy, cliName, newVersion, hasGpu) {
  const stripped = stripHelm(raw);
  const doc = yaml.load(stripped) || {};
  const meta = doc.metadata || {};
  const spec = { ...(doc.spec || {}) };
  const resourceSpec = { ...spec };

  delete spec.subCharts;
  delete spec.requiredCpu;
  delete spec.limitedCpu;
  delete spec.requiredMemory;
  delete spec.limitedMemory;
  delete spec.requiredDisk;
  delete spec.limitedDisk;
  delete spec.requiredGpu;
  delete spec.limitedGpu;

  spec.onlyAdmin = true;
  spec.accelerator = buildAccelerator(resourceSpec);

  spec.versionName = newVersion;
  if (!spec.upgradeDescription?.includes('v3')) {
    const note = `v${newVersion}: migrate to apiVersion v3 shared chart (Olares 1.12.6+). Flatten server+client; workloadReplicas; no OLARES_USER_* in templates.\n`;
    spec.upgradeDescription = note + (spec.upgradeDescription || '');
  }

  delete meta.appid;

  const sharedEntrances = (doc.sharedEntrances || []).map((se) => ({
    ...se,
    name: se.name === `${appName}api` || se.name?.endsWith('api') ? appName : se.name,
  }));

  const options = {
    shared: true,
    apiTimeout: doc.options?.apiTimeout ?? 0,
    dependencies: [{
      name: 'olares',
      version: '>=1.12.6-0',
      type: 'system',
    }],
  };
  if (hasGpu && (doc.envs || []).some((e) => ['LLM_CONTEXT_WINDOW', 'LLM_API_KEY'].includes(e.envName))) {
    options.LLMGatewaySupported = true;
  }
  if (doc.options?.conflicts) options.conflicts = doc.options.conflicts;
  if (doc.options?.allowedOutboundPorts) options.allowedOutboundPorts = doc.options.allowedOutboundPorts;

  const out = {
    'olaresManifest.version': '0.12.0',
    'olaresManifest.type': doc['olaresManifest.type'] || 'app',
    apiVersion: 'v3',
    workloadReplicas: {
      [serverDeploy]: 1,
      [cliName]: 1,
    },
    metadata: {
      ...meta,
      name: appName,
      version: newVersion,
    },
    sharedEntrances,
    entrances: doc.entrances,
    spec,
    permission: {
      appData: doc.permission?.appData ?? true,
      appCache: doc.permission?.appCache,
      appCommon: doc.permission?.appCommon,
      externalData: doc.permission?.externalData,
      userData: doc.permission?.userData,
    },
    envs: fixManifestEnvs(doc.envs),
    options,
  };

  // clean undefined permission keys
  Object.keys(out.permission).forEach((k) => out.permission[k] === undefined && delete out.permission[k]);

  let yamlOut = yaml.dump(out, { lineWidth: -1, noRefs: true, quotingType: "'", forceQuotes: false });
  yamlOut = normalizeSemverFields(yamlOut);
  return `---\n${yamlOut}`;
}

function mergeValues(appDir, appName, serverDeploy, cliName, existing) {
  const valuesPath = path.join(appDir, 'values.yaml');
  let base = {};
  if (fs.existsSync(valuesPath)) {
    try { base = yaml.load(fs.readFileSync(valuesPath, 'utf8')) || {}; } catch { base = {}; }
  }
  const olaresEnv = { ...(base.olaresEnv || {}) };
  for (const [oldName, newName] of ENV_MAP) {
    if (olaresEnv[oldName] !== undefined) {
      olaresEnv[newName] = olaresEnv[oldName];
      delete olaresEnv[oldName];
    }
    if (olaresEnv[newName] === undefined) olaresEnv[newName] = '';
  }
  const workloads = {
    [serverDeploy]: { replicaCount: 1 },
    [cliName]: { replicaCount: 1 },
  };
  const out = { ...base, workloads, olaresEnv };
  fs.writeFileSync(valuesPath, yaml.dump(out, { lineWidth: -1, noRefs: true }));
}

function removeStaleSubchartTemplates(appDir) {
  for (const entry of fs.readdirSync(appDir, { withFileTypes: true })) {
    if (!entry.isDirectory()) continue;
    if (entry.name === 'templates' || entry.name === 'i18n' || entry.name === 'docker' || entry.name === 'app') continue;
    const tpl = path.join(appDir, entry.name, 'templates');
    if (!fs.existsSync(tpl)) continue;
    for (const f of fs.readdirSync(tpl)) {
      if (f.endsWith('.yaml') || f.endsWith('.yml')) {
        fs.unlinkSync(path.join(tpl, f));
      }
    }
  }
}

function stripChartDependencies(chartPath) {
  if (!fs.existsSync(chartPath)) return;
  const raw = fs.readFileSync(chartPath, 'utf8').replace(/^apiVersion: v2\n(?=apiVersion: v2\n)/, '');
  const chart = yaml.load(raw);
  if (chart.dependencies) delete chart.dependencies;
  fs.writeFileSync(chartPath, yaml.dump(chart, { lineWidth: -1, noRefs: true }));
}

function scanYamlFiles(dir) {
  const files = [];
  if (!fs.existsSync(dir)) return files;
  for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
    const p = path.join(dir, entry.name);
    if (entry.isDirectory()) files.push(...scanYamlFiles(p));
    else if (entry.name.endsWith('.yaml') || entry.name.endsWith('.yml')) files.push(p);
  }
  return files;
}

function migrateApp(appName, { dryRun = false } = {}) {
  const appDir = path.join(REPO, appName);
  const manifestPath = path.join(appDir, 'OlaresManifest.yaml');
  if (!fs.existsSync(manifestPath)) return { appName, status: 'skip', reason: 'no manifest' };

  const rawManifest = fs.readFileSync(manifestPath, 'utf8');
  const isV3 = /apiVersion:\s*'?v3'?/.test(rawManifest);
  const hasSubCharts = rawManifest.includes('subCharts:');
  const needsScrub = chartHasOlaresUser(appDir);

  if (isV3 && !hasSubCharts && !needsScrub) {
    return { appName, status: 'skip', reason: 'already v3 clean' };
  }

  if (isV3 && !hasSubCharts && needsScrub) {
    const [, oldVersion] = readChartVersion(path.join(appDir, 'Chart.yaml'));
    const newVersion = bumpPatch(oldVersion);
    const doc = yaml.load(rawManifest) || {};
    const wr = doc.workloadReplicas || {};
    const keys = Object.keys(wr);
    const serverDeploy = keys.find((k) => !k.endsWith('cli')) || appName;
    const cliName = keys.find((k) => k.endsWith('cli')) || `${appName}cli`;
    if (dryRun) return { appName, status: 'dry-run', newVersion, mode: 'scrub-v3' };
    removeStaleSubchartTemplates(appDir);
    scrubChartOlaresUser(appDir);
    fs.writeFileSync(manifestPath, bumpManifestVersion(rawManifest, newVersion));
    mergeValues(appDir, appName, serverDeploy, cliName);
    updateChartVersion(path.join(appDir, 'Chart.yaml'), newVersion);
    stripChartDependencies(path.join(appDir, 'Chart.yaml'));
    if (chartHasOlaresUser(appDir)) {
      return { appName, status: 'error', reason: 'OLARES_USER still in chart files after scrub' };
    }
    return { appName, status: 'ok', newVersion, mode: 'scrub-v3' };
  }

  const manifest = yaml.load(stripHelm(rawManifest)) || {};
  if (!manifest.spec?.subCharts?.length) {
    return { appName, status: 'skip', reason: 'not shared split' };
  }

  const serverChart = findServerChart(appDir, manifest);
  if (!serverChart) return { appName, status: 'skip', reason: 'no server chart' };

  const srvDeployPath = path.join(appDir, serverChart, 'templates', 'deployment.yaml');
  const rootServerPath = path.join(appDir, 'templates', 'server.yaml');
  const deploySrc = fs.existsSync(rootServerPath) ? rootServerPath : srvDeployPath;
  if (!fs.existsSync(deploySrc)) return { appName, status: 'skip', reason: 'no deployment' };

  const serverDeploy = deploymentNameFromYaml(fs.readFileSync(deploySrc, 'utf8'));
  if (!serverDeploy) return { appName, status: 'skip', reason: 'no deployment name' };

  const clientPath = path.join(appDir, 'templates', 'clientproxy.yaml');
  if (!fs.existsSync(clientPath)) return { appName, status: 'skip', reason: 'no clientproxy' };

  const clientRaw = fs.readFileSync(clientPath, 'utf8');
  const backend = proxyBackendFromClient(clientRaw);
  const port = backend?.port || '8080';
  const cliName = clientCliName(manifest, appName);
  const hasGpu = Boolean(manifest.spec?.requiredGpu || manifest.spec?.limitedGpu);

  const [, oldVersion] = readChartVersion(path.join(appDir, 'Chart.yaml'));
  const newVersion = bumpPatch(oldVersion);

  if (dryRun) return { appName, status: 'dry-run', newVersion, serverDeploy, cliName };

  const templatesDir = path.join(appDir, 'templates');
  fs.mkdirSync(templatesDir, { recursive: true });

  // server
  let serverContent = fs.readFileSync(deploySrc, 'utf8');
  serverContent = stripAdminGuard(serverContent);
  serverContent = fixOlaresUserRefs(serverContent);
  serverContent = fixServerReplicas(serverContent, serverDeploy);
  fs.writeFileSync(rootServerPath, serverContent);

  const srvGhcr = path.join(appDir, serverChart, 'templates', 'ghcr-pull-secret.yaml');
  const rootGhcr = path.join(templatesDir, 'ghcr-pull-secret.yaml');
  if (fs.existsSync(srvGhcr) || fs.existsSync(rootGhcr)) {
    const src = fs.existsSync(srvGhcr) ? srvGhcr : rootGhcr;
    let ghcr = stripAdminGuard(fs.readFileSync(src, 'utf8'));
    ghcr = fixOlaresUserRefs(ghcr);
    fs.writeFileSync(rootGhcr, ghcr);
  }

  // client
  let clientFixed = fixClientProxy(clientRaw, appName, serverDeploy, port, cliName);
  fs.writeFileSync(clientPath, clientFixed);

  // fix any other templates (configmap etc.)
  for (const f of scanYamlFiles(path.join(appDir, 'templates'))) {
    if (f.endsWith('server.yaml') || f.endsWith('clientproxy.yaml')) continue;
    const fixed = fixOlaresUserRefs(fs.readFileSync(f, 'utf8'));
    if (fixed !== fs.readFileSync(f, 'utf8')) fs.writeFileSync(f, fixed);
  }

  removeStaleSubchartTemplates(appDir);

  const newManifest = rewriteManifest(rawManifest, appName, serverDeploy, cliName, newVersion, hasGpu);
  fs.writeFileSync(manifestPath, newManifest);

  mergeValues(appDir, appName, serverDeploy, cliName);
  updateChartVersion(path.join(appDir, 'Chart.yaml'), newVersion);
  stripChartDependencies(path.join(appDir, 'Chart.yaml'));
  updateChartVersion(path.join(appDir, serverChart, 'Chart.yaml'), newVersion);
  const clientChart = path.join(appDir, path.basename(appDir), 'Chart.yaml');
  updateChartVersion(clientChart, newVersion);

  scrubChartOlaresUser(appDir);

  if (chartHasOlaresUser(appDir)) {
    return { appName, status: 'error', reason: 'OLARES_USER still in chart files after migrate' };
  }

  return { appName, status: 'ok', newVersion, serverDeploy, cliName, mode: 'full-v3' };
}

const dryRun = process.argv.includes('--dry-run');
const only = process.argv.find((a) => a.startsWith('--app='))?.split('=')[1];

const apps = fs.readdirSync(REPO, { withFileTypes: true })
  .filter((e) => e.isDirectory())
  .map((e) => e.name)
  .filter((name) => fs.existsSync(path.join(REPO, name, 'OlaresManifest.yaml')))
  .filter((name) => !only || name === only)
  .sort();

const results = [];
for (const app of apps) {
  try {
    results.push(migrateApp(app, { dryRun }));
  } catch (err) {
    results.push({ appName: app, status: 'error', reason: err.message });
  }
}

for (const r of results) {
  if (r.status === 'ok' || r.status === 'dry-run') {
    console.log(`${r.status === 'ok' ? 'migrated' : 'would migrate'} ${r.appName} → v${r.newVersion}${r.serverDeploy ? ` (${r.serverDeploy} + ${r.cliName})` : r.mode ? ` [${r.mode}]` : ''}`);
  } else if (r.status === 'skip') {
    if (only) console.log(`SKIP ${r.appName}: ${r.reason}`);
  } else if (r.status === 'error') {
    console.error(`ERROR ${r.appName}: ${r.reason}`);
  }
}

const ok = results.filter((r) => r.status === 'ok').length;
const skipped = results.filter((r) => r.status === 'skip').length;
const errors = results.filter((r) => r.status === 'error').length;
console.log(`\nDone: ${ok} migrated, ${skipped} skipped, ${errors} errors`);
