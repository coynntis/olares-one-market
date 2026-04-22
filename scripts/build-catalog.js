const fs = require('fs');
const path = require('path');
const crypto = require('crypto');
const yaml = require('js-yaml');

// Scan app charts from this repo's root
const APPS_REPO = path.resolve(__dirname, '..');
const OUTPUT = path.resolve(__dirname, '../src/catalog.json');

// --- Helpers ---

function generateAppId(name) {
  return crypto.createHash('md5').update(name).digest('hex').substring(0, 8);
}

function parseCpu(value) {
  if (!value) return '0';
  const str = String(value);
  if (str.endsWith('m')) return String(parseInt(str) / 1000);
  return str;
}

function parseBytes(value) {
  if (!value) return '0';
  const str = String(value);
  const units = { Ki: 1024, Mi: 1048576, Gi: 1073741824, Ti: 1099511627776 };
  for (const [suffix, mult] of Object.entries(units)) {
    if (str.endsWith(suffix)) return String(parseInt(str) * mult);
  }
  return str;
}

// Strip Helm template directives from YAML.
// Keeps the "if" branch (admin), removes "else" branch (user proxy).
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
    // Remove inline template expressions
    result.push(line.replace(/\{\{.*?\}\}/g, ''));
  }
  return result.join('\n');
}

// --- Read i18n locales ---

function readI18n(appDir) {
  const i18n = {};
  const i18nDir = path.join(appDir, 'i18n');
  if (!fs.existsSync(i18nDir)) return i18n;

  for (const locale of fs.readdirSync(i18nDir, { withFileTypes: true })) {
    if (!locale.isDirectory()) continue;
    const manifestPath = path.join(i18nDir, locale.name, 'OlaresManifest.yaml');
    if (!fs.existsSync(manifestPath)) continue;
    try {
      const raw = fs.readFileSync(manifestPath, 'utf8');
      i18n[locale.name] = yaml.load(stripHelmTemplates(raw));
    } catch (e) {
      console.warn(`  Warning: failed to parse i18n/${locale.name}: ${e.message}`);
    }
  }
  return i18n;
}

// --- Scan all app directories ---

function scanApps() {
  const entries = fs.readdirSync(APPS_REPO, { withFileTypes: true });
  const apps = {};

  for (const entry of entries) {
    if (!entry.isDirectory()) continue;
    if (entry.name.startsWith('.')) continue;

    const appDir = path.join(APPS_REPO, entry.name);
    const chartPath = path.join(appDir, 'Chart.yaml');
    const manifestPath = path.join(appDir, 'OlaresManifest.yaml');

    if (!fs.existsSync(chartPath) || !fs.existsSync(manifestPath)) continue;

    console.log(`Processing: ${entry.name}`);

    let chart, manifest;
    try {
      chart = yaml.load(fs.readFileSync(chartPath, 'utf8'));
      const rawManifest = fs.readFileSync(manifestPath, 'utf8');
      manifest = yaml.load(stripHelmTemplates(rawManifest));
    } catch (e) {
      console.warn(`  Skipping ${entry.name}: ${e.message}`);
      continue;
    }

    const meta = manifest.metadata || {};
    const spec = manifest.spec || {};
    const appName = chart.name || meta.name || entry.name;
    const appId = generateAppId(appName);
    const i18n = readI18n(appDir);
    const categories = meta.categories || [];

    // Simplified entry for /api/v1/appstore/info
    const summary = {
      id: appId,
      name: appName,
      version: meta.version || chart.version,
      category: categories[0] || '',
      description: meta.description || '',
      icon: meta.icon || '',
      screenshots: null,
      tags: null,
      metadata: null,
      source: 1,
      updated_at: new Date().toISOString(),
    };

    // Full entry for /api/v1/applications/info
    const detail = {
      id: appId,
      name: appName,
      cfgType: manifest['olaresManifest.type'] || 'app',
      chartName: `${appName}-${chart.version}.tgz`,
      icon: meta.icon || '',
      description: meta.description || '',
      appID: appId,
      title: meta.title || appName,
      version: meta.version || chart.version,
      categories,
      versionName: spec.versionName || chart.appVersion || meta.version || '',
      fullDescription: spec.fullDescription || meta.description || '',
      upgradeDescription: spec.upgradeDescription || '',
      promoteImage: spec.promoteImage || null,
      promoteVideo: spec.promoteVideo || '',
      subCategory: spec.subCategory || '',
      locale: Object.keys(i18n).length > 0
        ? ['en-US', ...Object.keys(i18n).filter(l => l !== 'en-US')]
        : spec.locale || ['en-US'],
      developer: spec.developer || '',
      requiredMemory: parseBytes(spec.requiredMemory),
      requiredDisk: parseBytes(spec.requiredDisk),
      supportClient: spec.supportClient || {},
      supportArch: spec.supportArch || [],
      requiredGPU: parseBytes(spec.requiredGpu),
      requiredCPU: parseCpu(spec.requiredCpu),
      rating: 0,
      target: spec.target || '',
      permission: manifest.permission || {},
      entrances: (manifest.entrances || []).map(e => ({
        name: e.name || '',
        host: e.host || '',
        port: e.port || 0,
        title: e.title || '',
        icon: e.icon || '',
        authLevel: e.authLevel || 'private',
        invisible: e.invisible || false,
        openMethod: e.openMethod || '',
        disablePreload: e.disablePreload || false,
      })),
      middleware: manifest.middleware || null,
      options: manifest.options || {},
      submitter: spec.submitter || 'orales-market',
      doc: spec.doc || '',
      website: spec.website || '',
      featuredImage: spec.featuredImage || '',
      sourceCode: spec.sourceCode || '',
      license: spec.license || [],
      legal: spec.legal || null,
      i18n: Object.fromEntries(
        Object.entries(i18n).map(([locale, lm]) => [locale, {
          metadata: {
            title: (lm.metadata || {}).title || '',
            description: (lm.metadata || {}).description || '',
          },
          entrances: null,
          spec: {
            fullDescription: (lm.spec || {}).fullDescription || '',
            upgradeDescription: (lm.spec || {}).upgradeDescription || '',
          },
        }])
      ),
      namespace: '',
      onlyAdmin: spec.onlyAdmin || false,
      lastCommitHash: '',
      createTime: 0,
      updateTime: 0,
      count: null,
      versionHistory: [{
        appName: appName,
        version: meta.version || chart.version,
        versionName: chart.appVersion || '',
        mergedAt: new Date().toISOString(),
        upgradeDescription: '',
      }],
      screenshots: null,
      tags: null,
      metadata: null,
      updated_at: new Date().toISOString(),
    };

    apps[appId] = { summary, detail };
    console.log(`  -> ${appName} (${appId}) v${summary.version}`);
  }

  return apps;
}

// --- Build charts.json from charts/ directory ---

function buildCharts() {
  const chartsDir = path.resolve(__dirname, '../charts');
  const chartsOutput = path.resolve(__dirname, '../src/charts.json');
  const charts = {};

  if (fs.existsSync(chartsDir)) {
    for (const file of fs.readdirSync(chartsDir)) {
      if (!file.endsWith('.tgz')) continue;
      charts[file] = fs.readFileSync(path.join(chartsDir, file)).toString('base64');
      console.log(`Chart: ${file} (${Math.round(fs.statSync(path.join(chartsDir, file)).size / 1024)}KB)`);
    }
  }

  const newContent = JSON.stringify(charts);
  let existing = '';
  try { existing = fs.readFileSync(chartsOutput, 'utf8'); } catch {}
  if (newContent !== existing) {
    fs.writeFileSync(chartsOutput, newContent);
    console.log(`Charts written to ${chartsOutput}`);
  } else {
    console.log('Charts unchanged, skipping write.');
  }
  console.log();
  return charts;
}

// --- Build icons.json from icons/ directory ---

function buildIcons() {
  const iconsDir = path.resolve(__dirname, '../icons');
  const iconsOutput = path.resolve(__dirname, '../src/icons.json');
  const icons = {};

  if (fs.existsSync(iconsDir)) {
    for (const file of fs.readdirSync(iconsDir)) {
      if (!file.endsWith('.png')) continue;
      const name = file.replace(/\.png$/, '');
      icons[name] = fs.readFileSync(path.join(iconsDir, file)).toString('base64');
      console.log(`Icon: ${name} (${Math.round(fs.statSync(path.join(iconsDir, file)).size / 1024)}KB)`);
    }
  }

  const newContent = JSON.stringify(icons, null, 2);
  let existing = '';
  try { existing = fs.readFileSync(iconsOutput, 'utf8'); } catch {}
  if (newContent !== existing) {
    fs.writeFileSync(iconsOutput, newContent);
    console.log(`Icons written to ${iconsOutput}`);
  } else {
    console.log('Icons unchanged, skipping write.');
  }
  console.log();
  return icons;
}

// --- Main ---

console.log('Building catalog from', APPS_REPO);
console.log();

buildCharts();
buildIcons();
const apps = scanApps();

const summaries = {};
const details = {};
const latest = [];

for (const [id, app] of Object.entries(apps)) {
  summaries[id] = app.summary;
  details[id] = app.detail;
  latest.push(app.summary.name);
}

// Sort latest: newest apps first (by version, higher = newer)
latest.sort((a, b) => {
  const va = apps[Object.keys(apps).find(k => apps[k].summary.name === a)]?.summary?.version || '0';
  const vb = apps[Object.keys(apps).find(k => apps[k].summary.name === b)]?.summary?.version || '0';
  // Apps at v1.0.0 are newest (just created), higher patch = more updates
  const pa = va.split('.').map(Number);
  const pb = vb.split('.').map(Number);
  // Lower version = newer app (v1.0.0 just created vs v1.0.27 old)
  return (pa[0]*10000+pa[1]*100+pa[2]) - (pb[0]*10000+pb[1]*100+pb[2]);
});

// Deterministic hash based on app content only (no timestamps)
const catalogPayload = JSON.stringify({ summaries, details, latest });
const hash = crypto.createHash('md5').update(catalogPayload).digest('hex');

const catalog = { hash, summaries, details, latest };
const newContent = JSON.stringify(catalog, null, 2);

// Only write if content actually changed (avoids infinite wrangler rebuild loop)
let existingContent = '';
try { existingContent = fs.readFileSync(OUTPUT, 'utf8'); } catch {}

if (newContent === existingContent) {
  console.log('\nCatalog unchanged, skipping write.');
} else {
  fs.mkdirSync(path.dirname(OUTPUT), { recursive: true });
  fs.writeFileSync(OUTPUT, newContent);
  console.log(`\nCatalog written to ${OUTPUT}`);
}

console.log(`Apps: ${Object.keys(summaries).length}, Hash: ${hash}`);
