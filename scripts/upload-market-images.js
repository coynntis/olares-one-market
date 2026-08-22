#!/usr/bin/env node
/**
 * Upload generated icons/featured PNGs to Olares CDN and patch OlaresManifest.yaml.
 * Reads scripts/apps-icons.json (from generate-app-icons.py).
 *
 * Env overrides:
 *   OLARES_UPLOAD_HOST, OLARES_UPLOAD_KEY, OLARES_UPLOAD_PICBED, OLARES_UPLOAD_CONFIG_NAME
 */

const fs = require('fs');
const path = require('path');
const crypto = require('crypto');

const ROOT = path.resolve(__dirname, '..');
const META_PATH = path.join(__dirname, 'apps-icons.json');
const ICONS_DIR = path.join(ROOT, 'icons');
const FEATURED_DIR = path.join(ROOT, 'featured');

const UPLOAD_HOST = process.env.OLARES_UPLOAD_HOST || 'https://test-pic-uploader.olares.com/upload';
const UPLOAD_KEY = process.env.OLARES_UPLOAD_KEY || 'Olares2024@';
const UPLOAD_PICBED = process.env.OLARES_UPLOAD_PICBED || 'aws-s3-plist';
const UPLOAD_CONFIG = process.env.OLARES_UPLOAD_CONFIG_NAME || 'Default';

const DRY_RUN = process.argv.includes('--dry-run');
const FORCE = process.argv.includes('--force');

function sha256File(filePath) {
  return crypto.createHash('sha256').update(fs.readFileSync(filePath)).digest('hex');
}

function uploadQuery() {
  return new URLSearchParams({
    key: UPLOAD_KEY,
    picbed: UPLOAD_PICBED,
    configName: UPLOAD_CONFIG,
  }).toString();
}

async function uploadFiles(filePaths) {
  if (!filePaths.length) return [];
  const form = new FormData();
  for (let i = 0; i < filePaths.length; i++) {
    const p = filePaths[i];
    const buf = fs.readFileSync(p);
    form.append(`file${i}`, new Blob([buf]), path.basename(p));
  }
  const url = `${UPLOAD_HOST}?${uploadQuery()}`;
  const res = await fetch(url, { method: 'POST', body: form });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`Upload HTTP ${res.status}: ${text.slice(0, 500)}`);
  }
  const data = await res.json();
  if (!data.success || !Array.isArray(data.result)) {
    throw new Error(`Upload failed: ${JSON.stringify(data).slice(0, 500)}`);
  }
  return data.result;
}

function patchManifestIconUrls(content, iconUrl) {
  return content.replace(/^(\s*icon:\s*).+$/gm, `$1${iconUrl}`);
}

function patchManifestFeatured(content, featuredUrl) {
  if (/^\s*featuredImage:\s*/m.test(content)) {
    return content.replace(/^(\s*featuredImage:\s*).+$/m, `$1${featuredUrl}`);
  }
  if (/^(\s*versionName:\s*.+)$/m.test(content)) {
    return content.replace(/^(\s*versionName:\s*.+)$/m, `$1\n  featuredImage: ${featuredUrl}`);
  }
  return content.replace(/^spec:\s*$/m, `spec:\n  featuredImage: ${featuredUrl}`);
}

function patchAppManifest(appId, iconUrl, featuredUrl) {
  const manifestPath = path.join(ROOT, appId, 'OlaresManifest.yaml');
  if (!fs.existsSync(manifestPath)) {
    console.warn(`  skip manifest: ${appId}/OlaresManifest.yaml not found`);
    return false;
  }
  let content = fs.readFileSync(manifestPath, 'utf8');
  const before = content;
  if (iconUrl) content = patchManifestIconUrls(content, iconUrl);
  if (featuredUrl) content = patchManifestFeatured(content, featuredUrl);
  if (content === before) return false;
  if (DRY_RUN) {
    console.log(`  [dry-run] would patch ${path.relative(ROOT, manifestPath)}`);
    return true;
  }
  fs.writeFileSync(manifestPath, content);
  console.log(`  patched ${path.relative(ROOT, manifestPath)}`);
  return true;
}

async function uploadFile(filePath) {
  const [url] = await uploadFiles([filePath]);
  return url;
}

async function processApp(row) {
  const appId = row.id;
  const iconPath = path.join(ICONS_DIR, `${appId}.png`);
  const featuredPath = path.join(FEATURED_DIR, `${appId}.png`);
  if (!fs.existsSync(iconPath) || !fs.existsSync(featuredPath)) {
    const miss = [
      !fs.existsSync(iconPath) && `icons/${appId}.png`,
      !fs.existsSync(featuredPath) && `featured/${appId}.png`,
    ].filter(Boolean);
    throw new Error(`${appId}: missing required market art: ${miss.join(', ')} (run npm run generate:icons — BOTH icon + featured required)`);
  }

  const iconHash = row.icon_hash || sha256File(iconPath);
  const featuredHash = row.featured_hash || sha256File(featuredPath);
  let iconUrl = row.icon_url || '';
  let featuredUrl = row.featured_url || '';

  if (iconUrl && featuredUrl && iconUrl === featuredUrl) {
    console.warn(`  warn ${appId}: featured_url === icon_url — will re-upload featured`);
  }

  const needIcon = FORCE || !iconUrl || row.icon_hash !== iconHash;
  const needFeatured = FORCE || !featuredUrl || row.featured_hash !== featuredHash
    || (iconUrl && featuredUrl === iconUrl);

  if (needIcon) {
    if (DRY_RUN) {
      console.log(`  [dry-run] would upload ${appId} icon`);
    } else {
      console.log(`  uploading ${appId} icon`);
      iconUrl = await uploadFile(iconPath);
    }
  }
  if (needFeatured) {
    if (DRY_RUN) {
      console.log(`  [dry-run] would upload ${appId} featured`);
    } else {
      console.log(`  uploading ${appId} featured`);
      featuredUrl = await uploadFile(featuredPath);
    }
  }
  if (!needIcon && !needFeatured) {
    console.log(`  skip upload ${appId}: hashes unchanged`);
  }

  patchAppManifest(appId, iconUrl, featuredUrl);

  return {
    ...row,
    icon_hash: iconHash,
    featured_hash: featuredHash,
    icon_url: iconUrl,
    featured_url: featuredUrl,
  };
}

async function main() {
  if (!fs.existsSync(META_PATH)) {
    console.error('Missing scripts/apps-icons.json — run: npm run generate:icons');
    process.exit(1);
  }

  const rows = JSON.parse(fs.readFileSync(META_PATH, 'utf8'));
  console.log(`Upload host: ${UPLOAD_HOST}`);
  console.log(`Apps: ${rows.length}${DRY_RUN ? ' (dry-run)' : ''}${FORCE ? ' (force)' : ''}\n`);

  const updated = [];
  for (const row of rows) {
    updated.push(await processApp(row));
  }

  if (!DRY_RUN) {
    fs.writeFileSync(META_PATH, JSON.stringify(updated, null, 2) + '\n');
    console.log(`\nUpdated ${path.relative(ROOT, META_PATH)}`);
  }
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
