#!/usr/bin/env node
/**
 * Wire OLARES_USER_GITHUB_TOKEN → imagePullSecrets for any chart pulling ghcr.io images.
 */
'use strict';

const fs = require('fs');
const path = require('path');

const REPO = path.resolve(__dirname, '..');

const GHCR_MANIFEST_ENVS = `  - envName: GITHUB_TOKEN
    required: false
    applyOnChange: true
    valueFrom:
      envName: OLARES_USER_GITHUB_TOKEN
  - envName: GHCR_USER
    required: false
    applyOnChange: true
    valueFrom:
      envName: OLARES_USER_GITHUB_USERNAME`;

const VALUES_OLARES_GHCR = `  GITHUB_TOKEN: ""
  GHCR_USER: ""`;

function walk(dir, fn) {
  for (const ent of fs.readdirSync(dir, { withFileTypes: true })) {
    const p = path.join(dir, ent.name);
    if (ent.isDirectory()) walk(p, fn);
    else fn(p);
  }
}

function appNameFromDeploy(deployPath) {
  const rel = path.relative(REPO, deployPath);
  const top = rel.split(path.sep)[0];
  return top;
}

function srvTemplatesDir(deployPath) {
  return path.dirname(deployPath);
}

function ghcrSecretTemplate(appName, extraCondition = '') {
  const cond = extraCondition ? `\n{{- if ${extraCondition} }}` : '';
  const endExtra = extraCondition ? '\n{{- end }}' : '';
  return `{{- if and .Values.admin .Values.bfl.username (eq .Values.admin .Values.bfl.username) }}
{{- $token := .Values.olaresEnv.GITHUB_TOKEN | trim -}}
{{- $user := .Values.olaresEnv.GHCR_USER | trim -}}
{{- if and $token $user }}${cond}
{{- $auth := printf "%s:%s" $user $token | b64enc -}}
apiVersion: v1
kind: Secret
metadata:
  name: ${appName}-ghcr
  namespace: "{{ .Release.Namespace }}"
type: kubernetes.io/dockerconfigjson
data:
  .dockerconfigjson: {{ printf "{\\"auths\\":{\\"ghcr.io\\":{\\"username\\":\\"%s\\",\\"password\\":\\"%s\\",\\"auth\\":\\"%s\\"}}}" $user $token $auth | b64enc }}
{{- end }}${endExtra}
{{- end }}
`;
}

function pullSecretsBlock(appName, extraCondition = '') {
  const cond = extraCondition ? `\n      {{- if ${extraCondition} }}` : '';
  const endExtra = extraCondition ? '\n      {{- end }}' : '';
  return `      {{- $token := .Values.olaresEnv.GITHUB_TOKEN | trim -}}
      {{- if $token }}${cond}
      imagePullSecrets:
        - name: ${appName}-ghcr
      {{- end }}${endExtra}`;
}

function patchDeployment(deployPath, appName, extraCondition) {
  let d = fs.readFileSync(deployPath, 'utf8');
  if (!d.includes('ghcr.io/')) return false;

  const block = pullSecretsBlock(appName, extraCondition);
  if (d.includes(`name: ${appName}-ghcr`)) return false;

  // Insert at pod spec root, before initContainers/containers/volumes
  const re = /(    spec:\n)(      (?:initContainers|containers|volumes):)/;
  if (!re.test(d)) {
    console.warn('[add-ghcr-pull-secrets] skip deploy (no insert point):', deployPath);
    return false;
  }
  d = d.replace(re, `$1${block}\n$2`);
  fs.writeFileSync(deployPath, d);
  return true;
}

function patchGhcrSecretFile(templatesDir, appName, extraCondition) {
  const out = path.join(templatesDir, 'ghcr-pull-secret.yaml');
  if (fs.existsSync(out)) return false;
  const body = ghcrSecretTemplate(appName, extraCondition);
  fs.writeFileSync(out, body);
  return true;
}

function patchManifest(appDir) {
  const manifestPath = path.join(appDir, 'OlaresManifest.yaml');
  if (!fs.existsSync(manifestPath)) return false;
  let m = fs.readFileSync(manifestPath, 'utf8');
  if (m.includes('envName: GITHUB_TOKEN')) return false;
  if (!m.includes('envs:')) {
    m = m.replace(/(permission:\n  appData: true\n)/, `envs:\n${GHCR_MANIFEST_ENVS}\npermission:\n  appData: true\n`);
  } else {
    m = m.replace(/(envs:\n)/, `$1${GHCR_MANIFEST_ENVS}\n`);
  }
  fs.writeFileSync(manifestPath, m);
  return true;
}

function patchValues(appDir) {
  const valuesPath = path.join(appDir, 'values.yaml');
  if (!fs.existsSync(valuesPath)) return false;
  let v = fs.readFileSync(valuesPath, 'utf8');
  if (v.includes('GITHUB_TOKEN')) return false;
  if (/olaresEnv:\s*\{\}\s*/.test(v)) {
    v = v.replace(/olaresEnv:\s*\{\}\s*/, `olaresEnv:\n${VALUES_OLARES_GHCR}\n`);
  } else if (v.includes('olaresEnv:')) {
    v = v.replace(/(olaresEnv:\n)/, `$1${VALUES_OLARES_GHCR}\n`);
  } else {
    v += `\nolaresEnv:\n${VALUES_OLARES_GHCR}\n`;
  }
  fs.writeFileSync(valuesPath, v);
  return true;
}

function patchSrvValues(srvDir) {
  const valuesPath = path.join(srvDir, 'values.yaml');
  if (!fs.existsSync(valuesPath)) return false;
  let v = fs.readFileSync(valuesPath, 'utf8');
  if (v.includes('GITHUB_TOKEN')) return false;
  if (/olaresEnv:\s*\{\}\s*/.test(v)) {
    v = v.replace(/olaresEnv:\s*\{\}\s*/, `olaresEnv:\n${VALUES_OLARES_GHCR}\n`);
  } else if (v.includes('olaresEnv:')) {
    v = v.replace(/(olaresEnv:\n)/, `$1${VALUES_OLARES_GHCR}\n`);
  } else {
    v += `\nolaresEnv:\n${VALUES_OLARES_GHCR}\n`;
  }
  fs.writeFileSync(valuesPath, v);
  return true;
}

// Apps with ghcr.io container images (discovered from deployments)
const EXTRA_CONDITIONS = {
  locateanything3bone: 'not .Values.deps.bootstrapOnDevice',
};

const deployFiles = [];
walk(REPO, (file) => {
  if (!file.endsWith(path.join('templates', 'deployment.yaml'))) return;
  const content = fs.readFileSync(file, 'utf8');
  if (content.includes('ghcr.io/') && content.includes('image:')) {
    deployFiles.push(file);
  }
});

const apps = new Map();
for (const deployPath of deployFiles) {
  const appName = appNameFromDeploy(deployPath);
  if (!apps.has(appName)) apps.set(appName, []);
  apps.get(appName).push(deployPath);
}

let changed = 0;
for (const [appName, deploys] of apps) {
  const extra = EXTRA_CONDITIONS[appName] || '';
  const appDir = path.join(REPO, appName);
  for (const deployPath of deploys) {
    const templatesDir = srvTemplatesDir(deployPath);
    if (patchGhcrSecretFile(templatesDir, appName, extra)) {
      console.log(`[add-ghcr-pull-secrets] secret: ${path.relative(REPO, templatesDir)}/ghcr-pull-secret.yaml`);
      changed++;
    }
    if (patchDeployment(deployPath, appName, extra)) {
      console.log(`[add-ghcr-pull-secrets] deployment: ${path.relative(REPO, deployPath)}`);
      changed++;
    }
    const srvDir = path.dirname(templatesDir);
    if (srvDir.endsWith('srv') || srvDir.includes('srv')) {
      patchSrvValues(srvDir);
    }
  }
  if (patchManifest(appDir)) {
    console.log(`[add-ghcr-pull-secrets] manifest: ${appName}/OlaresManifest.yaml`);
    changed++;
  }
  if (patchValues(appDir)) {
    console.log(`[add-ghcr-pull-secrets] values: ${appName}/values.yaml`);
    changed++;
  }
}

console.log(`[add-ghcr-pull-secrets] done (${changed} file updates)`);
