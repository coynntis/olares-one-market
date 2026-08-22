#!/usr/bin/env node
/**
 * Split single-chart apps into Olares shared server + client proxy subcharts.
 * Names capped at 30 chars (srv/cli suffixes).
 */
const fs = require('fs');
const path = require('path');
const yaml = require('js-yaml');

const REPO = path.resolve(__dirname, '..');
const MAX_LEN = 30;
const SERVER_SUFFIX = 'srv';
const CLIENT_SUFFIX = 'cli';
const ADMIN_GUARD = '{{- if and .Values.admin .Values.bfl.username (eq .Values.admin .Values.bfl.username) }}';
const ADMIN_GUARD_END = '{{- end }}';

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

function fitSuffix(base, suffix) {
  const allowed = MAX_LEN - suffix.length;
  return `${String(base).slice(0, allowed)}${suffix}`;
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

function hasAdminGuard(content) {
  return content.includes('.Values.admin .Values.bfl.username');
}

function wrapAdminGuard(content) {
  const trimmed = content.replace(/^\s+/, '');
  if (!trimmed || hasAdminGuard(content)) return content;
  return `${ADMIN_GUARD}\n${trimmed.replace(/\s*$/, '\n')}${ADMIN_GUARD_END}\n`;
}

function parseServices(deployContent) {
  const stripped = stripHelm(deployContent);
  const docs = yaml.loadAll(stripped).filter(Boolean);
  let primary = null;
  let sharedHost = null;
  for (const doc of docs) {
    if (doc?.kind !== 'Service') continue;
    const name = doc.metadata?.name;
    if (!name) continue;
    if (name.startsWith('sharedentrances-')) {
      sharedHost = name;
      continue;
    }
    const port = doc.spec?.ports?.[0]?.port;
    const targetPort = doc.spec?.ports?.[0]?.targetPort ?? port;
    if (!primary) {
      primary = {
        name,
        port,
        targetPort,
        selector: doc.spec?.selector?.['io.kompose.service'] || name,
      };
    }
  }
  return { primary, sharedHost };
}

function clientProxyYaml({ appName, clientSvc, serverChart, backendSvc, backendPort }) {
  const upstream = `http://${backendSvc}.${serverChart}-shared:${backendPort}`;
  return `---
apiVersion: v1
data:
  nginx.conf: |
    server {

      listen 8080;
      server_name _;
      access_log /opt/bitnami/openresty/nginx/logs/access.log;
      error_log  /opt/bitnami/openresty/nginx/logs/error.log;

      proxy_connect_timeout                          600s;
      proxy_send_timeout                             600s;
      proxy_read_timeout                             1800s;
      proxy_buffering off;
      proxy_cache off;
      chunked_transfer_encoding on;
      proxy_set_header      host                      $host;
      proxy_set_header      x-forwarded-host          $http_host;

      proxy_http_version 1.1;

      proxy_set_header upgrade $http_upgrade;
      proxy_set_header connection "upgrade";

      location / {
        add_header X-Frame-Options "";
        proxy_pass ${upstream};
      }
    }

kind: ConfigMap
metadata:
  name: nginx-config
  namespace: {{ .Release.Namespace }}

---
apiVersion: apps/v1
kind: Deployment
metadata:
  labels:
    io.kompose.service: ${clientSvc}
  name: ${appName}
  namespace: '{{ .Release.Namespace }}'
spec:
  replicas: 1
  selector:
    matchLabels:
      io.kompose.service: ${clientSvc}
  template:
    metadata:
      labels:
        io.kompose.network/chrome-default: "true"
        io.kompose.service: ${clientSvc}
    spec:
      volumes:
        - name: nginx-config
          configMap:
            name: nginx-config
            defaultMode: 438
            items:
              - key: nginx.conf
                path: nginx.conf
      containers:
        - name: nginx
          image: "docker.io/beclab/aboveos-bitnami-openresty:1.25.3-2"
          ports:
            - containerPort: 8080
              protocol: TCP
          startupProbe:
            tcpSocket:
              port: 8080
            failureThreshold: 30
            periodSeconds: 10
          resources:
            limits:
              cpu: 500m
              memory: 500Mi
            requests:
              cpu: 10m
              memory: 64Mi
          volumeMounts:
            - name: nginx-config
              mountPath: /opt/bitnami/openresty/nginx/conf/server_blocks/nginx.conf
              subPath: nginx.conf

---
apiVersion: v1
kind: Service
metadata:
  name: ${clientSvc}
  namespace: {{ .Release.Namespace }}
spec:
  type: ClusterIP
  selector:
    io.kompose.service: ${clientSvc}
  ports:
    - name: ${clientSvc}
      protocol: TCP
      port: 8080
      targetPort: 8080
`;
}

function ensureSharedServiceInDeploy(deployContent, primary) {
  if (!primary) return deployContent;
  const sharedName = `sharedentrances-${primary.name}`;
  if (deployContent.includes(`name: ${sharedName}`)) return deployContent;
  const block = `---
apiVersion: v1
kind: Service
metadata:
  labels:
    io.kompose.service: ${primary.selector}
  name: ${sharedName}
  namespace: "{{ .Release.Namespace }}"
spec:
  ports:
    - name: "${primary.name}"
      port: 80
      targetPort: ${primary.targetPort}
  selector:
    io.kompose.service: ${primary.selector}
`;
  return `${deployContent.replace(/\s*$/, '\n')}${block}`;
}

function patchManifest(raw, {
  appName, serverChart, clientSvc, clientChart, sharedHost, primaryEntranceName, newVersion, oldVersion,
}) {
  let out = bumpVersionsInRaw(raw, oldVersion, newVersion);

  const sharedHostVal = sharedHost || `sharedentrances-${primaryEntranceName || appName}`;
  out = out.replace(
    /(sharedEntrances:\n(?:  - [^\n]+\n)*?    host:\s*)(['"]?)[^'"\n]+\2/g,
    `$1'${sharedHostVal}'`,
  );
  out = out.replace(
    /(sharedEntrances:\n(?:  - [^\n]+\n)*?    host:[^\n]+\n    port:\s*)(['"]?)[^'"\n]+\2/g,
    `$10`,
  );

  // Primary UI entrance -> client proxy
  const entMatch = out.match(/^entrances:\n((?:  - .*\n(?:    .*\n)*)+)/m);
  if (entMatch) {
    const block = entMatch[1];
    const first = block.match(/^  - name:\s*([^\n]+)\n((?:    [^\n]+\n)*)/m);
    if (first) {
      const rest = first[2]
        .replace(/^    port:\s*.*$/m, '    port: 8080')
        .replace(/^    host:\s*.*$/m, `    host: ${clientSvc}`);
      const replacement = `  - name: ${clientSvc}\n${rest}`;
      out = out.replace(block, block.replace(first[0], replacement));
    }
  }

  // provider.entrance -> client entrance
  out = out.replace(/^(\s+entrance:\s*)(['"]?)[^'"\n]+\2(?=\n\s+paths:)/gm, `$1'${clientSvc}'`);

  if (!/^  subCharts:/m.test(out)) {
    const subBlock = `  subCharts:\n  - name: ${serverChart}\n    shared: true\n  - name: ${clientChart}\n`;
    if (/  supportArch:\n(?:    - .*\n)+/m.test(out)) {
      out = out.replace(/(  supportArch:\n(?:    - .*\n)+)/m, `$1\n${subBlock}`);
    } else {
      out = out.replace(/^permission:/m, `${subBlock}permission:`);
    }
  } else {
    out = out.replace(
      /^  subCharts:\n(?:  - name: [^\n]+\n(?:    shared: true\n)?)+/m,
      `  subCharts:\n  - name: ${serverChart}\n    shared: true\n  - name: ${clientChart}\n`,
    );
  }

  const appScopeBlock = `  appScope:
  ${ADMIN_GUARD}
    clusterScoped: true
    appRef:
      - ${appName}
  {{- else }}
    clusterScoped: false
  ${ADMIN_GUARD_END}`;

  if (/^\s+appScope:/m.test(out)) {
    out = out.replace(/^  appScope:\n(?:  .*\n)+?(?=  dependencies:)/m, `${appScopeBlock}\n`);
  } else if (/^options:/m.test(out)) {
    out = out.replace(/^options:\n/m, `options:\n${appScopeBlock}\n`);
  }

  const depTail = `  {{- if and .Values.admin .Values.bfl.username (eq .Values.admin .Values.bfl.username) }}
  {{- else }}
    - name: ${appName}
      type: application
      version: '>=1.0.0'
      mandatory: true
  {{- end }}`;

  if (/mandatory: true/.test(out) && out.includes('type: application')) {
    // already has non-admin app dependency
  } else {
    out = out.replace(
      /(  dependencies:\n(?:    - .*\n(?:      .*\n)*)+)/m,
      `$1${depTail}\n`,
    );
  }

  return out;
}

function removeOldSubcharts(appDir, keepServer, keepClient) {
  for (const entry of fs.readdirSync(appDir, { withFileTypes: true })) {
    if (!entry.isDirectory()) continue;
    if (entry.name === 'templates' || entry.name === 'i18n' || entry.name === 'docker' || entry.name === 'files' || entry.name === 'scripts' || entry.name === 'app') continue;
    const chartPath = path.join(appDir, entry.name, 'Chart.yaml');
    if (!fs.existsSync(chartPath)) continue;
    if (entry.name === keepServer || entry.name === keepClient) continue;
    fs.rmSync(path.join(appDir, entry.name), { recursive: true, force: true });
  }
}

function writeChartYaml(chartDir, name, description, version) {
  fs.mkdirSync(path.join(chartDir, 'templates'), { recursive: true });
  fs.writeFileSync(path.join(chartDir, 'Chart.yaml'), yaml.dump({
    apiVersion: 'v2',
    appVersion: version,
    description,
    name,
    type: 'application',
    version,
  }, { lineWidth: -1, noRefs: true }));
  const valuesPath = path.join(chartDir, 'values.yaml');
  if (!fs.existsSync(valuesPath)) fs.writeFileSync(valuesPath, 'admin: ""\n');
}

let updated = 0;
let skipped = 0;
const nameReport = [];

for (const entry of fs.readdirSync(REPO, { withFileTypes: true }).filter((e) => e.isDirectory())) {
  const appDir = path.join(REPO, entry.name);
  const manifestPath = path.join(appDir, 'OlaresManifest.yaml');
  const chartPath = path.join(appDir, 'Chart.yaml');
  const templatesDir = path.join(appDir, 'templates');
  if (!fs.existsSync(manifestPath) || !fs.existsSync(chartPath)) continue;

  const rawManifest = fs.readFileSync(manifestPath, 'utf8');
  let manifest;
  try {
    manifest = yaml.load(stripHelm(rawManifest));
  } catch (e) {
    console.log(`skip ${entry.name}: manifest parse — ${e.message}`);
    skipped++;
    continue;
  }

  const appName = manifest.metadata?.name || entry.name;
  const serverChart = fitSuffix(appName, SERVER_SUFFIX);
  const clientChart = appName.length <= MAX_LEN ? appName : fitSuffix(appName, '');
  const clientSvc = fitSuffix(appName, CLIENT_SUFFIX);

  for (const [label, val] of [['server', serverChart], ['clientChart', clientChart], ['clientSvc', clientSvc], ['app', appName]]) {
    if (val.length > MAX_LEN) {
      console.log(`ABORT ${entry.name}: ${label}=${val} (${val.length})`);
      process.exit(1);
    }
  }

  const rootTemplates = fs.existsSync(templatesDir)
    ? fs.readdirSync(templatesDir).filter((f) => f.endsWith('.yaml') || f.endsWith('.yml'))
    : [];
  const serverDir = path.join(appDir, serverChart);
  const clientDir = path.join(appDir, clientChart);
  const alreadySplit = fs.existsSync(serverDir) && fs.existsSync(clientDir) && rootTemplates.length <= 1;

  if (alreadySplit && manifest.spec?.subCharts?.some((s) => s.shared)) {
    // Reconcile names only
    const oldVersion = String(manifest.metadata?.version || '1.0.0').replace(/['"]/g, '');
    const newVersion = bumpPatch(oldVersion);
    const primary = manifest.entrances?.[0];
    const { primary: svc } = parseServices(fs.readFileSync(path.join(serverDir, 'templates/deployment.yaml'), 'utf8'));
    const out = patchManifest(rawManifest, {
      appName,
      serverChart,
      clientSvc,
      clientChart,
      sharedHost: manifest.sharedEntrances?.[0]?.host,
      primaryEntranceName: svc?.name || primary?.host || appName,
      newVersion,
      oldVersion,
    });
    fs.writeFileSync(manifestPath, out);
    updateChartVersion(chartPath, newVersion);
    const clientProxyPath = path.join(clientDir, 'templates/clientproxy.yaml');
    if (fs.existsSync(clientProxyPath)) {
      fs.mkdirSync(templatesDir, { recursive: true });
      fs.writeFileSync(path.join(templatesDir, 'clientproxy.yaml'), fs.readFileSync(clientProxyPath, 'utf8'));
      const keep = path.join(templatesDir, 'keep');
      if (fs.existsSync(keep)) fs.unlinkSync(keep);
    }
    console.log(`reconcile ${entry.name}: srv=${serverChart} cli=${clientSvc} v${newVersion}`);
    nameReport.push({ app: entry.name, serverChart, clientSvc, len: serverChart.length });
    updated++;
    continue;
  }

  if (!fs.existsSync(path.join(templatesDir, 'deployment.yaml')) && !fs.existsSync(serverDir)) {
    console.log(`skip ${entry.name}: no deployment.yaml`);
    skipped++;
    continue;
  }

  const oldVersion = String(manifest.metadata?.version || '1.0.0').replace(/['"]/g, '');
  const newVersion = bumpPatch(oldVersion);

  removeOldSubcharts(appDir, serverChart, clientChart);

  const deploySrc = fs.existsSync(path.join(templatesDir, 'deployment.yaml'))
    ? path.join(templatesDir, 'deployment.yaml')
    : path.join(serverDir, 'templates/deployment.yaml');
  let deployContent = fs.readFileSync(deploySrc, 'utf8');
  const { primary, sharedHost } = parseServices(deployContent);
  if (!primary) {
    console.log(`skip ${entry.name}: no primary Service in deployment`);
    skipped++;
    continue;
  }

  deployContent = ensureSharedServiceInDeploy(deployContent, primary);
  deployContent = wrapAdminGuard(deployContent);

  writeChartYaml(serverDir, serverChart, `${appName} shared server`, newVersion);
  if (fs.existsSync(path.join(appDir, 'values.yaml'))) {
    fs.copyFileSync(path.join(appDir, 'values.yaml'), path.join(serverDir, 'values.yaml'));
  }

  const serverTemplates = path.join(serverDir, 'templates');
  fs.mkdirSync(serverTemplates, { recursive: true });
  fs.writeFileSync(path.join(serverTemplates, 'deployment.yaml'), deployContent);

  for (const file of rootTemplates) {
    if (file === 'deployment.yaml' || file === 'keep') continue;
    const src = path.join(templatesDir, file);
    let content = fs.readFileSync(src, 'utf8');
    content = wrapAdminGuard(content);
    fs.writeFileSync(path.join(serverTemplates, file), content);
    fs.unlinkSync(src);
  }

  writeChartYaml(clientDir, clientChart, `${appName} client proxy`, '1.0.0');
  fs.writeFileSync(path.join(clientDir, 'values.yaml'), 'admin: ""\n');
  fs.writeFileSync(
    path.join(clientDir, 'templates/clientproxy.yaml'),
    clientProxyYaml({
      appName,
      clientSvc,
      serverChart,
      backendSvc: primary.name,
      backendPort: primary.port,
    }),
  );

  fs.mkdirSync(templatesDir, { recursive: true });
  fs.writeFileSync(path.join(templatesDir, 'clientproxy.yaml'), fs.readFileSync(path.join(clientDir, 'templates/clientproxy.yaml'), 'utf8'));
  const keepPath = path.join(templatesDir, 'keep');
  if (fs.existsSync(keepPath)) fs.unlinkSync(keepPath);

  const out = patchManifest(rawManifest, {
    appName,
    serverChart,
    clientSvc,
    clientChart,
    sharedHost: sharedHost || manifest.sharedEntrances?.[0]?.host,
    primaryEntranceName: primary.name,
    newVersion,
    oldVersion,
  });
  fs.writeFileSync(manifestPath, out);
  updateChartVersion(chartPath, newVersion);

  console.log(`split ${entry.name}: srv=${serverChart} cli=${clientSvc} backend=${primary.name}:${primary.port} v${newVersion}`);
  nameReport.push({ app: entry.name, serverChart, clientSvc, len: serverChart.length });
  updated++;
}

console.log(`\nDone: ${updated} updated, ${skipped} skipped`);
const over = nameReport.filter((r) => r.serverChart.length > MAX_LEN || r.clientSvc.length > MAX_LEN);
if (over.length) {
  console.error('NAME LIMIT VIOLATIONS:', over);
  process.exit(1);
}
console.log('All names <=', MAX_LEN);
