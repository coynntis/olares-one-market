#!/usr/bin/env node
/**
 * Mount Common/comfyui/model/llms and reuse GGUFs there (official ComfyUI
 * migrates External/olares/ai/model/llms → that tree, often a subdir).
 * Skip redownload via symlink. Idempotent — safe to re-run.
 */
const fs = require('fs');
const path = require('path');

const ROOT = path.resolve(__dirname, '..');

const HELPER = `              find_comfy_llms() {
                local name="$1" src
                [ -n "$name" ] || return 1
                [ -d /comfyui-llms ] || return 1
                src="$(find /comfyui-llms -maxdepth 4 -type f -name "$name" 2>/dev/null | head -n 1)"
                [ -n "$src" ] && [ -f "$src" ] || return 1
                printf '%s\\n' "$src"
              }
              link_from_comfy_llms() {
                local dest="$1" name src
                name="$(basename "$dest")"
                [ -n "$name" ] || return 1
                src="$(find_comfy_llms "$name")" || return 1
                if [ -f "$dest" ] && [ ! -L "$dest" ]; then
                  return 1
                fi
                echo "[comfy-reuse] $src → $dest (symlink, skip redownload)" >&2
                rm -f "$dest"
                ln -sfn "$src" "$dest"
                return 0
              }
`;

const HELPER_RE =
  /              find_comfy_llms\(\) \{\n(?:.*\n)*?              \}\n              link_from_comfy_llms\(\) \{\n(?:.*\n)*?                return 0\n              \}\n/g;

function bumpPatch(ver) {
  const m = String(ver).match(/^(\d+)\.(\d+)\.(\d+)/);
  if (!m) return null;
  return `${m[1]}.${m[2]}.${Number(m[3]) + 1}`;
}

function patchValues(appDir) {
  const p = path.join(appDir, 'values.yaml');
  if (!fs.existsSync(p)) return false;
  let t = fs.readFileSync(p, 'utf8');
  if (/^\s+appCommon:/m.test(t)) return false;
  if (/^userspace:/m.test(t)) {
    t = t.replace(/(userspace:\n(?:  \w+:.*\n)*)/, (block) => {
      if (/appCommon:/.test(block)) return block;
      return block.replace(/\n$/, '\n  appCommon: \'\'\n');
    });
  } else {
    t = `userspace:\n  appData: ''\n  appCache: ''\n  userData: ''\n  appCommon: ''\n` + t;
  }
  fs.writeFileSync(p, t);
  return true;
}

function patchManifest(appDir, newVer) {
  const p = path.join(appDir, 'OlaresManifest.yaml');
  if (!fs.existsSync(p)) return;
  let t = fs.readFileSync(p, 'utf8');
  if (!/permission:\n/.test(t)) return;
  if (!/appCommon:\s*true/.test(t)) {
    t = t.replace(/permission:\n((?:  .+\n)*)/, (m, body) => {
      if (/appCommon:/.test(body)) return m;
      return `permission:\n${body}  appCommon: true\n`;
    });
  }
  if (newVer) {
    t = t.replace(/^(\s+version:\s+)[\d.]+/m, `$1${newVer}`);
    t = t.replace(/^(\s+versionName:\s+)[\d.]+/m, `$1${newVer}`);
    if (/upgradeDescription:\s*\|/.test(t) && !t.includes('ComfyUI Common GGUF reuse')) {
      t = t.replace(
        /(upgradeDescription:\s*\|\n)/,
        `$1    v${newVer}: reuse GGUFs already in Common/comfyui/model/llms (and subdirs) via symlink — official ComfyUI migrates External/ai/model/llms there; skip redownload.\n`
      );
    }
  }
  fs.writeFileSync(p, t);
}

function patchChartYaml(appDir) {
  const p = path.join(appDir, 'Chart.yaml');
  if (!fs.existsSync(p)) return null;
  let t = fs.readFileSync(p, 'utf8');
  const m = t.match(/^version:\s*([\d.]+)/m);
  if (!m) return null;
  const next = bumpPatch(m[1]);
  if (!next) return null;
  t = t.replace(/^version:\s*.+$/m, `version: ${next}`);
  t = t.replace(/^appVersion:\s*.+$/m, `appVersion: "${next}"`);
  fs.writeFileSync(p, t);
  return next;
}

function ensureVolume(yaml) {
  if (yaml.includes('name: comfyui-llms')) return yaml;
  const vol = `        - name: comfyui-llms
          hostPath:
            path: "{{ .Values.userspace.appCommon }}/comfyui/model/llms"
            type: DirectoryOrCreate
`;
  const re =
    /(        - name: shared-models\n          hostPath:\n            path: "\/olares\/share\/ai\/model"\n            type: DirectoryOrCreate\n)/;
  if (!re.test(yaml)) return yaml;
  return yaml.replace(re, `$1${vol}`);
}

function ensureMounts(yaml) {
  yaml = yaml.replace(
    /(            - mountPath: "\/shared-models"\n              name: shared-models\n)(?!            - mountPath: "\/comfyui-llms")/g,
    `$1            - mountPath: "/comfyui-llms"\n              name: comfyui-llms\n`
  );
  // Downloaders that only mount /models (e.g. TQ3_4S) still need ComfyUI tree.
  yaml = yaml.replace(
    /(            - mountPath: "\/models"\n              name: models\n)(?!            - mountPath:)/g,
    `$1            - mountPath: "/comfyui-llms"\n              name: comfyui-llms\n`
  );
  return yaml;
}

function replaceHelpers(yaml) {
  if (!HELPER_RE.test(yaml) && !yaml.includes('find_comfy_llms()')) {
    return yaml;
  }
  HELPER_RE.lastIndex = 0;
  yaml = yaml.replace(HELPER_RE, HELPER);
  while (yaml.includes(HELPER + HELPER)) {
    yaml = yaml.replace(HELPER + HELPER, HELPER);
  }
  return yaml;
}

function injectHelper(yaml) {
  if (yaml.includes('find_comfy_llms()')) return yaml;
  const patterns = [
    /(              SHARED_DIR="\/shared-models\/llms"\n              MODELS_DIR="\/models"\n              mkdir -p "\$SHARED_DIR" "\$MODELS_DIR"\n)/,
    /(              SHARED_DIR=\$\{SHARED_DIR:-\/shared-models\/llms\}\n              mkdir -p "\$MODELS_DIR" "\$SHARED_DIR"\n)/,
    /(              mkdir -p "\$SHARED_DIR" "\$MODELS_DIR"\n)/,
    /(              mkdir -p "\$SHARED" "\$MODELS_DIR"\n)/,
    /(              mkdir -p "\$MODELS_DIR" "\$SHARED_DIR"\n)/,
  ];
  let out = yaml;
  for (const re of patterns) {
    if (re.test(out)) {
      out = out.replace(re, (m) => m + HELPER);
      break;
    }
  }
  return out;
}

function injectHelperBeforeModelPath(yaml) {
  const call = yaml.indexOf('link_from_comfy_llms "$MODEL_PATH"');
  if (call < 0) return yaml;
  const def = yaml.indexOf('find_comfy_llms()');
  if (def >= 0 && def < call) return yaml;
  return yaml.replace(
    /(              set -eu\n\n              MODEL_PATH="\$SHARED_DIR\/\$\{MODEL_FILE\}"\n)/,
    `              set -eu
              SHARED_DIR="\${SHARED_DIR:-/models}"
              mkdir -p "$SHARED_DIR"
${HELPER}              MODEL_PATH="$SHARED_DIR/\${MODEL_FILE}"
`
  );
}

function injectDownloadHooks(yaml) {
  if (yaml.includes('download() {') && !yaml.includes('link_from_comfy_llms "$dest"')) {
    yaml = yaml.replace(
      /download\(\) \{\n((?:[^\n]*\n)*?)(                if file_ok "\$dest")/,
      (m, body, ifok) => {
        if (body.includes('link_from_comfy_llms')) return m;
        return `download() {\n${body}                if type link_from_comfy_llms >/dev/null 2>&1; then
                  if link_from_comfy_llms "$dest"; then
                    if file_ok "$dest" "$min_bytes" "$exact_bytes"; then
                      log "$label OK via ComfyUI Common — skip download"
                      return 0
                    fi
                    rm -f "$dest"
                  fi
                fi
${ifok}`;
      }
    );
  }
  if (yaml.includes('download_one() {') && !yaml.includes('link_from_comfy_llms "$dest"')) {
    yaml = yaml.replace(
      /(                dest="\$SHARED_DIR\/\$name"\n)/g,
      `$1                if type link_from_comfy_llms >/dev/null 2>&1; then
                  if link_from_comfy_llms "$dest"; then
                    if file_ok "$dest" "$min_bytes" "$exact_bytes"; then
                      log "ok $name via ComfyUI Common — skip download"
                      rm -f "\${dest}.part" "\${dest}.tmp" "\${dest}.tmp.tmp" 2>/dev/null || true
                      return 0
                    fi
                    rm -f "$dest"
                  fi
                fi
`
    );
  }
  if (yaml.includes('MODEL_PATH="$SHARED_DIR/${MODEL_FILE}"') && !yaml.includes('link_from_comfy_llms "$MODEL_PATH"')) {
    yaml = yaml.replace(
      /(              MODEL_PATH="\$SHARED_DIR\/\$\{MODEL_FILE\}"\n)/g,
      `$1              if type link_from_comfy_llms >/dev/null 2>&1; then
                link_from_comfy_llms "$MODEL_PATH" || true
              fi
`
    );
  }
  if (yaml.includes('MMPROJ_PATH="$SHARED_DIR/${MMPROJ_FILE}"') && !yaml.includes('link_from_comfy_llms "$MMPROJ_PATH"')) {
    yaml = yaml.replace(
      /(              MMPROJ_PATH="\$SHARED_DIR\/\$\{MMPROJ_FILE\}"\n)/g,
      `$1              if type link_from_comfy_llms >/dev/null 2>&1; then
                link_from_comfy_llms "$MMPROJ_PATH" || true
              fi
`
    );
  }
  return yaml;
}

function injectOkHooks(yaml) {
  if (!yaml.includes('find_comfy_llms()')) return yaml;
  if (!yaml.includes('link_from_comfy_llms "$SHARED_DIR/$TARGET_FILE"')) {
    yaml = yaml.replace(
      /(              if \[ ! -f "\$SHARED_DIR\/\$TARGET_FILE\.ok" \]; then\n)/,
      `              if type link_from_comfy_llms >/dev/null 2>&1; then
                if link_from_comfy_llms "$SHARED_DIR/$TARGET_FILE"; then
                  if [ -s "$SHARED_DIR/$TARGET_FILE" ]; then
                    echo "[comfy-reuse] $TARGET_FILE present — stamp .ok" >&2
                    if [ -n "\${REV:-}" ]; then
                      echo "$REV" > "$SHARED_DIR/$TARGET_FILE.ok"
                    else
                      touch "$SHARED_DIR/$TARGET_FILE.ok"
                    fi
                  fi
                fi
              fi
$1`
    );
  }
  if (yaml.includes('$DRAFT_FILE.ok') && !yaml.includes('link_from_comfy_llms "$SHARED_DIR/$DRAFT_FILE"')) {
    yaml = yaml.replace(
      /(              if \[ ! -f "\$SHARED_DIR\/\$DRAFT_FILE\.ok" \]; then\n)/,
      `              if type link_from_comfy_llms >/dev/null 2>&1; then
                if link_from_comfy_llms "$SHARED_DIR/$DRAFT_FILE"; then
                  if [ -s "$SHARED_DIR/$DRAFT_FILE" ]; then
                    echo "[comfy-reuse] $DRAFT_FILE present — stamp .ok" >&2
                    touch "$SHARED_DIR/$DRAFT_FILE.ok"
                  fi
                fi
              fi
$1`
    );
  }
  return yaml;
}

function patchColibri(yaml) {
  const broken = `              if [ ! -f "$INDEX" ]; then
                for d in /comfyui-llms/$MODEL_NAME /comfyui-llms/*/$MODEL_NAME; do
                  if [ -f "$d/model.safetensors.index.json" ]; then
                    echo "reuse ComfyUI Common checkpoint dir $d → $MODEL_DIR"
                    rm -rf "$MODEL_DIR"
                    ln -sfn "$d" "$MODEL_DIR"
                    break
                  fi
                done
              fi
              INDEX="\${MODEL_DIR}/model.safetensors.index.json"`;
  const fixed = `              INDEX="\${MODEL_DIR}/model.safetensors.index.json"
              if [ ! -f "$INDEX" ]; then
                for d in /comfyui-llms/$MODEL_NAME /comfyui-llms/*/$MODEL_NAME /comfyui-llms/*/*/$MODEL_NAME; do
                  if [ -f "$d/model.safetensors.index.json" ]; then
                    echo "reuse ComfyUI Common checkpoint dir $d → $MODEL_DIR"
                    rm -rf "$MODEL_DIR"
                    ln -sfn "$d" "$MODEL_DIR"
                    INDEX="\${MODEL_DIR}/model.safetensors.index.json"
                    break
                  fi
                done
              fi`;
  if (yaml.includes(broken)) return yaml.replace(broken, fixed);
  if (yaml.includes('MODEL_DIR_NAME') && !yaml.includes('reuse ComfyUI Common checkpoint dir')) {
    const needle = `              INDEX="\${MODEL_DIR}/model.safetensors.index.json"`;
    if (yaml.includes(needle)) {
      return yaml.replace(
        needle,
        `              INDEX="\${MODEL_DIR}/model.safetensors.index.json"
              if [ ! -f "$INDEX" ]; then
                for d in /comfyui-llms/$MODEL_NAME /comfyui-llms/*/$MODEL_NAME /comfyui-llms/*/*/$MODEL_NAME; do
                  if [ -f "$d/model.safetensors.index.json" ]; then
                    echo "reuse ComfyUI Common checkpoint dir $d → $MODEL_DIR"
                    rm -rf "$MODEL_DIR"
                    ln -sfn "$d" "$MODEL_DIR"
                    INDEX="\${MODEL_DIR}/model.safetensors.index.json"
                    break
                  fi
                done
              fi`
      );
    }
  }
  return yaml;
}

function isLlmChart(dir) {
  const name = path.basename(dir);
  return (
    name.startsWith('llamacpp') ||
    name === 'gemma4e2bone' ||
    name === 'qwen36a3bvisionone' ||
    name === 'colibridsv4flash0731one'
  );
}

function main() {
  const apps = fs
    .readdirSync(ROOT, { withFileTypes: true })
    .filter((d) => d.isDirectory() && isLlmChart(path.join(ROOT, d.name)))
    .map((d) => d.name)
    .sort()
    .map((name) => path.join(ROOT, name));

  const changed = [];
  const skipped = [];
  for (const appDir of apps) {
    const server = path.join(appDir, 'templates', 'server.yaml');
    if (!fs.existsSync(server)) continue;
    let yaml = fs.readFileSync(server, 'utf8');
    if (!yaml.includes('/olares/share/ai/model')) {
      skipped.push(path.basename(appDir) + ' (no shared-models)');
      continue;
    }
    const orig = yaml;
    yaml = ensureVolume(yaml);
    yaml = ensureMounts(yaml);
    yaml = replaceHelpers(yaml);
    yaml = injectHelper(yaml);
    yaml = injectHelperBeforeModelPath(yaml);
    yaml = injectDownloadHooks(yaml);
    yaml = injectOkHooks(yaml);
    yaml = patchColibri(yaml);
    yaml = replaceHelpers(yaml);

    const alreadyBumped = fs
      .readFileSync(path.join(appDir, 'OlaresManifest.yaml'), 'utf8')
      .includes('ComfyUI Common GGUF reuse');

    if (yaml !== orig) {
      fs.writeFileSync(server, yaml);
    }
    patchValues(appDir);
    if (!alreadyBumped) {
      const ver = patchChartYaml(appDir);
      patchManifest(appDir, ver);
      changed.push(`${path.basename(appDir)} ${ver || ''} (bumped)`);
    } else {
      patchManifest(appDir, null);
      changed.push(`${path.basename(appDir)} (fixed, same ver)`);
    }
  }
  console.log(`patched ${changed.length} LLM charts:`);
  for (const c of changed) console.log(' ', c);
  if (skipped.length) {
    console.log('skipped:', skipped.join(', '));
  }
}

main();
