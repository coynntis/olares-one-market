# LingBot chart bootstrap lessons (from Krea / vLLM soft-ready)

Battle-tested on Olares One. Apply to every LingBot chart.

## Helm ConfigMap (must)

1. Never put raw `{{` / `}}` in files injected into `templates/configmap.yaml` — Helm parses them as templates (`bad character U+002D '-'` on CSS like `body{{font-family…}}`).
2. Soft-ready HTML: plain string + single CSS braces (see Krea), not Python f-string brace-doubling.
3. Inject escape (two-phase placeholder) in `inject-lingbot-configmaps.js` as safety net.

## Soft-ready (must)

1. Bind `:7860` with **stdlib-only** `soft_ready.py` **before** uv/pip/HF download.
2. Olares install gate ~30m — soft-ready keeps TCP healthy while deps install.
3. `GET /health` returns JSON `{status, ready, phase, detail, elapsed_s, attempts}`.
4. Phase file: `/workspace/<app>/.boot-phase` + append-only `/workspace/<app>/bootstrap.log`.
5. Stop soft-ready, free port, then `exec` real Gradio/FastAPI on same port.

## uv + target site-packages (must)

1. Bootstrap `uv` into `/workspace/<app>/bin/uv` (persist binary + `uv-cache`).
2. Install with `uv pip install --python "$BASE_PY" --target "$SITE_PKGS"`.
3. Never venv that shadows image torch. Image owns `torch`/`torchvision`.
4. After every uv install: **purge** `$SITE_PKGS/torch*` / `nvidia*` / `triton*` (Krea v1.0.18).
5. Deps marker: `.deps-ok-v1-<req_hash>` — skip reinstall when imports OK and no torch shadow.
6. Optional `UV_INDEX_URL`; auto Tsinghua when `HF_ENDPOINT` looks like China HF mirror.

## Model download progress (must)

1. Prefer `huggingface_hub.snapshot_download` with `tqdm` + log every file.
2. Set `HF_HOME=/models/huggingface`, `HF_HUB_ENABLE_HF_TRANSFER=0`, prefer Xet if available.
3. Record attempts: timestamp, repo_id, bytes, elapsed, error → `bootstrap.log` + phase detail.
4. Resume-safe local dir under `/models/<app>/…`; skip when marker + size OK.

## Probes

- `startupProbe` TCP `:7860` — long `failureThreshold` (first boot can be hours for World/Video).
- Soft-ready satisfies probe during install; real app must bind same port after swap.

## Per-model VRAM notes (5090M 24GB)

| App | Strategy |
|-----|----------|
| depth / vision / map | Full GPU, tiny |
| video dense 1.3B | BF16 GPU; text encoder CPU if needed |
| VLA 6B | BF16 GPU |
| VA | Official: UMT5+VAE CPU offload (~18–24GB) |
| World NF4 | T5 CPU + NF4 DiT; official ~32GB — sequential / layer offload for 24GB |
