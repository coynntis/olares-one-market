# Build SplatLab One image via dockerbuilderone MCP

SplatLab does **not** build its own image. Use **dockerbuilderone** on Olares from Cursor.

## 1. Prepare zip

**Recommended** — from repo:

```bash
./splatlabone/docker/build-zip.sh
```

Writes **`splatlabone/docker/splatlabone-docker.zip`** (and copies to `/tmp/splatlabone-docker.zip` for MCP upload). Syncs current `app/` + Dockerfile, entrypoint, requirements, `fetch_backends.py`.

Manual alternative from repo root:

```bash
cd splatlabone/docker
zip -r /tmp/splatlabone-docker.zip Dockerfile .dockerignore entrypoint.sh requirements.txt
# Copy app into docker context for image bake:
cp -r ../app/* .
zip -r /tmp/splatlabone-docker.zip .
```

Dockerfile expects `requirements.txt`, `entrypoint.sh`, and `app/` files at build context root. Easiest: use `build-zip.sh` (rsync app, then overlay docker files).

**Recommended zip layout** (Dockerfile at root):

```
Dockerfile
.dockerignore
entrypoint.sh
requirements.txt
main.py
mcp_server.py
deps.py
api/
jobs/
pipeline/
static/
presets/
```

## 2. Cursor + dockerbuilderone MCP

1. Install **dockerbuilderone** on Olares; configure MCP (`/mcp/mcp` on `dockerbuildermcp` entrance).
2. Base64-encode zip or use MCP `upload_project`.
3. Prompt example:

```
upload_project("splatlabone", <zip_base64>)
start_build("splatlabone", "ghcr.io/coynntis/splatlabone:1.0.0", "Dockerfile")
```

Poll `get_build_logs` until push completes.

## 3. Install splatlabone chart

- Set `OLARES_USER_GITHUB_TOKEN` (read:packages) for ghcr pull
- Chart image: `ghcr.io/coynntis/splatlabone:1.0.0`

## Troubleshooting

### `Signed-By` / `ubuntu2204` apt error on Kaniko

```
E: Conflicting values set for option Signed-By regarding source ... cuda/repos/ubuntu2204/ ...
```

Cause: old Dockerfile used `nvidia/cuda:*-ubuntu22.04`; duplicate CUDA apt drop-ins break `apt-get`.

Fix (current `Dockerfile`):

- Bases: `nvidia/cuda:12.8.1-devel-ubuntu24.04` + `12.8.1-runtime-ubuntu24.04`
- Before each `apt-get update`: `rm -f /etc/apt/sources.list.d/cuda*.list /etc/apt/sources.list.d/nvidia*.list`

Re-zip and rebuild. If log still shows `ubuntu2204`, uploaded zip is stale.

### `unknown system group 'messagebus' in statoverride file`

Cause: `nvidia/cuda:*-ubuntu24.04` minimal image ships stale `/var/lib/dpkg/statoverride` referencing `messagebus` before dbus packages create that group.

Fix (current `Dockerfile`): before each `apt-get update`:

```dockerfile
sed -i '/messagebus/d' /var/lib/dpkg/statoverride 2>/dev/null || true
```

Re-zip and rebuild.

### Kaniko log floods with `Hash components for file: /var/lib/dpkg/...`

Not an apt failure — Kaniko `--verbosity=debug` snapshots the layer after `RUN apt-get`. Apt often already succeeded (`statoverride` size 0, packages like `zlib1g-dev` listed).

Current `Dockerfile` mitigations:

- One builder `RUN` that compiles COLMAP/GLOMAP then **purges `-dev` packages** so the snapshotted layer is much smaller.
- gsplat built in **devel** stage; runtime only `COPY` site-packages (no `pip install gsplat` on runtime — no nvcc).

If build still dies here: scroll up for `error building image` / exit 137 (OOM) / `no space left on device`. Consider `KANIKO_VERBOSITY=info` on dockerbuilderone to cut log noise.

### `libcuda.so.1: cannot open shared object file` during Kaniko build

Cause: Kaniko has **no NVIDIA driver**. `pip install gsplat` runs `python` which loads torch → needs `libcuda.so.1`.

Fix (current `Dockerfile`):

- **python-builder**: symlink `libcuda.so` stub → `libcuda.so.1` before any pip; `LD_LIBRARY_PATH` includes `/usr/local/cuda/lib64/stubs`
- **final stage**: **no RUN at all** — only COPY (even `test`/`chmod`/`python` can fail under Kaniko)
- **COLMAP** copied to `/opt/colmap/` (not `/usr/local/lib`) so build layers stay clean
- CUDA/colmap/torch verified in `entrypoint.sh` when the GPU pod starts

**Kaniko layer cache** may replay an old Dockerfile layer. After updating the zip:

1. Re-upload project (`upload_project`) — do not rely on stale project folder
2. Bump image tag (e.g. `1.1.1`) or set `CACHEBUST` in Dockerfile
3. Optional: delete `appData/dockerbuilder/kaniko-cache` on Olares if build still fails

Re-zip: `./splatlabone/docker/build-zip.sh`

Re-zip and rebuild.

### Build dies during apt `Setting up ...` (tzdata, ca-certificates, libbsd)

**Not a normal apt failure.** `nvidia/cuda:*-devel` + Kaniko `--snapshot-mode=redo` hashes all of `/var/lib/dpkg` after `RUN apt-get`. Log stops mid-`Setting up`; exit 1/128.

Current `Dockerfile` uses **zero `apt-get`**:

- `pytorch/pytorch:2.11.0-cuda12.8-cudnn9-devel` — compile gsplat + pip deps (PEP 668: `PIP_BREAK_SYSTEM_PACKAGES=1`)
- `colmap/colmap:latest` — COLMAP binaries copied into `pytorch:...-runtime` final image

ffmpeg optional (video ingest warns if missing). Redeploy dockerbuilderone so Kaniko defaults are `snapshot-mode=time`, `verbosity=info`.

### Build dies during apt `Setting up ...` (openmpi, llvm, cmake-data)

Usually **not apt failing** — Kaniko `--snapshot-mode=redo` hashes entire `/var/lib/dpkg` after the layer. Log stops mid-`Setting up`; exit code 1 or 128.

Current mitigations:

- **No COLMAP compile** — copy `colmap/colmap:latest` (COLMAP 4.1 + `global_mapper`).
- Builder apt: python + libomp only (~10 packages, no `libflann-dev` / OpenMPI chain).
- dockerbuilderone: `KANIKO_SNAPSHOT_MODE=time`, `KANIKO_VERBOSITY=info`.

Look for `=== PYTHON BUILDER APT DONE ===` in logs to confirm apt finished.

### `kaniko failed with exit code 128`

Usually **git** fatal (clone blocked or failed on Olares). Current `Dockerfile` uses **curl tarballs only** — no `git clone`, GLOMAP `FETCH_COLMAP=OFF` / `FETCH_POSELIB=OFF` so CMake does not git-fetch.

Also dropped `libcgal-dev` (`-DCGAL_ENABLED=OFF`) to avoid pulling OpenMPI during apt.

### `externally-managed-environment` on `pip install`

PyTorch 2.11 images use **Ubuntu 24.04 system Python 3.12**, not a conda venv. First `pip` without `--break-system-packages` fails PEP 668.

Fix (current `Dockerfile`):

```dockerfile
ENV PIP_BREAK_SYSTEM_PACKAGES=1
RUN rm -f /usr/lib/python3.12/EXTERNALLY-MANAGED ...
RUN python -m pip install --break-system-packages ...
```

Final stage is `pytorch:2.11.0-cuda12.8-cudnn9-runtime` (Python included). Do **not** `COPY /opt/conda` — that path is stale on 2.11 tags.

If log still shows bare `pip install --upgrade pip` with no `--break-system-packages`, re-zip context (stale zip).

## Dev loop

| Change | Action |
|--------|--------|
| Python / static UI | ConfigMap inject (`npm run build:catalog`) — no image rebuild |
| Dockerfile / COLMAP / torch | Re-zip → dockerbuilderone rebuild |
