# Build on Olares One → push ghcr.io

## 1. On Olares One (native amd64 + CUDA)

Copy chart dir or clone repo, then:

```bash
export GHCR_TOKEN=ghp_your_pat_with_write_packages
chmod +x locateanything3bone/scripts/build-and-push-ghcr.sh
./locateanything3bone/scripts/build-and-push-ghcr.sh
```

Uses `docker` or `nerdctl`. MagiAttention compile ~30–60 min first time.

**Olares One (Blackwell):** install script sets `MAGI_ATTENTION_PREBUILD_FFA=0`, `MAGI_ATTENTION_SKIP_MAGI_ATTN_COMM_BUILD=1` (no InfiniBand/RDMA headers needed for single GPU), and `MAX_JOBS=4` by default. Lower if OOM: `MAGI_MAX_JOBS=2`. If `dockerbuilderone` pod restarts mid-build, check `kubectl describe pod` for `OOMKilled`.

Skip MTP (faster):

```bash
INSTALL_MAGIATTENTION=0 ./locateanything3bone/scripts/build-and-push-ghcr.sh
```

Custom tag:

```bash
IMAGE=ghcr.io/coynntis/locate-anything:0.1.2 ./locateanything3bone/scripts/build-and-push-ghcr.sh
```

## 2. Make image pullable on Olares

- GitHub → Packages → `locateanything3bone` → **Public**, or
- Add `imagePullSecrets` for private ghcr

## 3. Install chart

**Market default (v1.0.5+)** — pre-built ghcr image:

```yaml
image:
  repository: ghcr.io/coynntis/locate-anything
  tag: "0.1.2"
deps:
  bootstrapOnDevice: false
```

**On-device bootstrap** (no ghcr pull; ~30–60 min first start):

```yaml
image:
  repository: pytorch/pytorch
  tag: "2.12.0-cuda13.0-cudnn9-devel"
deps:
  bootstrapOnDevice: true
```

## Mac

Do not cross-build here. Use Olares One script above or any native linux/amd64 GPU host.
