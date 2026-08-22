# Ideogram 4 NF4 Docker image

Pre-builds Python deps so pod startup skips pip install.

```bash
docker build -t ghcr.io/<user>/ideogram4nf4one:1.0.0 \
  -f ideogram4nf4one/docker/Dockerfile \
  ideogram4nf4one
```

In `values.yaml`:

```yaml
image:
  repository: ghcr.io/<user>/ideogram4nf4one
  tag: "1.0.0"
```

Default chart uses `pytorch/pytorch:2.12.0-cuda13.0-cudnn9-devel` and installs deps at boot from ConfigMap.
