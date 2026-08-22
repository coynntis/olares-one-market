# SplatLab One

3D Gaussian Splatting pipeline for **Olares One** (RTX 5090M, 96 GB RAM). Web UI + REST + SSE + MCP.

**Full guide:** [app/static/guide.html](app/static/guide.html) (architecture, pipeline, libraries, API, viewers).

## What it does

```
Import photos/video/COLMAP
  → SfM (COLMAP global_mapper) OR feed-forward geometry (VGGT-Omega, DA3, LingBot-Map, InstantSplat)
  → COLMAP sparse/0  (common format for all backends)
  → gsplat train (3DGS or 2DGS+MCMC)
  → export .splat / .ply / .ply_compressed + .pt checkpoint
  → view (sparse points, DA3 infer_gs, gsplat Viser, SuperSplat)
```

**COLMAP is the hub.** Geometry networks produce poses (+ sparse points); gsplat reads `--data-type colmap`. DA3 also exports `infer_gs.ply` for instant Gaussian preview before refine train.

## Architecture

| Component | Role |
|-----------|------|
| `splatlabonesrv/` | GPU server — FastAPI :7860, pipeline, `/data` volumes |
| `splatlabone/` | Client nginx :8080 → shared server |
| `app/main.py` | HTTP + static UI + job worker + MCP |
| `app/pipeline/runner.py` | SfM → geometry → train → export orchestration |
| `app/jobs/worker.py` | Single-job queue, SSE events |

**Data dirs** (under `DATA_DIR=/data`): `uploads/` datasets, `workspaces/` per-job COLMAP tree, `outputs/` train + stage artifacts, `splatlab/cache/` model weights.

## Libraries

| Library | Purpose |
|---------|---------|
| **gsplat** | Train 3DGS/2DGS, export splats, Viser viewer (`/opt/gsplat`) |
| **COLMAP 4** | SfM: features, matching, global_mapper / mapper |
| **VGGT-Omega** | Feed-forward pose + depth (robust preset) |
| **Depth Anything 3** | Pose + depth + infer_gs Gaussians (fast preset) |
| **LingBot-Map** | Streaming video geometry (stream preset) |
| **InstantSplat + MASt3R** | Sparse-view init (sparse preset) |
| **PyTorch cu128** | GPU inference + gsplat CUDA |
| **viser** | 3D viewer stack (via gsplat `simple_viewer.py`) |
| **FastAPI / MCP** | REST, SSE, agent tools |

Backend source baked at image build (`docker/fetch_backends.py`). Weights prefetched at pod start (initContainer).

## Presets

| Preset | Pipeline | Status |
|--------|----------|--------|
| **quality** | glomap → 3DGS (30k) | live |
| **quality_calibrated** | view_graph_calibrator + glomap → 3DGS | live |
| **robust** | VGGT-Omega → 2DGS + MCMC + joint pose | live |
| **fast** | DA3 infer_gs → 3DGS (7k) | live |
| **sparse** | InstantSplat/MASt3R → 3DGS | live |
| **stream** | LingBot-Map chunks → 3DGS | live |
| **quality_hloc** | HLoc SuperPoint+LightGlue → glomap | experimental |
| **scale_fastmap** | FastMap SfM (COLMAP DB → headless) | experimental |
| **robust_gluemap** | GlueMap → COLMAP | experimental |
| **fast_dacpp** | DA3.cpp → COLMAP | experimental |
| **fast_hybrid** | DA3.cpp + Python infer_gs | experimental |
| **dense_tracks** | LoFTR dense match → glomap | experimental |

## API surface

- **Ingest:** `POST /api/v1/ingest/{images,images-zip,video,colmap}`
- **Jobs:** `POST /api/v1/jobs`, SSE `GET /jobs/{id}/events`
- **Scenes:** downloads + `GET /scenes/{id}/artifacts`, `/infer_gs`
- **Viewer:** `POST /scenes/{id}/viewer/start` + WS/HTTP proxy to gsplat Viser
- **MCP:** `/mcp/mcp` — ingest, create_job, get_scene_urls, etc.

## Viewers

| Output | How |
|--------|-----|
| Sparse COLMAP points | Viewer → Three.js preview |
| DA3 `infer_gs.ply` | SuperSplat |
| Trained checkpoint | gsplat Viser (GPU) |
| Edit export | `scene.ply_compressed` → SuperSplat |

## Realtime modes

- `none` — full pipeline
- `geometry_preview` — geometry only, skip train
- `progressive_splat` — live Viser + checkpoint SSE during train

## Install on Olares

1. Add your deployed worker URL as a market source (see repo root README).
2. Build image via **dockerbuilderone MCP** — [docker/README.md](docker/README.md)
3. Install chart; set `OLARES_USER_GITHUB_TOKEN` (ghcr pull)
4. Optional: `OLARES_USER_HUGGINGFACE_TOKEN` for gated weights

## Dev

```bash
cd splatlabone/app
pip install -r requirements.txt
DATA_DIR=/tmp/splatlab uvicorn main:app --reload --port 7860
npm run build:catalog   # from repo root, after manifest changes
node scripts/inject-splatlab-configmap.js   # hot-patch app into Helm ConfigMap
```

## Layout

```
splatlabone/
├── app/              FastAPI, pipeline, static UI, presets
├── splatlabonesrv/   GPU Helm server chart
├── splatlabone/      nginx client chart
├── docker/           Dockerfile, fetch_backends.py, entrypoint
├── templates/        ConfigMap (generated from app/)
└── OlaresManifest.yaml
```
