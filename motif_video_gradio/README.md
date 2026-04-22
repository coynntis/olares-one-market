# Motif Video 2B Gradio

Gradio interface for:
- Text-to-video generation
- Image-to-video generation

Uses `diffusers` + `Motif-Technologies/Motif-Video-2B` with the model-card memory-efficient mode for 24GB GPUs:

- `enable_model_cpu_offload()`
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`

## Local run

```bash
cd motif_video_gradio
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
pip install -r requirements.txt
python app.py
```

Then open [http://localhost:7860](http://localhost:7860).

## Docker run

This uses the requested base image: `pytorch/pytorch:2.11.0-cuda13.0-cudnn9-devel`.

```bash
cd motif_video_gradio
docker build -t motif-video-gradio .
docker run --rm -it --gpus all -p 7860:7860 \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  motif-video-gradio
```

## Notes

- First run downloads model weights from Hugging Face and can take a while.
- A valid Hugging Face token may be required depending on access settings:
  - `export HF_TOKEN=...`
- CPU offload is enabled by default for GPUs with <=24GB VRAM in the UI.
