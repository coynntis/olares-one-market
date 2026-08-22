# Optional pre-built image for ideogram4nf4one (CUDA 13 / Olares One).
# Build: docker build -t ghcr.io/<user>/ideogram4nf4one:1.0.0 -f ideogram4nf4one/docker/Dockerfile ideogram4nf4one
# Then set values.yaml: image.repository / image.tag and deploy.

FROM pytorch/pytorch:2.12.0-cuda13.0-cudnn9-devel

WORKDIR /app

ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    TORCHDYNAMO_DISABLE=1 \
    GRADIO_TEMP_DIR=/output/gradio \
    SERVER_PORT=7860 \
    HF_HOME=/models/huggingface

COPY app/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r /app/requirements.txt

COPY app/app.py /app/app.py

EXPOSE 7860

CMD ["python", "/app/app.py"]
