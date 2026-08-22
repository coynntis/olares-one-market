# --- Web client (Vite) ---
FROM node:22-bookworm AS web
WORKDIR /web
COPY client/package.json ./
RUN npm install --no-audit --no-fund
COPY client/ ./
RUN npm run build

# --- Python bridge + ASGI static ---
FROM python:3.11-slim-bookworm

RUN apt-get update \
    && apt-get install -y --no-install-recommends libopus0 libopus-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ /app/src/
COPY --from=web /web/dist /app/web/dist

ENV PYTHONPATH=/app/src
ENV PYTHONUNBUFFERED=1
ENV STATIC_DIR=/app/web/dist

EXPOSE 8000

CMD ["uvicorn", "xiaozhi_bridge.asgi:app", "--host", "0.0.0.0", "--port", "8000", "--ws-ping-interval", "30", "--ws-ping-timeout", "600"]
