#!/usr/bin/env python3
"""Time a single 2048x2048 T2I via LightLLM OpenAI-compatible API (in-pod benchmark)."""

from __future__ import annotations

import argparse
import json
import os
import time
import urllib.error
import urllib.request

DEFAULT_URL = "http://127.0.0.1:8000/v1/chat/completions"
DEFAULT_MODEL = "sensenova-u1"

GENERATION_SYSTEM_PROMPT = (
    "You are an image generation assistant. Produce a high-quality image from the user prompt. "
    "Match the user's language for any visible text."
)


def main() -> None:
    p = argparse.ArgumentParser(description="Benchmark SenseNova T2I latency (LightLLM path)")
    p.add_argument("--url", default=os.getenv("BENCHMARK_URL", DEFAULT_URL))
    p.add_argument("--model", default=os.getenv("BENCHMARK_MODEL", DEFAULT_MODEL))
    p.add_argument("--prompt", default="A minimal infographic chart comparing solar vs wind energy, clean flat design.")
    p.add_argument("--aspect-ratio", default="1:1")
    p.add_argument("--image-size", default="2K")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--warmup", action="store_true", help="Run one untimed request first")
    args = p.parse_args()

    payload = {
        "model": args.model,
        "messages": [
            {"role": "system", "content": GENERATION_SYSTEM_PROMPT},
            {"role": "user", "content": args.prompt},
        ],
        "modalities": ["image"],
        "stream": False,
        "n": 1,
        "temperature": 0.8,
        "top_p": 0.95,
        "max_tokens": 4096,
        "chat_template_kwargs": {"enable_thinking": False},
        "image_config": {
            "aspect_ratio": args.aspect_ratio,
            "image_size": args.image_size,
            "image_type": "jpeg",
            "seed": args.seed,
            "dynamic_resolution": False,
            "height": 2048,
            "width": 2048,
        },
    }
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        args.url,
        data=body,
        headers={"Content-Type": "application/json", "Authorization": "Bearer dummy"},
        method="POST",
    )

    def once(label: str) -> float:
        t0 = time.perf_counter()
        with urllib.request.urlopen(req, timeout=900) as resp:
            raw = resp.read()
        dt = time.perf_counter() - t0
        data = json.loads(raw)
        choices = data.get("choices") or []
        msg = (choices[0].get("message") if choices else {}) or {}
        n_img = len(msg.get("images") or [])
        print(f"[{label}] wall_s={dt:.2f} images={n_img} content_len={len(msg.get('content') or '')}")
        return dt

    if args.warmup:
        print("warmup...")
        once("warmup")

    try:
        dt = once("timed")
    except urllib.error.HTTPError as exc:
        print(f"HTTP {exc.code}: {exc.read().decode('utf-8', errors='replace')[:2000]}")
        raise SystemExit(1) from exc

    print(json.dumps({"backend": "lightllm", "wall_seconds": round(dt, 3), "resolution": "2048x2048"}))


if __name__ == "__main__":
    main()
