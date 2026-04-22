#!/usr/bin/env python3
"""
Voice Pipeline: STT → Gemma 4 (Vision + LLM) → TTS
Runs entirely on Olares One via local APIs.

Usage:
  python voice_pipeline.py "path/to/audio.wav"
  python voice_pipeline.py --record        # record from mic
  python voice_pipeline.py --text "Hello"  # skip STT, go direct to LLM → TTS
"""

import argparse
import io
import json
import os
import sys
import time
import wave
import httpx

# === CONFIG ===
# Auto-discovers services via kubectl, or set URLs manually via env vars.
OLARES_HOST = os.getenv("OLARES_HOST", "192.168.1.29")
OLARES_USER = os.getenv("OLARES_USER", "aurelien")

# Override with env vars if set, otherwise auto-discover
STT_URL = os.getenv("STT_URL", "")
LLM_URL = os.getenv("LLM_URL", "")
TTS_URL = os.getenv("TTS_URL", "")

def _get_svc_url(app_name: str, port: int = 8000) -> str:
    """Get cluster IP for an app via kubectl."""
    import subprocess
    ns = f"{app_name}-{OLARES_USER}"
    try:
        result = subprocess.run(
            ["ssh", f"olares@{OLARES_HOST}",
             f"kubectl get svc {app_name} -n {ns} -o jsonpath='{{.spec.clusterIP}}' 2>/dev/null"],
            capture_output=True, text=True, timeout=5
        )
        ip = result.stdout.strip().strip("'")
        if ip:
            return f"http://{ip}:{port}"
    except:
        pass
    return ""

def _setup_port_forward(app_name: str, local_port: int, remote_port: int = 8000) -> str:
    """Setup SSH port-forward and return local URL."""
    import subprocess
    ns = f"{app_name}-{OLARES_USER}"
    # Kill existing
    subprocess.run(["pkill", "-f", f"kubectl port-forward.*{app_name}"], capture_output=True)
    # Start new
    subprocess.Popen(
        ["ssh", f"olares@{OLARES_HOST}",
         f"kubectl port-forward -n {ns} svc/{app_name} {local_port}:{remote_port} --address 0.0.0.0"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
    time.sleep(2)
    return f"http://{OLARES_HOST}:{local_port}"

def discover_services():
    """Auto-discover service URLs via port-forward."""
    global STT_URL, LLM_URL, TTS_URL
    if not STT_URL:
        print("🔍 Setting up STT port-forward (vllmvoxtralrt4bone:9001)...")
        STT_URL = _setup_port_forward("vllmvoxtralrt4bone", 9001)
    if not LLM_URL:
        print("🔍 Setting up LLM port-forward (gemma426ba4bone:9002)...")
        LLM_URL = _setup_port_forward("gemma426ba4bone", 9002, remote_port=8080)
    if not TTS_URL:
        print("🔍 Setting up TTS port-forward (vllmvoxtraltts4bone:9003)...")
        TTS_URL = _setup_port_forward("vllmvoxtraltts4bone", 9003)

# Models
STT_MODEL = "voxtral-realtime-4b"
LLM_MODEL = "gemma-4-26b-a4b"
TTS_MODEL = "voxtral-tts-4b"
TTS_VOICE = "fr_male"

# System prompt for Gemma
SYSTEM_PROMPT = """Tu es un assistant vocal intelligent qui tourne en local sur un Olares One.
Tu réponds de manière concise et naturelle, comme dans une conversation orale.
Tes réponses doivent être courtes (1-3 phrases) car elles seront lues à voix haute."""

TIMEOUT = 300.0


def transcribe(audio_path: str) -> str:
    """STT: Audio file → text via Voxtral"""
    print(f"🎤 Transcription de {audio_path}...")
    t0 = time.time()

    with open(audio_path, "rb") as f:
        audio_bytes = f.read()

    # OpenAI-compatible transcription endpoint
    response = httpx.post(
        f"{STT_URL}/v1/audio/transcriptions",
        files={"file": (os.path.basename(audio_path), audio_bytes, "audio/wav")},
        data={"model": STT_MODEL, "language": "fr"},
        timeout=TIMEOUT,
    )
    response.raise_for_status()
    text = response.json().get("text", "")

    print(f"📝 [{time.time()-t0:.1f}s] \"{text}\"")
    return text


def ask_llm(text: str, image_path: str = None) -> str:
    """LLM: Text (+ optional image) → response via Gemma 4"""
    print(f"🧠 Gemma 4 réfléchit...")
    t0 = time.time()

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": text},
    ]

    # If image provided, use vision
    if image_path:
        import base64
        with open(image_path, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode()
        ext = os.path.splitext(image_path)[1].lstrip(".")
        mime = f"image/{ext}" if ext in ("png", "jpeg", "jpg", "gif", "webp") else "image/png"
        messages[-1] = {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{img_b64}"}},
                {"type": "text", "text": text},
            ],
        }

    response = httpx.post(
        f"{LLM_URL}/v1/chat/completions",
        json={
            "model": LLM_MODEL,
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": 256,
        },
        timeout=TIMEOUT,
    )
    response.raise_for_status()
    reply = response.json()["choices"][0]["message"]["content"]

    # Strip thinking tags if present
    if "</think>" in reply:
        reply = reply.split("</think>")[-1].strip()

    print(f"💬 [{time.time()-t0:.1f}s] \"{reply}\"")
    return reply


def speak(text: str, output_path: str = "/tmp/pipeline_output.wav") -> str:
    """TTS: Text → audio via Voxtral TTS"""
    print(f"🔊 Synthèse vocale ({TTS_VOICE})...")
    t0 = time.time()

    response = httpx.post(
        f"{TTS_URL}/v1/audio/speech",
        json={
            "input": text,
            "model": TTS_MODEL,
            "voice": TTS_VOICE,
            "response_format": "wav",
        },
        timeout=TIMEOUT,
    )
    response.raise_for_status()

    with open(output_path, "wb") as f:
        f.write(response.content)

    # Get audio duration
    with wave.open(output_path) as w:
        duration = w.getnframes() / w.getframerate()

    print(f"🎵 [{time.time()-t0:.1f}s] {duration:.1f}s audio → {output_path}")
    return output_path


def record_audio(duration: int = 5, output_path: str = "/tmp/pipeline_input.wav") -> str:
    """Record audio from microphone"""
    try:
        import sounddevice as sd
        import numpy as np
    except ImportError:
        print("pip install sounddevice numpy")
        sys.exit(1)

    sample_rate = 16000
    print(f"🎙️  Enregistrement ({duration}s)... Parlez !")
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype="int16")
    sd.wait()
    print("✅ Enregistrement terminé")

    with wave.open(output_path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(audio.tobytes())

    return output_path


def play_audio(path: str):
    """Play audio file"""
    if sys.platform == "darwin":
        os.system(f"afplay {path}")
    elif sys.platform == "linux":
        os.system(f"aplay {path} 2>/dev/null || paplay {path} 2>/dev/null")
    else:
        print(f"Audio saved to {path}")


def pipeline(audio_path: str = None, text: str = None, image_path: str = None,
             record: bool = False, record_duration: int = 5, play: bool = True):
    """Full pipeline: STT → LLM → TTS"""
    print("=" * 50)
    print("🚀 Voice Pipeline: STT → Gemma 4 → TTS")
    print("=" * 50)

    discover_services()
    t_start = time.time()

    # Step 1: Get input text
    if text:
        user_text = text
        print(f"📝 Input: \"{user_text}\"")
    elif record:
        audio_path = record_audio(duration=record_duration)
        user_text = transcribe(audio_path)
    elif audio_path:
        user_text = transcribe(audio_path)
    else:
        print("Error: provide --audio, --record, or --text")
        sys.exit(1)

    if not user_text.strip():
        print("❌ Aucun texte détecté")
        return

    # Step 2: LLM
    reply = ask_llm(user_text, image_path=image_path)

    # Step 3: TTS
    output_path = speak(reply)

    # Summary
    total = time.time() - t_start
    print(f"\n⏱️  Pipeline total: {total:.1f}s")

    # Play
    if play:
        play_audio(output_path)

    return reply


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Voice Pipeline: STT → Gemma 4 → TTS")
    parser.add_argument("audio", nargs="?", help="Audio file to transcribe")
    parser.add_argument("--text", "-t", help="Skip STT, send text directly to LLM")
    parser.add_argument("--image", "-i", help="Image file for vision (sent with the question)")
    parser.add_argument("--record", "-r", action="store_true", help="Record from microphone")
    parser.add_argument("--duration", "-d", type=int, default=5, help="Recording duration in seconds")
    parser.add_argument("--voice", "-v", default=TTS_VOICE, help=f"TTS voice (default: {TTS_VOICE})")
    parser.add_argument("--no-play", action="store_true", help="Don't play audio output")
    args = parser.parse_args()

    TTS_VOICE = args.voice

    pipeline(
        audio_path=args.audio,
        text=args.text,
        image_path=args.image,
        record=args.record,
        record_duration=args.duration,
        play=not args.no_play,
    )
