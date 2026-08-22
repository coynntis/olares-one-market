#!/usr/bin/env python3
"""Generate Olares market icons (256x256) and featured banners (1440x900)."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
ICONS_DIR = ROOT / "icons"
FEATURED_DIR = ROOT / "featured"
ICON_SIZE = 256
FEATURED_W, FEATURED_H = 1440, 900
MARKET_BASE = "https://orales-one-market.coynntis.workers.dev"

# Hugging Face–style pipeline tags (canonical id → short pill label)
PIPELINE_LABELS: dict[str, str] = {
    "text-generation": "Text Gen",
    "text-to-image": "T2I",
    "text-to-video": "T2V",
    "image-to-video": "I2V",
    "image-text-to-text": "VLM",
    "text-to-speech": "TTS",
    "automatic-speech-recognition": "ASR",
    "any-to-any": "Any-to-Any",
    "visual-grounding": "Grounding",
    "agent-tools": "Agent Tools",
    "dev-tools": "Dev Tools",
    "speech-bridge": "Voice Bridge",
    "feature-extraction": "Embedding",
}

# Per-app display overrides (title, backend pill, accent RGB)
APPS: dict[str, tuple[str, str, tuple[int, int, int]]] = {
    "browserlessone": ("Browserless One", "Browserless", (59, 130, 246)),
    "dflashqwen3627bone": ("Qwen36 27B DFlash", "Lucebox DFlash", (234, 88, 12)),
    "dockerbuilderone": ("Docker Builder", "Kaniko", (14, 165, 233)),
    "dshone": ("DeepSeek Harness", "dsh npm", (4, 120, 87)),
    "fastwanqad13bone": ("FastWan QAD 1.3B", "FastVideo NVFP4", (124, 58, 237)),
    "fastwanqad13bsa2one": ("FastWan QAD 1.3B SA2", "FastVideo NVFP4", (168, 85, 247)),
    "fastwanqad13fp8one": ("FastWan QAD 1.3B FP8", "FastVideo FP8", (139, 92, 246)),
    "gemma4e2bone": ("Gemma 4 E2B", "llama.cpp", (34, 197, 94)),
    "ideogram4nf4one": ("Ideogram 4 NF4", "diffusers", (168, 85, 247)),
    "llamacppbonsai8bone": ("Bonsai 8B", "llama.cpp", (34, 197, 94)),
    "llamacppbonsai27bone": ("Bonsai 27B", "PrismML VLM", (16, 185, 129)),
    "llamacpptbonsai27bone": ("Ternary Bonsai 27B", "PrismML VLM", (22, 163, 74)),
    "llamacppnanbeige423bone": ("Nanbeige 4.2 3B", "Nanbeige llama.cpp", (14, 165, 233)),
    "llamacpplagunas21one": ("Laguna S 2.1", "poolside cpu-moe", (249, 115, 22)),
    "llamacppdsv4flash0731one": ("DSV4 Flash 0731", "cpu-moe IQ2", (249, 115, 22)),
    "colibridsv4flash0731one": ("DSV4 Flash Colibri", "SSD stream", (14, 165, 233)),
    "llamacppdiffusiongemma26a4bone": ("DiffusionGemma 26B", "vLLM", (129, 140, 248)),
    "llamacppgemma412agent1": ("Gemma 4 12B Agentic", "llama.cpp", (34, 197, 94)),
    "llamacppagentsa1one": ("Agents-A1 Vision", "llama.cpp", (34, 197, 94)),
    "llamacppornith35bone": ("Ornith 35B One", "llama.cpp", (34, 197, 94)),
    "llamacppornith9bone": ("Ornith 9B One", "llama.cpp", (34, 197, 94)),
    "llamacppthinkingcap27bone": ("ThinkingCap Qwen36 27B", "llama.cpp", (34, 197, 94)),
    "llamacppnemotrondiffusion14b1": ("Nemotron Diffusion 14B", "buun self-spec", (118, 185, 0)),
    "llamacppqwable35bone": ("Qwable 35B MoE", "buun MTP", (34, 197, 94)),
    "llamacppqwen3627btq34sone": ("Qwen36 27B TQ34S", "llama.cpp", (34, 197, 94)),
    "llamacppqwen3635ba3btq34sone": ("Qwen35 35B Vision", "llama.cpp", (34, 197, 94)),
    "llamacppqwen36a3bone": ("Qwen36 35B-A3B MTP", "buun MTP + vision", (34, 197, 94)),
    "llamacppqwen36a3bdflashone": ("Qwen36 35B DFlash", "upstream DFlash + vision", (16, 185, 129)),
    "llamacppqwen3827bmtpone": ("Qwen38 27B MTP", "buun MTP + vision", (59, 130, 246)),
    "llamacppqwen36fable27bone": ("Qwen36 Fable 27B", "MTP + vision tools", (234, 179, 8)),
    "llamacppgrug27bone": ("Grug 27B MTP", "short-think + vision", (163, 230, 53)),
    "llamacppkatcoderv25one": ("KAT Coder V25", "MoE coding MTP", (251, 146, 60)),
    "llamacppqwen36beellamaone": ("Qwen36 27B BeeLlama", "BeeLlama DFlash", (34, 197, 94)),
    "llamacppqwen36beellamavision1": ("Qwen36 27B Vision", "BeeLlama DFlash", (34, 197, 94)),
    "llamacppqwen36mtpone": ("Qwen36 27B MTP", "buun draft-MTP", (34, 197, 94)),
    "llamacppqwopus27coder1": ("Qwopus 27B Coder", "buun llama.cpp", (34, 197, 94)),
    "llamacppqwopus27mtpone": ("Qwopus 27B MTP v2", "buun draft-MTP", (34, 197, 94)),
    "llamacppqwythos9bone": ("Qwythos 9B", "buun draft-MTP", (34, 197, 94)),
    "locateanything3bone": ("LocateAnything 3B", "PyTorch + MagiAttn", (168, 85, 247)),
    "motifvideo2bone": ("Motif Video 2B", "Diffusers GGUF", (168, 85, 247)),
    "minimaxh3nvfp4one": ("MiniMax H3 NVFP4", "Diffusers Modular", (236, 72, 153)),
    "ltx23one": ("LTX-2.3 FP8", "LTX Distilled", (236, 72, 153)),
    "krea2turboone": ("Krea-2 Turbo", "Diffusers LoRA", (249, 115, 22)),
    "mageflowone": ("Mage Flow Turbo", "Mage-Flow 4B", (0, 120, 212)),
    "sensenovasi15one": ("SenseNova SI 1.5", "InternVL VQA", (59, 130, 246)),
    "sensenovavisionone": ("SenseNova Vision", "Vision 7B MoT", (14, 165, 233)),
    "consistcomposeone": ("ConsistCompose BAGEL", "BAGEL NF4", (139, 92, 246)),
    "nemotronlabselastic30bnvfp4one": ("Nemotron 30B NVFP4", "vLLM", (118, 185, 0)),
    "omnivoiceone": ("OmniVoice TTS", "OmniVoice", (236, 72, 153)),
    "sensevoiceone": ("SenseVoice STT One", "FunASR SenseVoice", (59, 130, 246)),
    "cosyvoice2yueone": ("CosyVoice2 Yue TTS", "CosyVoice2", (234, 88, 12)),
    "cosyvoice3one": ("CosyVoice3 TTS One", "Fun-CosyVoice3", (249, 115, 22)),
    "mossttslocalone": ("MOSS-TTS Local One", "MOSS-TTS 1.5", (14, 165, 233)),
    "voxcpmone": ("VoxCPM2 TTS One", "VoxCPM2", (168, 85, 247)),
    "openwebsearchone": ("Open WebSearch", "Open Web Search", (100, 116, 139)),
    "lingbotdepthone": ("LingBot Depth", "Depth ViT-L", (14, 165, 233)),
    "lingbotvisionone": ("LingBot Vision", "Vision ViT-L", (59, 130, 246)),
    "lingbotmapone": ("LingBot Map", "3D Map", (16, 185, 129)),
    "lingbotvideoone": ("LingBot Video Dense", "T2V 1.3B", (236, 72, 153)),
    "lingbotvlaone": ("LingBot VLA 2.0", "VLA 6B", (249, 115, 22)),
    "lingbotvaone": ("LingBot VA", "VA Offload", (168, 85, 247)),
    "lingbotworldone": ("LingBot World NF4", "World NF4", (234, 88, 12)),
    "pipecatxiaozhione": ("Pipecat Xiaozhi Bridge", "Pipecat", (20, 184, 166)),
    "qwen36a3bvisionone": ("Qwen3.6 Vision", "llama.cpp", (34, 197, 94)),
    "qwen3ttstone": ("Qwen3-TTS 1.7B", "Qwen3-TTS", (244, 63, 94)),
    "sensenovau1lightllmone": ("SenseNova U1 Light", "LightLLM FA3", (59, 130, 246)),
    "sensenovau1serveone": ("SenseNova U1 Serve", "SenseNova PyTorch", (139, 92, 246)),
    "sensenovau1infov2one": ("SenseNova Infog V2", "Infographic MoT", (168, 85, 247)),
    "sglangernieimageone": ("ERNIE-Image", "SGLang", (249, 115, 22)),
    "sglanglfm258ba1bone": ("LFM2-5 8B", "SGLang", (249, 115, 22)),
    "sglangminicpm51bone": ("MiniCPM5 1B", "SGLang", (249, 115, 22)),
    "sglangsanasprintone": ("Sana Sprint 1.6B", "SGLang", (249, 115, 22)),
    "sglangkrea2turboone": ("SGLang Krea Turbo", "SGLang DiT", (234, 88, 12)),
    "sglangltx23one": ("SGLang LTX-2-3", "SGLang Video", (219, 39, 119)),
    "splatlabone": ("SplatLab One", "3D Gaussian", (14, 165, 233)),
    "vllmgemma31bitnvfp4one": ("Gemma 4 31B NVFP4", "vLLM", (129, 140, 248)),
    "vllmgemma426ba4bvisionone": ("Gemma 4 26B Vision", "vLLM MTP", (129, 140, 248)),
    "vllmgemma4dflashone": ("Gemma 4 26B DFlash", "vLLM DFlash", (129, 140, 248)),
    "vllmgemma4e4bone": ("Gemma 4 E4B Multi", "vLLM MTP", (129, 140, 248)),
    "vllmtess427bone": ("Tess-4-27B NVFP4", "vLLM", (129, 140, 248)),
    "vllmnemotronaudex30bone": ("Nemotron Audex 30B", "vLLM", (118, 185, 0)),
    "vllmgepardone": ("Gepard TTS One", "vLLM", (129, 140, 248)),
    "llamacppagentworld35bone": ("AgentWorld 35B", "llama.cpp", (34, 197, 94)),
    "vllmqwen3627bnvfp4one": ("Qwen36 27B NVFP4", "vLLM TQ", (129, 140, 248)),
    "vllmqwen3827bnvfp4one": ("Qwen38 27B NVFP4", "vLLM TQ", (129, 140, 248)),
    "vllmqwen3635bnvfp4fone": ("Qwen36 35B NVFP4 Fast", "vLLM TQ", (129, 140, 248)),
    "vllmgemma431bnvfp4one": ("Gemma4 31B Unsloth NVFP4", "vLLM TQ", (129, 140, 248)),
    "sndrqwen3627bone": ("SNDR Qwen36 27B", "SNDR MTP K=4", (16, 185, 129)),
    "sndrqwen3635ba3bone": ("SNDR Qwen36 35B", "SNDR MTP K=5", (5, 150, 105)),
    "sndrgemma426ba4bone": ("SNDR Gemma4 26B", "SNDR AWQ", (4, 120, 87)),
    "sndrdiffusiongemma26bone": ("SNDR DiffusionGemma", "SNDR Diff", (6, 95, 70)),
}

PIPELINE_BY_APP: dict[str, list[str]] = {
    "browserlessone": ["agent-tools"],
    "dflashqwen3627bone": ["text-generation"],
    "dockerbuilderone": ["dev-tools"],
    "dshone": ["agent-tools"],
    "fastwanqad13bone": ["text-to-video"],
    "fastwanqad13bsa2one": ["text-to-video"],
    "fastwanqad13fp8one": ["text-to-video"],
    "gemma4e2bone": ["text-generation"],
    "ideogram4nf4one": ["text-to-image"],
    "llamacppbonsai8bone": ["text-generation"],
    "llamacppbonsai27bone": ["image-text-to-text", "text-generation"],
    "llamacpptbonsai27bone": ["image-text-to-text", "text-generation"],
    "llamacppnanbeige423bone": ["text-generation"],
    "llamacpplagunas21one": ["text-generation"],
    "llamacppdsv4flash0731one": ["text-generation"],
    "colibridsv4flash0731one": ["text-generation"],
    "llamacppdiffusiongemma26a4bone": ["text-generation"],
    "llamacppgemma412agent1": ["text-generation"],
    "llamacppagentsa1one": ["image-text-to-text"],
    "llamacppornith35bone": ["text-generation"],
    "llamacppornith9bone": ["text-generation"],
    "llamacppthinkingcap27bone": ["image-text-to-text"],
    "llamacppnemotrondiffusion14b1": ["text-generation"],
    "llamacppqwable35bone": ["text-generation"],
    "llamacppqwen3627btq34sone": ["image-text-to-text"],
    "llamacppqwen3635ba3btq34sone": ["image-text-to-text"],
    "llamacppqwen36a3bone": ["image-text-to-text"],
    "llamacppqwen36a3bdflashone": ["image-text-to-text"],
    "llamacppqwen3827bmtpone": ["image-text-to-text"],
    "llamacppqwen36fable27bone": ["image-text-to-text"],
    "llamacppgrug27bone": ["image-text-to-text"],
    "llamacppkatcoderv25one": ["text-generation"],
    "llamacppqwen36beellamaone": ["text-generation"],
    "llamacppqwen36beellamavision1": ["image-text-to-text"],
    "llamacppqwen36mtpone": ["text-generation"],
    "llamacppqwopus27coder1": ["text-generation"],
    "llamacppqwopus27mtpone": ["text-generation"],
    "llamacppqwythos9bone": ["text-generation"],
    "locateanything3bone": ["visual-grounding"],
    "motifvideo2bone": ["text-to-video", "image-to-video"],
    "minimaxh3nvfp4one": ["text-to-video", "image-to-video"],
    "ltx23one": ["text-to-video", "image-to-video"],
    "krea2turboone": ["text-to-image"],
    "mageflowone": ["text-to-image", "image-edit"],
    "sensenovasi15one": ["image-text-to-text"],
    "sensenovavisionone": ["image-text-to-text", "depth-estimation", "image-segmentation"],
    "consistcomposeone": ["text-to-image"],
    "nemotronlabselastic30bnvfp4one": ["text-generation"],
    "omnivoiceone": ["text-to-speech"],
    "sensevoiceone": ["automatic-speech-recognition"],
    "cosyvoice2yueone": ["text-to-speech"],
    "cosyvoice3one": ["text-to-speech"],
    "mossttslocalone": ["text-to-speech"],
    "voxcpmone": ["text-to-speech"],
    "openwebsearchone": ["agent-tools"],
    "pipecatxiaozhione": ["speech-bridge"],
    "qwen36a3bvisionone": ["image-text-to-text"],
    "qwen3ttstone": ["text-to-speech"],
    "sensenovau1lightllmone": ["text-to-image"],
    "sensenovau1serveone": ["any-to-any"],
    "sensenovau1infov2one": ["text-to-image", "any-to-any"],
    "sglangernieimageone": ["text-to-image"],
    "sglanglfm258ba1bone": ["text-generation"],
    "sglangminicpm51bone": ["text-generation"],
    "sglangsanasprintone": ["text-to-image"],
    "sglangkrea2turboone": ["text-to-image"],
    "sglangltx23one": ["text-to-video", "image-to-video"],
    "splatlabone": ["dev-tools"],
    "vllmgemma31bitnvfp4one": ["text-generation"],
    "vllmgemma426ba4bvisionone": ["image-text-to-text"],
    "vllmgemma4dflashone": ["text-generation"],
    "vllmgemma4e4bone": ["any-to-any"],
    "vllmtess427bone": ["image-text-to-text"],
    "vllmnemotronaudex30bone": ["any-to-any"],
    "vllmgepardone": ["text-to-speech"],
    "llamacppagentworld35bone": ["text-generation", "agent-tools"],
    "vllmqwen3627bnvfp4one": ["image-text-to-text"],
    "vllmqwen3827bnvfp4one": ["image-text-to-text"],
    "sndrqwen3627bone": ["text-generation"],
    "sndrqwen3635ba3bone": ["text-generation"],
    "sndrgemma426ba4bone": ["image-text-to-text", "text-generation"],
    "sndrdiffusiongemma26bone": ["text-generation"],
    "vllmqwen3635bnvfp4fone": ["image-text-to-text"],
    "vllmgemma431bnvfp4one": ["image-text-to-text"],
}

SPEC_OVERRIDES: dict[str, dict[str, str | bool]] = {
    "browserlessone": {"context": "—", "speed": "—", "vision": False},
    "dflashqwen3627bone": {"context": "32K", "speed": "—", "vision": False},
    "dockerbuilderone": {"context": "—", "speed": "—", "vision": False},
    "dshone": {"context": "—", "speed": "npx pin", "vision": False},
    "fastwanqad13bone": {"context": "—", "speed": "~1.8s clip", "vision": False},
    "fastwanqad13bsa2one": {"context": "—", "speed": "~2s clip", "vision": False},
    "gemma4e2bone": {"context": "8K", "speed": "—", "vision": False},
    "ideogram4nf4one": {"context": "—", "speed": "—", "vision": False},
    "llamacppbonsai8bone": {"context": "—", "speed": "—", "vision": False},
    "llamacppbonsai27bone": {"context": "32K", "speed": "~4.5GB", "vision": True},
    "llamacpptbonsai27bone": {"context": "32K", "speed": "~8GB", "vision": True},
    "llamacppnanbeige423bone": {"context": "131K", "speed": "~5GB", "vision": False},
    "llamacpplagunas21one": {"context": "16K", "speed": "IQ4 cpu-moe", "vision": False},
    "llamacppdsv4flash0731one": {"context": "16K", "speed": "IQ2 cpu-moe", "vision": False},
    "colibridsv4flash0731one": {"context": "—", "speed": "SSD stream", "vision": False},
    "llamacppdiffusiongemma26a4bone": {"context": "—", "speed": "—", "vision": False},
    "llamacppgemma412agent1": {"context": "128K", "speed": "—", "vision": False},
    "llamacppornith35bone": {"context": "64K", "speed": "—", "vision": False},
    "llamacppornith9bone": {"context": "128K", "speed": "—", "vision": False},
    "llamacppthinkingcap27bone": {"context": "64K", "speed": "—", "vision": True},
    "vllmtess427bone": {"context": "64K", "speed": "—", "vision": True},
    "llamacppqwen3627btq34sone": {"context": "131K", "speed": "—", "vision": True},
    "llamacppqwen3635ba3btq34sone": {"context": "128K", "speed": "—", "vision": True},
    "locateanything3bone": {"context": "—", "speed": "—", "vision": True},
    "motifvideo2bone": {"context": "—", "speed": "—", "vision": False},
    "minimaxh3nvfp4one": {"context": "—", "speed": "native audio", "vision": False},
    "ltx23one": {"context": "—", "speed": "8-step FP8", "vision": False},
    "krea2turboone": {"context": "—", "speed": "8-step Turbo", "vision": False},
    "mageflowone": {"context": "—", "speed": "4-step Turbo", "vision": False},
    "sglangkrea2turboone": {"context": "—", "speed": "SGLang offload", "vision": False},
    "sglangltx23one": {"context": "—", "speed": "layerwise DiT", "vision": False},
    "sensenovasi15one": {"context": "—", "speed": "—", "vision": True},
    "sensenovavisionone": {"context": "—", "speed": "offload 24GB", "vision": True},
    "sensenovau1infov2one": {"context": "—", "speed": "infographic", "vision": True},
    "consistcomposeone": {"context": "—", "speed": "NF4 compose", "vision": False},
    "openwebsearchone": {"context": "—", "speed": "—", "vision": False},
    "pipecatxiaozhione": {"context": "—", "speed": "—", "vision": False},
    "qwen36a3bvisionone": {"context": "32K", "speed": "131 t/s", "vision": True},
    "qwen3ttstone": {"context": "—", "speed": "—", "vision": False},
    "sensevoiceone": {"context": "—", "speed": "—", "vision": False},
    "cosyvoice2yueone": {"context": "—", "speed": "—", "vision": False},
    "cosyvoice3one": {"context": "—", "speed": "—", "vision": False},
    "mossttslocalone": {"context": "—", "speed": "—", "vision": False},
    "voxcpmone": {"context": "—", "speed": "—", "vision": False},
    "sglangernieimageone": {"context": "—", "speed": "—", "vision": False},
    "sglanglfm258ba1bone": {"context": "8K", "speed": "—", "vision": False},
    "sglangminicpm51bone": {"context": "8K", "speed": "—", "vision": False},
    "sglangsanasprintone": {"context": "—", "speed": "—", "vision": False},
}

BACKEND_COLORS: dict[str, tuple[int, int, int]] = {
    "llama.cpp": (34, 197, 94),
    "SGLang": (249, 115, 22),
    "vLLM": (129, 140, 248),
}

COLORS = {
    "bg": (15, 23, 42),
    "title": (248, 250, 252),
    "muted": (148, 163, 184),
    "label": (100, 116, 139),
    "value": (226, 232, 240),
    "pill_text": (15, 23, 42),
    "pipeline_bg": (30, 41, 59),
    "pipeline_border": (71, 85, 105),
    "row_divider": (51, 65, 85),
    "yes": (74, 222, 128),
    "no": (251, 113, 133),
    "highlight": (250, 204, 21),
}


@dataclass
class AppCardMeta:
    id: str
    title: str
    backend: str
    accent: tuple[int, int, int]
    pipelines: list[str] = field(default_factory=list)
    context: str = "—"
    speed: str = "—"
    vision: bool = False
    shared_entrance: bool = False
    description: str = ""
    size_label: str = ""
    badge: str = ""
    vram: str = ""
    stack: str = ""
    hero_label: str = ""
    tool_calling: bool = False
    mtp_enabled: bool = False
    mtp_accept: str = ""
    hero_value: str = ""
    categories: list[str] = field(default_factory=list)

    def to_json(self, *, icon_hash: str = "", featured_hash: str = "") -> dict:
        return {
            "id": self.id,
            "title": self.title,
            "backend": self.backend,
            "pipelines": self.pipelines,
            "context": self.context,
            "speed": self.speed,
            "vision": self.vision,
            "shared_entrance": self.shared_entrance,
            "icon": f"icons/{self.id}.png",
            "icon_hash": icon_hash,
            "featured": f"featured/{self.id}.png",
            "featured_hash": featured_hash,
            "icon_url": "",
            "featured_url": "",
        }


def _load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = []
    if bold:
        candidates.extend(
            [
                "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                "/Library/Fonts/Arial Bold.ttf",
            ]
        )
    else:
        candidates.extend(
            [
                "/System/Library/Fonts/Supplemental/Arial.ttf",
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                "/Library/Fonts/Arial.ttf",
                "/System/Library/Fonts/Supplemental/Courier New.ttf",
            ]
        )
    for path in candidates:
        p = Path(path)
        if p.exists():
            return ImageFont.truetype(str(p), size=size)
    return ImageFont.load_default()


def _strip_helm(raw: str) -> str:
    lines: list[str] = []
    skip = False
    for line in raw.splitlines():
        if "{{- if" in line and "else" in line:
            skip = True
            continue
        if "{{- else" in line:
            skip = False
            continue
        if "{{- end" in line:
            skip = False
            continue
        if skip or "{{" in line:
            continue
        lines.append(line)
    return "\n".join(lines)


def _load_manifest(app_id: str) -> dict | None:
    path = ROOT / app_id / "OlaresManifest.yaml"
    if not path.exists():
        return None
    raw = _strip_helm(path.read_text(encoding="utf-8"))
    try:
        return yaml.safe_load(raw) or {}
    except yaml.YAMLError:
        return None


def _spec_value(bento: dict | None, label: str) -> str | None:
    if not bento:
        return None
    for spec in bento.get("specs") or []:
        if isinstance(spec, dict) and str(spec.get("label", "")).lower() == label.lower():
            return str(spec.get("value", "")).strip() or None
    return None


def _normalize_speed(hero: dict | None) -> str | None:
    if not hero or not isinstance(hero, dict):
        return None
    value = str(hero.get("value", "")).strip()
    label = str(hero.get("label", "")).strip()
    if not value:
        return None
    lower = value.lower()
    if "t/s" in lower:
        return value
    if re.search(r"\d", value) and re.search(r"\bs\b|sec|clip", lower + " " + label.lower()):
        return value
    if re.match(r"^~?\d+(\.\d+)?$", value.replace("~", "")):
        if re.search(r"\bt/s\b|speed|throughput|decode", label, re.I):
            return f"{value} t/s"
        return f"{value} t/s"
    if re.search(r"[a-zA-Z]{3,}", value) and not re.search(r"\d", value):
        return None
    return value


def _truncate(text: str, max_len: int = 118) -> str:
    text = re.sub(r"\s+", " ", text.strip())
    if len(text) <= max_len:
        return text
    cut = text[: max_len - 1].rsplit(" ", 1)[0]
    return f"{cut}…"


def _build_meta(app_id: str) -> AppCardMeta:
    title, backend, accent = APPS[app_id]
    manifest = _load_manifest(app_id) or {}
    meta = manifest.get("metadata") or {}
    bento = meta.get("bento") if isinstance(meta.get("bento"), dict) else None
    caps = (bento or {}).get("capabilities") or {}
    hero = (bento or {}).get("hero") if isinstance((bento or {}).get("hero"), dict) else {}

    pipelines = list(PIPELINE_BY_APP.get(app_id, ["text-generation"]))
    shared = bool(manifest.get("sharedEntrances"))

    ctx = _spec_value(bento, "context")
    speed = _normalize_speed(hero)
    vision = bool(caps.get("vision")) if caps else False
    vram = _spec_value(bento, "vram") or ""

    overrides = SPEC_OVERRIDES.get(app_id, {})
    if not ctx:
        ctx = str(overrides.get("context", "—"))
    if not speed:
        speed = str(overrides.get("speed", "—"))
    if not caps and "vision" in overrides:
        vision = bool(overrides["vision"])

    mtp = caps.get("mtp") if isinstance(caps.get("mtp"), dict) else {}
    mtp_enabled = bool(mtp.get("enabled"))
    mtp_accept = ""
    if mtp_enabled:
        accept = mtp.get("accept")
        if accept not in (None, "", 0, False):
            mtp_accept = f"{accept}% accept"

    return AppCardMeta(
        id=app_id,
        title=title,
        backend=backend,
        accent=accent,
        pipelines=pipelines,
        context=ctx,
        speed=speed,
        vision=vision,
        shared_entrance=shared,
        description=_truncate(str(meta.get("description", ""))),
        size_label=str((bento or {}).get("size_label", "")).strip(),
        badge=str((bento or {}).get("badge", "")).strip(),
        vram=vram,
        stack=str((bento or {}).get("stack", "")).strip(),
        hero_label=str(hero.get("label", "")).strip(),
        hero_value=str(hero.get("value", "")).strip(),
        tool_calling=bool(caps.get("tool_calling")) if caps else False,
        mtp_enabled=mtp_enabled,
        mtp_accept=mtp_accept,
        categories=[str(c) for c in (meta.get("categories") or [])[:3]],
    )


def _wrap_title(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, max_width: int) -> list[str]:
    words = text.split()
    if not words:
        return [text]
    lines: list[str] = []
    current = words[0]
    for word in words[1:]:
        trial = f"{current} {word}"
        box = draw.textbbox((0, 0), trial, font=font)
        if box[2] - box[0] <= max_width:
            current = trial
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines[:2]


def _draw_gradient(img: Image.Image, draw: ImageDraw.ImageDraw, accent: tuple[int, int, int]) -> None:
    w, h = img.size
    for y in range(h):
        t = y / h
        r = int(15 + (accent[0] - 15) * t * 0.35)
        g = int(23 + (accent[1] - 23) * t * 0.35)
        b = int(42 + (accent[2] - 42) * t * 0.35)
        draw.line([(0, y), (w, y)], fill=(r, g, b))


def _draw_pipeline_pills(
    draw: ImageDraw.ImageDraw,
    pipelines: list[str],
    x: int,
    y: int,
    max_width: int,
    font: ImageFont.ImageFont,
    *,
    pill_h: int = 19,
    gap: int = 5,
    pad_x: int = 7,
    radius: int = 6,
    text_y_offset: int = 4,
    max_pills: int = 2,
) -> int:
    cx = x
    cy = y
    for pipe_id in pipelines[:max_pills]:
        label = PIPELINE_LABELS.get(pipe_id, pipe_id.replace("-", " ").title())
        bbox = draw.textbbox((0, 0), label, font=font)
        tw = bbox[2] - bbox[0]
        pill_w = tw + pad_x * 2
        if cx + pill_w > x + max_width and cx > x:
            break
        draw.rounded_rectangle(
            [cx, cy, cx + pill_w, cy + pill_h],
            radius=radius,
            fill=COLORS["pipeline_bg"],
            outline=COLORS["pipeline_border"],
            width=1,
        )
        draw.text((cx + pad_x, cy + text_y_offset), label, fill=COLORS["value"], font=font)
        cx += pill_w + gap
    return cy + pill_h + 7


def _spec_rows(meta: AppCardMeta) -> list[tuple[str, str]]:
    return [
        ("context", meta.context),
        ("speed", meta.speed),
        ("vision", "yes" if meta.vision else "no"),
        ("shared API", "yes" if meta.shared_entrance else "no"),
    ]


def _draw_spec_table_vertical(
    draw: ImageDraw.ImageDraw,
    meta: AppCardMeta,
    x: int,
    y: int,
    width: int,
    label_font: ImageFont.ImageFont,
    value_font: ImageFont.ImageFont,
    *,
    row_h: int = 23,
    panel_pad_v: int = 7,
    panel_pad_h: int = 6,
    radius: int = 8,
) -> int:
    rows = _spec_rows(meta)
    panel_top = y
    panel_h = panel_pad_v * 2 + len(rows) * row_h
    draw.rounded_rectangle(
        [x - panel_pad_h, panel_top, x + width + panel_pad_h, panel_top + panel_h],
        radius=radius,
        fill=(22, 32, 52),
        outline=COLORS["row_divider"],
        width=1,
    )
    y = panel_top + panel_pad_v
    for i, (label, value) in enumerate(rows):
        ry = y + i * row_h
        draw.text((x, ry + 5), label, fill=COLORS["label"], font=label_font)
        vb = draw.textbbox((0, 0), value, font=value_font)
        vw = vb[2] - vb[0]
        draw.text((x + width - vw, ry + 4), value, fill=COLORS["value"], font=value_font)
        if i < len(rows) - 1:
            draw.line(
                [(x, ry + row_h - 1), (x + width, ry + row_h - 1)],
                fill=COLORS["row_divider"],
                width=1,
            )
    return panel_top + panel_h + 6


def _draw_backend_pill(
    draw: ImageDraw.ImageDraw,
    backend: str,
    accent: tuple[int, int, int],
    x: int,
    y: int,
    font: ImageFont.ImageFont,
    *,
    pad_x: int = 12,
    pad_y: int = 8,
    radius: int = 7,
) -> None:
    backend_color = BACKEND_COLORS.get(backend, accent)
    bbox = draw.textbbox((0, 0), backend, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.rounded_rectangle(
        [x - pad_x, y - pad_y, x + tw + pad_x, y + th + pad_y],
        radius=radius,
        fill=backend_color,
    )
    draw.text((x, y), backend, fill=COLORS["pill_text"], font=font)


def _value_color(value: str, kind: str) -> tuple[int, int, int]:
    lower = value.lower()
    if kind == "bool" or lower in ("yes", "no"):
        return COLORS["yes"] if lower == "yes" else COLORS["no"]
    if kind == "highlight":
        return COLORS["highlight"]
    if kind == "mtp" and lower != "no":
        return COLORS["yes"]
    return COLORS["value"]


def _draw_colored_value(
    draw: ImageDraw.ImageDraw,
    value: str,
    kind: str,
    x: int,
    y: int,
    font: ImageFont.ImageFont,
) -> None:
    draw.text((x, y), value, fill=_value_color(value, kind), font=font)


def _featured_stats(meta: AppCardMeta) -> list[tuple[str, str, str]]:
    """Return (label, value, kind) rows for featured banner. kind: text|bool|highlight|mtp."""
    stats: list[tuple[str, str, str]] = []
    if meta.context != "—":
        stats.append(("context", meta.context, "text"))
    if meta.speed != "—":
        stats.append(("speed", meta.speed, "highlight"))
    elif meta.hero_value and meta.hero_value not in ("—", ""):
        stats.append(("throughput", meta.hero_value, "text"))
    if meta.vram:
        stats.append(("vram", meta.vram, "text"))
    if meta.size_label:
        stats.append(("model size", meta.size_label, "text"))
    stats.append(("vision", "yes" if meta.vision else "no", "bool"))
    stats.append(("tool calling", "yes" if meta.tool_calling else "no", "bool"))
    if meta.mtp_enabled:
        stats.append(("MTP", meta.mtp_accept or "yes", "mtp"))
    else:
        stats.append(("MTP", "no", "bool"))
    stats.append(("shared API", "yes" if meta.shared_entrance else "no", "bool"))
    return stats


def _draw_meta_pills(
    draw: ImageDraw.ImageDraw,
    labels: list[str],
    x: int,
    y: int,
    max_width: int,
    font: ImageFont.ImageFont,
    accent: tuple[int, int, int],
    *,
    pill_h: int = 36,
    gap: int = 12,
    pad_x: int = 16,
) -> int:
    cx = x
    for label in labels:
        if not label:
            continue
        text = label.replace("_", " ")
        bbox = draw.textbbox((0, 0), text, font=font)
        tw = bbox[2] - bbox[0]
        pill_w = tw + pad_x * 2
        if cx + pill_w > x + max_width:
            break
        draw.rounded_rectangle(
            [cx, y, cx + pill_w, y + pill_h],
            radius=12,
            fill=(accent[0] // 4 + 20, accent[1] // 4 + 20, accent[2] // 4 + 30),
            outline=accent,
            width=2,
        )
        draw.text((cx + pad_x, y + 8), text, fill=COLORS["title"], font=font)
        cx += pill_w + gap
    return y + pill_h + 10 if labels else y


def _draw_featured_spec_grid(
    draw: ImageDraw.ImageDraw,
    meta: AppCardMeta,
    x: int,
    y: int,
    width: int,
    label_font: ImageFont.ImageFont,
    value_font: ImageFont.ImageFont,
) -> int:
    stats = _featured_stats(meta)
    if not stats:
        return y

    row_chunks = [stats[i : i + 4] for i in range(0, len(stats), 4)]
    cell_h = 132
    panel_h = len(row_chunks) * cell_h + 28
    draw.rounded_rectangle(
        [x, y, x + width, y + panel_h],
        radius=20,
        fill=(22, 32, 52),
        outline=COLORS["row_divider"],
        width=2,
    )

    for row_idx, row_stats in enumerate(row_chunks):
        cols = len(row_stats)
        cell_w = width // cols
        cy = y + 18 + row_idx * cell_h
        for col, (label, value, kind) in enumerate(row_stats):
            cx = x + col * cell_w + 28
            draw.text((cx, cy), label, fill=COLORS["label"], font=label_font)
            _draw_colored_value(draw, value, kind, cx, cy + 42, value_font)
            if col < cols - 1:
                divider_x = x + (col + 1) * cell_w
                draw.line(
                    [(divider_x, cy - 4), (divider_x, cy + cell_h - 20)],
                    fill=COLORS["row_divider"],
                    width=2,
                )
        if row_idx > 0:
            line_y = y + row_idx * cell_h + 8
            draw.line([(x + 20, line_y), (x + width - 20, line_y)], fill=COLORS["row_divider"], width=2)

    return y + panel_h


def _draw_backend_pill_right(
    draw: ImageDraw.ImageDraw,
    backend: str,
    accent: tuple[int, int, int],
    right_x: int,
    y: int,
    font: ImageFont.ImageFont,
    *,
    pad_x: int = 24,
    pad_y: int = 16,
    radius: int = 18,
) -> None:
    backend_color = BACKEND_COLORS.get(backend, accent)
    bbox = draw.textbbox((0, 0), backend, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x = right_x - tw - pad_x * 2
    draw.rounded_rectangle(
        [x, y - pad_y, right_x, y + th + pad_y],
        radius=radius,
        fill=backend_color,
    )
    draw.text((x + pad_x, y), backend, fill=COLORS["pill_text"], font=font)


def render_icon(meta: AppCardMeta) -> Image.Image:
    accent = meta.accent
    img = Image.new("RGB", (ICON_SIZE, ICON_SIZE), COLORS["bg"])
    draw = ImageDraw.Draw(img)
    _draw_gradient(img, draw, accent)
    draw.rectangle([0, 0, ICON_SIZE, 6], fill=accent)

    pad = 11
    inner_w = ICON_SIZE - 2 * pad
    pill_font = _load_font(10, bold=True)
    title_font = _load_font(20, bold=True)
    label_font = _load_font(12)
    value_font = _load_font(12, bold=True)
    backend_font = _load_font(13, bold=True)
    footer_font = _load_font(11)

    y = 11
    y = _draw_pipeline_pills(draw, meta.pipelines, pad, y, inner_w, pill_font)
    for line in _wrap_title(draw, meta.title, title_font, inner_w):
        draw.text((pad, y), line, fill=COLORS["title"], font=title_font)
        box = draw.textbbox((pad, y), line, font=title_font)
        y = box[3] + 2
    y += 3
    y = _draw_spec_table_vertical(draw, meta, pad, y, inner_w, label_font, value_font)
    _draw_backend_pill(draw, meta.backend, accent, pad, ICON_SIZE - 48, backend_font, pad_x=6, pad_y=4, radius=7)
    draw.text((pad, ICON_SIZE - 19), "Olares One", fill=COLORS["muted"], font=footer_font)
    return img


def render_featured(meta: AppCardMeta) -> Image.Image:
    accent = meta.accent
    img = Image.new("RGB", (FEATURED_W, FEATURED_H), COLORS["bg"])
    draw = ImageDraw.Draw(img)
    _draw_gradient(img, draw, accent)
    draw.rectangle([0, 0, FEATURED_W, 10], fill=accent)

    pad = 56
    content_w = FEATURED_W - pad * 2
    right_x = FEATURED_W - pad
    pill_font = _load_font(26, bold=True)
    title_font = _load_font(96, bold=True)
    desc_font = _load_font(30)
    meta_font = _load_font(24, bold=True)
    label_font = _load_font(28)
    value_font = _load_font(44, bold=True)
    backend_font = _load_font(42, bold=True)
    stack_font = _load_font(24)
    footer_font = _load_font(32)

    y = 42
    _draw_backend_pill_right(draw, meta.backend, accent, right_x, y + 4, backend_font)
    pipeline_max = content_w - 560
    y = _draw_pipeline_pills(
        draw,
        meta.pipelines,
        pad,
        y,
        max(pipeline_max, content_w // 2),
        pill_font,
        pill_h=52,
        gap=16,
        pad_x=20,
        radius=16,
        text_y_offset=12,
        max_pills=3,
    )

    for line in _wrap_title(draw, meta.title, title_font, content_w):
        draw.text((pad, y), line, fill=COLORS["title"], font=title_font)
        box = draw.textbbox((pad, y), line, font=title_font)
        y = box[3] + 6
    y += 8

    if meta.description:
        draw.text((pad, y), meta.description, fill=COLORS["muted"], font=desc_font)
        y += 40

    if meta.hero_label:
        draw.text((pad, y), meta.hero_label, fill=COLORS["highlight"], font=desc_font)
        y += 38

    meta_pills = [p for p in [meta.badge, meta.size_label, *meta.categories] if p]
    if meta_pills:
        y = _draw_meta_pills(draw, meta_pills, pad, y, content_w, meta_font, accent, pill_h=42, gap=14)
        y += 6

    y = _draw_featured_spec_grid(draw, meta, pad, y, content_w, label_font, value_font)
    y += 18

    if meta.stack:
        stack_text = _truncate(meta.stack, 132)
        draw.text((pad, y), stack_text, fill=COLORS["muted"], font=stack_font)
        y += 34

    draw.text((pad, FEATURED_H - 52), "Optimized for Olares One · RTX 5090M · 96 GB", fill=COLORS["muted"], font=footer_font)
    olares_bbox = draw.textbbox((0, 0), "Olares One", font=footer_font)
    draw.text(
        (right_x - (olares_bbox[2] - olares_bbox[0]), FEATURED_H - 52),
        "Olares One",
        fill=COLORS["title"],
        font=footer_font,
    )

    return img


def _cleanup_stale_icons() -> None:
    for stale in ICONS_DIR.glob("*-256.png"):
        stale.unlink(missing_ok=True)


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    ICONS_DIR.mkdir(parents=True, exist_ok=True)
    FEATURED_DIR.mkdir(parents=True, exist_ok=True)
    _cleanup_stale_icons()

    meta_path = ROOT / "scripts" / "apps-icons.json"
    prev_by_id: dict[str, dict] = {}
    if meta_path.exists():
        try:
            prev_by_id = {row["id"]: row for row in json.loads(meta_path.read_text(encoding="utf-8"))}
        except (json.JSONDecodeError, KeyError, TypeError):
            prev_by_id = {}

    manifest: list[dict] = []
    for app_id in sorted(APPS):
        meta = _build_meta(app_id)
        icon_path = ICONS_DIR / f"{app_id}.png"
        featured_path = FEATURED_DIR / f"{app_id}.png"
        render_icon(meta).save(icon_path, format="PNG", optimize=True)
        render_featured(meta).save(featured_path, format="PNG", optimize=True)
        icon_hash = _file_hash(icon_path)
        featured_hash = _file_hash(featured_path)
        row = meta.to_json(icon_hash=icon_hash, featured_hash=featured_hash)
        prev = prev_by_id.get(app_id, {})
        if prev.get("icon_hash") == icon_hash and prev.get("icon_url"):
            row["icon_url"] = prev["icon_url"]
        if prev.get("featured_hash") == featured_hash and prev.get("featured_url"):
            row["featured_url"] = prev["featured_url"]
        manifest.append(row)
        pipes = ", ".join(meta.pipelines)
        print(
            f"Wrote {icon_path.name} ({icon_path.stat().st_size // 1024}KB) "
            f"+ featured/{featured_path.name} ({featured_path.stat().st_size // 1024}KB) "
            f"[{pipes}] ctx={meta.context} speed={meta.speed}"
        )

    meta_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {meta_path} ({len(manifest)} apps)")


if __name__ == "__main__":
    main()
