"""Split streaming LLM text into speakable segments (xiaozhi-esp32-server rules)."""

from __future__ import annotations

import re

_PUNCT = ("。", "？", "?", "！", "!", "；", ";", "：", ":")
# Softer breaks — only used once enough chars buffered (avoids tiny first chunks).
_SOFT_PUNCT = ("，", "~", "、", ",", ".")
_OPEN = "<" + "think>"
_CLOSE = "</" + "think>"
_THINK_BLOCK_RE = re.compile(re.escape(_OPEN) + r".*?" + re.escape(_CLOSE), re.DOTALL | re.IGNORECASE)
_THINK_OPEN_RE = re.compile(re.escape(_OPEN) + r".*", re.DOTALL | re.IGNORECASE)
_TRAIL_PUNCT_RE = re.compile(r"^[\s。！？?!；;：:,，、~.]+|[\s。！？?!；;：:,，、~.]+$")
_DEFAULT_MIN_CHARS = 12
_CJK_RE = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf]")


def clean_for_tts(text: str) -> str:
    """Strip think blocks, emoji-ish noise, edge punctuation for TTS input."""
    if not text:
        return ""
    out = _THINK_BLOCK_RE.sub("", text)
    out = _THINK_OPEN_RE.sub("", out)
    out = _TRAIL_PUNCT_RE.sub("", out.strip())
    if out and all(c in "，。！？?!；;：:,，、~. " for c in out):
        return ""
    return out.strip()


def display_subtitle(text: str) -> str:
    """Subtitle text for sentence_start (keep readable, no think tags)."""
    return clean_for_tts(text) or text.strip()


def _char_len(text: str) -> int:
    return len(clean_for_tts(text) or text.strip())


def _is_cjk_heavy(text: str) -> bool:
    if not text:
        return False
    hits = len(_CJK_RE.findall(text))
    return hits >= max(1, len(text) // 4)


def _best_soft_cut(buf: str, max_chars: int) -> int:
    """Index to cut at (inclusive), never mid-Latin-word when possible. -1 = no cut."""
    if _char_len(buf) < max_chars:
        return -1
    # Work in raw string indices up to ~max_chars codepoints.
    window = buf[:max_chars]
    min_keep = max(4, max_chars // 3)

    cut = -1
    for punct in _SOFT_PUNCT + _PUNCT:
        pos = window.rfind(punct)
        if pos >= min_keep and (cut == -1 or pos > cut):
            cut = pos
    if cut >= min_keep:
        return cut

    # Prefer whitespace — never split "somew|here".
    space = window.rfind(" ")
    if space >= min_keep:
        return space

    if _is_cjk_heavy(window):
        # CJK syllable boundary ≈ char boundary.
        return max_chars - 1

    # Latin with no space (URL / camelCase): hold for more text rather than mid-word.
    return -1


class SentenceSegmenter:
    """Accumulate streamed tokens; emit segments at punctuation boundaries."""

    def __init__(self, *, min_chars: int = _DEFAULT_MIN_CHARS) -> None:
        self._buff = ""
        self._processed = 0
        self._first = True
        self._flush_on_end = False
        self._min_chars = max(1, int(min_chars))
        self._hold = ""

    def feed(self, text: str) -> None:
        if text:
            self._buff += text

    def mark_end(self) -> None:
        self._flush_on_end = True

    def _emit(self, raw: str) -> str | None:
        seg = clean_for_tts(raw)
        if not seg:
            return None
        if self._hold:
            seg = clean_for_tts(self._hold + seg) or (self._hold + seg).strip()
            self._hold = ""
        return seg if seg else None

    def _defer_short(self, raw: str, seg: str) -> str | None:
        if self._flush_on_end or _char_len(seg) >= self._min_chars:
            return seg
        self._hold = (self._hold + raw).strip()
        return None

    def pop_segment(self) -> str | None:
        while True:
            current = self._buff[self._processed :]
            if not current:
                if self._flush_on_end:
                    self._flush_on_end = False
                    if self._hold:
                        tail = clean_for_tts(self._hold)
                        self._hold = ""
                        return tail if tail else None
                    return None
                return None

            puncs: tuple[str, ...]
            if self._first and _char_len(self._hold + current) >= self._min_chars:
                puncs = _SOFT_PUNCT + _PUNCT
            else:
                puncs = _PUNCT

            cut_at = -1
            for punct in puncs:
                pos = current.find(punct)
                if pos != -1 and (cut_at == -1 or pos < cut_at):
                    cut_at = pos

            if cut_at == -1:
                if self._flush_on_end:
                    raw = current
                    self._processed += len(raw)
                    self._first = True
                    self._flush_on_end = False
                    seg = self._emit(raw)
                    if seg:
                        return seg
                    continue
                return None

            raw = current[: cut_at + 1]
            seg = self._emit(raw)
            self._processed += len(raw)
            if self._first:
                self._first = False
            if not seg:
                continue
            deferred = self._defer_short(raw, seg)
            if deferred:
                return deferred

    def flush(self) -> str | None:
        self.mark_end()
        return self.pop_segment()


def split_long_segment(text: str, *, max_chars: int = 40) -> list[str]:
    """Further split a long speakable unit — prefer punct/space, never mid-word."""
    cleaned = clean_for_tts(text)
    if not cleaned:
        return []
    if _char_len(cleaned) <= max_chars:
        return [cleaned]

    parts: list[str] = []
    buf = ""
    for ch in cleaned:
        buf += ch
        if ch in _SOFT_PUNCT + _PUNCT and _char_len(buf) >= max(4, max_chars // 3):
            piece = clean_for_tts(buf)
            if piece:
                parts.append(piece)
            buf = ""
            continue
        if _char_len(buf) < max_chars:
            continue
        cut = _best_soft_cut(buf, max_chars)
        if cut < 0:
            # No safe break yet — keep buffering (avoid mid-word).
            if _char_len(buf) < max_chars * 2:
                continue
            # Extreme: force at last space anywhere, else hard cut.
            space = buf.rfind(" ")
            cut = space if space > 0 else max_chars - 1
        piece = clean_for_tts(buf[: cut + 1])
        rest = buf[cut + 1 :]
        if piece:
            parts.append(piece)
        buf = rest
    tail = clean_for_tts(buf)
    if tail:
        parts.append(tail)
    return parts or [cleaned]


def expand_segments_for_streaming(
    segments: list[str],
    *,
    max_chars: int = 40,
    first_max_chars: int | None = None,
) -> list[str]:
    """Flatten segments into short TTS chunks for play-while-synthesize.

    first_max_chars (if set) only applies to the first emitted chunk so the cold
    post-LLM OmniVoice hit still yields hearable speech ASAP.
    """
    out: list[str] = []
    first_limit = first_max_chars if first_max_chars is not None else max_chars
    for seg in segments:
        if not out:
            pieces = split_long_segment(seg, max_chars=max(4, first_limit))
            if not pieces:
                continue
            out.append(pieces[0])
            for rest in pieces[1:]:
                out.extend(split_long_segment(rest, max_chars=max_chars))
        else:
            out.extend(split_long_segment(seg, max_chars=max_chars))
    return out
