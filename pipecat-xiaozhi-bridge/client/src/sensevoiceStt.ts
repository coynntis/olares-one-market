import type { SenseVoiceMeta } from "./types";

export type { SenseVoiceMeta };

const LANGS = new Set(["zh", "en", "yue", "ja", "ko", "nospeech"]);
const EVENTS = new Set([
  "Speech",
  "BGM",
  "Laughter",
  "Applause",
  "Cough",
  "Sneeze",
  "Cry",
  "Breath",
]);

const TAG_RE = /<\|([^|]+)\|>/g;

const LANG_LABEL: Record<string, string> = {
  yue: "Cantonese",
  zh: "Mandarin",
  en: "English",
  ja: "Japanese",
  ko: "Korean",
  nospeech: "No speech",
};

export function parseSenseVoiceTranscript(raw: string): { text: string; meta: SenseVoiceMeta } {
  if (!raw.includes("<|")) {
    return { text: raw.trim(), meta: {} };
  }
  const tags: string[] = [];
  const text = raw.replace(TAG_RE, (_, tag: string) => {
    tags.push(tag);
    return "";
  }).trim();
  const meta: SenseVoiceMeta = {};
  const extra: string[] = [];
  for (const tag of tags) {
    if (LANGS.has(tag)) meta.language = tag;
    else if (tag.startsWith("EMO_")) meta.emotion = tag.slice(4).toLowerCase().replace(/_/g, " ");
    else if (tag === "withitn" || tag === "woitn") meta.itn = tag === "withitn";
    else if (EVENTS.has(tag)) meta.event = tag.toLowerCase();
    else extra.push(tag);
  }
  if (extra.length) meta.tags = extra;
  return { text: text || raw.trim(), meta };
}

export function senseVoiceLanguageLabel(code?: string): string | null {
  if (!code) return null;
  return LANG_LABEL[code] ?? code;
}

export function formatSenseVoiceMeta(meta?: SenseVoiceMeta): string[] {
  if (!meta) return [];
  const chips: string[] = [];
  const lang = senseVoiceLanguageLabel(meta.language);
  if (lang) chips.push(lang);
  if (meta.event && meta.event !== "speech") chips.push(meta.event);
  if (meta.emotion && meta.emotion !== "unknown") chips.push(meta.emotion);
  if (meta.itn) chips.push("normalized");
  return chips;
}

export function normalizeStoredSttText(
  text: string,
  meta?: Record<string, unknown>
): { text: string; sensevoice?: SenseVoiceMeta } {
  const sv = meta?.sensevoice;
  if (sv && typeof sv === "object") {
    return { text, sensevoice: sv as SenseVoiceMeta };
  }
  if (text.includes("<|")) {
    const parsed = parseSenseVoiceTranscript(text);
    return { text: parsed.text, sensevoice: Object.keys(parsed.meta).length ? parsed.meta : undefined };
  }
  return { text };
}
