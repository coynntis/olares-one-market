"""Named LLM profiles (endpoint + model + system prompt) stored in config.json."""

from __future__ import annotations

import time
import uuid
from typing import Any

from xiaozhi_bridge.config import apply_patch, load_settings


def _now_ms() -> int:
    return int(time.time() * 1000)


def _normalize_profile(raw: dict[str, Any]) -> dict[str, Any] | None:
    if not isinstance(raw, dict):
        return None
    pid = str(raw.get("id") or "").strip()
    name = str(raw.get("name") or "").strip()
    if not pid or not name:
        return None
    return {
        "id": pid,
        "name": name,
        "llm_base_url": str(raw.get("llm_base_url") or "").strip(),
        "llm_model": str(raw.get("llm_model") or "").strip(),
        "system_prompt": str(raw.get("system_prompt") or "").strip(),
        "created_at": int(raw.get("created_at") or _now_ms()),
    }


def list_profiles() -> tuple[list[dict[str, Any]], str]:
    cfg = load_settings()
    profiles: list[dict[str, Any]] = []
    for item in cfg.llm_profiles or []:
        norm = _normalize_profile(item)
        if norm:
            profiles.append(norm)
    profiles.sort(key=lambda p: p.get("created_at", 0), reverse=True)
    return profiles, str(cfg.active_llm_profile_id or "").strip()


def create_profile(
    *,
    name: str,
    llm_base_url: str = "",
    llm_model: str = "",
    system_prompt: str = "",
    set_active: bool = False,
) -> dict[str, Any]:
    label = name.strip()
    if not label:
        raise ValueError("name required")
    cfg = load_settings()
    profile = {
        "id": uuid.uuid4().hex[:12],
        "name": label,
        "llm_base_url": llm_base_url.strip(),
        "llm_model": llm_model.strip(),
        "system_prompt": system_prompt.strip(),
        "created_at": _now_ms(),
    }
    profiles = [_normalize_profile(p) for p in (cfg.llm_profiles or [])]
    profiles = [p for p in profiles if p]
    profiles.append(profile)
    patch: dict[str, Any] = {"llm_profiles": profiles}
    if set_active:
        patch.update(_profile_to_settings_patch(profile))
        patch["active_llm_profile_id"] = profile["id"]
    apply_patch(patch)
    return profile


def update_profile(
    profile_id: str,
    *,
    name: str | None = None,
    llm_base_url: str | None = None,
    llm_model: str | None = None,
    system_prompt: str | None = None,
) -> dict[str, Any]:
    pid = profile_id.strip()
    if not pid:
        raise ValueError("profile_id required")
    cfg = load_settings()
    found: dict[str, Any] | None = None
    profiles: list[dict[str, Any]] = []
    for item in cfg.llm_profiles or []:
        norm = _normalize_profile(item)
        if not norm:
            continue
        if norm["id"] == pid:
            if name is not None:
                label = name.strip()
                if not label:
                    raise ValueError("name required")
                norm["name"] = label
            if llm_base_url is not None:
                norm["llm_base_url"] = llm_base_url.strip()
            if llm_model is not None:
                norm["llm_model"] = llm_model.strip()
            if system_prompt is not None:
                norm["system_prompt"] = system_prompt.strip()
            found = norm
        profiles.append(norm)
    if not found:
        raise LookupError("profile not found")
    patch: dict[str, Any] = {"llm_profiles": profiles}
    if cfg.active_llm_profile_id == pid:
        patch.update(_profile_to_settings_patch(found))
    apply_patch(patch)
    return found


def delete_profile(profile_id: str) -> bool:
    pid = profile_id.strip()
    if not pid:
        return False
    cfg = load_settings()
    profiles = [
        p
        for p in (_normalize_profile(item) for item in (cfg.llm_profiles or []))
        if p and p["id"] != pid
    ]
    patch: dict[str, Any] = {"llm_profiles": profiles}
    if cfg.active_llm_profile_id == pid:
        patch["active_llm_profile_id"] = ""
    apply_patch(patch)
    return len(profiles) < len(cfg.llm_profiles or [])


def _profile_to_settings_patch(profile: dict[str, Any]) -> dict[str, str]:
    return {
        "llm_base_url": str(profile.get("llm_base_url") or "").strip(),
        "llm_model": str(profile.get("llm_model") or "").strip(),
        "system_prompt": str(profile.get("system_prompt") or "").strip(),
    }


def activate_profile(profile_id: str) -> dict[str, Any]:
    pid = profile_id.strip()
    if not pid:
        raise ValueError("profile_id required")
    profiles, _ = list_profiles()
    match = next((p for p in profiles if p["id"] == pid), None)
    if not match:
        raise LookupError("profile not found")
    apply_patch({**_profile_to_settings_patch(match), "active_llm_profile_id": pid})
    return match


def save_current_as_profile(name: str) -> dict[str, Any]:
    cfg = load_settings()
    return create_profile(
        name=name,
        llm_base_url=cfg.llm_base_url,
        llm_model=cfg.llm_model,
        system_prompt=cfg.system_prompt,
        set_active=True,
    )
