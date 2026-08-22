#!/usr/bin/env python3
"""Soft-fail flash_attn → SDPA for SenseNova-Vision BAGEL navit (sm_120 Blackwell).

Replaces hard `from flash_attn import flash_attn_varlen_func` with an optional import
and an SDPA-based varlen fallback. Does NOT pin flash_attn==2.5.8.
"""

from __future__ import annotations

import argparse
import pathlib
import re
import sys

MARKER = "# === OLARES flash_attn soft-fail → SDPA ==="

PATCH_BLOCK = r'''
# === OLARES flash_attn soft-fail → SDPA ===
try:
    from flash_attn import flash_attn_varlen_func as _flash_attn_varlen_func
    _FLASH_ATTN_OK = True
except Exception as _flash_exc:  # noqa: BLE001
    _FLASH_ATTN_OK = False
    _flash_attn_varlen_func = None
    print(f"[olares] flash_attn unavailable ({_flash_exc}); using SDPA varlen fallback")

def flash_attn_varlen_func(
    q,
    k,
    v,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q=None,
    max_seqlen_k=None,
    causal=True,
    **kwargs,
):
    if _FLASH_ATTN_OK:
        return _flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            causal=causal,
            **kwargs,
        )
    # Packed varlen → per-sample SDPA (GQA-aware).
    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from torch.nn.functional import scaled_dot_product_attention

    if q.ndim != 3 or k.ndim != 3 or v.ndim != 3:
        raise RuntimeError(f"expected (T,H,D) tensors, got q={tuple(q.shape)}")
    cu_q = cu_seqlens_q.tolist()
    cu_k = cu_seqlens_k.tolist()
    outs = []
    nheads_q = q.shape[1]
    nheads_k = k.shape[1]
    group = max(1, nheads_q // max(1, nheads_k))
    for i in range(len(cu_q) - 1):
        qs, qe = int(cu_q[i]), int(cu_q[i + 1])
        ks, ke = int(cu_k[i]), int(cu_k[i + 1])
        qi = q[qs:qe].transpose(0, 1).unsqueeze(0)  # (1, Hq, Lq, D)
        ki = k[ks:ke].transpose(0, 1).unsqueeze(0)
        vi = v[ks:ke].transpose(0, 1).unsqueeze(0)
        if group > 1 and nheads_k != nheads_q:
            ki = ki.repeat_interleave(group, dim=1)
            vi = vi.repeat_interleave(group, dim=1)
        lq, lk = qi.shape[2], ki.shape[2]
        attn_mask = None
        use_causal = bool(causal) and lq == lk
        if causal and lq != lk:
            # Align queries to end of key sequence (flash_attn varlen convention).
            device = qi.device
            dtype = torch.bfloat16
            row = torch.arange(lq, device=device)[:, None] + (lk - lq)
            col = torch.arange(lk, device=device)[None, :]
            allow = col <= row
            attn_mask = torch.zeros((lq, lk), device=device, dtype=dtype)
            attn_mask = attn_mask.masked_fill(~allow, torch.finfo(dtype).min)
        with sdpa_kernel(backends=[SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]):
            out = scaled_dot_product_attention(
                qi.to(torch.bfloat16),
                ki.to(torch.bfloat16),
                vi.to(torch.bfloat16),
                attn_mask=attn_mask,
                is_causal=use_causal,
            )
        outs.append(out.squeeze(0).transpose(0, 1).to(q.dtype))  # (Lq, Hq, D)
    return torch.cat(outs, dim=0)
# === end OLARES flash_attn soft-fail ===
'''


def patch_file(navit_path: pathlib.Path) -> bool:
    text = navit_path.read_text(encoding="utf-8")
    if MARKER in text:
        print(f"[patch_flash_sdpa] already patched: {navit_path}")
        return False

    pattern = re.compile(
        r"^from flash_attn import flash_attn_varlen_func\s*$",
        re.MULTILINE,
    )
    if not pattern.search(text):
        raise SystemExit(
            f"[patch_flash_sdpa] expected hard flash_attn import not found in {navit_path}"
        )

    new_text = pattern.sub(PATCH_BLOCK.strip() + "\n", text, count=1)
    navit_path.write_text(new_text, encoding="utf-8")
    print(f"[patch_flash_sdpa] patched: {navit_path}")
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "repo_root",
        type=pathlib.Path,
        help="Path to cloned SenseNova-Vision (or ConsistCompose) repo root",
    )
    args = ap.parse_args()
    candidates = [
        args.repo_root / "modeling" / "bagel" / "qwen2_navit.py",
        args.repo_root / "modeling" / "bagel" / "siglip_navit.py",
        args.repo_root
        / "consist_compose"
        / "bagel_utils"
        / "modeling"
        / "bagel"
        / "qwen2_navit.py",
    ]
    found = [p for p in candidates if p.is_file()]
    if not found:
        raise SystemExit(f"[patch_flash_sdpa] no navit files under {args.repo_root}")
    for navit in found:
        patch_file(navit)


if __name__ == "__main__":
    main()
    sys.exit(0)
