#!/usr/bin/env python3
"""Stdlib-only soft-ready HTTP for SNDR / vLLM charts.

Binds :SERVER_PORT immediately so Olares/K8s pass the install gate while
uv → site-packages + sndr.apply + model download run in the background.
"""

from __future__ import annotations

import json
import os
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PORT = int(os.environ.get("SERVER_PORT", "8000"))
APP_NAME = os.environ.get("SNDR_APP", "sndr")
PHASE_FILE = os.environ.get(
    "SNDR_BOOT_PHASE_FILE",
    f"/shared-models/sndr/{APP_NAME}/.boot-phase",
)
ATTEMPTS_FILE = os.environ.get(
    "SNDR_BOOT_ATTEMPTS_FILE",
    f"/shared-models/sndr/{APP_NAME}/bootstrap.log",
)
MODEL_ALIAS = os.environ.get("MODEL_ALIAS", APP_NAME)
START = time.time()


def _tail_attempts(n: int = 16) -> list[str]:
    if not os.path.isfile(ATTEMPTS_FILE):
        return []
    try:
        with open(ATTEMPTS_FILE, encoding="utf-8", errors="replace") as fh:
            lines = [ln.rstrip() for ln in fh if ln.strip()]
        return lines[-n:]
    except Exception as exc:
        return [f"attempts_read_error:{exc}"]


def _phase() -> dict:
    status = "installing"
    detail = "starting"
    if os.path.isfile(PHASE_FILE):
        try:
            with open(PHASE_FILE, encoding="utf-8") as fh:
                raw = fh.read().strip()
            if raw:
                if raw.startswith("{"):
                    data = json.loads(raw)
                    data.setdefault("elapsed_s", round(time.time() - START, 1))
                    data.setdefault("ready", False)
                    data.setdefault("attempts_tail", _tail_attempts())
                    return data
                status, _, detail = raw.partition(":")
                status = status.strip() or "installing"
                detail = detail.strip() or status
        except Exception as exc:
            detail = f"phase_read_error:{exc}"
    return {
        "status": status,
        "ready": False,
        "phase": status,
        "detail": detail,
        "elapsed_s": round(time.time() - START, 1),
        "app": APP_NAME,
        "attempts_tail": _tail_attempts(),
        "hint": (
            f"Deps via uv → /shared-models/sndr/{APP_NAME}/site-packages (persists). "
            "Set UV_INDEX_URL for a PyPI mirror if downloads are slow. "
            f"See {ATTEMPTS_FILE} for bootstrap attempts."
        ),
    }


def _models_soft() -> dict:
    phase = _phase()
    return {
        "object": "list",
        "data": [
            {
                "id": MODEL_ALIAS,
                "object": "model",
                "created": int(START),
                "owned_by": "sndr-soft-ready",
                "permission": [],
                "root": MODEL_ALIAS,
                "parent": None,
                "status": "loading",
                "phase": phase.get("detail") or phase.get("phase"),
                "elapsed_s": phase.get("elapsed_s"),
            }
        ],
    }


# CSS as plain strings (single braces only) so Helm ConfigMap embed stays inert.
_CSS = (
    "body{font-family:system-ui,sans-serif;max-width:42rem;margin:3rem auto;padding:0 1rem;line-height:1.45}"
    "code{background:#f2f2f2;padding:.1rem .3rem;border-radius:4px}"
    ".muted{color:#666}"
    "pre{background:#111;color:#e8e8e8;padding:1rem;border-radius:8px;overflow:auto;font-size:12px}"
)

HTML_HEAD = (
    "<!doctype html>\n<html><head><meta charset=utf-8><meta http-equiv=refresh content=8>\n"
    f"<title>{APP_NAME} — installing</title>\n"
    f"<style>\n{_CSS}\n</style></head><body>\n"
    f"<h1>{APP_NAME}</h1>\n"
    "<p>Bootstrap in progress (uv site-packages + SNDR apply + model load). "
    "First boot can take a long time.</p>\n"
    f"<p>Site-packages: <code>/shared-models/sndr/{APP_NAME}/site-packages</code></p>\n"
    f"<p>Bootstrap log: <code>{ATTEMPTS_FILE}</code></p>\n"
    "<p class=muted>Page refreshes every 8s. API: <code>GET /health</code> · "
    "<code>GET /v1/models</code></p>\n"
    "<pre id=p>loading…</pre>\n<script>\n"
)

HTML_SCRIPT = (
    "fetch('/health').then(function(r){return r.json();}).then(function(j){"
    "document.getElementById('p').textContent=JSON.stringify(j,null,2);"
    "}).catch(function(e){document.getElementById('p').textContent=String(e);});"
)

HTML_TAIL = "\n</script>\n</body></html>\n"

HTML = HTML_HEAD + HTML_SCRIPT + HTML_TAIL


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt: str, *args) -> None:  # noqa: A003
        return

    def _send(self, code: int, body: bytes, content_type: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        path = self.path.split("?", 1)[0]
        if path in ("/health", "/health/", "/ready", "/v1/health"):
            body = json.dumps(_phase()).encode()
            self._send(200, body, "application/json")
            return
        if path in ("/v1/models", "/v1/models/"):
            body = json.dumps(_models_soft()).encode()
            self._send(200, body, "application/json")
            return
        if path.startswith("/ui") or path in ("/", "/docs", "/openapi.json"):
            self._send(200, HTML.encode(), "text/html; charset=utf-8")
            return
        # OpenAI-shaped soft error for chat while loading
        if path.startswith("/v1/"):
            phase = _phase()
            err = {
                "error": {
                    "message": (
                        f"Model still loading (phase={phase.get('detail') or phase.get('phase')}). "
                        f"See {ATTEMPTS_FILE}."
                    ),
                    "type": "server_error",
                    "param": None,
                    "code": "model_loading",
                }
            }
            self._send(503, json.dumps(err).encode(), "application/json")
            return
        self._send(200, json.dumps(_phase()).encode(), "application/json")

    def do_POST(self) -> None:  # noqa: N802
        self.do_GET()


def main() -> None:
    os.makedirs(os.path.dirname(PHASE_FILE) or ".", exist_ok=True)
    with open(PHASE_FILE, "w", encoding="utf-8") as fh:
        fh.write("installing:soft_ready_listening")
    httpd = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    print(f"[{APP_NAME}-soft-ready] listening 0.0.0.0:{PORT}", flush=True)
    httpd.serve_forever()


if __name__ == "__main__":
    main()
