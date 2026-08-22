#!/usr/bin/env python3
"""Stdlib-only soft-ready HTTP for SenseNova Vision One.

Binds :SERVER_PORT immediately so Olares/K8s pass the install gate while
uv → site-packages + HF model download run in the background.

IMPORTANT: Do not put raw double-brace sequences in this file — Helm parses
ConfigMap data as templates (see krea soft_ready: plain CSS braces, not f-string doubles).
"""

from __future__ import annotations

import json
import os
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PORT = int(os.environ.get("SERVER_PORT", "7860"))
APP_NAME = os.environ.get("APP_NAME", "sensenovavisionone")
PHASE_FILE = os.environ.get(
    "BOOT_PHASE_FILE",
    "/workspace/{}/.boot-phase".format(APP_NAME),
)
ATTEMPTS_FILE = os.environ.get(
    "BOOT_ATTEMPTS_FILE",
    "/workspace/{}/bootstrap.log".format(APP_NAME),
)
START = time.time()


def _tail_attempts(n: int = 12) -> list:
    if not os.path.isfile(ATTEMPTS_FILE):
        return []
    try:
        with open(ATTEMPTS_FILE, encoding="utf-8", errors="replace") as fh:
            lines = [ln.rstrip() for ln in fh if ln.strip()]
        return lines[-n:]
    except Exception as exc:
        return ["attempts_read_error:{}".format(exc)]


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
            detail = "phase_read_error:{}".format(exc)
    return {
        "status": status,
        "ready": False,
        "phase": status,
        "detail": detail,
        "elapsed_s": round(time.time() - START, 1),
        "app": APP_NAME,
        "attempts_tail": _tail_attempts(),
        "hint": (
            "Deps via uv → /workspace/{}/site-packages (persists). "
            "Set UV_INDEX_URL for a PyPI mirror if downloads are slow. "
            "See {} for bootstrap attempts."
        ).format(APP_NAME, ATTEMPTS_FILE),
    }


# Plain string — single braces for CSS/JS (Helm-safe). No Python f-string brace doubling.
HTML = """<!doctype html>
<html><head><meta charset=utf-8><meta http-equiv=refresh content=8>
<title>__APP__ — installing</title>
<style>
body{font-family:system-ui,sans-serif;max-width:42rem;margin:3rem auto;padding:0 1rem;line-height:1.45}
code{background:#f2f2f2;padding:.1rem .3rem;border-radius:4px}
.muted{color:#666}
pre{background:#111;color:#e8e8e8;padding:1rem;border-radius:8px;overflow:auto;font-size:12px}
</style></head><body>
<h1>__APP__</h1>
<p>Bootstrap in progress (uv site-packages + ~30GB model download). First boot can take a long time.</p>
<p>Site-packages: <code>/workspace/__APP__/site-packages</code></p>
<p>Bootstrap log: <code>__ATTEMPTS__</code></p>
<p class=muted>Page refreshes every 8s. API: <code>GET /health</code></p>
<pre id=p>loading…</pre>
<script>
fetch('/health').then(r=>r.json()).then(j=>{
  document.getElementById('p').textContent=JSON.stringify(j,null,2);
}).catch(e=>{document.getElementById('p').textContent=String(e);});
</script>
</body></html>
""".replace("__APP__", APP_NAME).replace("__ATTEMPTS__", ATTEMPTS_FILE)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):  # noqa: A003
        return

    def _send(self, code, body, content_type):
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):  # noqa: N802
        path = self.path.split("?", 1)[0]
        if path in ("/health", "/health/"):
            body = json.dumps(_phase()).encode()
            self._send(200, body, "application/json")
            return
        if path.startswith("/ui") or path in ("/", "/docs", "/openapi.json"):
            self._send(200, HTML.encode(), "text/html; charset=utf-8")
            return
        self._send(200, json.dumps(_phase()).encode(), "application/json")


def main():
    os.makedirs(os.path.dirname(PHASE_FILE) or ".", exist_ok=True)
    with open(PHASE_FILE, "w", encoding="utf-8") as fh:
        fh.write("installing:soft_ready_listening")
    httpd = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    print("[{}-soft-ready] listening 0.0.0.0:{}".format(APP_NAME, PORT), flush=True)
    httpd.serve_forever()


if __name__ == "__main__":
    main()
