#!/usr/bin/env python3
"""Stdlib-only soft-ready HTTP server for Mage-Flow boot."""

from __future__ import annotations

import json
import os
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PORT = int(os.environ.get("SERVER_PORT", "7860"))
PHASE_FILE = os.environ.get(
    "MAGE_BOOT_PHASE_FILE",
    "/workspace/mageflowone/.boot-phase",
)
START = time.time()


def _phase() -> dict:
    status = "installing"
    detail = "starting"
    if os.path.isfile(PHASE_FILE):
        try:
            with open(PHASE_FILE, encoding="utf-8") as fh:
                raw = fh.read().strip()
            if raw:
                if raw.startswith("{"):
                    return json.loads(raw)
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
        "hint": "Deps via uv → /workspace/mageflowone/site-packages. Set UV_INDEX_URL for a PyPI mirror.",
    }


HTML = """<!doctype html>
<html><head><meta charset=utf-8><meta http-equiv=refresh content=8>
<title>Mage Flow One — installing</title>
<style>
body{font-family:system-ui,sans-serif;max-width:40rem;margin:3rem auto;padding:0 1rem;line-height:1.45}
code{background:#f2f2f2;padding:.1rem .3rem;border-radius:4px}
.muted{color:#666}
</style></head><body>
<h1>Mage Flow One</h1>
<p>Environment install in progress (first boot can take a while).</p>
<p>Site-packages land in <code>/workspace/mageflowone/site-packages</code> and are reused next start.</p>
<p class=muted>This page refreshes every 8s. API: <code>GET /health</code></p>
<pre id=p>loading…</pre>
<script>
fetch('/health').then(r=>r.json()).then(j=>{
  document.getElementById('p').textContent=JSON.stringify(j,null,2);
}).catch(e=>{document.getElementById('p').textContent=String(e);});
</script>
</body></html>
"""


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
        if path in ("/health", "/healthz"):
            body = json.dumps(_phase()).encode()
            self._send(200, body, "application/json")
            return
        self._send(200, HTML.encode(), "text/html; charset=utf-8")


def main() -> None:
    httpd = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    httpd.serve_forever()


if __name__ == "__main__":
    main()
