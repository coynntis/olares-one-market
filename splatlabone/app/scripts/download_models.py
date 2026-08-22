#!/usr/bin/env python3
"""Prefetch geometry model weights (initContainer entrypoint)."""

from __future__ import annotations

import json
import logging
import os
import sys

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("download_models")

# Allow running from /app
sys.path.insert(0, "/app")

from pipeline.models.download import prefetch_all  # noqa: E402


def main() -> int:
    raw = os.environ.get("SPLATLAB_PREFETCH_MODELS", "vggt_omega,da3,lingbot,mast3r")
    keys = [k.strip() for k in raw.split(",") if k.strip()]
    log.info("prefetch models: %s", keys)
    results = prefetch_all(keys)
    print(json.dumps(results, indent=2))
    failed = [k for k, v in results.items() if str(v).startswith("error:")]
    if failed:
        log.warning("some prefetches failed: %s", failed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
