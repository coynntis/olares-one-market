"""Shared app state for HTTP API and MCP."""

from __future__ import annotations

import os
from pathlib import Path

from builder import BuildManager

DATA_DIR = Path(os.environ.get("DATA_DIR", "/data"))
STATIC_DIR = Path(os.environ.get("STATIC_DIR", str(Path(__file__).parent / "static")))

manager = BuildManager(DATA_DIR)
