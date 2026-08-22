#!/usr/bin/env python3
"""CLI entry for /opt/dense-sfm/run_matching.py (external dense_sfm fallback)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dense_sfm import run_matching


def main() -> None:
    p = argparse.ArgumentParser(description="Dense-SfM LoFTR matching → COLMAP DB")
    p.add_argument("--workspace", required=True)
    p.add_argument("--images", required=True)
    p.add_argument("--output", default="", help="unused; workspace holds database.db")
    args = p.parse_args()
    db = run_matching(args.workspace, args.images)
    print(f"ok database={db}")


if __name__ == "__main__":
    main()
