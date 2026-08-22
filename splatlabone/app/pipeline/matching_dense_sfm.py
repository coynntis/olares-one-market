"""Dense matching (LoFTR via hloc match_dense) → COLMAP database."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path
from typing import Callable

from pipeline.matching_hloc import _link_or_copy, _require_hloc, import_hloc_to_colmap_db


def _run_external_dense_sfm(
    workspace: Path, images: Path, log: Callable[[str], None]
) -> Path:
    """Fallback when hloc match_dense is unavailable but Dense-SfM is installed."""
    dense_root = Path("/opt/dense-sfm")
    if dense_root.is_dir() and str(dense_root) not in sys.path:
        sys.path.insert(0, str(dense_root))

    if importlib.util.find_spec("dense_sfm") is None:
        raise RuntimeError(
            "dense_sfm matching unavailable — install hloc with match_dense (LoFTR) "
            "or clone Dense-SfM to /opt/dense-sfm"
        )

    import dense_sfm  # type: ignore[import-untyped]

    if hasattr(dense_sfm, "run_matching"):
        log("dense_sfm: running external dense_sfm.run_matching…")
        return Path(dense_sfm.run_matching(workspace, images))

    script = dense_root / "run_matching.py"
    if script.is_file():
        out_dir = workspace / "dense_sfm"
        out_dir.mkdir(parents=True, exist_ok=True)
        log(f"dense_sfm: running {script}…")
        proc = subprocess.run(
            [
                sys.executable,
                str(script),
                "--workspace",
                str(workspace),
                "--images",
                str(images),
                "--output",
                str(out_dir),
            ],
            capture_output=True,
            text=True,
        )
        if proc.stdout:
            for line in proc.stdout.strip().splitlines()[-20:]:
                log(line)
        if proc.returncode != 0:
            err = (proc.stderr or proc.stdout or "")[-500:]
            raise RuntimeError(f"dense_sfm script failed ({proc.returncode}): {err}")
        db = workspace / "database.db"
        if not db.is_file() or db.stat().st_size == 0:
            raise RuntimeError("dense_sfm script did not produce a non-empty database.db")
        return db

    raise RuntimeError(
        "dense_sfm package found but exposes no run_matching(workspace, images) "
        "and /opt/dense-sfm/run_matching.py is missing"
    )


def run_dense_sfm_matching(
    workspace: Path, images: Path, log: Callable[[str], None]
) -> Path:
    """LoFTR semi-dense matching via hloc, imported into COLMAP database."""
    _require_hloc()

    if not images.is_dir():
        raise FileNotFoundError(f"Image directory not found: {images}")

    image_files = [
        p
        for p in images.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff"}
    ]
    if not image_files:
        raise FileNotFoundError(f"No images found in {images}")

    dense_conf = None
    dense_conf_name = None
    try:
        from hloc import match_dense
        from hloc import pairs_from_exhaustive

        for name in ("loftr_aachen", "loftr", "loftr_superpoint"):
            if name in match_dense.confs:
                dense_conf_name = name
                dense_conf = match_dense.confs[name]
                break
    except ImportError:
        match_dense = None  # type: ignore[assignment]
        pairs_from_exhaustive = None  # type: ignore[assignment]

    if dense_conf is None:
        log("hloc match_dense (LoFTR) not available — trying external dense_sfm…")
        db = _run_external_dense_sfm(workspace, images, log)
        sentinel = workspace / ".matching_dense_done"
        sentinel.write_text("backend=external_dense_sfm\n")
        log(f"dense_sfm matching complete — sentinel {sentinel}")
        return db

    hloc_dir = workspace / "hloc"
    hloc_dir.mkdir(parents=True, exist_ok=True)
    pairs_path = hloc_dir / "pairs.txt"
    features_link = hloc_dir / "features.h5"
    matches_link = hloc_dir / "matches.h5"

    log(f"dense_sfm: LoFTR config {dense_conf_name!r} on {len(image_files)} images…")

    # Optional SuperPoint anchors improve stability for larger scenes.
    feature_ref = None
    try:
        from hloc import extract_features

        sp_conf = extract_features.confs.get("superpoint_aachen")
        if sp_conf is not None:
            log("dense_sfm: extracting SuperPoint anchor features…")
            feature_ref = extract_features.main(sp_conf, images, hloc_dir)
    except Exception as exc:
        log(f"dense_sfm: SuperPoint anchors skipped ({exc})")

    log("dense_sfm: building exhaustive image pairs…")
    if feature_ref is not None:
        pairs_from_exhaustive.main(pairs_path, features=feature_ref)
    else:
        pairs_from_exhaustive.main(pairs_path, image_list=sorted(p.name for p in image_files))

    log(f"dense_sfm: running match_dense ({dense_conf_name})…")
    feature_path, match_path = match_dense.main(
        dense_conf,
        pairs_path,
        images,
        export_dir=hloc_dir,
        features_ref=feature_ref,
        overwrite=False,
    )
    _link_or_copy(feature_path, features_link)
    _link_or_copy(match_path, matches_link)
    log(f"dense_sfm: features → {feature_path}")
    log(f"dense_sfm: matches → {match_path}")

    db = import_hloc_to_colmap_db(workspace, images, pairs_path, feature_path, match_path, log)

    sentinel = workspace / ".matching_dense_done"
    sentinel.write_text(
        f"backend=loftr\nconfig={dense_conf_name}\n"
        f"features={feature_path.name}\nmatches={match_path.name}\n"
    )
    log(f"dense_sfm matching complete — sentinel {sentinel}")
    return db
