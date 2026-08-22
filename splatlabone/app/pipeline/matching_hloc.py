"""HLoc (SuperPoint + LightGlue) → COLMAP database for global_mapper / GLOMAP."""

from __future__ import annotations

from pathlib import Path
from typing import Callable


_HLOC_INSTALL_HINT = (
    "hloc not installed — clone https://github.com/cvg/Hierarchical-Localization "
    "and run: pip install -e /opt/hloc  (or pip install -e /path/to/Hierarchical-Localization)"
)


def _require_hloc():
    try:
        import hloc  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(_HLOC_INSTALL_HINT) from exc


def _link_or_copy(src: Path, dest: Path) -> None:
    if dest.exists() or dest.is_symlink():
        dest.unlink()
    try:
        dest.symlink_to(src.resolve())
    except OSError:
        import shutil

        shutil.copy2(src, dest)


def import_hloc_to_colmap_db(
    workspace: Path,
    images: Path,
    pairs: Path,
    features: Path,
    matches: Path,
    log: Callable[[str], None],
) -> Path:
    """Populate workspace/database.db from hloc feature/match HDF5 files."""
    import pycolmap
    from hloc.reconstruction import create_empty_db, get_image_ids, import_images
    from hloc.triangulation import (
        estimation_and_geometric_verification,
        import_features,
        import_matches,
    )

    db = workspace / "database.db"
    if db.exists():
        log(f"Removing existing database: {db}")
        db.unlink()

    log("Creating COLMAP database and importing images…")
    create_empty_db(db)
    import_images(images, db, pycolmap.CameraMode.AUTO)
    image_ids = get_image_ids(db)

    log("Importing hloc features and matches into COLMAP database…")
    with pycolmap.Database.open(db) as colmap_db:
        import_features(image_ids, colmap_db, features)
        import_matches(image_ids, colmap_db, pairs, matches)

    log("Running COLMAP geometric verification…")
    estimation_and_geometric_verification(db, pairs, verbose=False)

    if not db.is_file() or db.stat().st_size == 0:
        raise RuntimeError(f"COLMAP database import failed: {db}")

    log(f"COLMAP database ready ({db.stat().st_size} bytes): {db}")
    return db


def run_hloc_matching(workspace: Path, images: Path, log: Callable[[str], None]) -> Path:
    """Extract SuperPoint features, match with LightGlue, import into COLMAP database."""
    _require_hloc()

    from hloc import extract_features, match_features, pairs_from_exhaustive

    if not images.is_dir():
        raise FileNotFoundError(f"Image directory not found: {images}")

    image_files = [
        p
        for p in images.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff"}
    ]
    if not image_files:
        raise FileNotFoundError(f"No images found in {images}")

    hloc_dir = workspace / "hloc"
    hloc_dir.mkdir(parents=True, exist_ok=True)
    pairs_path = hloc_dir / "pairs.txt"
    features_link = hloc_dir / "features.h5"
    matches_link = hloc_dir / "matches.h5"

    feature_conf_name = "superpoint_aachen"
    matcher_conf_name = "superpoint+lightglue"
    if feature_conf_name not in extract_features.confs:
        raise RuntimeError(
            f"hloc feature config {feature_conf_name!r} not found; "
            f"available: {sorted(extract_features.confs)}"
        )
    if matcher_conf_name not in match_features.confs:
        raise RuntimeError(
            f"hloc matcher config {matcher_conf_name!r} not found; "
            f"available: {sorted(match_features.confs)}"
        )

    feature_conf = extract_features.confs[feature_conf_name]
    matcher_conf = match_features.confs[matcher_conf_name]

    log(f"hloc: extracting {feature_conf_name} features from {len(image_files)} images…")
    feature_path = extract_features.main(feature_conf, images, hloc_dir)
    _link_or_copy(feature_path, features_link)
    log(f"hloc: features → {feature_path}")

    log("hloc: building exhaustive image pairs…")
    pairs_from_exhaustive.main(pairs_path, features=feature_path)
    log(f"hloc: pairs → {pairs_path}")

    log(f"hloc: matching with {matcher_conf_name}…")
    match_path = match_features.main(matcher_conf, pairs_path, feature_path, hloc_dir)
    _link_or_copy(match_path, matches_link)
    log(f"hloc: matches → {match_path}")

    db = import_hloc_to_colmap_db(workspace, images, pairs_path, feature_path, match_path, log)

    sentinel = workspace / ".matching_hloc_done"
    sentinel.write_text(f"features={feature_path.name}\nmatches={match_path.name}\n")
    log(f"hloc matching complete — sentinel {sentinel}")

    return db
