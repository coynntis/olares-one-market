"""Thin Dense-SfM matching entry used by SplatLab (/opt/dense-sfm).

Primary path: hloc match_dense (LoFTR) → COLMAP database.db
Upstream CVPR DenseSfM-Refine (if present) lives alongside for refinement scripts.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

PathLike = Union[str, Path]


def run_matching(workspace: PathLike, images: PathLike) -> Path:
    """Match images with LoFTR via hloc; write workspace/database.db; return db path."""
    workspace = Path(workspace)
    images = Path(images)
    workspace.mkdir(parents=True, exist_ok=True)

    from hloc import match_dense, pairs_from_exhaustive
    from hloc.reconstruction import create_empty_db, get_image_ids, import_images
    from hloc.triangulation import (
        import_features,
        import_matches,
        estimation_and_geometric_verification,
    )
    import pycolmap

    feature_path = workspace / "feats-loftr.h5"
    match_path = workspace / "matches-loftr.h5"
    pairs_path = workspace / "pairs-exhaustive.txt"
    db = workspace / "database.db"

    image_files = sorted(
        p.name
        for p in images.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff"}
    )
    if not image_files:
        raise RuntimeError(f"no images in {images}")

    pairs_from_exhaustive.main(pairs_path, image_list=image_files)

    dense_conf = match_dense.confs.get("loftr") or match_dense.confs.get("loftr_ot")
    if dense_conf is None:
        # fall back to first conf
        dense_conf = next(iter(match_dense.confs.values()))

    match_dense.main(
        dense_conf,
        pairs_path,
        images,
        matches=match_path,
        features=feature_path,
        max_error=1,
        cell_size=1,
    )

    if db.exists():
        db.unlink()
    create_empty_db(db)
    import_images(images, db, pycolmap.CameraMode.SINGLE, image_list=image_files)
    image_ids = get_image_ids(db)
    import_features(image_ids, db, feature_path)
    import_matches(image_ids, db, pairs_path, match_path)
    estimation_and_geometric_verification(db, pairs_path)

    sentinel = workspace / ".matching_dense_done"
    sentinel.write_text("backend=dense_sfm_hloc_loftr\n")
    return db
