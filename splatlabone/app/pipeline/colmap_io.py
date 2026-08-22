"""COLMAP sparse model read/write helpers."""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np

try:
    from scipy.spatial.transform import Rotation
except ImportError:  # pragma: no cover
    Rotation = None  # type: ignore


def write_stub_sparse(sparse_dir: Path, width: int = 1920, height: int = 1080) -> None:
    sparse_dir.mkdir(parents=True, exist_ok=True)
    (sparse_dir / "cameras.txt").write_text(
        f"# splatlab\n1 SIMPLE_PINHOLE {width} {height} 1000 {width / 2:.1f} {height / 2:.1f}\n"
    )
    (sparse_dir / "images.txt").write_text("# splatlab\n")
    (sparse_dir / "points3D.txt").write_text("# splatlab\n")


def _as_c2w_4x4(ext: np.ndarray) -> np.ndarray:
    ext = np.asarray(ext, dtype=np.float64)
    if ext.shape == (3, 4):
        out = np.eye(4, dtype=np.float64)
        out[:3, :4] = ext
        return out
    if ext.shape == (4, 4):
        return ext
    raise ValueError(f"unsupported extrinsic shape {ext.shape}")


def c2w_to_colmap_w2c(ext_c2w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert camera-from-world (OpenCV) to COLMAP world-to-camera quaternion + translation."""
    w2c = np.linalg.inv(_as_c2w_4x4(ext_c2w))
    r = w2c[:3, :3]
    t = w2c[:3, 3]
    if Rotation is None:
        # Fallback: identity quaternion if scipy missing (should not happen in image)
        return np.array([1.0, 0.0, 0.0, 0.0]), t
    q_xyzw = Rotation.from_matrix(r).as_quat()
    return np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]]), t


def _intrinsic_to_pinhole(k: np.ndarray) -> tuple[float, float, float, float, int, int]:
    k = np.asarray(k, dtype=np.float64)
    fx, fy = float(k[0, 0]), float(k[1, 1])
    cx, cy = float(k[0, 2]), float(k[1, 2])
    # Approximate image size from principal point (2x cx, 2x cy) when H/W unknown
    width = max(int(round(cx * 2)), 1)
    height = max(int(round(cy * 2)), 1)
    return fx, fy, cx, cy, width, height


def write_colmap_model(
    sparse_dir: Path,
    *,
    image_names: list[str],
    extrinsics_c2w: np.ndarray,
    intrinsics: np.ndarray,
    points3d: np.ndarray | None = None,
    points_rgb: np.ndarray | None = None,
    shared_camera: bool = False,
) -> None:
    """Write COLMAP text model from estimated cameras (+ optional point cloud)."""
    sparse_dir.mkdir(parents=True, exist_ok=True)
    n = len(image_names)
    if n == 0:
        raise ValueError("write_colmap_model: no images")

    ext = np.asarray(extrinsics_c2w, dtype=np.float64)
    if ext.ndim == 2:
        ext = ext[None, ...]
    if ext.shape[0] != n:
        raise ValueError(f"extrinsics count {ext.shape[0]} != images {n}")

    k_arr = np.asarray(intrinsics, dtype=np.float64)
    if k_arr.ndim == 2:
        k_arr = np.repeat(k_arr[None, ...], n, axis=0)
    if k_arr.shape[0] not in (1, n):
        raise ValueError(f"intrinsics count {k_arr.shape[0]} invalid for {n} images")

    cam_lines = ["# splatlab\n"]
    img_lines = ["# splatlab\n"]
    camera_ids: list[int] = []

    for i in range(n):
        k = k_arr[0 if shared_camera else i]
        fx, fy, cx, cy, width, height = _intrinsic_to_pinhole(k)
        cam_id = 1 if shared_camera else i + 1
        camera_ids.append(cam_id)
        if not shared_camera:
            cam_lines.append(
                f"{cam_id} PINHOLE {width} {height} {fx:.6f} {fy:.6f} {cx:.6f} {cy:.6f}\n"
            )

    if shared_camera:
        k = k_arr[0]
        fx, fy, cx, cy, width, height = _intrinsic_to_pinhole(k)
        cam_lines.append(
            f"1 PINHOLE {width} {height} {fx:.6f} {fy:.6f} {cx:.6f} {cy:.6f}\n"
        )

    for i, name in enumerate(image_names):
        q, t = c2w_to_colmap_w2c(ext[i])
        basename = Path(name).name
        img_lines.append(
            f"{i + 1} {q[0]:.8f} {q[1]:.8f} {q[2]:.8f} {q[3]:.8f} "
            f"{t[0]:.8f} {t[1]:.8f} {t[2]:.8f} {camera_ids[i]} {basename}\n\n"
        )

    (sparse_dir / "cameras.txt").write_text("".join(cam_lines))
    (sparse_dir / "images.txt").write_text("".join(img_lines))

    pts_lines = ["# splatlab\n"]
    if points3d is not None and len(points3d) > 0:
        pts = np.asarray(points3d, dtype=np.float64).reshape(-1, 3)
        colors = None
        if points_rgb is not None:
            colors = np.asarray(points_rgb).reshape(-1, 3)
        for pid, p in enumerate(pts, start=1):
            r, g, b = (128, 128, 128)
            if colors is not None and pid - 1 < len(colors):
                c = colors[pid - 1]
                r, g, b = int(c[0]), int(c[1]), int(c[2])
            pts_lines.append(
                f"{pid} {p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {r} {g} {b} 0.0\n"
            )
    (sparse_dir / "points3D.txt").write_text("".join(pts_lines))


def copy_colmap_sparse(src: Path, dst: Path) -> None:
    """Copy COLMAP sparse/0 text or binary model into workspace."""
    dst.mkdir(parents=True, exist_ok=True)
    if (src / "sparse" / "0").is_dir():
        src = src / "sparse" / "0"
    elif src.name != "0" and (src / "0").is_dir():
        src = src / "0"
    for name in ("cameras.txt", "images.txt", "points3D.txt", "cameras.bin", "images.bin", "points3D.bin"):
        f = src / name
        if f.is_file():
            shutil.copy2(f, dst / name)


def unproject_depth_points(
    depth: np.ndarray,
    intrinsic: np.ndarray,
    extrinsic_c2w: np.ndarray,
    *,
    conf: np.ndarray | None = None,
    conf_percentile: float = 40.0,
    max_points: int = 100_000,
) -> tuple[np.ndarray, np.ndarray]:
    """Unproject depth map to world points (OpenCV cam convention)."""
    d = np.asarray(depth, dtype=np.float64).squeeze()
    k = np.asarray(intrinsic, dtype=np.float64)
    c2w = _as_c2w_4x4(extrinsic_c2w)
    h, w = d.shape[:2]
    ys, xs = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    valid = d > 1e-4
    if conf is not None:
        c = np.asarray(conf).squeeze()
        if c.shape == d.shape:
            thr = np.percentile(c[valid], conf_percentile) if valid.any() else 0
            valid &= c >= thr
    xs, ys, zs = xs[valid], ys[valid], d[valid]
    if xs.size == 0:
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=np.uint8)
    fx, fy, cx, cy = k[0, 0], k[1, 1], k[0, 2], k[1, 2]
    x_cam = (xs - cx) * zs / fx
    y_cam = (ys - cy) * zs / fy
    pts_cam = np.stack([x_cam, y_cam, zs, np.ones_like(zs)], axis=-1)
    pts_world = (c2w @ pts_cam.T).T[:, :3]
    if pts_world.shape[0] > max_points:
        idx = np.linspace(0, pts_world.shape[0] - 1, max_points, dtype=int)
        pts_world = pts_world[idx]
    rgb = np.full((pts_world.shape[0], 3), 180, dtype=np.uint8)
    return pts_world, rgb


def _read_colmap_points3d_txt(path: Path) -> tuple[np.ndarray, np.ndarray] | None:
    if not path.is_file():
        return None
    pts: list[list[float]] = []
    colors: list[list[int]] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 7:
            continue
        try:
            pts.append([float(parts[1]), float(parts[2]), float(parts[3])])
            colors.append([int(parts[4]), int(parts[5]), int(parts[6])])
        except ValueError:
            continue
    if not pts:
        return None
    return np.asarray(pts, dtype=np.float64), np.asarray(colors, dtype=np.uint8)


def write_points_ply(path: Path, points: np.ndarray, colors: np.ndarray | None = None) -> None:
    """Write ASCII PLY point cloud."""
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = colors
    if cols is None:
        cols = np.full((pts.shape[0], 3), 180, dtype=np.uint8)
    cols = np.asarray(cols).reshape(-1, 3)
    n = pts.shape[0]
    lines = [
        "ply",
        "format ascii 1.0",
        f"element vertex {n}",
        "property float x",
        "property float y",
        "property float z",
        "property uchar red",
        "property uchar green",
        "property uchar blue",
        "end_header",
    ]
    body = [
        f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {int(c[0])} {int(c[1])} {int(c[2])}"
        for p, c in zip(pts, cols)
    ]
    path.write_text("\n".join(lines + body) + "\n")


def export_sparse_pointcloud_ply(sparse_dir: Path, out_ply: Path, *, max_points: int = 500_000) -> bool:
    """Export COLMAP sparse points3D to viewable PLY."""
    sparse_dir = sparse_dir if sparse_dir.name == "0" else sparse_dir / "0" if (sparse_dir / "0").is_dir() else sparse_dir
    parsed = _read_colmap_points3d_txt(sparse_dir / "points3D.txt")
    if parsed is None:
        return False
    pts, colors = parsed
    if pts.shape[0] > max_points:
        idx = np.linspace(0, pts.shape[0] - 1, max_points, dtype=int)
        pts, colors = pts[idx], colors[idx]
    write_points_ply(out_ply, pts, colors)
    return True


# Deprecated — kept for callers migrating; raises if used accidentally
def write_sparse_from_poses(*args, **kwargs):  # type: ignore
    raise RuntimeError("write_sparse_from_poses is removed — use write_colmap_model with real poses")
